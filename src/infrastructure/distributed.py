from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from src.config.schema import Config, DistributedConfig

logger = logging.getLogger(__name__)


class DistributedSetup:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.dist_cfg = cfg.distributed
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.is_distributed = self.world_size > 1

        use_fsdp = cfg.distributed.strategy == "fsdp"
        if self.is_distributed or use_fsdp:
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
            if not self.is_distributed and use_fsdp and not torch.cuda.is_available():
                logger.info(
                    "Skipping process group initialization for local CPU fsdp strategy; "
                    "no distributed world_size and no CUDA available."
                )
            elif not dist.is_initialized():
                if not self.is_distributed:
                    os.environ.setdefault("MASTER_ADDR", "localhost")
                    os.environ.setdefault("MASTER_PORT", "29500")
                    os.environ.setdefault("WORLD_SIZE", "1")
                    os.environ.setdefault("RANK", "0")
                    os.environ.setdefault("LOCAL_RANK", "0")
                backend = "nccl" if torch.cuda.is_available() else "gloo"
                try:
                    dist.init_process_group(backend=backend)
                except RuntimeError as exc:
                    if not self.is_distributed and backend == "gloo":
                        logger.warning(
                            "Local process group init failed for gloo backend; continuing without distributed setup: %s",
                            exc,
                        )
                    else:
                        raise

    def is_main_process(self) -> bool:
        return self.rank == 0

    def get_training_args(
        self,
        output_dir: str,
        per_device_batch_size: int = 1,
        grad_accum_steps: int = 1,
    ) -> Dict[str, Any]:
        # Pull stage-aware defaults from the typed config instead of hardcoding
        pretrain = getattr(self.cfg.training, "pretrain", None)
        tcfg = pretrain if pretrain and getattr(pretrain, "learning_rate", None) is not None else self.cfg.training
        args: Dict[str, Any] = {
            "output_dir": output_dir,
            "per_device_train_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": grad_accum_steps,
            "learning_rate": float(getattr(tcfg, "learning_rate", 1e-4)),
            "weight_decay": float(getattr(self.cfg.training, "weight_decay", 0.1)),
            "warmup_steps": int(getattr(self.cfg.training, "warmup_steps", 200)),
            "logging_steps": int(getattr(self.cfg.training, "logging_steps", 10)),
            "save_steps": int(getattr(self.cfg.training, "save_steps", 500)),
            "save_total_limit": 10,
            "eval_strategy": "no",
            "save_strategy": "steps",
            "save_only_model": True,
            "ddp_find_unused_parameters": False,
            "report_to": "none",
            "remove_unused_columns": False,
            "ignore_data_skip": True,
            "dataloader_pin_memory": True,
            "accelerator_config": {"dispatch_batches": False},
        }

        if self.cfg.model.dtype == "bfloat16" and torch.cuda.is_available():
            args["bf16"] = True
        elif self.cfg.model.dtype == "float16" and torch.cuda.is_available():
            args["fp16"] = True

        if self.dist_cfg.strategy == "fsdp":
            fsdp_cfg = self.dist_cfg.fsdp
            if not torch.cuda.is_available():
                logger.warning(
                    "FSDP strategy requested but no CUDA devices were detected; falling back to non-FSDP training."
                )
            else:
                fsdp_args = [fsdp_cfg.sharding_strategy, "auto_wrap"]
                fsdp_config: Dict[str, Any] = {
                    "transformer_layer_cls_to_wrap": [fsdp_cfg.transformer_layer_cls],
                    "backward_prefetch": fsdp_cfg.backward_prefetch,
                    "forward_prefetch": fsdp_cfg.forward_prefetch,
                    "activation_checkpointing": fsdp_cfg.activation_checkpointing,
                    "use_orig_params": fsdp_cfg.use_orig_params,
                    "sync_module_states": fsdp_cfg.sync_module_states,
                    "limit_all_gathers": fsdp_cfg.limit_all_gathers,
                    "mixed_precision": fsdp_cfg.mixed_precision,
                }
                if fsdp_cfg.cpu_offload:
                    fsdp_args.append("offload")
                    fsdp_config["cpu_offload"] = True
                if not self.is_distributed:
                    fsdp_args.append("no_shard")
                args["fsdp"] = " ".join(fsdp_args)
                args["fsdp_config"] = fsdp_config
                if not args.get("bf16") and fsdp_cfg.mixed_precision == "bf16":
                    args["bf16"] = True
                if not args.get("fp16") and fsdp_cfg.mixed_precision == "fp16":
                    args["fp16"] = True
 
        elif self.dist_cfg.strategy == "deepspeed" and self.is_distributed:
            ds_cfg = self.dist_cfg.deepspeed
            if ds_cfg:
                use_bf16 = bool(args.get("bf16"))
                use_fp16 = bool(args.get("fp16"))
                ds_config = {
                    "zero_optimization": {
                        "stage": ds_cfg.zero_stage,
                        "offload_optimizer": {"device": ds_cfg.offload_optimizer} if ds_cfg.offload_optimizer else {},
                        "offload_param": {"device": ds_cfg.offload_params} if ds_cfg.offload_params else {},
                    },
                    "bf16": {"enabled": use_bf16},
                    "fp16": {"enabled": use_fp16 and not use_bf16},
                    "gradient_accumulation_steps": grad_accum_steps,
                    "gradient_clipping": 1.0,
                    "train_batch_size": self.world_size * per_device_batch_size * grad_accum_steps,
                    "train_micro_batch_size_per_gpu": per_device_batch_size,
                }
                ds_path = Path(output_dir) / "ds_config.json"
                ds_path.parent.mkdir(parents=True, exist_ok=True)
                ds_path.write_text(json.dumps(ds_config))
                args["deepspeed"] = str(ds_path)

        return args

    def auto_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.local_rank}")
        return torch.device("cpu")

    def num_gpus(self) -> int:
        return torch.cuda.device_count()

    def effective_batch_size(self, per_device: int, grad_accum: int) -> int:
        return per_device * self.world_size * grad_accum
