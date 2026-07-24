from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    MixtralConfig,
    MixtralForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.config.schema import Config, ModelArchitectureConfig

logger = logging.getLogger(__name__)


ARCH_CONFIG_MAP = {
    "llama": (LlamaConfig, LlamaForCausalLM),
    "mixtral": (MixtralConfig, MixtralForCausalLM),
    "qwen2_moe": (AutoConfig, AutoModelForCausalLM),
    "deepseek_v2": (AutoConfig, AutoModelForCausalLM),
    "nslt": (None, None),
}

ARCH_FSDP_LAYER_MAP = {
    "llama": "LlamaDecoderLayer",
    "mixtral": "MixtralDecoderLayer",
    "qwen2_moe": "Qwen2MoeDecoderLayer",
    "deepseek_v2": "DeepseekV2DecoderLayer",
    "nslt": "NSLTModel",
}


class ModelFactory:
    @staticmethod
    def build_model_config(
        vocab_size: int,
        arch: ModelArchitectureConfig,
        max_seq_length: int,
    ) -> Any:
        model_type = arch.model_type
        rope_scaling = None
        if arch.rope_scaling and arch.rope_scaling.factor > 1:
            rope_scaling = {
                "type": arch.rope_scaling.type,
                "factor": arch.rope_scaling.factor,
                "target_max_length": arch.rope_scaling.target_max_length,
                "original_max_position_embeddings": arch.rope_scaling.original_max_position_embeddings,
            }

        if model_type == "llama":
            config = LlamaConfig(
                vocab_size=vocab_size,
                hidden_size=arch.hidden_size,
                intermediate_size=arch.intermediate_size,
                num_hidden_layers=arch.num_hidden_layers,
                num_attention_heads=arch.num_attention_heads,
                num_key_value_heads=arch.num_key_value_heads,
                max_position_embeddings=arch.max_position_embeddings,
                rope_theta=arch.rope_theta,
                rope_scaling=rope_scaling,
                tie_word_embeddings=arch.tie_word_embeddings,
                attention_bias=arch.attention_bias,
                attention_dropout=arch.attention_dropout,
                hidden_act=arch.hidden_act,
                rms_norm_eps=arch.rms_norm_eps,
                initializer_range=arch.initializer_range,
                pretraining_tp=arch.pretraining_tp,
                mlp_bias=arch.mlp_bias,
            )
        elif model_type == "mixtral":
            moe = arch.moe
            config = MixtralConfig(
                vocab_size=vocab_size,
                hidden_size=arch.hidden_size,
                intermediate_size=arch.intermediate_size,
                num_hidden_layers=arch.num_hidden_layers,
                num_attention_heads=arch.num_attention_heads,
                num_key_value_heads=arch.num_key_value_heads,
                max_position_embeddings=arch.max_position_embeddings,
                rope_theta=arch.rope_theta,
                rope_scaling=rope_scaling,
                tie_word_embeddings=arch.tie_word_embeddings,
                attention_bias=arch.attention_bias,
                attention_dropout=arch.attention_dropout,
                hidden_act=arch.hidden_act,
                rms_norm_eps=arch.rms_norm_eps,
                initializer_range=arch.initializer_range,
                mlp_bias=arch.mlp_bias,
                num_local_experts=moe.num_experts,
                num_experts_per_tok=moe.top_k,
                router_aux_loss_coef=moe.router_aux_loss_coef,
                norm_topk_prob=moe.norm_topk_prob,
                output_router_logits=moe.output_router_logits,
            )
        elif model_type == "qwen2_moe":
            moe = arch.moe
            config = AutoConfig.for_model(
                "qwen2_moe",
                vocab_size=vocab_size,
                hidden_size=arch.hidden_size,
                intermediate_size=arch.intermediate_size,
                num_hidden_layers=arch.num_hidden_layers,
                num_attention_heads=arch.num_attention_heads,
                num_key_value_heads=arch.num_key_value_heads,
                max_position_embeddings=arch.max_position_embeddings,
                rope_theta=arch.rope_theta,
                tie_word_embeddings=arch.tie_word_embeddings,
                hidden_act=arch.hidden_act,
                rms_norm_eps=arch.rms_norm_eps,
                num_experts=moe.num_experts,
                top_k=moe.top_k,
                router_aux_loss_coef=moe.router_aux_loss_coef,
            )
        elif model_type == "deepseek_v2":
            config = AutoConfig.for_model(
                "deepseek_v2",
                vocab_size=vocab_size,
                hidden_size=arch.hidden_size,
                intermediate_size=arch.intermediate_size,
                num_hidden_layers=arch.num_hidden_layers,
                num_attention_heads=arch.num_attention_heads,
                max_position_embeddings=arch.max_position_embeddings,
                rope_theta=arch.rope_theta,
                rope_scaling=rope_scaling,
                tie_word_embeddings=arch.tie_word_embeddings,
                hidden_act=arch.hidden_act,
                rms_norm_eps=arch.rms_norm_eps,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        setattr(config, "_attn_implementation", arch.attention_implementation)
        return config

    @staticmethod
    def architecture_spec(cfg: Config) -> Dict[str, Any]:
        arch = cfg.model.architecture
        spec = {
            "model_type": arch.model_type,
            "hidden_size": arch.hidden_size,
            "intermediate_size": arch.intermediate_size,
            "num_hidden_layers": arch.num_hidden_layers,
            "num_attention_heads": arch.num_attention_heads,
            "num_key_value_heads": arch.num_key_value_heads,
            "max_position_embeddings": arch.max_position_embeddings,
            "rope_theta": arch.rope_theta,
            "attention_bias": arch.attention_bias,
            "attention_dropout": arch.attention_dropout,
            "hidden_act": arch.hidden_act,
            "tie_word_embeddings": arch.tie_word_embeddings,
            "rms_norm_eps": arch.rms_norm_eps,
            "mlp_bias": arch.mlp_bias,
        }
        if arch.model_type == "mixtral" or arch.model_type == "qwen2_moe":
            spec["num_local_experts"] = arch.moe.num_experts
            spec["num_experts_per_tok"] = arch.moe.top_k
        return spec

    @staticmethod
    def saved_architecture_spec(saved: Dict[str, Any]) -> Dict[str, Any]:
        rope_params = saved.get("rope_scaling") or saved.get("rope_parameters") or {}
        spec = {
            "model_type": saved.get("model_type"),
            "hidden_size": saved.get("hidden_size"),
            "intermediate_size": saved.get("intermediate_size"),
            "num_hidden_layers": saved.get("num_hidden_layers"),
            "num_attention_heads": saved.get("num_attention_heads"),
            "num_key_value_heads": saved.get("num_key_value_heads", saved.get("num_attention_heads")),
            "max_position_embeddings": saved.get("max_position_embeddings"),
            "rope_theta": rope_params.get("rope_theta") or saved.get("rope_theta"),
            "attention_bias": saved.get("attention_bias"),
            "attention_dropout": saved.get("attention_dropout"),
            "hidden_act": saved.get("hidden_act"),
            "tie_word_embeddings": saved.get("tie_word_embeddings"),
            "rms_norm_eps": saved.get("rms_norm_eps"),
            "mlp_bias": saved.get("mlp_bias"),
        }
        num_local_experts = saved.get("num_local_experts") or saved.get("num_experts")
        num_experts_per_tok = saved.get("num_experts_per_tok") or saved.get("top_k")
        if num_local_experts:
            spec["num_local_experts"] = num_local_experts
        if num_experts_per_tok:
            spec["num_experts_per_tok"] = num_experts_per_tok
        return spec

    @staticmethod
    def is_compatible(cfg: Config, tokenizer: PreTrainedTokenizerBase, model_path: Path) -> bool:
        config_file = model_path / "config.json"
        if not config_file.exists():
            return False
        try:
            with open(config_file) as f:
                saved = json.load(f)
            current = ModelFactory.architecture_spec(cfg)
            saved_arch = ModelFactory.saved_architecture_spec(saved)
            checks = {**current, "vocab_size": len(tokenizer)}
            saved_arch["vocab_size"] = saved.get("vocab_size")
            for key, current_val in checks.items():
                saved_val = saved_arch.get(key)
                if saved_val is not None and saved_val != current_val:
                    logger.warning("Architecture mismatch: %s saved=%s current=%s", key, saved_val, current_val)
                    return False
            return True
        except Exception as e:
            logger.error("Compatibility check failed: %s", e)
            return False

    @staticmethod
    def create_model(cfg: Config, tokenizer: PreTrainedTokenizerBase) -> Any:
        model_type = cfg.model.architecture.model_type
        dtype = ModelFactory._resolve_dtype(cfg.model.dtype)

        if model_type == "nslt":
            from src.nslt import NSLTModel
            arch = cfg.model.architecture
            nslt_cfg = arch.nslt
            model = NSLTModel(
                vocab_size=len(tokenizer),
                d_model=arch.hidden_size,
                d_state=nslt_cfg.d_state,
                d_hidden=nslt_cfg.d_hidden,
                n_ssm_layers=nslt_cfg.n_ssm_layers,
                max_seq_len=arch.max_position_embeddings,
                rope_base=arch.rope_theta,
                sparsity_pct=nslt_cfg.sparsity_pct,
                n_ode_steps=nslt_cfg.n_ode_steps,
                n_trajectories=nslt_cfg.n_trajectories,
                n_sim_steps=nslt_cfg.n_sim_steps,
                use_efficient_sandbox=nslt_cfg.use_efficient_sandbox,
                dtype=dtype,
            )
            return model

        arch_cfg = ModelFactory.build_model_config(
            vocab_size=len(tokenizer),
            arch=cfg.model.architecture,
            max_seq_length=cfg.training.max_seq_length,
        )

        _, model_cls = ARCH_CONFIG_MAP.get(model_type, (AutoConfig, AutoModelForCausalLM))
        model = model_cls.from_config(arch_cfg, torch_dtype=dtype)

        for param in model.parameters():
            param.requires_grad = True
        model.config.use_cache = False
        return model

    @staticmethod
    def get_fsdp_layer_cls(model_type: str) -> str:
        return ARCH_FSDP_LAYER_MAP.get(model_type, "LlamaDecoderLayer")

    @staticmethod
    def estimate_model_size(
        arch: ModelArchitectureConfig,
        tokenizer_or_vocab: Optional[Any] = None,
    ) -> Dict[str, Any]:
        vocab_size = arch.vocab_size
        if tokenizer_or_vocab is not None:
            if isinstance(tokenizer_or_vocab, PreTrainedTokenizerBase):
                vocab_size = len(tokenizer_or_vocab)
            elif isinstance(tokenizer_or_vocab, int):
                vocab_size = tokenizer_or_vocab
            elif isinstance(tokenizer_or_vocab, dict) and "vocab_size" in tokenizer_or_vocab:
                vocab_size = int(tokenizer_or_vocab["vocab_size"])

        if arch.model_type == "nslt":
            nslt = arch.nslt
            d_model = arch.hidden_size
            d_state = nslt.d_state
            d_hidden = nslt.d_hidden
            n_ssm = nslt.n_ssm_layers

            # Embedding
            embed = vocab_size * d_model

            # Per SSM block: 3 linear projections + conv1d
            ssm_block = 2 * d_model * (d_model * 2) + 2 * d_model * nslt.n_ode_steps + d_state * d_model
            ssm_total = n_ssm * ssm_block

            # LTC layer
            ltc = (d_state + d_hidden) * d_hidden * 3 + d_hidden * d_hidden

            # Sandbox
            sandbox_d_latent = min(d_hidden // 4, 4096)
            sandbox = (d_hidden * sandbox_d_latent * 2) + (sandbox_d_latent * d_hidden)

            # Sparse output
            sparse_output = d_hidden * (d_hidden // 2) + (d_hidden // 2) * vocab_size

            total = embed + ssm_total + ltc + sandbox + sparse_output
            active = total

            return {
                "total_params_b": round(total / 1e9, 2),
                "active_params_b": round(active / 1e9, 2),
                "layers": n_ssm,
                "experts": 1,
                "top_k": 1,
                "architecture": "nslt",
            }

        hidden = arch.hidden_size
        layers = arch.num_hidden_layers
        heads = arch.num_attention_heads
        kv_heads = arch.num_key_value_heads
        intermediate = arch.intermediate_size
        num_experts = arch.moe.num_experts
        experts_per_tok = arch.moe.top_k

        embed_params = vocab_size * hidden * 2
        per_layer_attn = 4 * hidden * hidden + 2 * hidden * (hidden // heads) * kv_heads
        if num_experts > 1:
            expert_params = num_experts * 3 * hidden * intermediate
            shared_params = 3 * hidden * intermediate
            per_layer_mlp = expert_params / experts_per_tok + shared_params
        else:
            per_layer_mlp = 3 * hidden * intermediate
            shared_params = 0
        per_layer_norm = 2 * hidden
        total_params = embed_params + layers * (per_layer_attn + per_layer_mlp + per_layer_norm) + hidden * vocab_size
        active_params = embed_params + layers * (per_layer_attn + (expert_params / experts_per_tok if num_experts > 1 else per_layer_mlp) + per_layer_norm) + hidden * vocab_size

        return {
            "total_params_b": round(total_params / 1e9, 2),
            "active_params_b": round(active_params / 1e9, 2),
            "layers": layers,
            "experts": num_experts,
            "top_k": experts_per_tok,
        }

    @staticmethod
    def load_tokenizer(path: str | Path = "models/tokenizer") -> PreTrainedTokenizerBase:
        path = Path(path)
        if not path.exists():
            raise RuntimeError(f"Tokenizer not found at {path}. Train it first.")
        tokenizer = ModelFactory._try_load_tokenizer(path)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
        return tokenizer

    @staticmethod
    def _try_load_tokenizer(path: Path) -> PreTrainedTokenizerBase:
        from transformers import PreTrainedTokenizerFast
        try:
            return AutoTokenizer.from_pretrained(str(path), trust_remote_code=False)
        except Exception:
            pass
        try:
            return PreTrainedTokenizerFast.from_pretrained(str(path))
        except Exception:
            pass
        vocab_file = path / "vocab.json"
        merges_file = path / "merges.txt"
        if vocab_file.exists() and merges_file.exists():
            from tokenizers import ByteLevelBPETokenizer
            from transformers import PreTrainedTokenizerFast
            backend = ByteLevelBPETokenizer(
                str(vocab_file), str(merges_file)
            )
            return PreTrainedTokenizerFast(
                tokenizer_object=backend._tokenizer,
                bos_token="<s>",
                eos_token="</s>",
                unk_token="<unk>",
                pad_token="<pad>",
            )
        raise RuntimeError(
            f"Unable to load tokenizer from {path}. "
            "Run tokenizer training first or check the tokenizer files."
        )

    @staticmethod
    def load_model(
        path: str | Path,
        cfg: Config,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        path = Path(path)
        if tokenizer is None:
            tokenizer = ModelFactory.load_tokenizer()

        config_file = path / "config.json"
        model_type = None
        if config_file.exists():
            with open(config_file) as f:
                saved_config = json.load(f)
            model_type = saved_config.get("model_type") or saved_config.get("architecture")
        elif cfg.model.architecture.model_type == "nslt" and (path / "pytorch_model.bin").exists():
            model_type = "nslt"

        if model_type == "nslt":
            from src.nslt import NSLTModel
            import torch.nn as nn
            state_dict = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)
            arch = cfg.model.architecture
            nslt_cfg = arch.nslt
            model = NSLTModel(
                vocab_size=len(tokenizer),
                d_model=arch.hidden_size,
                d_state=nslt_cfg.d_state,
                d_hidden=nslt_cfg.d_hidden,
                n_ssm_layers=nslt_cfg.n_ssm_layers,
                max_seq_len=arch.max_position_embeddings,
                rope_base=arch.rope_theta,
                sparsity_pct=nslt_cfg.sparsity_pct,
                n_ode_steps=nslt_cfg.n_ode_steps,
                n_trajectories=nslt_cfg.n_trajectories,
                n_sim_steps=nslt_cfg.n_sim_steps,
                use_efficient_sandbox=nslt_cfg.use_efficient_sandbox,
                dtype=ModelFactory._resolve_dtype(cfg.model.dtype),
            )
            model.load_state_dict(state_dict, strict=False)
            for param in model.parameters():
                param.requires_grad = True
            return model, tokenizer

        load_dtype = ModelFactory._resolve_dtype(cfg.model.dtype)
        is_distributed = any(os.environ.get(v) is not None for v in ["WORLD_SIZE", "LOCAL_RANK", "RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if is_distributed and world_size <= 1:
            is_distributed = False

        device_map = None
        if torch.cuda.is_available() and not is_distributed:
            device_map = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            str(path),
            torch_dtype=load_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        for param in model.parameters():
            param.requires_grad = True

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

        return model, tokenizer

    @staticmethod
    def _resolve_dtype(dtype_str: str) -> torch.dtype:
        has_cuda = torch.cuda.is_available()
        if dtype_str == "bfloat16" and has_cuda:
            return torch.bfloat16
        elif dtype_str == "float16" and has_cuda:
            return torch.float16
        return torch.float32
