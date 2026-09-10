from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import transformers as _transformers
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

_TRANSFORMERS_MAJOR = int(_transformers.__version__.split(".")[0])
_DTYPE_CONFIG_ATTR = "dtype" if _TRANSFORMERS_MAJOR >= 5 else "torch_dtype"
_DTYPE_KWARG = _DTYPE_CONFIG_ATTR


ARCH_CONFIG_MAP = {
    "llama": (LlamaConfig, LlamaForCausalLM),
    "mixtral": (MixtralConfig, MixtralForCausalLM),
    "qwen2_moe": (AutoConfig, AutoModelForCausalLM),
    "deepseek_v2": (AutoConfig, AutoModelForCausalLM),
    "nslt": (None, None),
    "dhara_v3": (None, None),
}

ARCH_FSDP_LAYER_MAP = {
    "llama": "LlamaDecoderLayer",
    "mixtral": "MixtralDecoderLayer",
    "qwen2_moe": "Qwen2MoeDecoderLayer",
    "deepseek_v2": "DeepseekV2DecoderLayer",
    "nslt": "SSMCompressionEngine",
    "dhara_v3": "HierarchicalSSMStack",
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
        if arch.model_type == "dhara_v3":
            spec["dhara_v3"] = arch.dhara_v3.model_dump(mode="python")
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

        if model_type == "dhara_v3":
            from src.dhara import DharaModel
            from src.dhara.model import DharaConfig as DharaModelConfig
            arch = cfg.model.architecture
            v3 = arch.dhara_v3
            model_config = DharaModelConfig(
                vocab_size=len(tokenizer),
                hidden_size=arch.hidden_size,
                d_state=v3.d_state,
                d_hidden=v3.d_hidden,
                max_position_embeddings=arch.max_position_embeddings,
                rope_theta=arch.rope_theta,
                sparsity_pct=v3.sparsity_pct,
                n_ssm_layers=v3.n_ssm_layers,
                n_hssm_levels=v3.n_hssm_levels,
                n_ode_steps=v3.n_ode_steps,
                n_trajectories=v3.n_trajectories,
                n_sim_steps=v3.n_sim_steps,
                use_efficient_sandbox=v3.use_efficient_sandbox,
                n_token_categories=v3.n_token_categories,
                n_languages=v3.n_languages,
                n_doc_roles=v3.n_doc_roles,
                n_task_types=v3.n_task_types,
                n_difficulty_levels=v3.n_difficulty_levels,
                n_reasoning_types=v3.n_reasoning_types,
                max_subgoals=v3.max_subgoals,
                n_domains=v3.n_domains,
                max_reasoning_steps=v3.max_reasoning_steps,
                n_semantic_concepts=v3.n_semantic_concepts,
                working_mem_capacity=v3.working_mem_capacity,
                max_episodes=v3.max_episodes,
                n_context_adapter_blocks=v3.n_context_adapter_blocks,
                n_language_groups=v3.n_language_groups,
                adaptive_top_k_min=v3.adaptive_top_k_min,
                adaptive_top_k_max=v3.adaptive_top_k_max,
                n_experts=v3.n_experts,
                n_debate_rounds=v3.n_debate_rounds,
                max_refinement_passes=v3.max_refinement_passes,
                max_repair_iters=v3.max_repair_iters,
                max_entities=v3.max_entities,
                n_relation_types=v3.n_relation_types,
                max_events=v3.max_events,
                n_tool_types=v3.n_tool_types,
                loss_weights=getattr(v3, "loss_weights", None),
                enable_executive=v3.enable_executive,
                enable_world_model=v3.enable_world_model,
                enable_tools=v3.enable_tools,
                enable_curiosity=v3.enable_curiosity,
                enable_aux_losses=v3.enable_aux_losses,
                executive_gate_threshold=v3.executive_gate_threshold,
                qa_max_passes=v3.qa_max_passes,
                qa_converge_threshold=v3.qa_converge_threshold,
            )
            setattr(model_config, _DTYPE_CONFIG_ATTR, dtype)
            model = DharaModel(config=model_config)
            return model

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
        if _TRANSFORMERS_MAJOR >= 5 and not hasattr(model_cls, "from_config"):
            model_cls = AutoModelForCausalLM
        model = model_cls.from_config(arch_cfg, **{_DTYPE_KWARG: dtype})

        for param in model.parameters():
            param.requires_grad = True
        model.config.use_cache = False
        return model

    @staticmethod
    def get_fsdp_layer_cls(model_type: str) -> str:
        return ARCH_FSDP_LAYER_MAP.get(model_type, "LlamaDecoderLayer")

    @staticmethod
    def estimate_model_size(arch: ModelArchitectureConfig, tokenizer_or_vocab: Any = None) -> Dict[str, Any]:
        if arch.model_type == "nslt":
            nslt = arch.nslt
            d_model = arch.hidden_size
            d_state = nslt.d_state
            d_hidden = nslt.d_hidden
            n_ssm = nslt.n_ssm_layers
            if tokenizer_or_vocab is not None:
                vocab_size = len(tokenizer_or_vocab) if hasattr(tokenizer_or_vocab, "__len__") else int(tokenizer_or_vocab)
            else:
                vocab_size = arch.vocab_size

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

        if arch.model_type == "dhara_v3":
            v3 = arch.dhara_v3
            d_model = arch.hidden_size
            d_hidden = v3.d_hidden
            if tokenizer_or_vocab is not None:
                vocab_size = len(tokenizer_or_vocab) if hasattr(tokenizer_or_vocab, "__len__") else int(tokenizer_or_vocab)
            else:
                vocab_size = arch.vocab_size
            embed = vocab_size * d_model
            ssm_params = v3.n_ssm_layers * (4 * d_model ** 2 + d_model * v3.d_state * 3 + d_model * v3.d_state)
            memory_params = d_model * d_hidden * 4 + d_hidden * d_hidden * 2
            intent_params = d_model * (v3.n_task_types + v3.n_difficulty_levels + v3.n_reasoning_types + 2)
            planner_params = v3.max_subgoals * d_model * 4 + d_model * d_model
            reasoning_params = v3.n_domains * ((v3.d_state + d_hidden) * d_hidden * 2 + d_hidden * d_hidden)
            workspace_params = d_hidden * d_hidden * 4
            specialist_params = 7 * (d_hidden * d_hidden * 4 + d_hidden)
            decoder_params = d_hidden * vocab_size + d_hidden * v3.n_language_groups
            total = embed + ssm_params + memory_params + intent_params + planner_params + reasoning_params + workspace_params + specialist_params + decoder_params
            return {
                "total_params_b": round(total / 1e9, 2),
                "active_params_b": round(total / 1e9, 2),
                "layers": v3.n_ssm_layers,
                "experts": 1,
                "top_k": 1,
                "architecture": "dhara_v3",
            }

        vocab_size = arch.vocab_size
        hidden = arch.hidden_size
        layers = arch.num_hidden_layers
        heads = arch.num_attention_heads
        kv_heads = arch.num_key_value_heads
        intermediate = arch.intermediate_size
        num_experts = arch.moe.num_experts
        experts_per_tok = arch.moe.top_k

        # vocab*hidden counted once for embeddings; lm_head is tied when
        # tie_word_embeddings is true (schema default), otherwise separate.
        tie = bool(getattr(arch, "tie_word_embeddings", True))
        embed_params = vocab_size * hidden
        lm_head_params = 0 if tie else vocab_size * hidden
        per_layer_attn = 4 * hidden * hidden + 2 * hidden * (hidden // heads) * kv_heads
        if num_experts > 1:
            expert_params = num_experts * 3 * hidden * intermediate
            shared_params = 3 * hidden * intermediate
            per_layer_mlp = expert_params / experts_per_tok + shared_params
        else:
            per_layer_mlp = 3 * hidden * intermediate
            shared_params = 0
        per_layer_norm = 2 * hidden
        total_params = embed_params + lm_head_params + layers * (per_layer_attn + per_layer_mlp + per_layer_norm)
        # active = dense + per-token expert slice + shared expert
        if num_experts > 1:
            active_mlp = (expert_params / num_experts) * experts_per_tok + shared_params
        else:
            active_mlp = per_layer_mlp
        active_params = embed_params + lm_head_params + layers * (per_layer_attn + active_mlp + per_layer_norm)

        return {
            "total_params_b": round(total_params / 1e9, 2),
            "active_params_b": round(active_params / 1e9, 2),
            "layers": layers,
            "experts": num_experts,
            "top_k": experts_per_tok,
        }

    @staticmethod
    def load_tokenizer(path: str | Path = "models/tokenizer", cfg: Optional[Any] = None) -> PreTrainedTokenizerBase:
        path = Path(path)
        if not path.exists():
            raise RuntimeError(f"Tokenizer not found at {path}. Train it first.")
        verified = ModelFactory._verify_tokenizer_cache(path)
        if verified is not None:
            logger.info("Tokenizer cache verified (hash match) — loading offline, no network access")
            tokenizer = ModelFactory._try_load_tokenizer(path, offline=True)
        else:
            ModelFactory._write_tokenizer_manifest(path)
            tokenizer = ModelFactory._try_load_tokenizer(path, offline=False)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
        if not hasattr(tokenizer, "padding_side") or tokenizer.padding_side != "right":
            tokenizer.padding_side = "right"
        report = ModelFactory.special_tokens_report(tokenizer)
        logger.info("Special-token table: %s",
                    {k: v for k, v in report.items() if k != "pad_aliases_eos"})
        if report["pad_aliases_eos"]:
            logger.warning(
                "PAD token id == EOS token id (%d) — packed sequences pad with "
                "EOS. Safe while loss labels mask these positions (-100); any "
                "code path that feeds real padding tokens to the model will "
                "see EOS semantics. Insert a distinct <pad> to change this.",
                report["eos_token_id"],
            )
        max_len = getattr(cfg, 'model', None) and getattr(cfg.model, 'architecture', None) and cfg.model.architecture.max_position_embeddings
        if (not hasattr(tokenizer, "model_max_length")
                or tokenizer.model_max_length is None
                or tokenizer.model_max_length > 1_000_000):
            tokenizer.model_max_length = max_len or 4096
        return tokenizer

    @staticmethod
    def _tokenizer_manifest_path(path: Path) -> Path:
        return path / "tokenizer_version.json"

    @staticmethod
    def _hash_tokenizer_files(path: Path) -> Dict[str, str]:
        import hashlib
        out: Dict[str, str] = {}
        for f in sorted(path.iterdir()):
            if f.is_file() and f.name != "tokenizer_version.json":
                out[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
        return out

    @staticmethod
    def special_tokens_report(tokenizer) -> Dict[str, Any]:
        """Resolved special-token table plus the PAD/EOS aliasing flag.

        Phase 10 audit: packing pads sequences with ``eos_token_id``, so every
        log must state plainly whether ``pad_token_id == eos_token_id`` in the
        active tokenizer (deepseek-style tokenizers alias them)."""
        table = {
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "unk_token_id": getattr(tokenizer, "unk_token_id", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "mask_token_id": getattr(tokenizer, "mask_token_id", None),
        }
        table["pad_aliases_eos"] = (
            table["pad_token_id"] is not None
            and table["eos_token_id"] is not None
            and table["pad_token_id"] == table["eos_token_id"]
        )
        return table

    @staticmethod
    def _verify_tokenizer_cache(path: Path) -> Optional[Dict[str, Any]]:
        """Return the manifest if the on-disk tokenizer files match the stored
        hashes (i.e. the tokenizer is exactly the version we trained and can be
        loaded offline). Returns None otherwise."""
        manifest_path = ModelFactory._tokenizer_manifest_path(path)
        if not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("version") != 1:
                return None
            current = ModelFactory._hash_tokenizer_files(path)
            if manifest.get("files") != current:
                logger.warning("Tokenizer files changed since manifest was written — re-verifying")
                return None
            return manifest
        except Exception as e:
            logger.debug("Tokenizer manifest check failed: %s", e)
            return None

    @staticmethod
    def _write_tokenizer_manifest(path: Path) -> None:
        try:
            manifest = {"version": 1, "files": ModelFactory._hash_tokenizer_files(path)}
            ModelFactory._tokenizer_manifest_path(path).write_text(
                json.dumps(manifest, indent=2), encoding="utf-8")
            logger.info("Tokenizer manifest written (%d files)", len(manifest["files"]))
        except Exception as e:
            logger.warning("Could not write tokenizer manifest: %s", e)

    @staticmethod
    def _try_load_tokenizer(path: Path, offline: bool = False) -> PreTrainedTokenizerBase:
        from transformers import PreTrainedTokenizerFast
        try:
            if offline:
                return AutoTokenizer.from_pretrained(str(path), trust_remote_code=False, local_files_only=True)
            return AutoTokenizer.from_pretrained(str(path), trust_remote_code=False)
        except Exception:
            pass
        try:
            if offline:
                return PreTrainedTokenizerFast.from_pretrained(str(path), local_files_only=True)
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
        strict: bool = False,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        path = Path(path)
        if tokenizer is None:
            tokenizer = ModelFactory.load_tokenizer(cfg=cfg)

        config_file = path / "config.json"
        saved_config = None
        model_type = None
        if config_file.exists():
            with open(config_file) as f:
                saved_config = json.load(f)
            model_type = saved_config.get("model_type") or saved_config.get("architecture")
        elif cfg.model.architecture.model_type in ("nslt", "dhara_v3") and ((path / "pytorch_model.bin").exists() or (path / "model.safetensors").exists()):
            model_type = cfg.model.architecture.model_type

        if model_type == "dhara_v3":
            from src.dhara import DharaModel
            from src.dhara.model import DharaConfig as DharaModelConfig
            bin_path = path / "pytorch_model.bin"
            st_path = path / "model.safetensors"
            if bin_path.exists():
                state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
            elif st_path.exists():
                from safetensors.torch import load_file
                state_dict = load_file(str(st_path), device="cpu")
            else:
                raise FileNotFoundError(
                    f"No model weights found in {path} (expected pytorch_model.bin or model.safetensors). "
                    "Was the model saved with save_pretrained()?"
                )
            if saved_config is not None:
                model_config = DharaModelConfig.from_pretrained(str(path))
                model_config.vocab_size = len(tokenizer)
            else:
                arch = cfg.model.architecture
                v3 = arch.dhara_v3
                model_config = DharaModelConfig(
                    vocab_size=len(tokenizer),
                    hidden_size=arch.hidden_size,
                    d_state=v3.d_state,
                    d_hidden=v3.d_hidden,
                    max_position_embeddings=arch.max_position_embeddings,
                    rope_theta=arch.rope_theta,
                    sparsity_pct=v3.sparsity_pct,
                    n_ssm_layers=v3.n_ssm_layers,
                    n_hssm_levels=v3.n_hssm_levels,
                    n_ode_steps=v3.n_ode_steps,
                    n_trajectories=v3.n_trajectories,
                    n_sim_steps=v3.n_sim_steps,
                    use_efficient_sandbox=v3.use_efficient_sandbox,
                    n_token_categories=v3.n_token_categories,
                    n_languages=v3.n_languages,
                    n_doc_roles=v3.n_doc_roles,
                    n_task_types=v3.n_task_types,
                    n_difficulty_levels=v3.n_difficulty_levels,
                    n_reasoning_types=v3.n_reasoning_types,
                    max_subgoals=v3.max_subgoals,
                    n_domains=v3.n_domains,
                    max_reasoning_steps=v3.max_reasoning_steps,
                    n_semantic_concepts=v3.n_semantic_concepts,
                    working_mem_capacity=v3.working_mem_capacity,
                    max_episodes=v3.max_episodes,
                    n_context_adapter_blocks=v3.n_context_adapter_blocks,
                    n_language_groups=v3.n_language_groups,
                    adaptive_top_k_min=v3.adaptive_top_k_min,
                    adaptive_top_k_max=v3.adaptive_top_k_max,
                    n_experts=v3.n_experts,
                    n_debate_rounds=v3.n_debate_rounds,
                    max_refinement_passes=v3.max_refinement_passes,
                    max_repair_iters=v3.max_repair_iters,
                    max_entities=v3.max_entities,
                    n_relation_types=v3.n_relation_types,
                    max_events=v3.max_events,
                    n_tool_types=v3.n_tool_types,
                    enable_curiosity=getattr(v3, "enable_curiosity", True),
                    loss_weights=getattr(v3, "loss_weights", None),
                    executive_gate_threshold=getattr(v3, "executive_gate_threshold", 0.3),
                    qa_max_passes=getattr(v3, "qa_max_passes", 5),
                )
            setattr(model_config, _DTYPE_CONFIG_ATTR, ModelFactory._resolve_dtype(cfg.model.dtype))
            model = DharaModel(config=model_config)
            model_state = model.state_dict()
            filtered = {}
            shape_skipped = []
            for k, v in state_dict.items():
                if k in model_state and v.shape == model_state[k].shape:
                    filtered[k] = v
                else:
                    shape_skipped.append(k)
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            if missing:
                msg = f"Checkpoint missing {len(missing)} keys: {sorted(missing)[:8]}"
                if strict:
                    raise RuntimeError(msg)
                logger.info("Random init for %d new parameters — %s", len(missing), sorted(missing)[:4])
            if shape_skipped:
                msg = (f"Checkpoint has {len(shape_skipped)} incompatible/unknown weights "
                       f"(skipped): {sorted(shape_skipped)[:8]}")
                if strict:
                    raise RuntimeError(msg)
                logger.info("%s", msg)
            for param in model.parameters():
                param.requires_grad = True
            return model, tokenizer

        if model_type == "nslt":
            from src.nslt import NSLTModel
            bin_path = path / "pytorch_model.bin"
            st_path = path / "model.safetensors"
            if bin_path.exists():
                state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
            elif st_path.exists():
                from safetensors.torch import load_file
                state_dict = load_file(str(st_path), device="cpu")
            else:
                raise FileNotFoundError(
                    f"No model weights found in {path} (expected pytorch_model.bin or model.safetensors). "
                    "Was the model saved with save_pretrained()?"
                )
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
            model_state = model.state_dict()
            # Filter BEFORE load_state_dict: torch raises on size mismatches
            # even with strict=False.
            filtered = {k: v for k, v in state_dict.items()
                        if k in model_state and v.shape == model_state[k].shape}
            shape_skipped = sorted(set(state_dict) - set(filtered))
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            if strict and (missing or shape_skipped):
                raise RuntimeError(
                    f"Strict load failed: {len(missing)} missing keys, "
                    f"{len(shape_skipped)} incompatible/unknown weights"
                )
            if shape_skipped:
                logger.info("Skipped %d incompatible checkpoint weights", len(shape_skipped))
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
            **{_DTYPE_KWARG: load_dtype},
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
