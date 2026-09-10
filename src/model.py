from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.config.schema import Config, load_config
from src.models.factory import ModelFactory

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


class SpecializedCoderModel:
    def __init__(self, config_path: str | Path = _DEFAULT_CONFIG_PATH) -> None:
        self._config_path = Path(config_path)
        self.config_dict: Dict[str, Any] = self._load_yaml(self._config_path)
        self.cfg: Config = self._resolve_config()
        self.device: str = self.config_dict.get("model", {}).get("device", "cuda")
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        import yaml
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def _resolve_config(self) -> Config:
        try:
            return load_config(self._config_path)
        except Exception:
            return Config.model_validate(self.config_dict)

    def load_base_model(self) -> None:
        self.tokenizer = ModelFactory.load_tokenizer()
        self.model = ModelFactory.create_model(self.cfg, self.tokenizer)
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        use_fsdp = world_size > 1
        if not use_fsdp:
            device = self.cfg.model.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            elif device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                device = "cpu"
            self.model = self.model.to(device)
        if hasattr(self.model, "config"):
            self._orig_use_cache = getattr(self.model.config, "use_cache", True)
            self.model.config.use_cache = False
        logger.info("Model initialized with %s parameters.",
                    f"{sum(p.numel() for p in self.model.parameters()):,}")

    def _current_architecture_spec(self) -> Dict[str, Any]:
        return ModelFactory.architecture_spec(self.cfg)

    @staticmethod
    def _saved_rope_theta(saved: Dict[str, Any]) -> Any:
        rope_params = saved.get("rope_parameters") or {}
        return rope_params.get("rope_theta") or saved.get("rope_theta")

    def _saved_architecture_spec(self, saved: Dict[str, Any]) -> Dict[str, Any]:
        return ModelFactory.saved_architecture_spec(saved)

    def is_compatible(self, path: str | Path) -> bool:
        import json
        path = Path(path)
        config_file = path / "config.json"
        if not config_file.exists():
            return False
        try:
            with open(config_file) as f:
                saved = json.load(f)
            if self.tokenizer is None:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(str(path))
                except Exception:
                    self.tokenizer = ModelFactory.load_tokenizer()
            return ModelFactory.is_compatible(self.cfg, self.tokenizer, path)
        except Exception as e:
            logger.error("Compatibility check failed: %s", e)
            return False

    def load_model(self, path: str | Path) -> None:
        self.model, self.tokenizer = ModelFactory.load_model(path, self.cfg, self.tokenizer)
        logger.info("Model loaded from %s", path)

    def generate_code(
        self,
        problem: str,
        language: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_base_model() first.")
        config_max = int(self.config_dict.get("generation", {}).get("max_new_tokens", 8192))
        max_new_tokens = min(max_new_tokens, config_max)
        prompt = (
            f"### Instruction\n"
            f"Write a {language} solution for the following problem:\n"
            f"{problem}\n\n"
            f"### Response\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        inputs.pop("token_type_ids", None)
        eos_ids = []
        if self.tokenizer.eos_token_id is not None:
            eos_ids.append(self.tokenizer.eos_token_id)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=temperature > 0,
            repetition_penalty=self.config_dict.get("generation", {}).get("repetition_penalty", 1.1),
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=getattr(self, "_orig_use_cache", True),
        )
        if eos_ids:
            gen_kwargs["eos_token_id"] = eos_ids
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][prompt_len:]
        code = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # Only truncate at instruction boundaries (previous code chopped any
        # "###" including markdown headers inside generated code).
        for marker in ("\n### Instruction", "\n### End"):
            idx = code.find(marker)
            if idx > 0:
                code = code[:idx].strip()
        return code

    def save_model(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(path, safe_serialization=True)
        else:
            torch.save(self.model.state_dict(), str(path / "pytorch_model.bin"))
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(path)
        logger.info("Model saved to %s", path)

    def print_trainable_parameters(self) -> None:
        if self.model is None:
            logger.warning("Model not loaded.")
            return
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_param = sum(p.numel() for p in self.model.parameters())
        print(f"trainable params: {trainable:,} || all params: {all_param:,} || trainable%: {100 * trainable / max(all_param, 1):.4f}")

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None
