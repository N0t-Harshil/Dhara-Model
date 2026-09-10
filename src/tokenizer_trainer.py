from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterable, Optional

import yaml
from tokenizers import ByteLevelBPETokenizer
from src.data.streaming import MassiveDataCollector
from src.config.schema import DatasetEntryConfig

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = [
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
    "### Instruction",
    "### Input",
    "### Response",
    "<thought>",
    "</thought>",
    "<fim_prefix>",
    "<fim_middle>",
    "<fim_suffix>",
    "```",
    "```python",
    "```javascript",
    "```typescript",
    "```rust",
    "```go",
    "```cpp",
    "```java",
    "```sql",
    "```bash",
]


def _load_training_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _format_sample(sample: dict) -> str:
    language = str(sample.get("language") or "python").strip().lower()
    instruction = str(sample.get("instruction") or f"Complete this {language} code:").strip()
    input_text = str(sample.get("input") or "").strip()
    output = str(sample.get("output") or "").strip()

    if input_text:
        instruction = f"{instruction}\n{input_text}"

    return (
        "### Instruction\n"
        f"Write a {language} solution for the following problem:\n"
        f"{instruction}\n\n"
        "### Response\n"
        f"```{language}\n{output}\n```"
    )


def _iter_tokenizer_corpus(collector: MassiveDataCollector, max_samples: int, theme: str) -> Iterable[str]:
    seen: set[str] = set()
    for i, sample in enumerate(collector.stream_samples(limit=max_samples, theme=theme)):
        text = _format_sample(sample)
        if len(text) < 80:
            continue
        fingerprint = hashlib.sha256(text.encode()).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        yield text
        if i and i % 1000 == 0:
            logger.info("Collected %d tokenizer samples...", i)


def download_tokenizer(
    output_dir: str = "models/tokenizer",
    model_id: str = "deepseek-ai/deepseek-coder-33b-instruct",
    force: bool = False,
) -> None:
    """Download a tokenizer from HuggingFace Hub and save it locally.

    Args:
        output_dir: Directory to save the tokenizer.
        model_id: HuggingFace model ID (e.g. 'deepseek-ai/deepseek-coder-33b-instruct').
        force: Overwrite existing tokenizer.
    """
    from transformers import AutoTokenizer

    output_path = Path(output_dir)
    if (output_path / "tokenizer.json").exists() and not force:
        logger.info("Tokenizer already exists at %s (use --force to overwrite)", output_dir)
        return

    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading tokenizer from HuggingFace: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    special_tokens = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>",
    }
    missing = {k for k, v in special_tokens.items() if getattr(tokenizer, k, None) is None}
    if missing:
        logger.info("Adding missing special tokens: %s", missing)
        tokenizer.add_special_tokens({k: v for k, v in special_tokens.items() if k in missing})

    tokenizer.save_pretrained(str(output_path))
    logger.info("Tokenizer downloaded and saved to %s", output_dir)


def train_custom_tokenizer(
    output_dir: str = "models/tokenizer",
    vocab_size: int | None = None,
    max_samples: int | None = None,
    theme: str | None = None,
    force: bool = False,
):
    """Train a custom BPE tokenizer on the project's dataset corpus.

    Args:
        output_dir: Directory to save the tokenizer.
        vocab_size: Vocabulary size.
        max_samples: Maximum samples to train on.
        theme: Dataset theme filter.
        force: Overwrite existing tokenizer.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if (output_path / "tokenizer.json").exists() and not force:
        logger.info("Custom tokenizer already exists at %s", output_dir)
        return

    config_path = Path("config.yaml")
    if not config_path.exists():
        config_path = Path("../config.yaml")

    config = _load_training_config(config_path)
    tok_cfg = config.get("tokenizer", {})
    vocab_size = int(vocab_size or tok_cfg.get("vocab_size", 64000))
    max_samples = int(max_samples or tok_cfg.get("max_samples", 50000))
    theme = str(theme or tok_cfg.get("theme", "all"))

    logger.info(
        "Starting custom tokenizer training: vocab=%d, samples=%d, theme=%s",
        vocab_size,
        max_samples,
        theme,
    )

    raw_datasets = config.get("data", {}).get("datasets", [])
    datasets_cfg = [DatasetEntryConfig(**d) if isinstance(d, dict) else d for d in raw_datasets]
    collector = MassiveDataCollector(datasets_cfg=datasets_cfg)
    samples = list(_iter_tokenizer_corpus(collector, max_samples=max_samples, theme=theme))
    if len(samples) < 100:
        raise RuntimeError(
            f"Tokenizer corpus is too small ({len(samples)} samples). "
            "Check dataset connectivity/config or lower tokenizer.max_samples for a smoke test."
        )

    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train_from_iterator(
        samples,
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
    )

    from transformers import PreTrainedTokenizerFast
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer._tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        additional_special_tokens=SPECIAL_TOKENS[5:],
    )
    fast_tokenizer.save_pretrained(str(output_path))
    logger.info("Custom tokenizer saved to %s", output_dir)


def ensure_tokenizer(
    output_dir: str = "models/tokenizer",
    config_path: str = "config.yaml",
    force: bool = False,
) -> None:
    """Ensure a tokenizer exists at output_dir, downloading or training if needed.

    Cache-aware: a warm, hash-verified tokenizer is never re-downloaded or
    re-trained (the manifest recorded by ``ModelFactory`` is the single source
    of truth for what a verified cache is). Only an explicit ``force``, a
    missing cache, or an unverified/changed cache triggers (re)acquisition.

    Reads config.yaml to determine the tokenizer source. If source is
    'huggingface', downloads from Hub. If 'custom', trains on the project
    dataset.
    """
    from src.config.schema import load_config
    cfg = load_config(config_path)
    tok_cfg = cfg.tokenizer

    from src.models.factory import ModelFactory
    output_path = Path(output_dir)
    verified = ModelFactory._verify_tokenizer_cache(output_path) is not None
    if force:
        logger.info("Tokenizer force-update requested for %s — re-acquiring", output_dir)
    elif verified:
        logger.info("Tokenizer cache verified (hash match) at %s — "
                    "loading offline, no download needed", output_dir)
        return
    elif (output_path / "tokenizer.json").exists():
        logger.warning("Tokenizer files present at %s but unverified/changed "
                       "(manifest mismatch) — re-acquiring", output_dir)
    else:
        logger.info("Tokenizer cache missing at %s — acquiring", output_dir)

    # ``force`` here doubles as "re-acquire even when files exist": the sub
    # functions' ``exists() and not force`` early-return must not let a
    # corrupt-but-present cache skip acquisition.
    needs_acquire = force or not verified

    if tok_cfg.source == "huggingface":
        download_tokenizer(
            output_dir=output_dir,
            model_id=tok_cfg.huggingface_model,
            force=needs_acquire,
        )
    else:
        train_custom_tokenizer(
            output_dir=output_dir,
            vocab_size=tok_cfg.vocab_size,
            max_samples=tok_cfg.max_samples,
            force=needs_acquire,
        )

    ModelFactory._write_tokenizer_manifest(output_path)
    logger.info("Tokenizer cache manifest written for %s", output_dir)
