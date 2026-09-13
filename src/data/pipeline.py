from __future__ import annotations

import fnmatch
import gc
import hashlib
import itertools
import json
import logging
import math
import multiprocessing
import os
import random
import re
import threading
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import numpy as np
import torch
from datasets import Dataset, IterableDataset, concatenate_datasets, load_dataset, load_from_disk
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizerBase

from src.config.schema import Config, DatasetEntryConfig
from src.data.quality import (
    ContaminationFilter,
    ExactDeduplicator,
    MinHashDeduplicator,
    QualityFilter,
    QualityScorer,
    SimHashDeduplicator,
    SemanticDeduplicator,
    _pool_quality_score,
    _pool_simhash_fp,
    document_quality_score,
    detect_language as detect_lang_quality,
)
from src.data.streaming import (
    MassiveDataCollector,
    ShardCoordinator,
    stream_dataset_with_fallbacks,
)
from src.data.registry import DatasetRegistry, DatasetInfo, build_registry, extract_text
from src.data.shards import ShardProgressStore, build_shard_plan
from src.data.ast_filter import (
    _pool_filter_code,
    extract_functions,
    filter_code,
    identifier_ratio,
    executable_ratio,
)
from src.data.function_sampler import (
    sample_functions_or_fallback,
    is_code_empty_or_trivial,
)
from src.data.health_reporter import DatasetHealthReport
from src.data.metadata_cache import DatasetMetadataCache, SCHEMA_VERSION as METADATA_CACHE_SCHEMA_VERSION
from src.data.sanity import run_sanity_checks
from src.data.drivers import (
    BuilderCache,
    DRIVER_KIND_FILE,
    DRIVER_KIND_LOCAL,
    DRIVER_KIND_SCRIPT,
    DRIVER_KIND_STREAMING,
    detect_driver,
)

logger = logging.getLogger(__name__)


CODE_LANGUAGE_WEIGHTS = {
    'python': 0.35, 'javascript': 0.12, 'typescript': 0.08,
    'cpp': 0.08, 'java': 0.08, 'rust': 0.08, 'go': 0.07,
    'c': 0.05, 'sql': 0.05, 'shell': 0.04,
}

DOMAIN_MARKERS: Dict[str, List[str]] = {
    'backend': ['server', 'api', 'rest', 'graphql', 'database', 'sql', 'http', 'endpoint'],
    'frontend': ['react', 'vue', 'angular', 'css', 'html', 'dom', 'component', 'ui'],
    'ml': ['tensor', 'layer', 'loss', 'gradient', 'train', 'inference', 'model', 'dataset'],
    'cv': ['image', 'pixel', 'convolution', 'filter', 'detect', 'segment', 'bbox'],
    'nlp': ['token', 'embedding', 'attention', 'transformer', 'bert', 'gpt', 'text'],
    'databases': ['select', 'insert', 'join', 'index', 'query', 'table', 'schema'],
    'networking': ['socket', 'tcp', 'udp', 'packet', 'protocol', 'connect', 'listen'],
    'os': ['memory', 'thread', 'process', 'signal', 'kernel', 'alloc', 'page', 'scheduler'],
    'security': ['auth', 'encrypt', 'hash', 'password', 'token', 'permission', 'access'],
    'embedded': ['gpio', 'i2c', 'spi', 'uart', 'interrupt', 'register', 'firmware'],
    'mobile': ['activity', 'view', 'intent', 'fragment', 'widget', 'screen', 'layout'],
    'web': ['route', 'middleware', 'template', 'session', 'cookie', 'static', 'url'],
    'compilers': ['parser', 'lexer', 'ast', 'ir', 'codegen', 'optimizer', 'compiler'],
    'cryptography': ['aes', 'rsa', 'sha', 'encrypt', 'decrypt', 'cipher', 'key'],
    'gamedev': ['sprite', 'mesh', 'shader', 'physics', 'collision', 'render', 'game'],
    'graphics': ['shader', 'vertex', 'fragment', 'raster', 'pipeline', 'opengl', 'vulkan'],
    'scientific': ['simulation', 'equation', 'numerical', 'matrix', 'vector', 'fourier'],
    'robotics': ['ros', 'joint', 'kinematics', 'sensor', 'actuator', 'pid', 'control'],
    'data_engineering': ['spark', 'hadoop', 'kafka', 'etl', 'pipeline', 'warehouse', 'stream'],
    'cloud': ['aws', 'gcp', 'azure', 'kubernetes', 'docker', 'deploy', 'scale'],
    'algorithms': ['algorithm', 'data structure', 'sort', 'search', 'graph', 'tree', 'dp'],
}

QUALITY_THRESHOLDS = {
    'code': 0.35, 'math': 0.30, 'science': 0.30,
    'web_text': 0.35, 'books': 0.40, 'docs': 0.35,
    'wiki': 0.35, 'structured_knowledge': 0.30,
    'long_context': 0.35,
}

# Hard cap for materializing any single dataset before training starts.
# Prevents `limit=None` from iterating an entire hub corpus (e.g. opc-fineweb-code-corpus
# is multi-TB / ~150M samples) in a `list(...)` during dataset construction.
# Matches the existing convention (TokenizerConfig.max_samples default = 100000).
DEFAULT_MAX_SAMPLES_PER_DATASET = 100_000

# Default size for the persistent cleanup/filter process pool when no explicit
# cleanup_pool_size is configured. Previously fell back to min(32, cpu_count),
# which oversubscribed many-core hosts (box log: "32 workers (fork context)")
# and fought the CUDA/IO threads for the same cores. 8 is a deliberate,
# observable bound for filter-stage parallelism.
DEFAULT_CLEANUP_POOL_WORKERS = 8

# Format epoch for EVERY packed-cache key (unit + registry + stage + legacy
# tokenized). Bump ONLY when the on-disk cache format/semantics change so the
# whole cache family invalidates in lockstep (mandate Phase 11: a metadata-
# cache format change must not silently reuse stale packed units).
PACKED_CACHE_FORMAT_VERSION = 1


class ShardedStreamIterator:
    """Iterator wrapper exposing a mutable stats holder on a plain generator
    (generator objects disallow attribute assignment, so the stats
    SimpleNamespace must be attached here instead)."""

    def __init__(self, gen: Generator, stats: "SimpleNamespace") -> None:
        self._gen = gen
        self.stats = stats

    def __iter__(self) -> "ShardedStreamIterator":
        return self

    def __next__(self) -> Dict[str, Any]:
        return next(self._gen)

    def close(self) -> None:
        self._gen.close()


def detect_domain(text: str, category: str) -> str:
    if category != 'code':
        return category
    lower = text.lower()
    scores = {}
    for domain, markers in DOMAIN_MARKERS.items():
        count = sum(1 for m in markers if m in lower)
        if count > 0:
            scores[domain] = count
    if scores:
        return max(scores, key=scores.get)
    return 'backend'


def detect_language(text: str, dataset_name: str = "") -> str:
    name_lower = dataset_name.lower()
    if 'python' in name_lower or '/py' in name_lower:
        return 'python'
    if 'javascript' in name_lower or '/js' in name_lower:
        return 'javascript'
    if 'typescript' in name_lower or '/ts' in name_lower:
        return 'typescript'
    if '/java' in name_lower and 'javascript' not in name_lower:
        return 'java'
    if '/rust' in name_lower:
        return 'rust'
    if '/go' in name_lower or 'golang' in name_lower:
        return 'go'
    if 'def ' in text and 'import ' in text:
        return 'python'
    if 'function ' in text and ('const ' in text or 'var ' in text):
        return 'javascript'
    return 'other'


def random_window_sample(tokens: list, max_seq_length: int) -> list:
    if len(tokens) <= max_seq_length:
        return tokens
    start = random.randint(0, len(tokens) - max_seq_length)
    return tokens[start:start + max_seq_length]


LICENSE_KEYWORDS = [
    'copyright', 'license', 'spdx', 'apache', 'mit license', 'mit ',
    'licensed under', 'all rights reserved', 'gnu general public',
    'gnu lesser general public', 'gnu affero general public',
    'redistribution and use', 'permission is hereby granted',
    'this file is part of', 'generated by', 'auto-generated',
    'do not edit', 'this code was generated', 'this file was generated',
    'source code is machine-generated', 'generated by the protocol',
    'generated by the swagger', 'generated automatically',
    'autogenerated', 'automatically generated', 'bsd license',
    'mozilla public license', 'the mit license', 'under mit',
    'dual license', 'lgpl', 'agpl', 'mpl 2.0', 'mpl-2.0',
    'apache 2.0', 'apache-2.0', 'apache license 2.0',
    'license 2.0', 'licensing', 'all rights reserved',
]


def remove_boilerplate(text: str, keywords: Optional[List[str]] = None) -> str:
    if keywords is None:
        keywords = LICENSE_KEYWORDS
    lines = text.split('\n')
    if not lines:
        return text
    comments = ('#', '//', '/*', '*', '*/', '"""', "'''", '--', ';')
    start_line = 0
    in_license_block = False
    for i, line in enumerate(lines[:50]):
        stripped = line.strip().lower()
        if not stripped or stripped in comments:
            continue
        if any(kw in stripped for kw in keywords):
            start_line = i + 1
            in_license_block = True
        elif in_license_block and any(stripped.startswith(c) for c in comments):
            start_line = i + 1
        else:
            break
    lines = lines[start_line:]
    text = '\n'.join(lines)
    text = re.sub(r'\n{4,}', '\n\n', text)
    text = '\n'.join(line.rstrip() for line in text.split('\n'))
    return text.strip()


BOILERPLATE_FILE_PATTERNS = [
    '_pb2.py', '_pb2_grpc.py', '_grpc_pb2.py', '_pb2.pyi',
    'generated/', 'build/', 'dist/', 'vendor/',
    'node_modules/', '.git/', '__pycache__/',
    '.min.js', '.min.css', '.bundle.js',
    'package-lock.json', 'yarn.lock', 'go.sum',
    'pnpm-lock.yaml', 'poetry.lock', 'gemfile.lock',
    'requirements.lock', 'pipfile.lock',
    '.tfstate', '.tfstate.backup',
    '.parquet', '.arrow',
    '.o', '.obj', '.bin', '.exe', '.dll', '.so', '.dylib',
]


def should_skip_file(file_path: str, patterns: Optional[List[str]] = None) -> bool:
    if not file_path:
        return False
    if patterns is None:
        patterns = BOILERPLATE_FILE_PATTERNS
    path_lower = file_path.lower()
    return any(pattern in path_lower for pattern in patterns)


def compute_quality_score(text: str, category: str, language: str = "text") -> float:
    return document_quality_score(text, category, language)["final"]


def passes_quality_filter(text: str, category: str, quality_score: Optional[float] = None, lang: str = "text") -> bool:
    threshold = quality_score if quality_score is not None else QUALITY_THRESHOLDS.get(category, 0.30)
    return compute_quality_score(text, category, lang) >= threshold


# â”€â”€ Process-pool workers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Pure, picklable worker functions used to parallelize the expensive
# per-sample stages. They live in src.data.quality / src.data.ast_filter
# (light import chains) so that worker processes start fast â€” importing
# this pipeline module inside a spawned worker would drag in the full
# torch/transformers/datasets stack. Nothing stateful runs in workers:
# deduplicators keep their running state in the main process.
# (workers: _pool_simhash_fp, _pool_quality_score, _pool_filter_code)


def pack_sequences(
    tokenized_samples: list,
    max_seq_length: int,
    eos_token_id: int,
) -> tuple:
    packed_examples = []
    current_tokens = []
    current_labels = []
    current_quality = 0.0
    current_segments = 0
    total_padded_tokens = 0
    total_slots = 0
    for sample in tokenized_samples:
        tokens = sample['input_ids']
        if isinstance(tokens, list):
            tokens = tokens
        else:
            tokens = tokens.tolist()
        if len(tokens) > max_seq_length:
            start = random.randint(0, len(tokens) - max_seq_length)
            tokens = tokens[start:start + max_seq_length]
        doc_quality = sample.get('quality_score', 0.5)
        sep = [eos_token_id]
        needed = len(tokens) + len(sep) if current_tokens else len(tokens)
        if len(current_tokens) + needed <= max_seq_length:
            if current_tokens:
                current_tokens.extend(sep)
                current_labels.extend([-100])
            current_tokens.extend(tokens)
            current_labels.extend(tokens)
            current_quality += doc_quality
            current_segments += 1
        else:
            if current_tokens:
                pad_len = max_seq_length - len(current_tokens)
                total_padded_tokens += pad_len
                total_slots += max_seq_length
                packed_examples.append({
                    'input_ids': current_tokens + [eos_token_id] * pad_len,
                    'labels': current_labels + [-100] * pad_len,
                    'attention_mask': [1] * len(current_tokens) + [0] * pad_len,
                    '_segments': current_segments,
                    '_avg_quality': round(current_quality / max(current_segments, 1), 4),
                })
            current_tokens = tokens
            current_labels = tokens[:]
            current_quality = doc_quality
            current_segments = 1
    if current_tokens:
        pad_len = max_seq_length - len(current_tokens)
        total_padded_tokens += pad_len
        total_slots += max_seq_length
        packed_examples.append({
            'input_ids': current_tokens + [eos_token_id] * pad_len,
            'labels': current_labels + [-100] * pad_len,
            'attention_mask': [1] * len(current_tokens) + [0] * pad_len,
            '_segments': current_segments,
            '_avg_quality': round(current_quality / max(current_segments, 1), 4),
        })
    packing_eff = 1.0 - (total_padded_tokens / max(total_slots, 1)) if total_slots else 0.0
    total_padding_pct = (total_padded_tokens / max(total_slots, 1)) * 100 if total_slots else 0.0
    logger.debug("Packing: %d seqs, eff=%.1f%%, padding=%.1f%%",
                 len(packed_examples), packing_eff * 100, total_padding_pct)
    return packed_examples, packing_eff


class WeightedMixedDataset(TorchDataset):
    def __init__(
        self,
        datasets_with_weights: list,
        total_samples: int,
        balance_languages: bool = False,
        lang_target: Optional[Dict[str, float]] = None,
        balance_domains: bool = False,
        domain_target: Optional[Dict[str, float]] = None,
        rng_seed: int = 42,
        quality_scores: Optional[List[float]] = None,
    ):
        self.balance_languages = balance_languages
        self.balance_domains = balance_domains
        self.lang_target = lang_target
        self.domain_target = domain_target
        self._rng_seed = rng_seed
        self.datasets = [(d, n) for d, _, n in datasets_with_weights]
        base_weights = [w for _, w, _ in datasets_with_weights]
        qs = quality_scores or ([1.0] * len(base_weights))
        # Effective weight = base_weight * quality_score
        eff = [bw * q for bw, q in zip(base_weights, qs)]
        total_w = sum(eff)
        if total_w <= 0:
            # All-zero weights (e.g. after normalization with zero category
            # targets) would divide by zero â€” fall back to uniform.
            logger.warning("All dataset weights are zero â€” using uniform weights")
            eff = [1.0] * len(eff)
            total_w = float(len(eff))
        self.weights = [w / total_w for w in eff]
        self.total_samples = total_samples
        self._built = False
        self._dataset_sizes: List[int] = []
        self._consumed: List[int] = []
        self.assignments: List[int] = []
        self.indices: List[Tuple[int, int]] = []

    def _ensure_ready(self) -> None:
        if self._built:
            return
        # All dataset traversal / sampling work is deferred out of __init__ so
        # constructing the mixed dataset never iterates or sizes the underlying
        # datasets before training actually needs them.
        start = time.perf_counter()
        self._dataset_sizes = [len(ds) for ds, _ in self.datasets]
        self._consumed = [0] * len(self.datasets)
        rng = np.random.default_rng(seed=self._rng_seed)

        if self.balance_languages and self.lang_target:
            lang_ds_map: Dict[str, list] = {}
            for di, (ds, name) in enumerate(self.datasets):
                lang = getattr(ds, 'language', None) or 'other'
                lang_ds_map.setdefault(lang, []).append(di)
            total_lang_w = sum(v for v in self.lang_target.values())
            lang_samples = {}
            for lang, frac in self.lang_target.items():
                lang_samples[lang] = int(self.total_samples * (frac / total_lang_w))
            self._build_from_stratification(rng, lang_ds_map, lang_samples)
        elif self.balance_domains and self.domain_target:
            domain_ds_map: Dict[str, list] = {}
            for di, (ds, name) in enumerate(self.datasets):
                domain = getattr(ds, 'domain', None) or 'other'
                domain_ds_map.setdefault(domain, []).append(di)
            total_domain_w = sum(v for v in self.domain_target.values())
            domain_samples = {}
            for domain, frac in self.domain_target.items():
                domain_samples[domain] = int(self.total_samples * (frac / total_domain_w))
            self._build_from_stratification(rng, domain_ds_map, domain_samples)
        else:
            self._sample_adaptive(rng)
        self._built = True
        logger.info("[TIMER] WeightedMixedDataset._ensure_ready: %.1fs (%d datasets, %d samples)",
                    time.perf_counter() - start, len(self.datasets), self.total_samples)

    def _effective_weights(self) -> List[float]:
        remaining_ratios = []
        for size, consumed in zip(self._dataset_sizes, self._consumed):
            remaining = max(0, size - consumed)
            ratio = remaining / max(size, 1)
            remaining_ratios.append(ratio)
        eff = [w * r for w, r in zip(self.weights, remaining_ratios)]
        total = sum(eff)
        return [w / total for w in eff] if total > 0 else self.weights

    def _all_exhausted(self) -> bool:
        """True when every constituent dataset has been fully consumed."""
        return all(c >= s for c, s in zip(self._consumed, self._dataset_sizes))

    def _sample_adaptive(self, rng, group_map=None, group_name="", num_samples=None):
        self.assignments = []
        self.indices = []
        n = self.total_samples if num_samples is None else num_samples
        for _ in range(n):
            if self._all_exhausted():
                # Every dataset fully consumed â€” start a fresh pass instead of
                # re-sampling consumed indices (that would fabricate an epoch
                # of ~100% duplicate samples).
                self._consumed = [0] * len(self.datasets)
            eff_w = self._effective_weights()
            if sum(eff_w) == 0:
                eff_w = [1.0 / len(eff_w)] * len(eff_w)
            if group_map is not None:
                ds_indices = group_map.get(group_name, list(range(len(self.datasets))))
                sub_w = [eff_w[di] for di in ds_indices]
                sw = sum(sub_w)
                sub_p = [w / sw for w in sub_w] if sw > 0 else [1.0 / len(sub_w)] * len(sub_w)
                chosen_di = rng.choice(ds_indices, p=sub_p)
            else:
                chosen_di = int(rng.choice(len(self.datasets), p=eff_w))
            self._consumed[chosen_di] += 1
            ds, _ = self.datasets[chosen_di]
            idx = int(rng.integers(0, len(ds)))
            self.assignments.append(chosen_di)
            self.indices.append((chosen_di, idx))

    def _build_from_stratification(self, rng, group_map, group_samples):
        self.assignments = []
        self.indices = []
        adjusted = sum(group_samples.values())
        # Correct overflow/underflow on a group that actually contributes
        # datasets (previously always the first key, even when that group's
        # dataset list was empty â†’ samples silently lost).
        def _adjust_key(delta: int):
            # Prefer the group with the most backing datasets.
            ranked = sorted(group_samples.keys(),
                            key=lambda g: len(group_map.get(g, [])), reverse=True)
            for k in ranked:
                if group_map.get(k):
                    group_samples[k] += delta
                    return
            # Fallback: no group has datasets (should not happen).
            group_samples[list(group_samples.keys())[0]] += delta

        if adjusted < self.total_samples:
            _adjust_key(self.total_samples - adjusted)
        elif adjusted > self.total_samples:
            _adjust_key(-(adjusted - self.total_samples))
        for group, count in group_samples.items():
            ds_indices = group_map.get(group, [])
            if not ds_indices:
                continue
            self._sample_adaptive(rng, group_map, group, num_samples=count)

    def __len__(self):
        self._ensure_ready()
        return self.total_samples

    def __getitem__(self, idx):
        self._ensure_ready()
        dataset_idx, sample_idx = self.indices[idx]
        dataset, name = self.datasets[dataset_idx]
        return dataset[sample_idx]

    def stats(self) -> Dict[str, Any]:
        self._ensure_ready()
        return {
            "total_samples": self.total_samples,
            "datasets": len(self.datasets),
            "weights": [round(w, 4) for w in self.weights],
            "consumed": self._consumed,
            "remaining_ratios": [round(1.0 - c / max(s, 1), 3) for s, c in zip(self._dataset_sizes, self._consumed)],
        }


class DataPipeline:
    def __init__(
        self,
        cfg: Config,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer
        if cfg.data.hf_token:
            os.environ.setdefault("HF_TOKEN", cfg.data.hf_token)
        dedup_method = cfg.data.quality.deduplication.method
        dedup_threshold = cfg.data.quality.deduplication.threshold
        if dedup_method == "simhash":
            self.dedup = SimHashDeduplicator(threshold=dedup_threshold)
        elif dedup_method == "minhash":
            self.dedup = MinHashDeduplicator(threshold=dedup_threshold)
        elif dedup_method == "embedding":
            self.dedup = SemanticDeduplicator(threshold=dedup_threshold)
        else:
            self.dedup = ExactDeduplicator()
        self.contamination = ContaminationFilter(benchmarks=cfg.data.quality.contamination.benchmarks)
        self.scorer = QualityScorer()
        self.exact_dedup = ExactDeduplicator()
        self._collector: Optional[MassiveDataCollector] = None
        self._cleanup_pool = None
        self._shard_store: Optional[ShardProgressStore] = None
        self.health_report = DatasetHealthReport(
            name=cfg.model.name,
            config_path=getattr(cfg, '_config_path', ''),
        )
        mc = cfg.data.metadata_cache
        self.meta_cache = DatasetMetadataCache(
            Path(mc.dir), enabled=mc.enabled, fingerprint_version=mc.fingerprint_version,
        )
        self._builder_cache = BuilderCache(
            Path(mc.dir) / "builders", enabled=mc.enabled,
            fingerprint_version=mc.fingerprint_version,
        )
        self._driver_cache: Dict[str, tuple] = {}
        self._driver_lock = threading.Lock()
        # Global cancellation: set on close()/shutdown so in-flight unit builds
        # stop at their next checkpoint instead of running to completion on
        # daemon threads while the process tears down.
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        """Request cancellation of all in-flight builds. Cooperative: builds
        observe the flag at chunk/loop boundaries and raise
        ``UnitBuildCancelled``. Idempotent."""
        if getattr(self, "_cancelled", None) is None:
            # __new__-constructed test doubles skip __init__; materialize the
            # flag lazily so cancellation still works.
            self._cancelled = threading.Event()
        self._cancelled.set()

    def raise_if_cancelled(self, cancel_event=None) -> None:
        """Raise ``UnitBuildCancelled`` when the pipeline is shutting down or
        the slot's owner cancelled it. Checked at cheap, bounded cadence so a
        build never runs away past a cancellation request."""
        if self._is_cancelled(cancel_event):
            # Lazy import: ``src.training/__init__`` eagerly re-exports
            # ``TrainingPipeline``, which imports back into this module, so a
            # top-level import here forms a circular-import hazard that breaks
            # ``tests/test_data_pipeline.py`` / ``test_async_pipeline_overlap.py``
            # when this module loads first in a fresh process.
            from src.training.asyncprefetch import UnitBuildCancelled
            raise UnitBuildCancelled()

    def _is_cancelled(self, cancel_event=None) -> bool:
        cancelled = getattr(self, "_cancelled", None)
        return ((cancelled is not None and cancelled.is_set())
                or (cancel_event is not None and cancel_event.is_set()))

    def processing_signature(self) -> str:
        """Signature of all preprocessing/quality settings that affect what a
        dataset produces. Used as part of the metadata-cache fingerprint so any
        change invalidates cached resolutions."""
        pp = self.cfg.data.preprocessing
        q = self.cfg.data.quality
        return "|".join([
            q.deduplication.method, f"{q.deduplication.threshold:.4f}",
            str(self.cfg.data.ast_filter.code_filtering),
            str(self.cfg.data.function_sampling.enabled),
            str(pp.remove_boilerplate), str(pp.min_text_length),
            "|".join(sorted(pp.boilerplate_file_patterns or [])),
            "|".join(sorted(set(pp.autogen_patterns or []))),
            str(self.cfg.training.max_seq_length),
            str(self.cfg.model.architecture.vocab_size),
        ])

    @property
    def tokenizer_signature(self) -> str:
        tok = self.tokenizer
        n = getattr(tok, "vocab_size", None)
        if n is None:
            try:
                n = len(tok)  # type: ignore[arg-type]
            except Exception:
                n = 0
        return f"{getattr(tok, 'name_or_path', '')}|{n}"

    @property
    def collector(self) -> MassiveDataCollector:
        if self._collector is None:
            self._collector = MassiveDataCollector(self.cfg.data.datasets)
        return self._collector

    def format_prompt(
        self,
        language: str,
        problem: str,
        style: str = "standard",
        constitution: Optional[List[str]] = None,
    ) -> str:
        if language.strip().lower() == "text":
            prompt = f"### Instruction\nAnswer the following request:\n{problem}\n\n"
        else:
            prompt = f"### Instruction\nWrite a {language} solution for the following problem:\n{problem}\n\n"
        if style == "constitutional" and constitution:
            block = "\n".join(f"- {p}" for p in constitution if p.strip())
            prompt = f"### Constitution\n{block}\n\n{prompt}### Alignment Note\nFollow the constitution above.\n\n### Response\n"
        else:
            prompt += "### Response\n"
        return prompt

    def tokenize_supervised(
        self,
        texts: List[Tuple[str, str]],
        max_length: int,
    ) -> Dict[str, Any]:
        batch_input_ids: List[List[int]] = []
        batch_attention_mask: List[List[int]] = []
        batch_labels: List[List[int]] = []
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
        for prompt, response in texts:
            full_text = f"{prompt}{response}{self.tokenizer.eos_token or ''}"
            prompt_ids = self.tokenizer(
                prompt, add_special_tokens=False, truncation=True, max_length=max_length,
            )["input_ids"]
            full_ids = self.tokenizer(
                full_text, add_special_tokens=False, truncation=True, max_length=max_length,
            )["input_ids"]
            attention_mask = [1] * len(full_ids)
            labels = list(full_ids)
            prompt_len = len(prompt_ids)
            labels[:prompt_len] = [-100] * prompt_len
            pad_len = max_length - len(full_ids)
            if pad_len > 0:
                full_ids.extend([pad_id] * pad_len)
                attention_mask.extend([0] * pad_len)
                labels.extend([-100] * pad_len)
            batch_input_ids.append(full_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
        }

    def build_sft_dataset(
        self,
        data: List[Dict[str, str]],
        style: str = "standard",
        constitution: Optional[List[str]] = None,
        apply_quality_filter: bool = True,
        apply_dedup: bool = True,
        apply_contamination: bool = True,
    ) -> Dataset:
        max_len = self.cfg.training.max_seq_length
        pairs: List[Tuple[str, str]] = []
        for ex in data:
            lang = ex.get("language", "python")
            prob = ex.get("instruction") or ex.get("problem") or ""
            inp = ex.get("input", "")
            sol = ex.get("output") or ex.get("solution") or ""
            if not sol:
                continue
            if apply_quality_filter:
                content = f"{prob} {inp} {sol}"
                if not QualityFilter.check_length(content, self.cfg.data.quality.min_length, self.cfg.data.quality.max_length):
                    continue
                if not QualityFilter.is_high_quality_content(content):
                    continue
            if apply_contamination and self.contamination.is_contaminated(f"{prob} {sol}"):
                continue
            if inp:
                prob = f"{prob}\n{inp}"
            prompt = self.format_prompt(lang, prob, style, constitution)
            pairs.append((prompt, sol))
        if apply_dedup:
            unique: List[Tuple[str, str]] = []
            for p, r in pairs:
                if not self.dedup.is_duplicate(r):
                    unique.append((p, r))
            pairs = unique
        if not pairs:
            logger.warning("No valid samples after filtering!")
            return Dataset.from_dict({"input_ids": [], "attention_mask": [], "labels": []})
        ds = Dataset.from_dict({"idx": list(range(len(pairs)))})
        ds = ds.map(
            lambda examples: self.tokenize_supervised(
                [(pairs[i][0], pairs[i][1]) for i in examples["idx"]],
                max_length=max_len,
            ),
            batched=True,
            remove_columns=["idx"],
        )
        logger.info("Built SFT dataset: %d samples", len(ds))
        return ds

    def build_preference_dataset(
        self,
        data: List[Dict[str, Any]],
    ) -> Dataset:
        max_len = self.cfg.training.max_seq_length
        chosen_list, rejected_list = [], []
        for ex in data:
            chosen = ex.get("chosen") or ex.get("response") or ex.get("output") or ""
            rejected = ex.get("rejected") or ex.get("response_rejected") or ""
            if not chosen or not rejected:
                continue
            prompt_text = ex.get("prompt") or ex.get("instruction") or ""
            chosen_list.append((prompt_text, chosen))
            rejected_list.append((prompt_text, rejected))
        if not chosen_list:
            return Dataset.from_dict({"input_ids": [], "attention_mask": [], "labels": []})
        chosen_ds = Dataset.from_dict({"idx": list(range(len(chosen_list)))})
        chosen_ds = chosen_ds.map(
            lambda ex: self.tokenize_supervised(
                [(chosen_list[i][0], chosen_list[i][1]) for i in ex["idx"]],
                max_len,
            ),
            batched=True, remove_columns=["idx"],
        )
        rejected_ds = Dataset.from_dict({"idx": list(range(len(rejected_list)))})
        rejected_ds = rejected_ds.map(
            lambda ex: self.tokenize_supervised(
                [(rejected_list[i][0], rejected_list[i][1]) for i in ex["idx"]],
                max_len,
            ),
            batched=True, remove_columns=["idx"],
        )
        combined = Dataset.from_dict({
            "chosen_input_ids": chosen_ds["input_ids"],
            "chosen_attention_mask": chosen_ds["attention_mask"],
            "chosen_labels": chosen_ds["labels"],
            "rejected_input_ids": rejected_ds["input_ids"],
            "rejected_attention_mask": rejected_ds["attention_mask"],
            "rejected_labels": rejected_ds["labels"],
        })
        logger.info("Built preference dataset: %d pairs", len(combined))
        return combined

    def build_stage_dataset(
        self,
        raw_samples: List[Dict[str, str]],
    ) -> Dataset:
        if not raw_samples:
            return Dataset.from_dict({"input_ids": [], "attention_mask": [], "labels": []})
        ds = Dataset.from_list(raw_samples)
        ds = ds.map(
            self._preprocess_batch,
            batched=True,
            batch_size=256,
            remove_columns=[c for c in ["instruction", "input", "output", "language"] if c in ds.column_names],
            desc="Tokenizing stage dataset",
        )
        return ds

    def _get_cache_key(self, ds_info: DatasetEntryConfig, stage_name: str) -> str:
        # Dataset identity + preprocessing/quality config must all be part of
        # the key, otherwise changing dedup/quality/filtering settings (or the
        # dataset name) silently reuses the previously cached packed dataset.
        pp = self.cfg.data.preprocessing
        q = self.cfg.data.quality
        raw = "|".join([
            ds_info.path, ds_info.name or "", ds_info.split, str(ds_info.max_samples),
            str(self.cfg.training.max_seq_length),
            str(self.cfg.model.architecture.vocab_size),
            q.deduplication.method, f"{q.deduplication.threshold:.4f}",
            str(self.cfg.data.ast_filter.code_filtering),
            str(pp.remove_boilerplate), str(pp.min_text_length),
            f"v{PACKED_CACHE_FORMAT_VERSION}",  # family-wide epoch (Phase 11)
        ])
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _cached_dataset_path(self, ds_info: DatasetEntryConfig, stage_name: str) -> Path:
        ck = self._get_cache_key(ds_info, stage_name)
        return Path(self.cfg.data.cache_dir) / "tokenized" / stage_name / ck

    def _registry_cache_key(self, info: DatasetInfo, ds_limit: int, accepted_target: int = 0) -> str:
        """Config-sensitive cache key for the registry packed-dataset cache.
        Any change that would alter filtering or packing invalidates the key."""
        pp = self.cfg.data.preprocessing
        q = self.cfg.data.quality
        sig = "|".join([
            info.path, info.name or "", str(info.weight), str(info.quality_score),
            str(ds_limit), str(accepted_target),
            str(self.cfg.training.max_seq_length),
            str(self.cfg.model.architecture.vocab_size),
            q.deduplication.method, f"{q.deduplication.threshold:.4f}",
            str(self.cfg.data.ast_filter.code_filtering),
            str(self.cfg.data.function_sampling.enabled),
            str(pp.remove_boilerplate), str(pp.min_text_length),
            "|".join(sorted(pp.boilerplate_file_patterns or [])),
            f"{QUALITY_THRESHOLDS.get(info.category, 0.30):.4f}",
            f"v{PACKED_CACHE_FORMAT_VERSION}",  # family-wide epoch (Phase 11)
        ])
        return hashlib.sha256(sig.encode()).hexdigest()[:16]

    def _registry_cache_path(self, info: DatasetInfo, cache_key: str) -> Path:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{info.path}_{info.name or 'default'}")
        return Path(self.cfg.data.cache_dir) / "registry" / f"{safe_name}_{cache_key}.pt"

    def _load_registry_dataset_cache(self, cache_path: Path, cache_key: str, ds_key: str, info: DatasetInfo):
        """Try to load a packed-dataset cache entry. Returns (Dataset, meta) or None."""
        try:
            if not cache_path.exists():
                return None
            obj = torch.load(cache_path)
            if obj.get("cache_key") != cache_key:
                logger.warning("    [CACHE] key mismatch for %s â€” rebuilding", ds_key)
                return None
            if obj.get("version") != 1:
                logger.warning("    [CACHE] version mismatch for %s â€” rebuilding", ds_key)
                return None
            packed = obj["packed"]
            meta = obj["meta"]
            ds = Dataset.from_list(packed)
            logger.info("[TIMER] cache load %s: %d packed results", ds_key, len(packed))
            return ds, meta
        except Exception as e:
            logger.warning("    [CACHE] load failed for %s (%s) â€” rebuilding", ds_key, e)
            return None

    def _save_registry_dataset_cache(self, cache_path: Path, cache_key: str, packed: list, meta: dict) -> None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"version": 1, "cache_key": cache_key, "packed": packed, "meta": meta}, cache_path)
        except Exception as e:
            logger.warning("    [CACHE] save failed for %s (%s)", cache_path, e)

    def _stage_cache_key(self, stage_cfg, stage_index: int) -> str:
        """Config-sensitive cache key for a stage-level packed-dataset cache."""
        pp = self.cfg.data.preprocessing
        q = self.cfg.data.quality
        tok = self.tokenizer
        cats = sorted(stage_cfg.categories or [])
        wts = sorted((stage_cfg.weights or {}).items())
        sig = "|".join([
            f"stage{stage_index}",
            stage_cfg.name or "",
            "|".join(cats),
            "|".join(f"{k}:{v}" for k, v in wts),
            str(stage_cfg.steps),
            str(stage_cfg.max_samples_per_dataset),
            str(self.cfg.training.max_seq_length),
            str(self.cfg.model.architecture.vocab_size),
            str(getattr(tok, "name_or_path", "")), str(getattr(tok, "vocab_size", "")),
            q.deduplication.method, f"{q.deduplication.threshold:.4f}",
            str(self.cfg.data.ast_filter.code_filtering),
            str(self.cfg.data.function_sampling.enabled),
            str(pp.remove_boilerplate), str(pp.min_text_length),
            "|".join(sorted(pp.boilerplate_file_patterns or [])),
            "|".join(f"{c}:{QUALITY_THRESHOLDS.get(c, 0.30):.4f}" for c in cats),
            str(self.cfg.data.sampler.balance_by),
            f"v{PACKED_CACHE_FORMAT_VERSION}",  # family-wide epoch (Phase 11)
        ])
        return hashlib.sha256(sig.encode()).hexdigest()[:16]

    def _load_stage_dataset_cache(self, stage_dir: Path, cache_key: str, stage_cfg):
        """Try to load a stage cache. Returns (WeightedMixedDataset, meta) or None."""
        try:
            packed_path = stage_dir / "packed.pt"
            if not packed_path.exists():
                return None
            obj = torch.load(packed_path)
            if obj.get("cache_key") != cache_key:
                logger.warning("[STAGE] key mismatch for %s â€” rebuilding", stage_dir)
                return None
            if obj.get("version") != 1:
                logger.warning("[STAGE] version mismatch for %s â€” rebuilding", stage_dir)
                return None
            entries = obj.get("datasets") or []
            if not entries:
                logger.warning("[STAGE] empty cache %s â€” rebuilding", stage_dir)
                return None
            all_tokenized = [
                (Dataset.from_list(e["packed"]), e["weight"], e["path"], e["category"], e["avg_qs"])
                for e in entries
            ]
            result = self._unit_or_mixed(all_tokenized)
            result._entries = all_tokenized
            result._dataset_metas = [e.get("meta") or {} for e in entries]
            meta = obj.get("meta") or {}
            self._log_registry_summary(
                meta.get("n_listed", len(entries)), all_tokenized,
                meta.get("global_stats", {}),
                meta.get("lang_dist", {}), meta.get("domain_dist", {}),
                meta.get("rejection_reasons", {}),
                fallbacks_used=meta.get("fallbacks_used"),
            )
            return result, meta
        except Exception as e:
            logger.warning("[STAGE] cache load failed for %s (%s) â€” rebuilding", stage_dir, e)
            return None

    def _save_stage_dataset_cache(self, stage_dir: Path, cache_key: str, stage_cfg,
                                  entries: list, metas: list, meta: dict) -> None:
        try:
            stage_dir.mkdir(parents=True, exist_ok=True)
            datasets = []
            for (ds, weight, path, cat, avg_qs), dmeta in zip(entries, metas):
                datasets.append({
                    "path": path, "category": cat,
                    "weight": float(weight), "avg_qs": float(avg_qs),
                    "meta": dmeta, "packed": ds.to_list(),
                })
            torch.save({"version": 1, "cache_key": cache_key, "datasets": datasets, "meta": meta},
                       stage_dir / "packed.pt")
            (stage_dir / "stage_meta.json").write_text(json.dumps({
                "version": 1,
                "cache_key": cache_key,
                "stage_index": meta.get("stage_index"),
                "stage_name": meta.get("stage_name"),
                "categories": meta.get("categories"),
                "global_stats": meta.get("global_stats", {}),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, indent=2, default=str), encoding="utf-8")
            logger.info("[TIMER] stage cache saved: %s (%d datasets)", stage_dir, len(datasets))
        except Exception as e:
            logger.warning("[STAGE] cache save failed for %s (%s)", stage_dir, e)

    def build_pretrain_stage_dataset(self, stage_cfg, stage_index: int, total_stages: int):
        """Build the dataset for one pretraining stage (filtered by its categories),
        with its own disk cache under <stage_cache_dir>/stage<stage_index>/.

        Returns (WeightedMixedDataset, meta_dict). On a cache hit no streaming or
        tokenization happens â€” packed sequences are loaded straight from disk."""
        stage_dir = Path(self.cfg.training.pretrain.staging.stage_cache_dir) / f"stage{stage_index}"
        cache_key = self._stage_cache_key(stage_cfg, stage_index)
        if self.cfg.data.use_packed_cache:
            cached = self._load_stage_dataset_cache(stage_dir, cache_key, stage_cfg)
            if cached is not None:
                dataset, meta = cached
                logger.info("[STAGE] %d/%d %s â€” cache hit (no streaming)",
                            stage_index, total_stages, stage_cfg.name or "")
                return dataset, meta
        logger.info("[STAGE] %d/%d %s â€” building (no cache)",
                    stage_index, total_stages, stage_cfg.name or "")
        dataset = self.build_pretrain_dataset_from_registry(
            dataset_filter=stage_cfg.categories or None,
            max_samples_per_dataset=stage_cfg.max_samples_per_dataset,
        )
        entries = getattr(dataset, "_entries", None)
        if not entries:
            logger.warning("[STAGE] per-dataset entries unavailable â€” skipping stage cache save")
            return dataset, getattr(dataset, "_global_stats", {})
        meta = {
            "cache_key": cache_key,
            "stage_index": stage_index,
            "stage_name": stage_cfg.name or "",
            "categories": sorted(stage_cfg.categories or []),
            "n_listed": len(entries),
            "global_stats": getattr(dataset, "_global_stats", {}),
            "lang_dist": getattr(dataset, "_lang_dist", {}),
            "domain_dist": getattr(dataset, "_domain_dist", {}),
            "rejection_reasons": getattr(dataset, "_rejection_reasons", {}),
            "fallbacks_used": getattr(dataset, "_fallbacks_used", {}),
        }
        self._save_stage_dataset_cache(
            stage_dir, cache_key, stage_cfg,
            entries, getattr(dataset, "_dataset_metas", []) or [{}] * len(entries), meta,
        )
        return dataset, meta

    def unit_cache_key(self, info, ds_limit: Optional[int] = None) -> str:
        if ds_limit is None:
            ds_limit = (info.max_samples
                        or self.cfg.data.max_samples_per_dataset
                        or DEFAULT_MAX_SAMPLES_PER_DATASET)
        return hashlib.sha256("|".join([
            "unit-v1", info.path, info.name or "", info.split,
            info.category, str(ds_limit),
            self.processing_signature(), self.tokenizer_signature,
            # Phase 11: the metadata-cache schema and the packed-cache family
            # epoch are part of the unit key, so a metadata format/web fork
            # change and a packed-format change BOTH invalidate every unit.
            f"mcs{METADATA_CACHE_SCHEMA_VERSION}",
            f"pcf{PACKED_CACHE_FORMAT_VERSION}",
        ]).encode()).hexdigest()[:16]

    def unit_cache_dir(self, stage_index: int, unit_index: int) -> Path:
        staging = self.cfg.training.pretrain.staging
        return Path(staging.stage_cache_dir) / f"stage{stage_index}" / f"u{unit_index:03d}"

    def unit_cache_hit(self, info, stage_index: int, unit_index: int) -> bool:
        """True if the unit's packed cache exists and its key still matches the
        current dataset identity + preprocessing/tokenizer signature."""
        if not self.cfg.data.use_packed_cache:
            return False
        unit_dir = self.unit_cache_dir(stage_index, unit_index)
        rec_path = unit_dir / "unit_meta.json"
        try:
            if not (unit_dir / "packed.pt").exists() or not rec_path.exists():
                return False
            rec = json.loads(rec_path.read_text(encoding="utf-8"))
            return rec.get("key") == self.unit_cache_key(info)
        except Exception:
            return False

    def unit_cache_packed_count(self, info, stage_index: int, unit_index: int) -> Optional[int]:
        """Packed-sample count of a valid unit cache (None when not cached)."""
        if not self.unit_cache_hit(info, stage_index, unit_index):
            return None
        try:
            rec = json.loads((self.unit_cache_dir(stage_index, unit_index) / "unit_meta.json")
                             .read_text(encoding="utf-8"))
            return int(rec.get("packed_count", 0))
        except Exception:
            return None

    def has_metadata_record(self, info) -> bool:
        """True if a verified driver record (file metadata OR script builder)
        exists for this dataset â€” no HF resolution needed to stream it."""
        ppsig = self.processing_signature()
        toksig = self.tokenizer_signature
        return (
            self.meta_cache.verify(info, ppsig, toksig) is not None
            or self._builder_cache.verify(info, ppsig, toksig) is not None
        )

    def warm_metadata_cache(self, info) -> bool:
        """Resolve + cache a dataset's driver record (file list OR script
        builder identity) without downloading any data. Returns True when a
        usable record exists after the call. Never raises."""
        try:
            drv, diag = self._driver_for(
                info, self.processing_signature(), self.tokenizer_signature)
            return drv.record is not None or diag.get("driver_kind") != "streaming"
        except Exception as e:
            logger.warning("Metadata warm failed for %s/%s (%s)",
                           info.path, info.name or "default", e)
            return False

    def _health_stats_from_meta(self, meta: dict, info, max_len: int) -> dict:
        """Map a persisted dataset meta dict to add_dataset_stats kwargs.

        The per-text language/domain distributions stored in the meta are the
        authoritative source for the health report. The collapsed single-key
        ``{lang_d: accepted}`` form is only a legacy fallback when the meta has
        no distribution â€” otherwise a mixed dataset whose static
        ``info.language`` is None would collapse to a single 'unknown' bucket
        on cache replay (the box-run "unknown 44548 (100.0%)" regression).
        """
        cached_lang = meta.get("lang_d") or info.language
        cached_domain = meta.get("domain_d") or info.domain
        lang_dist = meta.get("lang_dist")
        domain_dist = meta.get("domain_dist")
        accepted = meta.get("accepted", 0)
        lang_report = dict(lang_dist) if lang_dist else (
            {cached_lang: accepted} if cached_lang else {})
        domain_report = dict(domain_dist) if domain_dist else (
            {cached_domain: accepted} if cached_domain else {})
        return {
            "path": info.path,
            "category": info.category,
            "weight": info.weight,
            "raw_count": meta.get("loaded", 0),
            "after_boilerplate": accepted,
            "after_quality": accepted,
            "after_dedup": accepted,
            "packed_count": meta.get("packed_count", 0),
            "total_tokens": meta.get("total_tokens", 0),
            "duplicate_removed": meta.get("rejected_dedup", 0),
            "rejection_reasons": meta.get("ledger", {}) or {},
            "quality_scores": meta.get("quality_scores", []),
            "lang_dist": lang_report,
            "domain_dist": domain_report,
            "token_lengths": meta.get("token_lengths", []),
            "max_seq_length": max_len,
        }

    def build_pretrain_dataset_unit(self, info, stage_index: int, unit_index: int,
                                 unit_total: int,
                                 cancel_event=None):
        """Build the packed dataset for ONE registry dataset ("training unit")
        with its own disk cache at <stage_cache_dir>/stage<N>/u<M>/.

        The unit cache key covers the dataset identity + sample cap + full
        preprocessing/tokenizer signature, so a warm cache skips streaming,
        tokenization and HF resolution entirely. Returns (WeightedMixedDataset,
        meta_dict).

        ``cancel_event`` (threading.Event) enables cooperative cancellation:
        when set, the build aborts at its next checkpoint by raising
        ``UnitBuildCancelled`` (the prefetch worker treats that as a quiet
        out-of-band stop, never as a failure).
        """
        self.raise_if_cancelled(cancel_event)
        staging = self.cfg.training.pretrain.staging
        unit_dir = Path(staging.stage_cache_dir) / f"stage{stage_index}" / f"u{unit_index:03d}"
        ds_limit = (info.max_samples
                    or self.cfg.data.max_samples_per_dataset
                    or DEFAULT_MAX_SAMPLES_PER_DATASET)
        key = self.unit_cache_key(info, ds_limit)

        if self.cfg.data.use_packed_cache:
            packed_path = unit_dir / "packed.pt"
            rec_path = unit_dir / "unit_meta.json"
            if packed_path.exists() and rec_path.exists():
                try:
                    rec = json.loads(rec_path.read_text(encoding="utf-8"))
                    if rec.get("key") == key:
                        obj = torch.load(packed_path)
                        entries = [(Dataset.from_list(obj["packed"]), obj["weight"],
                                    info.path, info.category, obj["avg_qs"])]
                        meta_from_cache = obj.get("meta") or {}
                        result = self._unit_or_mixed(entries, info)
                        result._entries = entries
                        result._dataset_metas = [meta_from_cache]
                        # Warm unit cache replays its persisted dataset meta
                        # (which carries the per-text lang/domain distributions)
                        # so the health report reflects this run instead of
                        # silently showing "no datasets" on resumed runs.
                        if meta_from_cache:
                            self.health_report.add_dataset_stats(
                                **self._health_stats_from_meta(
                                    meta_from_cache, info,
                                    self.cfg.training.max_seq_length))
                        logger.info("[UNIT] %d/%d %s/%s â€” cache hit (no stream, no tokenize)",
                                    unit_index, unit_total, info.path, info.name or "default")
                        return result, rec
                except Exception as e:
                    logger.warning("[UNIT] cache load failed for %s (%s) â€” rebuilding", unit_dir, e)

        logger.info("[UNIT] %d/%d %s/%s â€” building (no cache)",
                    unit_index, unit_total, info.path, info.name or "default")
        try:
            dataset = self.build_pretrain_dataset_from_registry(
                include=[(info.path, info.name)],
                max_samples_per_dataset=info.max_samples or self.cfg.data.max_samples_per_dataset,
                cancel_event=cancel_event,
            )
        except Exception as e:  # noqa: BLE001 â€” phase-tagged for the prefetch worker
            from src.training.asyncprefetch import _tag_phase
            _tag_phase(e, "construction",
                       dataset=info.path, name=info.name or "",
                       split=info.split, unit=f"{stage_index}/{unit_index}",
                       intended_type="Dataset")
            raise
        entries = getattr(dataset, "_entries", None)
        if not entries:
            raise RuntimeError(
                f"Unit build produced no dataset: {info.path}/{info.name or ''} â€” "
                "check HF access/fallbacks for this entry")
        ds0, weight0, path0, cat0, avg_qs0 = entries[0]
        meta = {"key": key, "path": path0, "name": info.name, "category": cat0,
                "weight": weight0, "avg_qs": avg_qs0, "packed_count": len(ds0)}
        if self.cfg.data.use_packed_cache:
            try:
                self.raise_if_cancelled(cancel_event)
                unit_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "packed": ds0.to_list(),
                    "weight": weight0,
                    "avg_qs": avg_qs0,
                    "meta": (getattr(dataset, "_dataset_metas", []) or [{}])[0],
                }, unit_dir / "packed.pt")
                (unit_dir / "unit_meta.json").write_text(
                    json.dumps(meta, indent=2, default=str), encoding="utf-8")
                logger.info("[UNIT] %d/%d unit cache saved: %s", unit_index, unit_total, unit_dir)
            except Exception as e:
                logger.warning("[UNIT] cache save failed (%s)", e)
        return dataset, meta

    def get_active_datasets_for_stage(self, stage) -> List[DatasetEntryConfig]:
        if stage.dataset_filter is None:
            return self.collector.get_dataset_list()
        all_ds = self.collector.get_dataset_list()
        return [d for d in all_ds if d.category in stage.dataset_filter]

    def build_pretrain_dataset(
        self,
        stage_name: str = "pretrain",
        dataset_filter: Optional[List[str]] = None,
        max_samples_per_dataset: Optional[int] = None,
    ) -> Dataset:
        # When use_registry is enabled, delegate to the registry method
        if self.cfg.data.use_registry:
            return self.build_pretrain_dataset_from_registry(
                dataset_filter=dataset_filter,
                max_samples_per_dataset=max_samples_per_dataset,
            )
        if dataset_filter is not None:
            datasets = [d for d in self.collector.get_dataset_list() if d.category in dataset_filter]
        else:
            datasets = self.collector.get_dataset_list()
        all_tokenized: List[Tuple[Dataset, DatasetEntryConfig]] = []
        total_raw = 0
        total_after_boilerplate = 0
        total_after_quality = 0
        total_after_dedup = 0
        total_after_ast = 0
        total_packed = 0
        total_tokens = 0
        lang_dist: Dict[str, int] = defaultdict(int)
        domain_dist: Dict[str, int] = defaultdict(int)
        rejection_reasons: Dict[str, int] = defaultdict(int)
        ast_filter_cfg = self.cfg.data.ast_filter
        func_sampling_cfg = self.cfg.data.function_sampling
        pp = self.cfg.data.preprocessing
        kw = pp.license_keywords
        patterns = pp.boilerplate_file_patterns
        min_text_len = pp.min_text_length

        for ds_idx, ds_info in enumerate(datasets, 1):
            ds_name = ds_info.path
            category = ds_info.category
            logger.info("")
            logger.info("--- Dataset [%d/%d]: %s (cat=%s, weight=%.2f, qs=%s) ---",
                         ds_idx, len(datasets), ds_name, category, ds_info.weight, ds_info.quality_score)
            cache_path = self._cached_dataset_path(ds_info, stage_name)
            if cache_path.exists():
                logger.info("    Loading cached from %s", cache_path)
                ds = load_from_disk(str(cache_path))
                logger.info("    Loaded %d packed samples from cache", len(ds))
                all_tokenized.append((ds, ds_info))
                total_packed += len(ds)
                continue
            try:
                limit = max_samples_per_dataset or ds_info.max_samples or DEFAULT_MAX_SAMPLES_PER_DATASET
                samples = list(self.collector.stream_single_dataset(ds_info, limit=limit, raw_text=True))
            except Exception as e:
                logger.error("    FAILED: %s â€” skipping", e)
                self.health_report.add_error(f"{ds_info.path}: {e}")
                continue
            if not samples:
                logger.warning("    No valid samples â€” skipping")
                continue
            raw_count = len(samples)
            total_raw += raw_count
            ledger = {"boilerplate_skip": 0, "too_short": 0, "quality_fail": 0,
                      "dedup": 0, "ast_reject": 0, "empty_trivial": 0, "contaminated": 0}
            cleaned: List[str] = []
            for s in samples:
                text = s.get("output") or s.get("text") or s.get("content") or ""
                if not text:
                    ledger["boilerplate_skip"] += 1
                    continue
                if pp.remove_boilerplate:
                    text = remove_boilerplate(text, keywords=kw)
                if should_skip_file(s.get("file_path", ""), patterns=patterns):
                    ledger["boilerplate_skip"] += 1
                    continue
                if len(text.strip()) < min_text_len:
                    ledger["too_short"] += 1
                    continue
                category = ds_info.category or s.get("category", "code")
                ds_qs = ds_info.quality_score
                if not passes_quality_filter(text, category, quality_score=ds_qs):
                    ledger["quality_fail"] += 1
                    continue
                if self.contamination.is_contaminated(text):
                    ledger["contaminated"] += 1
                    rejection_reasons["contamination"] += 1
                    continue
                if category in ('code',) and ast_filter_cfg.code_filtering:
                    lang = ds_info.language or detect_language(text, ds_name)
                    ok, reason = filter_code(text, lang, ast_filter_cfg)
                    if not ok:
                        ledger["ast_reject"] += 1
                        rejection_reasons[reason] += 1
                        continue
                if self.exact_dedup.is_duplicate(text):
                    ledger["dedup"] += 1
                    rejection_reasons["exact_dedup"] += 1
                    continue
                # Dedup: use whichever method was configured (the old code only
                # consulted SimHash even when minhash/exact/embedding was selected,
                # so those methods were constructed-but-never-called dummies).
                if hasattr(self.dedup, "is_duplicate") and self.dedup is not self.exact_dedup:
                    if self.dedup.is_duplicate(text):
                        ledger["dedup"] += 1
                        rejection_reasons[f"{self.cfg.data.quality.deduplication.method}_dedup"] += 1
                        continue
                if category in ('code',) and func_sampling_cfg.enabled and ds_info.function_sampling:
                    lang = ds_info.language or detect_language(text, ds_name)
                    extracted = sample_functions_or_fallback(text, lang, func_sampling_cfg)
                    if is_code_empty_or_trivial(extracted):
                        ledger["empty_trivial"] += 1
                        continue
                    text = extracted
                cleaned.append(text)
            total_after_boilerplate += raw_count - ledger["boilerplate_skip"] - ledger["too_short"]
            total_after_quality += raw_count - ledger["boilerplate_skip"] - ledger["too_short"] - ledger["quality_fail"]
            dedup_total = ledger["dedup"] + ledger["ast_reject"] + ledger["empty_trivial"] + ledger["contaminated"]
            total_after_dedup += raw_count - ledger["boilerplate_skip"] - ledger["too_short"] - ledger["quality_fail"] - dedup_total
            total_after_ast += len(cleaned)
            logger.info("    Boilerplate skipped: %d | too short: %d | quality fail: %d | contaminated: %d",
                        ledger["boilerplate_skip"], ledger["too_short"], ledger["quality_fail"], ledger["contaminated"])
            logger.info("    AST reject: %d | dedup: %d | empty/trivial: %d",
                        ledger["ast_reject"], ledger["dedup"], ledger["empty_trivial"])
            logger.info("    Survived: %d (%.1f%%)", len(cleaned), len(cleaned) / max(raw_count, 1) * 100)
            if not cleaned:
                logger.warning("    No samples survived â€” skipping")
                self.health_report.add_error(f"{ds_info.path}: all {raw_count} samples rejected")
                continue
            tokenized_samples = []
            eos_id = self.tokenizer.eos_token_id or 0
            max_len = self.cfg.training.max_seq_length
            quality_scores: List[float] = []
            ds_lang_dist: Dict[str, int] = defaultdict(int)
            ds_domain_dist: Dict[str, int] = defaultdict(int)
            for text in cleaned:
                lang = detect_language(text, ds_name)
                lang_dist[lang] += 1
                ds_lang_dist[lang] += 1
                domain = detect_domain(text, category)
                domain_dist[domain] += 1
                ds_domain_dist[domain] += 1
                tokens = self.tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
                tokens = random_window_sample(tokens, max_len)
                tokenized_samples.append({"input_ids": tokens})
            logger.info("    Tokenized: %d samples", len(tokenized_samples))
            packed, packing_eff = pack_sequences(tokenized_samples, max_len, eos_id)
            total_packed += len(packed)
            for p in packed:
                total_tokens += len(p["input_ids"])
            logger.info("    Packed: %d sequences (eff=%.1f%%)", len(packed), packing_eff * 100)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            ds = Dataset.from_list(packed)
            ds.save_to_disk(str(cache_path))
            logger.info("    Cached to %s", cache_path)
            all_tokenized.append((ds, ds_info))
            self.health_report.add_dataset_stats(
                path=ds_info.path,
                category=category,
                weight=ds_info.weight,
                raw_count=raw_count,
                after_boilerplate=raw_count - ledger["boilerplate_skip"] - ledger["too_short"],
                after_quality=raw_count - ledger["boilerplate_skip"] - ledger["too_short"] - ledger["quality_fail"],
                after_dedup=len(cleaned),
                packed_count=len(packed),
                total_tokens=sum(len(p["input_ids"]) for p in packed),
                duplicate_removed=ledger["dedup"],
                rejection_reasons=dict(ledger),
                quality_scores=quality_scores,
                lang_dist=dict(ds_lang_dist),
                domain_dist=dict(ds_domain_dist),
            )
            del samples, cleaned, tokenized_samples, packed
            gc.collect()
        logger.info("")
        logger.info("=" * 60)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info("Total raw: %d | boil: %d | quality: %d | dedup+AST: %d | packed: %d",
                     total_raw, total_after_boilerplate, total_after_quality, total_after_dedup, total_packed)
        logger.info("Total tokens: %d", total_tokens)
        logger.info("Language dist: %s", dict(sorted(lang_dist.items(), key=lambda x: -x[1])))
        top_rejections = sorted(rejection_reasons.items(), key=lambda x: -x[1])[:5]
        if top_rejections:
            logger.info("Top rejection reasons: %s", top_rejections)
        if total_packed:
            from src.utils.steps import (
                estimate_pretrain_steps,
                format_step_estimate,
            )
            ga = int(getattr(self.cfg.training.pretrain,
                             "gradient_accumulation_steps", None) or 1)
            _est = estimate_pretrain_steps(
                total_packed,
                self.cfg.training.pretrain.batch_size,
                gradient_accumulation_steps=ga,
                world_size=int(os.environ.get("WORLD_SIZE", "1") or 1),
            )
            logger.info("Est steps: %s", format_step_estimate(_est))
        if not all_tokenized:
            logger.warning("No datasets produced any samples!")
            self.health_report.compute_global_stats()
            if self.cfg.data.health_reporting.enabled:
                self.health_report.save(self.cfg.data.health_reporting.output_dir)
            return Dataset.from_dict({"input_ids": [], "attention_mask": [], "labels": []})
        if len(all_tokenized) == 1:
            result = all_tokenized[0][0]
        else:
            weighted = [(ds, ds_info.weight, ds_info.path) for ds, ds_info in all_tokenized]
            balance_tokens = self.cfg.data.sampler.balance_by == "tokens"
            if balance_tokens:
                token_counts = [len(ds) if hasattr(ds, '__len__') else 1 for ds, _ in all_tokenized]
                total_tok = sum(token_counts)
                if total_tok > 0:
                    weighted = [
                        (ds, ds_info.weight * (token_counts[i] / total_tok), ds_info.path)
                        for i, (ds, ds_info) in enumerate(all_tokenized)
                    ]
            total_est = min(ds.num_rows for ds, _ in all_tokenized) * len(all_tokenized) * 100
            lb = self.cfg.data.language_balancing
            db = self.cfg.data.domain_balancing
            logger.info("WeightedMixedDataset: sub-datasets=%d, total=%d, lang_bal=%s, domain_bal=%s",
                         len(weighted), total_est, lb.enabled, db.enabled)
            result = WeightedMixedDataset(
                weighted, total_est,
                balance_languages=lb.enabled,
                lang_target=lb.target_distribution,
                balance_domains=db.enabled,
                domain_target=db.include if db.enabled else None,
            )
        if self.cfg.data.sanity_checks.enabled:
            try:
                sc = run_sanity_checks(
                    result,
                    self.tokenizer,
                    num_samples=self.cfg.data.sanity_checks.num_samples,
                    max_decode=self.cfg.data.sanity_checks.max_decode_length,
                )
                logger.info("Sanity checks: %s", sc)
            except Exception as e:
                logger.warning("Sanity check failed: %s", e)
        self.health_report.compute_global_stats()
        if self.cfg.data.health_reporting.enabled:
            self.health_report.save(self.cfg.data.health_reporting.output_dir)
            logger.info("Health report:\n%s", self.health_report.summary_text())
        return result

    def _get_cleanup_pool(self):
        """Process pool for parallel cleanup/filtering, created once and kept
        alive for the lifetime of the pipeline (reused across datasets/stages
        instead of being recreated on every build).

        Worker topology is bounded and observable: the size is
        ``cleanup_pool_size`` when configured, otherwise a documented default
        cap (DEFAULT_CLEANUP_POOL_WORKERS) instead of every core on the host.

        The context is deterministic: `fork` is preferred where the platform
        provides it, but a fork that is *refused at pool creation* (e.g. the
        "os.fork is unsafe while filelock is changing descriptor ownership"
        guard) immediately falls back to `spawn` â€” never a silent
        "cleanup will run sequentially" followed by a second, contradictory
        32-worker fork attempt like the box log showed.
        """
        if self._cleanup_pool is None:
            n_cpu = os.cpu_count() or 2
            if n_cpu > 1:
                configured = getattr(self.cfg.data, "cleanup_pool_size", None)
                if configured and int(configured) > 0:
                    n_workers = min(32, int(configured))
                    source = "config"
                else:
                    n_workers = min(DEFAULT_CLEANUP_POOL_WORKERS, n_cpu)
                    source = "default-bound"
                pool = None
                ctx = None
                fork_note = None
                try:
                    ctx = multiprocessing.get_context("fork")
                    pool = ctx.Pool(n_workers)
                except ValueError:
                    fork_note = "fork backend unavailable on this platform"
                except Exception as e:  # noqa: BLE001 â€” guard/fork refusal
                    fork_note = f"fork refused at pool creation ({e})"
                if pool is None:
                    if fork_note:
                        logger.warning("[WORKERS] %s â€” cleanup pool falls back to spawn", fork_note)
                    try:
                        ctx = multiprocessing.get_context("spawn")
                        pool = ctx.Pool(n_workers)
                    except Exception as e:  # noqa: BLE001 â€” final fallback
                        logger.warning("Process pool unavailable (%s) â€” cleanup will run sequentially", e)
                        pool = None
                if pool is not None:
                    self._cleanup_pool = pool
                    logger.info("[WORKERS] cleanup pool: %d workers (context=%s, "
                                "source=%s) â€” kept alive for the run",
                                pool._processes, ctx.get_start_method(), source)
        return self._cleanup_pool

    def close(self) -> None:
        """Terminate the persistent cleanup pool and cancel any in-flight
        unit builds. Idempotent, safe to call at any point; used on pipeline
        shutdown so spawned workers never leak child processes and lone
        builds stop at their next checkpoint."""
        self.cancel()
        pool = self._cleanup_pool
        self._cleanup_pool = None
        if pool is not None:
            try:
                n_pool = getattr(pool, "_processes", "?")
            except Exception:  # noqa: BLE001
                n_pool = "?"
            logger.info("[POOL] closing shared cleanup pool (%s workers)", n_pool)
            try:
                pool.close()
                pool.join()  # drain: idle workers exit promptly on close()
            except Exception as e:
                logger.warning("Cleanup pool join failed (%s) â€” terminating", e)
            try:
                pool.terminate()  # backstop: never leak worker processes
            except Exception:
                pass
        # Deferred health-report emission: warm unit-cache runs never enter the
        # per-dataset build paths that save the report, so persist the
        # aggregated report here (guard against re-invocation races).
        hr = getattr(getattr(self.cfg, "data", None), "health_reporting", None)
        if hr is not None and getattr(hr, "enabled", False) and getattr(hr, "output_dir", None):
            try:
                self.health_report.compute_global_stats()
                out = self.health_report.save(hr.output_dir)
                logger.info("Health report saved at close: %s", out)
            except Exception as e:  # noqa: BLE001
                logger.warning("Health report close-save failed: %s", e)

    def _shard_progress_store(self) -> Optional[ShardProgressStore]:
        d = self.cfg.data
        if not d.resume_shards or not d.shard_progress_dir:
            return None
        if self._shard_store is None:
            self._shard_store = ShardProgressStore(d.shard_progress_dir)
        return self._shard_store

    def _driver_for(self, info: DatasetInfo, ppsig: str, toksig: str):
        """Pick (and cache) the dataset driver for one registry entry.

        Detection order: local filesystem -> file (metadata record) ->
        script (builder record) -> streaming fallback. Resolution happens
        once per dataset; concurrent calls (prefetch thread vs. main loop)
        are serialized on a lock so the main loop waits for the background
        resolution to finish instead of duplicating it."""
        key = f"{info.path}/{info.name or 'default'}"
        with self._driver_lock:
            if key not in self._driver_cache:
                drv, diag = detect_driver(
                    info, self.meta_cache, self._builder_cache,
                    ppsig, toksig)
                self._driver_cache[key] = (drv, diag)
            return self._driver_cache[key]

    def _prefetch_driver(self, info: DatasetInfo) -> None:
        """Warm the driver record for an upcoming dataset on a background
        thread so its resolution (network) overlaps the current dataset's
        streaming/processing â€” CPU/GPU/network overlap across datasets."""
        key = f"{info.path}/{info.name or 'default'}"
        if key in self._driver_cache:
            return

        def _bg() -> None:
            try:
                self._driver_for(info, self.processing_signature(),
                                 self.tokenizer_signature)
            except Exception as e:
                logger.debug("Driver prefetch failed for %s (%s)", key, e)

        threading.Thread(target=_bg, daemon=True, name=f"driver-{key}").start()

    def _dataset_policy(self, info: DatasetInfo):
        """First enabled policy whose glob matches the dataset path, else a
        no-op default (no accepted target, default workers, sequential)."""
        d = self.cfg.data
        for p in d.dataset_policies:
            if p.enabled and fnmatch.fnmatchcase(info.path, p.path):
                return p
        return SimpleNamespace(
            path="*", enabled=True, accepted_target=0, text_fields=None,
            shard_workers=d.shard_workers or 4, shard_order="sequential",
        )

    def _stream_sharded(
        self,
        info: DatasetInfo,
        registry: DatasetRegistry,
        limit: Optional[int],
        policy,
    ) -> "ShardedStreamIterator":
        """Shard-parallel, resume-capable streaming for one registry entry.

        Walks the fallback chain exactly like stream_dataset_with_fallbacks:
        each entry streams via a cached metadata record when one exists
        (shard plan + parallel workers + persisted progress), otherwise via
        the plain streaming path. Yields the same gated samples as the
        sequential path ({**sample, "text": text}) plus _shard/_raw_seq tags.

        Callers read ``streamer.stats`` after iteration:
          .timings (dict)  .raw_rows (int)  .gated (int)  .plan (list)
          .record (dict|None)  .shard_stats (dict)
        """
        stats = SimpleNamespace(
            timings={"metadata_resolve": 0.0, "shard_select": 0.0},
            raw_rows=0, gated=0, plan=[], record=None,
            shard_stats={}, shard_streamed={}, shard_accepted={},
            driver_kind=None, metadata_hit=False, builder_hit=False,
            repo_resolution_skipped=False, first_resolution=False,
            resolve_error=None, script_resumed=False,
        )
        gen = self._stream_sharded_gen(info, registry, limit, policy, stats)
        return ShardedStreamIterator(gen, stats)

    def _stream_sharded_gen(
        self,
        info: DatasetInfo,
        registry: DatasetRegistry,
        limit: Optional[int],
        policy,
        stats: SimpleNamespace,
    ) -> Generator[Dict[str, Any], None, None]:
        ppsig = self.processing_signature()
        toksig = self.tokenizer_signature
        store = self._shard_progress_store()

        chain = [info]
        for fb_path in info.fallbacks:
            fb = registry.get_by_path_category(fb_path, info.category)
            if fb is not None and fb not in chain:
                chain.append(fb)

        tried: Set[str] = set()
        for entry in chain:
            key = f"{entry.path}/{entry.name or 'default'}"
            if key in tried:
                continue
            tried.add(key)
            logger.info("Loading dataset: %s/%s (cat=%s, weight=%.3f, qs=%.2f)",
                        entry.path, entry.name or "default", entry.category,
                        entry.weight, entry.quality_score)
            t0 = time.perf_counter()
            drv, diag = self._driver_for(entry, ppsig, toksig)
            stats.timings["metadata_resolve"] += time.perf_counter() - t0
            stats.driver_kind = diag.get("driver_kind")
            stats.metadata_hit = diag.get("metadata_hit", False)
            stats.builder_hit = diag.get("builder_hit", False)
            stats.repo_resolution_skipped = diag.get("repo_resolution_skipped", False)
            stats.first_resolution = diag.get("first_resolution", False)
            stats.resolve_error = diag.get("resolve_error")
            stats.record = getattr(drv, "record", None)
            text_fields = policy.text_fields or entry.text_fields
            try:
                if (drv.kind in (DRIVER_KIND_FILE, DRIVER_KIND_LOCAL)
                        and drv.record is not None and drv.record.get("files")):
                    if stats.metadata_hit:
                        logger.info("Dataset cache found: %s/%s", entry.path,
                                    entry.name or "default")
                        logger.info("  Repository unchanged | metadata reused (%d shards)",
                                    len(drv.record.get("files") or []))
                        logger.info("  Arrow reused â€” direct iterable, no HF resolution")
                        logger.info("  Streaming begins (no HF resolution)")
                    yield from self._stream_record(entry, drv.record, limit, policy,
                                                   text_fields, store, ppsig,
                                                   stats)
                    if stats.gated > 0:
                        return
                    logger.warning("Dataset %s returned 0 samples, trying fallback %s",
                                   key, entry.fallbacks if entry is info else "none")
                    continue
                if drv.kind == DRIVER_KIND_SCRIPT and drv.record is not None:
                    yield from self._stream_script(entry, drv, limit, text_fields,
                                                   stats)
                    if stats.gated > 0:
                        return
                    logger.warning("Dataset %s returned 0 samples, trying fallback %s",
                                   key, entry.fallbacks if entry is info else "none")
                    continue
                if stats.resolve_error:
                    # Detection failed (gated/404/offline) â€” log access
                    # guidance once, then try the NEXT fallback entry instead
                    # of giving up on the whole chain.
                    from src.data.metadata_cache import _handle_load_error
                    _handle_load_error(entry.path, RuntimeError(stats.resolve_error))
                    continue
                yield from self._stream_cold(entry, limit, text_fields, stats)
                return
            except Exception:
                logger.exception("Stream failed for %s", key)
                continue
        logger.error("All fallbacks exhausted for %s/%s", info.path, info.name or "default")

    def _stream_record(
        self,
        entry: DatasetInfo,
        rec: Dict[str, Any],
        limit: Optional[int],
        policy,
        text_fields: Optional[List[str]],
        store: Optional[ShardProgressStore],
        ppsig: str,
        stats: SimpleNamespace,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream one record's shards via the ShardCoordinator, persisting
        per-shard progress so interrupted runs resume exactly."""
        files = rec.get("files") or []
        n_shards = len(files)
        if n_shards == 0:
            raise RuntimeError(f"No file sources in record for {rec.get('repo')}")
        fingerprint = self._shard_progress_fingerprint(entry, rec, ppsig)
        prect = (store.load(entry.path, entry.name, entry.split, fingerprint)
                 if store is not None else None)
        t_sel = time.perf_counter()
        if prect is not None:
            plan = build_shard_plan(n_shards, prect, policy.shard_order)
        else:
            plan = [(i, 0) for i in range(n_shards)]
        stats.timings["shard_select"] = time.perf_counter() - t_sel
        stats.plan = plan
        if not plan:
            return
        coord = ShardCoordinator(
            rec, plan,
            workers=int(policy.shard_workers or self.cfg.data.shard_workers or 4),
            text_fields=text_fields,
            token=os.environ.get("HF_TOKEN"),
            limit=limit,
        )
        coord_start = time.perf_counter()
        try:
            for sample in coord:
                yield sample
        finally:
            coord.close()
            stats.timings["stream_sec"] = time.perf_counter() - coord_start
            stats.timings.update(coord.timings())
            stats.raw_rows = coord.raw_rows()
            stats.gated = coord.gated_count()
            stats.shard_stats = self._shard_stats_from_stream(coord)
            if store is not None and prect is not None:
                self._persist_shard_progress(
                    entry, store, prect, coord, n_shards,
                    stats.shard_streamed or {},
                    stats.shard_accepted or {},
                )

    def _shard_progress_fingerprint(
        self, entry: DatasetInfo, rec: Dict[str, Any], ppsig: str,
    ) -> str:
        """Shard-progress key fingerprint.

        The record fingerprint (repo/name/split + resolved FILE LIST +
        revision + preprocess/tokenizer signatures) is folded into the
        shard-progress key: when upstream files change, a fresh progress
        record starts instead of re-using stale per-shard offsets.
        """
        rec_fp = (rec or {}).get("fingerprint") or ""
        return hashlib.sha256(
            f"{entry.path}|{entry.name or ''}|{entry.split}|{ppsig}|"
            f"{self.tokenizer_signature}|{rec_fp}"
            .encode()).hexdigest()

    def _shard_stats_from_stream(self, coord: ShardCoordinator) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        for sidx, raw in coord._raw_done.items():
            stats[sidx] = {"raw": raw}
        return stats

    def _persist_shard_progress(
        self,
        entry: DatasetInfo,
        store: ShardProgressStore,
        prect: Dict[str, Any],
        coord: ShardCoordinator,
        n_shards: int,
        streamed: Dict[int, int],
        accepted: Dict[int, int],
    ) -> None:
        try:
            state = coord.progress_state()
            stats = prect.setdefault("stats", {})
            failed = set(coord.failed_shards())
            # Finished shards are marked complete (yield-order safe) â€” except
            # FAILED shards, which must NOT be skipped on resume: their
            # progress is reset so the next run retries them from scratch
            # instead of silently losing their rows.
            for sidx in coord._raw_done:
                st = stats.setdefault(str(sidx), {})
                if sidx in failed:
                    st["complete"] = 0
                    st.pop("streamed", None)
                    st.pop("accepted", None)
                else:
                    st["complete"] = 1
            if all(str(i) in stats and stats[str(i)].get("complete")
                   for i in range(n_shards)):
                store.delete(entry.path, entry.name, entry.split,
                             prect.get("fingerprint", ""))
                return
            if state[0] >= 0:
                prect["resume"] = {"shard": int(state[0]), "offset": int(state[1])}
                prect["last_shard"] = int(state[0]) - 1
                prect["last_offset"] = int(state[1])
            for sidx, n in streamed.items():
                if n:
                    store.update_shard_stats(prect, sidx, {"streamed": n})
            for sidx, n in accepted.items():
                if n:
                    store.update_shard_stats(prect, sidx, {"accepted": n})
            if state[1] > 0:
                store.update_shard_stats(prect, state[0], {"raw": int(state[1])})
            store.save(prect)
        except Exception as e:
            logger.warning("Shard progress persistence failed for %s: %s",
                           entry.path, e)

    def _stream_script(
        self,
        entry: DatasetInfo,
        driver,
        limit: Optional[int],
        text_fields: Optional[List[str]],
        stats: SimpleNamespace,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream a script/IterableDataset through its builder with exact
        iterator resume (raw-row offsets persisted in the builder record).

        Mirrors ShardCoordinator semantics: gated samples carry (_shard=0,
        _raw_seq) and stats; the resume state is persisted when the stream is
        interrupted (limit / accepted target) and reset when the dataset is
        exhausted naturally â€” the same delete-on-complete contract as the
        shard progress store."""
        stats.script_resumed = driver.resumed
        if stats.builder_hit:
            logger.info("Builder cache found: %s/%s", entry.path,
                        entry.name or "default")
            logger.info("  Repository unchanged | builder reused (script=%s)",
                        (driver.record or {}).get("builder_class"))
            logger.info("  Streaming begins (no HF resolution)")
        if driver.resumed:
            logger.info("Iterator resumed at raw row %d", driver.raw_consumed())
        count = 0
        try:
            for sample in driver.stream(
                limit=limit,
                text_fields=text_fields,
                token=os.environ.get("HF_TOKEN"),
            ):
                yield sample
                count += 1
        finally:
            stats.gated = count
            stats.raw_rows = driver.raw_consumed()
            if driver.natural_end():
                driver.reset_resume()
                logger.info("Script dataset exhausted â€” iterator state reset "
                            "(next run starts fresh)")
            else:
                driver.save_resume()
                if driver.raw_consumed() > 0:
                    logger.info("Iterator state persisted at raw row %d",
                                driver.raw_consumed())

    def _stream_cold(
        self,
        entry: DatasetInfo,
        limit: Optional[int],
        text_fields: Optional[List[str]],
        stats: SimpleNamespace,
    ) -> Generator[Dict[str, Any], None, None]:
        """Fallback for datasets without a usable driver record (detection
        failed / streaming family): the original resolution-based streaming
        path, still caching file metadata opportunistically mid-stream so the
        next run is shard-parallel."""
        from src.data.drivers import StreamingDatasetDriver

        driver = StreamingDatasetDriver(
            info=entry,
            meta_cache=self.meta_cache,
            preprocess_sig=self.processing_signature(),
            token_sig=self.tokenizer_signature,
        )
        count = 0
        for sample in driver.stream(
            limit=limit,
            text_fields=text_fields,
            token=os.environ.get("HF_TOKEN"),
        ):
            yield sample
            count += 1
        stats.gated = count

    def _report_dataset_diagnostics(
        self,
        ds_key: str,
        streamer,
        stage_acc: Dict[str, float],
        ledger: Dict[str, int],
        tokenize_pack_sec: float,
    ) -> None:
        """Timing report + bottleneck detection + low-acceptance investigation."""
        d = self.cfg.data
        t = streamer.stats.timings
        driver_kind = getattr(streamer.stats, "driver_kind", None)
        if driver_kind:
            hits = []
            if getattr(streamer.stats, "metadata_hit", False):
                hits.append("metadata-hit")
            if getattr(streamer.stats, "builder_hit", False):
                hits.append("builder-hit")
            if getattr(streamer.stats, "repo_resolution_skipped", False):
                hits.append("resolution-skipped")
            if getattr(streamer.stats, "first_resolution", False):
                hits.append("first-resolution")
            if getattr(streamer.stats, "script_resumed", False):
                hits.append("iterator-resumed")
            logger.info("[DIAG] %s driver: %s (%s)", ds_key, driver_kind,
                        ", ".join(hits) or "live")
        timings = {
            "metadata_resolve": t.get("metadata_resolve", 0.0),
            "shard_selection": t.get("shard_select", 0.0),
            "arrow_open": t.get("arrow_open_sec", 0.0),
            "arrow_open_worst_shard": t.get("arrow_open_max_sec", 0.0),
            "network_wait": t.get("network_wait_sec", 0.0),
            "extraction": t.get("extraction_sec", 0.0),
            "shard_stream": t.get("stream_sec", 0.0),
        }
        for k, v in stage_acc.items():
            timings[f"cleanup/{k}"] = v
        timings["tokenize+pack"] = tokenize_pack_sec
        logger.info("[DIAG] %s stage timings:", ds_key)
        for k, v in sorted(timings.items(), key=lambda x: -x[1]):
            logger.info("[DIAG]   %-26s %8.1fs", k, v)
        threshold = d.bottleneck_threshold_sec
        if threshold and threshold > 0:
            flagged = {k: v for k, v in timings.items() if v >= threshold}
            if flagged:
                logger.warning("BOTTLENECK DETECTED â€” %s (stage >= %.0fs):", ds_key, threshold)
                for k, v in sorted(flagged.items(), key=lambda x: -x[1]):
                    logger.warning("  %-26s %8.1fs", k, v)
                logger.warning("  Fixes: raise shard_workers, enable metadata/Arrow reuse, "
                               "raise accepted_target, or move quality/AST work off the network wait.")
        raw = int(streamer.stats.raw_rows or 0)
        inv_thr = d.acceptance_investigation_threshold
        if inv_thr and inv_thr > 0 and raw > 0:
            rate = ledger["accepted"] / raw
            if rate < inv_thr:
                logger.warning("LOW ACCEPTANCE â€” %s: %.2f%% (accepted=%d / raw=%d) â€” investigating:",
                               ds_key, rate * 100, ledger["accepted"], raw)
                filters = [
                    ("boilerplate", ledger["rejected_boilerplate"]),
                    ("too short", ledger["rejected_short"]),
                    ("low quality", ledger["rejected_quality"]),
                    ("duplicate", ledger["rejected_dedup"]),
                    ("AST", ledger["rejected_ast"]),
                    ("empty/trivial", ledger["rejected_empty"]),
                ]
                for name, n in sorted(filters, key=lambda x: -x[1]):
                    if n:
                        logger.warning("  rejected by %-16s %d (%.1f%% of raw)",
                                       name, n, 100.0 * n / raw)
                shard_stats = streamer.stats.shard_stats or {}
                if shard_stats:
                    worst = sorted(shard_stats.items(), key=lambda x: -x[1].get("raw", 0))[:5]
                    logger.warning("  highest-raw shards: %s",
                                   ", ".join(f"#{i}(raw={st.get('raw', 0)})"
                                             for i, st in worst))

    def build_pretrain_dataset_from_registry(
        self,
        max_samples_per_dataset: Optional[int] = None,
        dataset_filter: Optional[List[str]] = None,
        include: Optional[List[Tuple[str, Optional[str]]]] = None,
        cancel_event=None,
    ) -> Dataset:
        registry = build_registry(exclude=getattr(self.cfg.data, "registry_exclude", None))
        all_infos = registry.all_entries()
        if include:
            include_set = {(p, n or "") for p, n in include}
            all_infos = [i for i in all_infos if (i.path, i.name or "") in include_set]
        if dataset_filter:
            all_infos = [i for i in all_infos if i.category in dataset_filter]

        all_tokenized: List[Tuple[Dataset, float, str, str, float]] = []  # (ds, weight, path, cat, avg_quality)
        dataset_metas: List[dict] = []  # aligned with all_tokenized
        global_stats = {
            "total_raw": 0, "total_accepted": 0,
            "total_rejected_dup": 0, "total_rejected_quality": 0,
            "total_rejected_ast": 0, "total_rejected_boilerplate": 0,
            "total_rejected_short": 0, "total_rejected_empty": 0,
            "total_packed": 0, "total_tokens": 0,
        }
        rejection_reasons: Dict[str, int] = defaultdict(int)
        lang_dist: Dict[str, int] = defaultdict(int)
        domain_dist: Dict[str, int] = defaultdict(int)
        # Driver-layer cache accounting (exposed to callers via the result's
        # _driver_stats attribute â†’ surfaced in the ASYNC PIPELINE REPORT).
        driver_stats = {
            "metadata_hits": 0, "metadata_attempts": 0,
            "builder_hits": 0, "builder_attempts": 0,
            "local_entries": 0, "streaming_fallback": 0,
        }

        ast_filter_cfg = self.cfg.data.ast_filter
        func_sampling_cfg = self.cfg.data.function_sampling
        pp = self.cfg.data.preprocessing
        kw = pp.license_keywords
        patterns = pp.boilerplate_file_patterns
        min_text_len = pp.min_text_length
        max_len = self.cfg.training.max_seq_length
        eos_id = self.tokenizer.eos_token_id or 0

        pool = self._get_cleanup_pool()

        for idx, info in enumerate(all_infos, 1):
            self.raise_if_cancelled(cancel_event)
            ds_key = f"{info.path}/{info.name or 'default'}"
            cat = info.category
            logger.info("")
            logger.info("--- [%d/%d] %s (cat=%s, weight=%.3f, qs=%.2f) ---",
                         idx, len(all_infos), ds_key, cat, info.weight, info.quality_score)

            # CPU/GPU/network overlap: resolve the next datasets' driver
            # records (metadata/builder identity) in the background while this
            # dataset streams and processes.
            for nxt in all_infos[idx: idx + 2]:
                self._prefetch_driver(nxt)

            ledger = {"loaded": 0, "accepted": 0,
                      "rejected_boilerplate": 0, "rejected_short": 0,
                      "rejected_quality": 0, "rejected_dedup": 0,
                      "rejected_ast": 0, "rejected_empty": 0,
                      "rejected_contamination": 0}
            # quality_scores holds EVERY candidate's score (declared before the
            # quality threshold). Only a subset survives into cleaned_texts —
            # from the surviving (text, score) pairs we rebuild an
            # aligned `cleaned_quality` list so doc_qs / avg_qs / health stats
            # reflect the texts actually packed, never a mis-indexed candidate.
            quality_scores: List[float] = []
            cleaned_texts: List[str] = []
            cleaned_quality: List[float] = []
            shard_streamed: Dict[int, int] = defaultdict(int)
            shard_accepted: Dict[int, int] = defaultdict(int)

            ds_limit = (max_samples_per_dataset
                        or self.cfg.data.max_samples_per_dataset
                        or info.max_samples
                        or DEFAULT_MAX_SAMPLES_PER_DATASET)
            policy = self._dataset_policy(info)
            accepted_target = int(policy.accepted_target or 0)
            cache_key = self._registry_cache_key(info, ds_limit, accepted_target)
            cache_path = self._registry_cache_path(info, cache_key)
            if self.cfg.data.use_packed_cache:
                hit = self._load_registry_dataset_cache(cache_path, cache_key, ds_key, info)
                if hit is not None:
                    cached_ds, meta = hit
                    # Restore language/domain hints lost when the packed
                    # dataset was saved to disk (needed for stratification).
                    if info.language:
                        cached_ds.language = info.language
                    if info.domain:
                        cached_ds.domain = info.domain
                    all_tokenized.append((cached_ds, info.weight, info.path, info.category, meta["avg_qs"]))
                    dataset_metas.append(meta)
                    for k, v in meta["global_delta"].items():
                        global_stats[k] += v
                    for k, v in meta["lang_dist"].items():
                        lang_dist[k] += v
                    for k, v in meta["domain_dist"].items():
                        domain_dist[k] += v
                    for reason, n in meta["rejections"].items():
                        rejection_reasons[reason] += n
                    self.health_report.add_dataset_stats(
                        **self._health_stats_from_meta(meta, info, max_len))
                    logger.info("    CACHED: %s â€” %d packed results loaded from disk",
                                ds_key, meta["packed_count"])
                    continue

            try:
                t0 = time.perf_counter()
                chunk_size = max(1024, min(ds_limit, 8192))

                # Accepted-sample-driven streaming: pull chunks from the
                # shard-parallel streamer and run the identical filter chain
                # per chunk (same order, same pure functions, same stateful
                # dedup â€” sequential-equivalent, so the accepted set matches
                # the pre-optimization pipeline exactly). Stop pulling as soon
                # as the accepted target is met instead of materializing the
                # full ds_limit (The Stack V2: 20k streamed â†’ 79 accepted).
                t_clean = time.perf_counter()
                stage_acc: Dict[str, float] = {k: 0.0 for k in (
                    "extract_empty", "boilerplate", "length", "language",
                    "exact_dedup", "simhash_fingerprint", "simhash_dedup",
                    "quality_scoring", "ast_validation", "function_sampling",
                )}
                loop_total = 0.0
                stream_ok = True
                streamer = self._stream_sharded(info, registry, ds_limit, policy)
                try:
                    while True:
                        self.raise_if_cancelled(cancel_event)
                        chunk = list(itertools.islice(streamer, chunk_size))
                        if not chunk:
                            break
                        ledger["loaded"] += len(chunk)
                        global_stats["total_raw"] += len(chunk)
                        for s in chunk:
                            shard_streamed[s.get("_shard", -1)] += 1
                        t_stage = time.perf_counter()
                        cands: List[Tuple[str, str, int]] = []
                        for s in chunk:
                            text = s.get("output") or s.get("text") or s.get("content") or ""
                            if not text:
                                ledger["rejected_empty"] += 1
                                continue
                            t = time.perf_counter()
                            if pp.remove_boilerplate:
                                text = remove_boilerplate(text, keywords=kw)
                            if should_skip_file(s.get("file_path", ""), patterns=patterns):
                                ledger["rejected_boilerplate"] += 1
                                continue
                            t2 = time.perf_counter()
                            if len(text.strip()) < min_text_len:
                                ledger["rejected_short"] += 1
                                continue
                            t3 = time.perf_counter()
                            ds_lang = info.language or detect_language(text, info.path)
                            cands.append((text, ds_lang, s.get("_shard", -1)))
                            stage_acc["boilerplate"] += t2 - t
                            stage_acc["length"] += t3 - t2
                            stage_acc["language"] += time.perf_counter() - t3
                        loop_total += time.perf_counter() - t_stage

                        # Quality scoring, then AST validation â€” parallelized (pure functions).
                        # Stage order mirrors the original pipeline (quality -> AST -> dedup),
                        # so the accepted set is identical to pre-optimization behavior.
                        eff_threshold = max(QUALITY_THRESHOLDS.get(cat, 0.30), 0.15)
                        survivors: List[Tuple[str, str, int]] = cands
                        if survivors:
                            t_stage = time.perf_counter()
                            q_tasks = [(text, cat, ds_lang) for text, ds_lang, _s in survivors]
                            if pool is not None:
                                scores = pool.starmap(_pool_quality_score, q_tasks, chunksize=1024)
                            else:
                                scores = [_pool_quality_score(*a) for a in q_tasks]
                            stage_acc["quality_scoring"] += time.perf_counter() - t_stage

                            quality_kept: List[Tuple[str, str, float, int]] = []
                            for (text, ds_lang, _sh), comp_score in zip(survivors, scores):
                                quality_scores.append(comp_score)
                                if comp_score < eff_threshold:
                                    ledger["rejected_quality"] += 1
                                    continue
                                quality_kept.append((text, ds_lang, comp_score, _sh))

                            contamination_kept: List[Tuple[str, str, float, int]] = []
                            for text, ds_lang, comp_score, _sh in quality_kept:
                                if self.contamination.is_contaminated(text):
                                    ledger["rejected_contamination"] += 1
                                    rejection_reasons["contamination"] += 1
                                    continue
                                contamination_kept.append((text, ds_lang, comp_score, _sh))
                            quality_kept = contamination_kept

                            code_filtering = cat in ("code",) and ast_filter_cfg.code_filtering
                            if code_filtering and quality_kept:
                                t_stage = time.perf_counter()
                                a_tasks = [(text, ds_lang, ast_filter_cfg)
                                           for text, ds_lang, _sc, _s in quality_kept]
                                if pool is not None:
                                    ast_results = pool.starmap(_pool_filter_code, a_tasks, chunksize=1024)
                                else:
                                    ast_results = [_pool_filter_code(*a) for a in a_tasks]
                                stage_acc["ast_validation"] += time.perf_counter() - t_stage
                                accepted: List[Tuple[str, str, float, int]] = []
                                for (text, ds_lang, comp_score, _sh), (ok, reason) in zip(quality_kept, ast_results):
                                    if not ok:
                                        ledger["rejected_ast"] += 1
                                        rejection_reasons[reason] += 1
                                        continue
                                    accepted.append((text, ds_lang, comp_score, _sh))
                            else:
                                accepted = quality_kept
                        else:
                            accepted = []

                        # Duplicates last â€” same stage sequence as the original pipeline, so the
                        # accepted set is unchanged; dedup runs only on quality+AST survivors.
                        # Stateful dedup carries across chunks (sequential-equivalent).
                        t_stage = time.perf_counter()
                        deduped: List[Tuple[str, str, float, int]] = []
                        for text, ds_lang, comp_score, _sh in accepted:
                            if self.exact_dedup.is_duplicate(text):
                                ledger["rejected_dedup"] += 1
                                rejection_reasons["exact_dedup"] += 1
                                continue
                            deduped.append((text, ds_lang, comp_score, _sh))
                        stage_acc["exact_dedup"] += time.perf_counter() - t_stage

                        simhash_active = (self.cfg.data.quality.deduplication.method == "simhash"
                                          and isinstance(self.dedup, SimHashDeduplicator))
                        if simhash_active and deduped:
                            t_stage = time.perf_counter()
                            dedup_texts = [t for t, _l, _sc, _s in deduped]
                            if pool is not None:
                                fps = pool.map(_pool_simhash_fp, dedup_texts, chunksize=2048)
                            else:
                                fps = [_pool_simhash_fp(t) for t in dedup_texts]
                            stage_acc["simhash_fingerprint"] += time.perf_counter() - t_stage
                            t_stage = time.perf_counter()
                            final_kept: List[Tuple[str, str, float, int]] = []
                            dup_flags = self.dedup.process_batch(fps)
                            for (text, ds_lang, comp_score, _sh), is_dup in zip(deduped, dup_flags):
                                if is_dup:
                                    ledger["rejected_dedup"] += 1
                                    rejection_reasons["simhash_dedup"] += 1
                                    continue
                                final_kept.append((text, ds_lang, comp_score, _sh))
                            stage_acc["simhash_dedup"] += time.perf_counter() - t_stage
                            accepted = final_kept

                        # Function-level sampling (transform; last, on survivors only).
                        t_stage = time.perf_counter()
                        func_enabled = cat in ("code",) and func_sampling_cfg.enabled and info.function_sampling
                        if func_enabled:
                            final_texts: List[str] = []
                            final_scores: List[float] = []
                            final_shards: List[int] = []
                            for text, ds_lang, score, _sh in accepted:
                                lang = info.language or detect_language(text, info.path)
                                extracted = sample_functions_or_fallback(text, lang, func_sampling_cfg)
                                if is_code_empty_or_trivial(extracted):
                                    ledger["rejected_empty"] += 1
                                    continue
                                final_texts.append(extracted)
                                final_scores.append(score)
                                final_shards.append(_sh)
                        else:
                            final_texts = [t for t, _l, _sc, _sh in accepted]
                            final_scores = [_sc for _t, _l, _sc, _sh in accepted]
                            final_shards = [_sh for _t, _l, _sc, _sh in accepted]
                        stage_acc["function_sampling"] += time.perf_counter() - t_stage
                        cleaned_texts.extend(final_texts)
                        cleaned_quality.extend(final_scores)
                        for _sh in final_shards:
                            shard_accepted[_sh] += 1
                        ledger["accepted"] = len(cleaned_texts)
                        if accepted_target and ledger["accepted"] >= accepted_target:
                            logger.info("    Accepted target reached: %d/%d â€” stopping stream",
                                        ledger["accepted"], accepted_target)
                            break
                except Exception as e:
                    logger.error("    FAILED: %s â€” %s", ds_key, e)
                    self.health_report.add_error(f"{ds_key}: {e}")
                    stream_ok = False
                finally:
                    streamer.stats.shard_streamed = dict(shard_streamed)
                    streamer.stats.shard_accepted = dict(shard_accepted)
                    streamer.close()
                ds_drv = getattr(streamer.stats, "driver_kind", None)
                if ds_drv:
                    driver_stats["metadata_attempts"] += int(
                        ds_drv in (DRIVER_KIND_FILE, DRIVER_KIND_LOCAL))
                    driver_stats["builder_attempts"] += int(
                        ds_drv == DRIVER_KIND_SCRIPT)
                    driver_stats["local_entries"] += int(ds_drv == DRIVER_KIND_LOCAL)
                    driver_stats["streaming_fallback"] += int(
                        ds_drv == DRIVER_KIND_STREAMING)
                driver_stats["metadata_hits"] += int(
                    getattr(streamer.stats, "metadata_hit", False))
                driver_stats["builder_hits"] += int(
                    getattr(streamer.stats, "builder_hit", False))
                if not stream_ok:
                    continue
                stage_acc["extract_empty"] = max(
                    0.0, loop_total - stage_acc["boilerplate"]
                    - stage_acc["length"] - stage_acc["language"])
                logger.info("[TIMER] stream+materialize %s (limit=%s, raw=%d, gated=%d): %.1fs",
                            ds_key, ds_limit, streamer.stats.raw_rows, ledger["loaded"],
                            time.perf_counter() - t0)
            except Exception as e:
                logger.error("    FAILED: %s â€” %s", ds_key, e)
                self.health_report.add_error(f"{ds_key}: {e}")
                continue

            if ledger["loaded"] == 0:
                logger.warning("    EMPTY: %s â€” 0 samples loaded", ds_key)
                self.health_report.add_error(f"{ds_key}: empty")
                continue

            logger.info("    Loaded: %d", ledger["loaded"])
            if ledger["loaded"]:
                _acc_pct = 100.0 * ledger["accepted"] / ledger["loaded"]
                logger.info("    Accepted: %d (%.1f%% of %d loaded)",
                            ledger["accepted"], _acc_pct, ledger["loaded"])
            else:
                logger.info("    Accepted: %d", ledger["accepted"])
            for _stage, _secs in stage_acc.items():
                logger.info("[TIMER] cleanup/%s %s: %.1fs", _stage, ds_key, _secs)
            logger.info("[TIMER] cleanup/filter total %s: %.1fs", ds_key, time.perf_counter() - t_clean)
            logger.info("    Rejected:")
            logger.info("      boilerplate: %d", ledger["rejected_boilerplate"])
            logger.info("      too short:   %d", ledger["rejected_short"])
            logger.info("      low quality: %d", ledger["rejected_quality"])
            logger.info("      duplicate:   %d", ledger["rejected_dedup"])
            logger.info("      AST filter:  %d", ledger["rejected_ast"])
            logger.info("      empty/triv:  %d", ledger["rejected_empty"])

            if not cleaned_texts:
                logger.warning("    SKIP: %s â€” 0 accepted", ds_key)
                self.health_report.add_error(f"{ds_key}: all {ledger['loaded']} rejected")
                continue

            global_stats["total_accepted"] += ledger["accepted"]
            global_stats["total_rejected_dup"] += ledger["rejected_dedup"]
            global_stats["total_rejected_quality"] += ledger["rejected_quality"]
            global_stats["total_rejected_ast"] += ledger["rejected_ast"]
            global_stats["total_rejected_boilerplate"] += ledger["rejected_boilerplate"]
            global_stats["total_rejected_short"] += ledger["rejected_short"]
            global_stats["total_rejected_empty"] += ledger["rejected_empty"]

            t_tok = time.perf_counter()
            tokenized = []
            ds_lang_dist: Dict[str, int] = defaultdict(int)
            ds_domain_dist: Dict[str, int] = defaultdict(int)
            for text_idx, text in enumerate(cleaned_texts):
                self.raise_if_cancelled(cancel_event)
                tok = self.tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
                tok = random_window_sample(tok, max_len)
                doc_qs = cleaned_quality[text_idx] if text_idx < len(cleaned_quality) else 0.5
                tokenized.append({"input_ids": tok, "quality_score": doc_qs})
                lang_d = info.language or detect_language(text, info.path)
                lang_dist[lang_d] += 1
                ds_lang_dist[lang_d] += 1
                domain_d = info.domain or detect_domain(text, cat)
                domain_dist[domain_d] += 1
                ds_domain_dist[domain_d] += 1

            packed, packing_eff = pack_sequences(tokenized, max_len, eos_id)
            tok_pack_sec = time.perf_counter() - t_tok
            logger.info("[TIMER] tokenize+pack %s: %.1fs (%d tok -> %d packed)",
                        ds_key, tok_pack_sec, len(tokenized), len(packed))
            global_stats["total_packed"] += len(packed)
            for p in packed:
                global_stats["total_tokens"] += len(p["input_ids"])

            self._report_dataset_diagnostics(ds_key, streamer, stage_acc, ledger, tok_pack_sec)

            avg_qs = sum(cleaned_quality) / max(len(cleaned_quality), 1)
            if cleaned_quality:
                sorted_qs = sorted(cleaned_quality)
                p10 = sorted_qs[len(sorted_qs) // 10] if len(sorted_qs) >= 10 else sorted_qs[0]
                p90 = sorted_qs[(9 * len(sorted_qs)) // 10] if len(sorted_qs) >= 10 else sorted_qs[-1]
                logger.info("    Quality: mean=%.3f p10=%.3f p90=%.3f range=[%.3f, %.3f]",
                             avg_qs, p10, p90, sorted_qs[0], sorted_qs[-1])
            logger.info("    Packed: %d sequences (eff=%.1f%%, padding=%.1f%%)",
                         len(packed), packing_eff * 100, (1 - packing_eff) * 100)

            for p in packed:
                p["_dataset"] = info.path
                p["_category"] = cat
                p["_language"] = info.language or ""
                p["_domain"] = info.domain
            ds = Dataset.from_list(packed)
            # Attach language/domain hints so WeightedMixedDataset's
            # balance_languages/balance_domains stratification can group this
            # dataset instead of collapsing every entry into 'other'.
            if info.language:
                ds.language = info.language
            if info.domain:
                ds.domain = info.domain
            all_tokenized.append((ds, info.weight, info.path, info.category, avg_qs))
            raw_tok_lengths = [len(t["input_ids"]) for t in tokenized[:1000]]
            ds_meta = {
                "loaded": ledger["loaded"],
                "accepted": ledger["accepted"],
                "avg_qs": avg_qs,
                "packed_count": len(packed),
                "total_tokens": sum(len(p["input_ids"]) for p in packed),
                "rejected_dedup": ledger["rejected_dedup"],
                "quality_scores": cleaned_quality,
                "token_lengths": raw_tok_lengths,
                "lang_dist": dict(ds_lang_dist),
                "domain_dist": dict(ds_domain_dist),
                "lang_d": info.language,
                "domain_d": info.domain,
                "ledger": dict(ledger),
                "rejections": dict(rejection_reasons),
                "global_delta": {
                    "total_raw": ledger["loaded"], "total_accepted": ledger["accepted"],
                    "total_rejected_dup": ledger["rejected_dedup"],
                    "total_rejected_quality": ledger["rejected_quality"],
                    "total_rejected_ast": ledger["rejected_ast"],
                    "total_rejected_boilerplate": ledger["rejected_boilerplate"],
                    "total_rejected_short": ledger["rejected_short"],
                    "total_rejected_empty": ledger["rejected_empty"],
                    "total_packed": len(packed),
                    "total_tokens": sum(len(p["input_ids"]) for p in packed),
                },
            }
            dataset_metas.append(ds_meta)
            self.health_report.add_dataset_stats(
                path=info.path, category=cat, weight=info.weight,
                raw_count=ledger["loaded"], after_boilerplate=ledger["accepted"],
                after_quality=ledger["accepted"], after_dedup=ledger["accepted"],
                packed_count=len(packed),
                total_tokens=sum(len(p["input_ids"]) for p in packed),
                duplicate_removed=ledger["rejected_dedup"],
                rejection_reasons=dict(ledger),
                quality_scores=cleaned_quality,
                # Per-dataset distributions only — never the run-global
                # cumulative dict, or every dataset would report (and inflate
                # the aggregate by) the union of all datasets' languages.
                lang_dist=dict(ds_lang_dist), domain_dist=dict(ds_domain_dist),
                token_lengths=raw_tok_lengths,
                max_seq_length=max_len,
            )

            if self.cfg.data.use_packed_cache:
                self._save_registry_dataset_cache(cache_path, cache_key, packed, ds_meta)

        if pool is not None:
            logger.info("[TIMER] cleanup pool kept alive for next dataset/stage")

        self._log_registry_summary(
            len(all_infos), all_tokenized, global_stats,
            lang_dist, domain_dist, rejection_reasons,
            fallbacks_used=registry.summary().get("fallbacks_used", {}),
        )

        if not all_tokenized:
            logger.error("ALL DATASETS FAILED â€” no samples produced by any dataset in registry")
            if hasattr(self.cfg, '_registry_errors'):
                for err in self.cfg._registry_errors:
                    logger.error("  Error: %s", err)
            self.health_report.compute_global_stats()
            if self.cfg.data.health_reporting.enabled:
                self.health_report.save(self.cfg.data.health_reporting.output_dir)
            raise RuntimeError(
                "All datasets failed. No training possible. "
                "Check: (1) HF_TOKEN is set, (2) dataset access is granted, "
                "(3) network connectivity, (4) disk space."
            )

        try:
            result = self._unit_or_mixed(all_tokenized)
        except Exception as e:  # noqa: BLE001 â€” phase-tagged for the prefetch worker
            from src.training.asyncprefetch import _tag_phase
            _tag_phase(e, "wrapper",
                       n_datasets=len(all_tokenized),
                       paths=[t[2] for t in all_tokenized if len(t) > 2],
                       intended_type="Dataset")
            raise

        # Attach raw entries so callers (e.g. staged pretraining) can persist
        # the stage-level cache without re-streaming.
        result._entries = all_tokenized
        result._dataset_metas = dataset_metas
        result._global_stats = global_stats
        result._driver_stats = driver_stats
        result._lang_dist = dict(lang_dist)
        result._domain_dist = dict(domain_dist)
        result._rejection_reasons = dict(rejection_reasons)
        result._fallbacks_used = registry.summary().get("fallbacks_used", {})

        if self.cfg.data.sanity_checks.enabled:
            try:
                sc = run_sanity_checks(result, self.tokenizer,
                    num_samples=self.cfg.data.sanity_checks.num_samples,
                    max_decode=self.cfg.data.sanity_checks.max_decode_length)
                logger.info("Sanity checks: %s", sc)
            except Exception as e:
                logger.warning("Sanity check failed: %s", e)

        self.health_report.compute_global_stats()
        if self.cfg.data.health_reporting.enabled:
            self.health_report.save(self.cfg.data.health_reporting.output_dir)
            logger.info("Health report:\n%s", self.health_report.summary_text())
        return result

    def _log_registry_summary(
        self,
        n_listed: int,
        all_tokenized: List[Tuple],
        global_stats: Dict[str, int],
        lang_dist: Dict[str, int],
        domain_dist: Dict[str, int],
        rejection_reasons: Dict[str, int],
        fallbacks_used: Optional[Dict] = None,
    ) -> None:
        max_len = self.cfg.training.max_seq_length
        logger.info("")
        logger.info("=" * 70)
        logger.info("PIPELINE SUMMARY â€” REGISTRY MODE")
        logger.info("=" * 70)
        logger.info("  Datasets listed:   %d", n_listed)
        logger.info("  Datasets loaded:   %d", len(all_tokenized))
        logger.info("  Datasets skipped:  %d (all samples rejected)", n_listed - len(all_tokenized))
        logger.info("  Fallbacks used:    %s", fallbacks_used or {})
        logger.info("")
        logger.info("  Total raw:         %d", global_stats.get("total_raw", 0))
        logger.info("  Total accepted:    %d", global_stats.get("total_accepted", 0))
        logger.info("  Total rejected:")
        logger.info("    boilerplate:     %d", global_stats.get("total_rejected_boilerplate", 0))
        logger.info("    too short:       %d", global_stats.get("total_rejected_short", 0))
        logger.info("    low quality:     %d", global_stats.get("total_rejected_quality", 0))
        logger.info("    duplicate:       %d", global_stats.get("total_rejected_dup", 0))
        logger.info("    AST filter:      %d", global_stats.get("total_rejected_ast", 0))
        logger.info("    empty/trivial:   %d", global_stats.get("total_rejected_empty", 0))
        logger.info("")
        logger.info("  Total packed seqs: %d", global_stats.get("total_packed", 0))
        logger.info("  Total tokens:      %d", global_stats.get("total_tokens", 0))
        logger.info("  Acceptance rate:   %.1f%%", (global_stats.get("total_accepted", 0) / max(global_stats.get("total_raw", 0), 1)) * 100)
        logger.info("")
        logger.info("  Language dist:     %s", dict(sorted(lang_dist.items(), key=lambda x: -x[1])[:10]))
        logger.info("  Domain dist:       %s", dict(sorted(domain_dist.items(), key=lambda x: -x[1])[:10]))
        if rejection_reasons:
            logger.info("  Top rejections:    %s", sorted(rejection_reasons.items(), key=lambda x: -x[1])[:5])
        if global_stats.get("total_packed", 0):
            eff = global_stats["total_packed"] * max_len / max(global_stats.get("total_tokens", 0), 1)
            logger.info("  Overall util:      %.1f%%", eff * 100)
            from src.utils.steps import estimate_pretrain_steps, format_step_estimate
            ga = int(getattr(self.cfg.training.pretrain,
                             "gradient_accumulation_steps", None) or 1)
            _est = estimate_pretrain_steps(
                global_stats["total_packed"],
                self.cfg.training.pretrain.batch_size,
                gradient_accumulation_steps=ga,
                world_size=int(os.environ.get("WORLD_SIZE", "1") or 1),
            )
            logger.info("  Est training steps: %s", format_step_estimate(_est))
        logger.info("=" * 70)

    def _unit_or_mixed(
        self,
        all_tokenized: List[Tuple],
        info=None,
    ) -> Any:
        """Return the dataset to feed the trainer for ONE unit.

        A unit feeds a single registry dataset, so no mixing is needed: the raw
        packed dataset is returned directly (no WeightedMixedDataset layer, no
        assignment/index precomputation). Mixing is only applied when a unit
        actually contains multiple datasets."""
        if all_tokenized is None or len(all_tokenized) < 1:
            raise RuntimeError("_unit_or_mixed called without any datasets")
        if len(all_tokenized) == 1:
            return all_tokenized[0][0]
        return self._construct_mixed_dataset(all_tokenized)

    def _construct_mixed_dataset(self, all_tokenized: List[Tuple]) -> WeightedMixedDataset:
        weighted = [(ds, w, f"{path}_{cat}") for ds, w, path, cat, _ in all_tokenized]
        quality_scores_for_w = [aq for _, _, _, _, aq in all_tokenized]
        balance_tokens = self.cfg.data.sampler.balance_by == "tokens"
        if balance_tokens:
            token_counts = [len(ds) if hasattr(ds, '__len__') else 1 for ds, _, _, _, _ in all_tokenized]
            total_tok = sum(token_counts)
            if total_tok > 0:
                weighted = [
                    (ds, w * (tc / total_tok), f"{path}_{cat}")
                    for (ds, w, path, cat, _), tc in zip(all_tokenized, token_counts)
                ]
        total_est = min(len(ds) for ds, _, _, _, _ in all_tokenized) * len(all_tokenized) * 100
        lb = self.cfg.data.language_balancing
        db = self.cfg.data.domain_balancing
        t_mix = time.perf_counter()
        result = WeightedMixedDataset(
            weighted, total_est,
            balance_languages=lb.enabled,
            lang_target=lb.target_distribution,
            balance_domains=db.enabled,
            domain_target=db.include if db.enabled else None,
            quality_scores=quality_scores_for_w,
        )
        logger.info("[TIMER] WeightedMixedDataset construct (lazy): %.1fs, total_samples=%d",
                    time.perf_counter() - t_mix, total_est)
        return result

    def _get_dataset_weights(self) -> Dict[str, float]:
        try:
            return self.cfg.training.pretrain.data_mix
        except AttributeError:
            return {
                'code': 0.35, 'web_text': 0.25, 'books': 0.10,
                'math': 0.10, 'science': 0.10, 'conversations': 0.05,
                'tool_use': 0.03, 'preference': 0.02,
            }

    def get_datasets_for_step(self, current_step: int) -> List[DatasetEntryConfig]:
        if not self.cfg.data.curriculum.enabled:
            return self.collector.get_dataset_list()
        for stage in self.cfg.data.curriculum.stages:
            if current_step <= stage.max_steps:
                if stage.dataset_filter is None:
                    return self.collector.get_dataset_list()
                all_ds = self.collector.get_dataset_list()
                return [d for d in all_ds if d.category in stage.dataset_filter]
        return self.collector.get_dataset_list()

    def _preprocess_batch(self, examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        pairs: List[Tuple[str, str]] = []
        for i in range(len(examples.get("output", []))):
            lang = str(examples.get("language", ["python"] * len(examples["output"]))[i] or "python")
            problem = str(examples.get("instruction", [""] * len(examples["output"]))[i] or "")
            inp = str(examples.get("input", [""] * len(examples["output"]))[i] or "")
            sol = str(examples["output"][i] or "")
            if inp:
                problem = f"{problem}\n{inp}"
            prompt = self.format_prompt(lang, problem, self.cfg.alignment.prompt_style, self.cfg.alignment.constitution)
            pairs.append((prompt, sol))
        if not pairs:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        return self.tokenize_supervised(pairs, self.cfg.training.max_seq_length)
