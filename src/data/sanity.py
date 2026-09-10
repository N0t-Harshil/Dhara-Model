from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from transformers import PreTrainedTokenizerBase

from src.data.ast_filter import identifier_ratio, executable_ratio

logger = logging.getLogger(__name__)

CONTENT_MARKERS: Dict[str, List[str]] = {
    "function_bodies": ["def ", "function ", "fn ", "func ", "=>"],
    "classes": ["class ", "struct ", "trait ", "interface "],
    "algorithms": ["for ", "while ", "if ", "else ", "switch ", "match "],
    "documentation": ['"""', "'''", "///", "/**", "/*!", "..", ":param", ":return"],
    "mathematics": ["=", "+", "-", "*", "/", "%", "∑", "∫", "√", "π"],
    "natural_language": ["the ", "is ", "are ", "was ", "were ", "have ", "has "],
    "code_keywords": ["import ", "from ", "include ", "using ", "package ", "module "],
    "data_structures": ["list", "dict", "array", "vector", "map", "set", "tuple"],
    "import_statements": ["import ", "from ", "include ", "require ", "use "],
}


def check_packed_sample(
    input_ids: List[int],
    labels: List[int],
    attention_mask: List[int],
    tokenizer: PreTrainedTokenizerBase,
) -> Dict[str, Any]:
    decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
    eos_id = tokenizer.eos_token_id or 0
    # Don't count EOS padding (attention_mask==0) as document separators:
    # pack_sequences pads input_ids with eos_token_id while masking pads.
    num_segments = sum(1 for t, m in zip(input_ids, attention_mask) if t == eos_id and m == 1)
    pad_ratio = attention_mask.count(0) / max(len(attention_mask), 1)
    label_mask_ratio = labels.count(-100) / max(len(labels), 1)
    has_func = any(m in decoded for m in CONTENT_MARKERS["function_bodies"])
    has_class = any(m in decoded for m in CONTENT_MARKERS["classes"])
    has_algo = any(m in decoded for m in CONTENT_MARKERS["algorithms"])
    has_docs = any(m in decoded for m in CONTENT_MARKERS["documentation"])
    has_math = any(m in decoded for m in CONTENT_MARKERS["mathematics"])
    has_nl = any(m in decoded for m in CONTENT_MARKERS["natural_language"])
    has_code_kw = any(m in decoded for m in CONTENT_MARKERS["code_keywords"])
    has_imports = any(m in decoded for m in CONTENT_MARKERS["import_statements"])
    id_ratio = identifier_ratio(decoded)
    exec_ratio_val = executable_ratio(decoded, 'python')
    result = {
        "decoded_preview": decoded[:500],
        "num_segments": num_segments,
        "padding_ratio": round(pad_ratio, 4),
        "label_mask_ratio": round(label_mask_ratio, 4),
        "has_function_code": has_func,
        "has_classes": has_class,
        "has_algorithms": has_algo,
        "has_documentation": has_docs,
        "has_mathematics": has_math,
        "has_natural_language": has_nl,
        "has_code_keywords": has_code_kw,
        "has_imports": has_imports,
        "identifier_ratio": round(id_ratio, 4),
        "executable_ratio": round(exec_ratio_val, 4),
    }
    return result


def run_sanity_checks(
    dataset,
    tokenizer: PreTrainedTokenizerBase,
    num_samples: int = 10,
    max_decode: int = 512,
) -> Dict[str, Any]:
    logger.info("Running sanity checks on %d random samples...", num_samples)
    if len(dataset) == 0:
        return {"error": "empty_dataset", "samples_checked": 0}
    n = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), n)
    results: List[Dict[str, Any]] = []
    content_types: Dict[str, int] = {
        "function": 0,
        "class": 0,
        "algorithm": 0,
        "documentation": 0,
        "mathematics": 0,
        "natural_language": 0,
        "imports": 0,
    }
    total_padding = 0.0
    total_segments = 0
    only_license = 0
    for idx in indices:
        sample = dataset[idx]
        input_ids = sample.get("input_ids", [])
        labels = sample.get("labels", [])
        attention_mask = sample.get("attention_mask", [1] * len(input_ids))
        if isinstance(input_ids, (list, tuple)):
            ids = list(input_ids)
        else:
            ids = input_ids.tolist() if hasattr(input_ids, 'tolist') else list(input_ids)
        lbls = list(labels) if isinstance(labels, (list, tuple)) else labels.tolist() if hasattr(labels, 'tolist') else list(labels)
        mask = list(attention_mask) if isinstance(attention_mask, (list, tuple)) else attention_mask.tolist() if hasattr(attention_mask, 'tolist') else list(attention_mask)
        truncated = ids[:max_decode]
        lbls = lbls[:max_decode]
        mask = mask[:max_decode]
        check = check_packed_sample(truncated, lbls, mask, tokenizer)
        total_padding += check["padding_ratio"]
        total_segments += check["num_segments"]
        if check["has_function_code"]:
            content_types["function"] += 1
        if check["has_classes"]:
            content_types["class"] += 1
        if check["has_algorithms"]:
            content_types["algorithm"] += 1
        if check["has_documentation"]:
            content_types["documentation"] += 1
        if check["has_mathematics"]:
            content_types["mathematics"] += 1
        if check["has_natural_language"]:
            content_types["natural_language"] += 1
        if check["has_imports"]:
            content_types["imports"] += 1
        decoded_lower = check["decoded_preview"].lower()
        license_signals = ['copyright', 'license', 'all rights reserved', 'spdx', 'mit license']
        if all(s in decoded_lower for s in license_signals[:3]):
            only_license += 1
        results.append(check)
    summary = {
        "samples_checked": n,
        "average_padding_ratio": round(total_padding / n, 4) if n else 0,
        "average_segments_per_sample": round(total_segments / n, 2) if n else 0,
        "content_type_counts": content_types,
        "samples_with_only_license_headers": only_license,
        "pct_with_function": round(content_types["function"] / n * 100, 1),
        "pct_with_class": round(content_types["class"] / n * 100, 1),
        "pct_with_algorithm": round(content_types["algorithm"] / n * 100, 1),
        "pct_with_documentation": round(content_types["documentation"] / n * 100, 1),
        "pct_with_mathematics": round(content_types["mathematics"] / n * 100, 1),
        "pct_with_natural_language": round(content_types["natural_language"] / n * 100, 1),
    }
    if only_license > n * 0.5:
        summary["warning"] = "More than 50% of sampled samples contain only license headers!"
    if content_types["function"] == 0 and content_types["mathematics"] == 0:
        summary["warning"] = "No function or mathematics content found in samples — check code datasets"
    logger.info("Sanity check summary: %s", summary)
    return summary
