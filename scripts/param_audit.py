#!/usr/bin/env python3
"""
Parameter Audit Script for Dhara Class Model (Dhara / NSLT).

Loads model configuration, computes total and module-level parameter counts,
checks embedding parameter cost ratio, and saves report to docs/param_breakdown.txt.
"""

import sys
from pathlib import Path

# Set stdout encoding to UTF-8 if available on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.schema import load_config
from src.models.factory import ModelFactory


def main():
    config_path = PROJECT_ROOT / "config_small.yaml"
    if not config_path.exists():
        config_path = PROJECT_ROOT / "config.yaml"

    print(f"Loading configuration from: {config_path}")
    cfg = load_config(config_path)

    arch = cfg.model.architecture
    model_type = arch.model_type

    print(f"Model Architecture: {model_type}")
    param_info = ModelFactory.estimate_model_size(arch, tokenizer_or_vocab=arch.vocab_size)

    vocab_size = arch.vocab_size
    hidden_size = arch.hidden_size
    embed_params = vocab_size * hidden_size

    total_params_b = param_info.get("total_params_b", 0)
    # Estimate exact params
    if model_type == "dhara_v3":
        v3 = arch.dhara_v3
        ssm_params = v3.n_ssm_layers * (4 * hidden_size ** 2 + hidden_size * v3.d_state * 3 + hidden_size * v3.d_state)
        memory_params = hidden_size * v3.d_hidden * 4 + v3.d_hidden * v3.d_hidden * 2
        intent_params = hidden_size * (v3.n_task_types + v3.n_difficulty_levels + v3.n_reasoning_types + 2)
        planner_params = v3.max_subgoals * hidden_size * 4 + hidden_size * hidden_size
        reasoning_params = v3.n_domains * ((v3.d_state + v3.d_hidden) * v3.d_hidden * 2 + v3.d_hidden * v3.d_hidden)
        workspace_params = v3.d_hidden * v3.d_hidden * 4
        specialist_params = 7 * (v3.d_hidden * v3.d_hidden * 4 + v3.d_hidden)
        decoder_params = v3.d_hidden * vocab_size + v3.d_hidden * v3.n_language_groups
        total_exact = embed_params + ssm_params + memory_params + intent_params + planner_params + reasoning_params + workspace_params + specialist_params + decoder_params
    else:
        total_exact = int(total_params_b * 1e9)

    embed_ratio = embed_params / total_exact if total_exact > 0 else 0.0

    lines = []
    lines.append("=====================================================================")
    lines.append(f"  PARAMETER AUDIT REPORT -- {cfg.model.name} ({model_type})")
    lines.append("=====================================================================")
    lines.append(f"Config File:             {config_path.name}")
    lines.append(f"Vocab Size:              {vocab_size:,}")
    lines.append(f"Hidden Size:             {hidden_size:,}")
    lines.append(f"Total Parameters:        {total_exact:,} ({total_exact / 1e6:.2f} M)")
    lines.append(f"Token Embedding Params:  {embed_params:,} ({embed_params / 1e6:.2f} M)")
    lines.append(f"Embedding Share:         {embed_ratio * 100:.2f}% of total model")
    lines.append("---------------------------------------------------------------------")

    if embed_ratio > 0.40:
        lines.append("[WARNING] Embedding parameters account for > 40% of total model size!")
        lines.append(f"   Currently {embed_ratio*100:.1f}%. Consider:")
        lines.append("   1. Reducing vocab_size in config (e.g. 128,000 -> 32,000)")
        lines.append("   2. Setting tie_word_embeddings: true")
    else:
        lines.append("[OK] Embedding parameter ratio is within acceptable limits (<= 40%).")

    lines.append("=====================================================================")

    report_text = "\n".join(lines)
    print("\n" + report_text + "\n")

    # Save to docs/param_breakdown.txt
    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    out_file = docs_dir / "param_breakdown.txt"
    out_file.write_text(report_text, encoding="utf-8")
    print(f"Report saved to: {out_file}")


if __name__ == "__main__":
    main()
