##!/usr/bin/env bash
set -euo pipefail

# Production validation script — run once on the Linux training server.
# Prerequisites:
#   export HF_TOKEN=your_token_here
#   cd /path/to/specialized-coding-model

echo "========================================"
echo " PHASE 1: Dataset verification"
echo "========================================"
python scripts/verify_datasets.py --token "$HF_TOKEN" --max-samples 5

echo ""
echo "========================================"
echo " PHASE 2: Build documentation corpus"
echo "========================================"
python src/data/doc_builder.py --output-dir data/docs

echo ""
echo "========================================"
echo " PHASE 3-7: Full production validation"
echo "========================================"
python scripts/production_validation.py \
    --token "$HF_TOKEN" \
    --report output/production_report.txt \
    --skip-smoke \
    --max-samples 5000

echo ""
echo "========================================"
echo " PHASE 8: Training smoke test (100 steps)"
echo "========================================"
python scripts/run_training.py \
    --config config_foundation.yaml \
    --max-steps 100 \
    --logging-steps 1 \
    --save-steps 1000 2>&1 | tail -50

echo ""
echo "========================================"
echo " PHASE 9: Checkpoint verification"
echo "========================================"
python -c "
from pathlib import Path
import torch

# Find latest checkpoint
ckpt_dir = Path('checkpoints')
if not ckpt_dir.exists():
    print('No checkpoints directory found')
    exit(1)
ckpts = sorted(ckpt_dir.glob('step_*'))
if not ckpts:
    print('No checkpoint steps found in', ckpt_dir)
    exit(1)
latest = ckpts[-1]
print(f'Latest checkpoint: {latest}')

# Verify checkpoint contents
ckpt_path = latest / 'training_state.pt'
if not ckpt_path.exists():
    print(f'  training_state.pt not found in {latest}')
    exit(1)
state = torch.load(str(ckpt_path), map_location='cpu')
print(f'  optimizer keys: {list(state.get(\"optimizer\", {}).keys())[:3]}')
print(f'  scheduler keys: {list(state.get(\"scheduler\", {}).keys())[:3]}')
print(f'  step: {state.get(\"step\", \"?\")}')
print(f'  loss: {state.get(\"loss\", \"?\")}')
print(f'  CHECKPOINT VERIFIED OK')

# Test resume
print()
print('Testing checkpoint resume...')
import sys
sys.path.insert(0, '.')
from src.config.schema import load_config
from src.training.trainer import create_trainer

cfg = load_config('config_foundation.yaml')
trainer = create_trainer(cfg, resume_from=str(latest))
print(f'  Resume OK — starting step: {trainer.current_step}')
print(f'  CHECKPOINT RESUME VERIFIED OK')
"

echo ""
echo "========================================"
echo " ALL VALIDATION PHASES COMPLETE"
echo "========================================"
echo ""
echo "Run the following to generate the full report:"
echo "  python scripts/production_validation.py --token \$HF_TOKEN --report output/production_report.txt"
echo ""
echo "Check output/production_report.txt for final readiness decision."
