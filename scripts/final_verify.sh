#!/usr/bin/env bash
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
echo " PHASE 8: Training smoke test (config validation + CLI)"
echo "========================================"
python main.py config-validate --config config_foundation.yaml
python main.py info
# A short real training run (Ctrl+C once the first steps log):
#   python main.py full-training --config config_foundation.yaml --gpu 0

echo ""
echo "========================================"
echo " PHASE 9: Checkpoint verification"
echo "========================================"
python -c "
from pathlib import Path

# Find latest checkpoint (matches the foundation config output layout)
ckpt_dir = Path('models/dhara-foundation/checkpoints')
if not ckpt_dir.exists():
    print('No checkpoints directory found at', ckpt_dir)
    exit(1)
ckpts = sorted(ckpt_dir.glob('checkpoint-*'))
if not ckpts:
    print('No checkpoint dirs found in', ckpt_dir)
    exit(1)
latest = ckpts[-1]
print(f'Latest checkpoint: {latest}')

for artifact in ('config.json', 'trainer_state.json'):
    p = latest / artifact
    if not p.exists():
        print(f'  {artifact} not found in {latest}')
        exit(1)
print(f'  config.json + trainer_state.json present')
weights = sorted(latest.glob('pytorch_model*.bin')) + sorted(latest.glob('model*.safetensors'))
if not weights:
    print('  WARNING: no weight files found in', latest)
    exit(1)
print(f'  weights: {weights[0].name}')
print('  CHECKPOINT VERIFIED OK')
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
