#!/usr/bin/env bash
# =============================================================================
# 4-GPU FSDP Training — NSLT / MethosV3 on 4x A100 80GB (320GB pooled)
# =============================================================================
# Usage:
#   bash scripts/train_4gpu.sh                        # start full training
#   bash scripts/train_4gpu.sh --fresh-start           # ignore checkpoints
#   bash scripts/train_4gpu.sh config-validate         # validate config
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${CONFIG:-$SCRIPT_DIR/config.yaml}"

# Pre-flight check: Verify CUDA / GPUs are available
if command -v nvidia-smi &> /dev/null; then
    if ! nvidia-smi &> /dev/null; then
        echo "ERROR: nvidia-smi failed. CUDA driver or GPU unavailable."
        exit 1
    fi
else
    echo "WARNING: nvidia-smi not found in PATH."
fi

# Auto-detect available GPUs (those with > 40 GiB free memory)
# Only runs when CUDA_VISIBLE_DEVICES is not already set by the user
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    AVAILABLE_GPUS=$(python3 -c "
import subprocess
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True)
    gpus = []
    for line in result.stdout.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split(', ')
        idx, free_mib = parts[0], parts[1]
        if int(free_mib) > 40000:
            gpus.append(idx)
    print(','.join(gpus))
except Exception:
    pass
" 2>/dev/null || true)
    if [ -n "${AVAILABLE_GPUS:-}" ]; then
        export CUDA_VISIBLE_DEVICES="$AVAILABLE_GPUS"
    fi
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# Set NUM_GPUS from CUDA_VISIBLE_DEVICES
IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_LIST[@]}
MASTER_PORT=${MASTER_PORT:-29500}

CMD="${1:-full-training}"
shift 2>/dev/null || true
export OMP_NUM_THREADS=8
# Force NCCL settings (don't use fallback — override container defaults)
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Calculate training stats for reference display
# Pretrain: 1M steps x 64 eff_batch x 4096 seq_len = ~262B tokens
# SFT: 100K steps x 16 eff_batch x 4096 seq_len = ~6.5B tokens
# Instruction: 50K steps x 16 eff_batch x 4096 seq_len = ~3.3B tokens
echo "====================================================================="
echo "  Launching FSDP training — Methos Class Model"
echo "  Config:   ${CONFIG}"
echo "  GPUs:     ${NUM_GPUS} GPUs (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "  Strategy: FSDP full_shard"
echo "  Command:  ${CMD}"
echo "---------------------------------------------------------------------"
echo "  Pretrain Phase:           1.00M steps | Eff Batch: 64 | Tokens: ~262B"
echo "  SFT Phase:              100.00K steps | Eff Batch: 16 | Tokens: ~6.5B"
echo "  Instruction Tuning:      50.00K steps | Eff Batch: 16 | Tokens: ~3.3B"
echo "  Total Pipeline Tokens:   ~272B tokens (~130x Chinchilla overtraining)"
echo "====================================================================="

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/main.py" \
    --config "${CONFIG}" \
    "${CMD}" \
    "$@"
