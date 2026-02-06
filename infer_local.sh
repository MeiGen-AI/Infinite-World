#!/bin/bash
# Infinite World - Local Inference Script (Single/Multi GPU)
# Usage: bash infer_local.sh [num_gpus]
# Example: bash infer_local.sh 1   (single GPU, no torchrun, avoids port conflict)
# Example: bash infer_local.sh 8   (8 GPUs via torchrun)
#
# Single GPU (num_gpus=1): runs "python scripts/..." directly, no port needed.
# Multi GPU: runs torchrun. If EADDRINUSE, set: export MASTER_PORT=29500

NUM_GPUS=${1:-1}
WORK_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/wuruiqi/infinite-world"

cd $WORK_DIR

echo "=============================================="
echo "Infinite World - Local Inference"
echo "=============================================="
echo "Using $NUM_GPUS GPU(s)"
echo "Working directory: $WORK_DIR"

if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU: run directly to avoid torchrun port (EADDRINUSE)
    python scripts/infworld_inference.py
else
    MASTER_PORT=${MASTER_PORT:-29400}
    echo "MASTER_PORT: $MASTER_PORT"
    torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS \
        --rdzv_id=100 --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:$MASTER_PORT \
        scripts/infworld_inference.py
fi
