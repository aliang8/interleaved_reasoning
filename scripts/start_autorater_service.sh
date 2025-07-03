#!/bin/bash

# Start AutoRater FastAPI Service
# This script should be run on the AutoRater VM

set -e

# Configuration
AUTORATER_VM_IP=${AUTORATER_VM_IP:-"0.0.0.0"}
AUTORATER_PORT=${AUTORATER_PORT:-80}
AUTORATER_CONFIG=${AUTORATER_CONFIG:-"configs/autorater_service_config.yaml"}
WORKERS=${WORKERS:-1}

# Auto-detect all available GPUs if not specified
if [ -z "$NUM_GPUS" ]; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "Auto-detected $NUM_GPUS GPUs"
fi

if [ -z "$GPU_IDS" ]; then
    # Create comma-separated list of all GPU IDs (0,1,2,...)
    GPU_IDS=$(seq -s, 0 $((NUM_GPUS-1)))
    echo "Using all GPUs: $GPU_IDS"
fi

# Environment setup
export CUDA_VISIBLE_DEVICES=$GPU_IDS  # Use specified GPUs
export VLLM_DISABLE_ASYNC_OUTPUT_PROC=1
export VLLM_USE_V1=0
export VLLM_DISABLE_SLEEP_MODE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Distributed training environment (standalone mode)
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

echo "======================================"
echo "Starting AutoRater FastAPI Service"
echo "======================================"
echo "Host: $AUTORATER_VM_IP"
echo "Port: $AUTORATER_PORT"
echo "Config: $AUTORATER_CONFIG"
echo "Workers: $WORKERS"
echo "Num GPUs: $NUM_GPUS"
echo "GPU IDs: $CUDA_VISIBLE_DEVICES"
echo "======================================"

# Check if config file exists
if [ ! -f "$AUTORATER_CONFIG" ]; then
    echo "Error: Configuration file not found: $AUTORATER_CONFIG"
    echo "Please create the configuration file or specify correct path"
    exit 1
fi

# Check GPU availability
if ! nvidia-smi > /dev/null 2>&1; then
    echo "Warning: nvidia-smi not available. GPU may not be accessible."
fi

# Create logs directory
mkdir -p logs

# Start the service
echo "Starting AutoRater service..."

# Option 1: Start with auto-initialization from config file
python verl/verl/workers/autorater/fastapi_autorater_service.py \
    --host "$AUTORATER_VM_IP" \
    --port "$AUTORATER_PORT" \
    --workers "$WORKERS" \
    --config "$AUTORATER_CONFIG" \
    2>&1 | tee logs/autorater_service.log

# Option 2: Start without auto-initialization (uncomment if preferred)
# python verl/verl/workers/autorater/fastapi_autorater_service.py \
#     --host "$AUTORATER_VM_IP" \
#     --port "$AUTORATER_PORT" \
#     --workers "$WORKERS" \
#     2>&1 | tee logs/autorater_service.log

echo "AutoRater service stopped." 