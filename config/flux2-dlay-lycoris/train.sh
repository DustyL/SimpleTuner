#!/bin/bash
# Flux2 LyCORIS Training Script for 4xB200 GPUs
# Model: DLAY man subject training with masked loss

set -e

# Configuration
export ACCELERATE_CONFIG_FILE="config/flux2-dlay-lycoris/accelerate_config.yaml"
export SIMPLETUNER_LOG_LEVEL=INFO
export PYTHONUNBUFFERED=1

# Optional: Enable TF32 for faster matrix operations on Ampere+ GPUs
export NVIDIA_TF32_OVERRIDE=1

# Optional: Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "Starting Flux2 LyCORIS Training"
echo "Model: DLAY man Dreambooth-style"
echo "Hardware: 4x B200 GPUs with FSDP2"
echo "=============================================="

# Login to HuggingFace if needed
# huggingface-cli login

# Login to Weights & Biases if using wandb tracking
# wandb login

# Launch training with accelerate
accelerate launch \
    --config_file "$ACCELERATE_CONFIG_FILE" \
    simpletuner/train.py \
    --config_backend=json \
    --config config/flux2-dlay-lycoris/config.json

echo "=============================================="
echo "Training completed!"
echo "=============================================="
