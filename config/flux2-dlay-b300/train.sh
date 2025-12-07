#!/bin/bash
# ============================================
# FLUX.2 LyCORIS LoKr Training Script
# Optimized for Single B300 GPU (275GB VRAM)
# Subject: DLAY man persona training
# ============================================

set -e

# Navigate to SimpleTuner directory
cd /root/SimpleTuner

# Activate virtual environment
source venv/bin/activate

# Configuration paths
export ACCELERATE_CONFIG_FILE="config/flux2-dlay-b300/accelerate_config.yaml"
export CONFIG_PATH="config/flux2-dlay-b300/config.json"
export SIMPLETUNER_LOG_LEVEL=INFO
export PYTHONUNBUFFERED=1

# Enable TF32 for faster matrix operations on Blackwell GPUs
export NVIDIA_TF32_OVERRIDE=1

# Memory optimization - expandable segments for large models
export PYTORCH_ALLOC_CONF=expandable_segments:True

# CUDA settings
export CUDA_VISIBLE_DEVICES=0

echo "=============================================="
echo "SimpleTuner - FLUX.2 LyCORIS LoKr Training"
echo "=============================================="
echo "Subject: DLAY man persona"
echo "Hardware: Single B300 GPU (275GB VRAM)"
echo "Dataset: /root/DATASETS/DLAY/subject (269 images)"
echo "Output: /root/output/simpletuner-flux2-dlay-lokr"
echo "=============================================="
echo ""

# Check if logged into wandb
if ! wandb status &>/dev/null; then
    echo "Warning: Not logged into wandb. Run 'wandb login' if you want tracking."
fi

# Launch training
echo "Starting training..."
accelerate launch \
    --config_file "$ACCELERATE_CONFIG_FILE" \
    simpletuner/train.py \
    --config_backend=json

echo ""
echo "=============================================="
echo "Training completed!"
echo "Output saved to: /root/output/simpletuner-flux2-dlay-lokr"
echo "=============================================="
