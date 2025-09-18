#!/bin/bash

# PartCrafter DPO Training Launch Script
# This script launches DPO training with appropriate settings

echo "🚀 Starting PartCrafter DPO Training"
echo "====================================="

# Set environment variables
export CUDA_VISIBLE_DEVICES=7
export WANDB_API_KEY="ce1816bb45de1fc10f92c8bc17f2d7cc9b1a8757"

# Activate conda environment
echo "📦 Activating conda environment..."
conda activate partcrafter

# Check if environment is activated
if [ "$CONDA_DEFAULT_ENV" != "partcrafter" ]; then
    echo "❌ Failed to activate partcrafter environment"
    exit 1
fi

echo "✅ Environment activated: $CONDA_DEFAULT_ENV"

# Set training parameters
MAX_SAMPLES=50  # Start with small dataset for testing
BATCH_SIZE=1
LEARNING_RATE=5e-5
MAX_TRAIN_STEPS=1000
DPO_BETA=0.1
OUTPUT_DIR="./outputs"
EXP_NAME="partcrafter_dpo_test"

echo "📊 Training Configuration:"
echo "   Max samples: $MAX_SAMPLES"
echo "   Batch size: $BATCH_SIZE"
echo "   Learning rate: $LEARNING_RATE"
echo "   Max train steps: $MAX_TRAIN_STEPS"
echo "   DPO beta: $DPO_BETA"
echo "   Output dir: $OUTPUT_DIR"
echo "   Experiment name: $EXP_NAME"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Launch training
echo "🏃 Starting training..."
python scripts/train_partcrafter_dpo.py \
    --max_samples "$MAX_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --max_train_steps "$MAX_TRAIN_STEPS" \
    --dpo_beta "$DPO_BETA" \
    --output_dir "$OUTPUT_DIR" \
    --exp_name "$EXP_NAME" \
    --log_interval 10 \
    --save_interval 100 \
    --seed 42

echo "✅ Training completed!"
