#!/bin/bash
# Wrapper script for FP8 training on cluster with addqueue
#
# This script sets up the environment and launches FP8 training
# using accelerate + Transformer Engine on H200 GPUs.
#
# Usage:
#   addqueue -q sondhigpu -s -n 30 -m 20 -c 'pt-fp8' \
#       --gpus 1 --gputype h200with141gb \
#       ./run_fp8_training.sh [config_file]
#
# Example:
#   addqueue -q sondhigpu -s -n 30 -m 20 -c 'pt-fp8' \
#       --gpus 1 --gputype h200with141gb \
#       ./run_fp8_training.sh configs/fineweb_edu_fp8.yaml

# Note: Don't use 'set -e' during verification stage to allow graceful degradation
# We'll check critical things explicitly and only fail if truly necessary

# ============================================================================
# Configuration
# ============================================================================

# Get config file from argument or use default
CONFIG=${1:-configs/fineweb_edu_fp8.yaml}

echo "=========================================="
echo "FP8 Training Job - Starting"
echo "=========================================="
echo "Config file: $CONFIG"
echo "Working directory: $PWD"
echo "Hostname: $HOSTNAME"
echo "Date: $(date)"
echo "=========================================="
echo ""

# ============================================================================
# Environment Setup
# ============================================================================

echo "Setting up environment..."

# Add user's local bin to PATH (for accelerate)
export PATH=$HOME/.local/bin:$PATH

# Add current directory to Python path
export PYTHONPATH=$PWD:$PYTHONPATH

# Disable tokenizers parallelism warning
export TOKENIZERS_PARALLELISM=false

# Load modules if needed (uncomment and adjust for your cluster)
# module load cuda/12.1
# module load python/3.10

echo "PATH: $PATH"
echo ""

# ============================================================================
# Verify Environment
# ============================================================================

echo "Verifying environment..."
echo ""

# Check Python
echo "Python version:"
python --version
echo ""

# Check GPU
echo "GPU information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || nvidia-smi -L
echo ""

# Check CUDA availability in PyTorch
echo "PyTorch CUDA check:"
python -c "import torch; print(f'  PyTorch version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA version: {torch.version.cuda}'); print(f'  GPU count: {torch.cuda.device_count()}')"
echo ""

# Check required packages
echo "Checking required packages..."
python -c "import transformers; print(f'  transformers: {transformers.__version__}')" || { echo "  ✗ ERROR: transformers not installed"; MISSING_DEPS=true; }
python -c "import datasets; print(f'  datasets: {datasets.__version__}')" || { echo "  ✗ ERROR: datasets not installed"; MISSING_DEPS=true; }

if [ "$MISSING_DEPS" = true ]; then
    echo ""
    echo "ERROR: Missing critical dependencies. Please install:"
    echo "  pip install --user transformers datasets"
    exit 1
fi

# Check accelerate (required for FP8)
if python -c "import accelerate; print(f'  accelerate: {accelerate.__version__}')" 2>/dev/null; then
    HAS_ACCELERATE=true
    echo "  ✓ Accelerate available"
else
    HAS_ACCELERATE=false
    echo "  ✗ Accelerate not found - will use BF16 instead of FP8"
fi

# Check Transformer Engine (optional for FP8)
if python -c "import transformer_engine; print(f'  transformer_engine: {transformer_engine.__version__}')" 2>/dev/null; then
    echo "  ✓ Transformer Engine available"
else
    echo "  ✗ Transformer Engine not found - FP8 may not work optimally"
fi

echo ""

# ============================================================================
# Verify Config File
# ============================================================================

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

echo "Config file contents:"
echo "----------------------------------------"
head -30 "$CONFIG"
echo "----------------------------------------"
echo ""

# ============================================================================
# Run Training
# ============================================================================

# Now enable strict error handling for the actual training
set -e

echo "=========================================="
echo "Starting Training"
echo "=========================================="
echo ""

# Check if accelerate command exists and if Transformer Engine is available
if command -v accelerate &> /dev/null && [ "$HAS_ACCELERATE" = true ]; then
    # Check if Transformer Engine is available for FP8
    if python -c "import transformer_engine" 2>/dev/null; then
        echo "Using accelerate with FP8 training (Transformer Engine available)"
        echo "Command: accelerate launch --mixed_precision fp8 --num_processes 1 train_fp8.py --config $CONFIG"
        echo ""

        accelerate launch \
            --mixed_precision fp8 \
            --num_processes 1 \
            train_fp8.py \
            --config $CONFIG

        TRAIN_EXIT_CODE=$?
    else
        echo "Transformer Engine not found - using BF16 with standard training script"
        echo "Note: BF16 is still ~4x faster than FP32, just not as fast as FP8 (2x slower than FP8)"
        echo "Command: python train.py --config $CONFIG"
        echo ""

        # Use regular train.py which doesn't require Transformer Engine
        python train.py --config $CONFIG

        TRAIN_EXIT_CODE=$?
    fi
else
    echo "WARNING: Accelerate not available, falling back to standard training (BF16)"
    echo "Command: python train.py --config $CONFIG"
    echo ""

    python train.py --config $CONFIG

    TRAIN_EXIT_CODE=$?
fi

# ============================================================================
# Completion
# ============================================================================

echo ""
echo "=========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code: $TRAIN_EXIT_CODE"
fi
echo "Completed at: $(date)"
echo "=========================================="

exit $TRAIN_EXIT_CODE
