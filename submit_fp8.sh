#!/bin/bash
#SBATCH --job-name=gpt2-fp8
#SBATCH --output=logs/fp8_train_%j.out
#SBATCH --error=logs/fp8_train_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=200G

# FP8 Training Job Submission Script for SLURM clusters
#
# This script submits a GPT-2 training job with FP8 precision on H200 GPUs.
#
# Usage:
#   # Submit job
#   sbatch submit_fp8.sh
#
#   # Or with custom config
#   sbatch --export=CONFIG=configs/fineweb_edu_fp8.yaml submit_fp8.sh
#
#   # Monitor job
#   squeue -u $USER
#   tail -f logs/fp8_train_<job_id>.out

# Create logs directory
mkdir -p logs

# Print job info
echo "=========================================="
echo "FP8 Training Job Started"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Load modules (adjust for your cluster)
module purge
module load cuda/12.1  # Adjust to your cluster's CUDA version
module load python/3.10  # Adjust to your Python version

# Or use conda if available
# source activate your_env_name

# Verify GPU
echo "Checking GPU..."
nvidia-smi
echo ""

# Default config
CONFIG=${CONFIG:-configs/fineweb_edu_fp8.yaml}

echo "Configuration: $CONFIG"
echo ""

# Install dependencies if needed (only on first run)
# Uncomment these lines if packages aren't installed
# pip install --user accelerate>=0.25.0
# pip install --user git+https://github.com/NVIDIA/TransformerEngine.git@stable

# Verify installations
echo "Verifying installations..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')" || echo "Warning: accelerate not installed"
python -c "import transformer_engine; print(f'Transformer Engine: {transformer_engine.__version__}')" || echo "Warning: transformer_engine not installed"
echo ""

# Run training with accelerate
echo "Starting FP8 training..."
echo "=========================================="

accelerate launch \
    --mixed_precision fp8 \
    --num_processes 1 \
    train_fp8.py \
    --config $CONFIG

# Alternative: Run without accelerate (uses BF16)
# python train.py --config $CONFIG

# Print completion info
echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
