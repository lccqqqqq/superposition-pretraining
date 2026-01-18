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

addqueue -q sondhigpu -s -n 30 -m 20 -c 'pt-fp8' \
    --gpus 1 --gputype h200with141gb \
    accelerate launch \
    --mixed_precision fp8 \
    --num_processes 1 \
    train_fp8.py \
    --config configs/fineweb_edu_fp8.yaml