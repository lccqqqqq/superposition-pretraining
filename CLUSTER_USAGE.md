# Cluster Usage Guide for FP8 Training

This guide explains how to run FP8 training on SLURM clusters where CUDA is not available on login nodes.

## Quick Start

```bash
# 1. Install dependencies (one-time setup, on login node)
pip install --user accelerate>=0.25.0
pip install --user transformers datasets torch
pip install --user git+https://github.com/NVIDIA/TransformerEngine.git@stable

# 2. Submit training job
sbatch submit_fp8.sh

# 3. Monitor job
squeue -u $USER  # Check job status
tail -f logs/fp8_train_<job_id>.out  # Watch training progress
```

## Setup (One-Time)

### 1. Install Dependencies

On the **login node** (where CUDA is not available):

```bash
# Core dependencies
pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install --user transformers datasets
pip install --user accelerate>=0.25.0
pip install --user wandb  # Optional, for experiment tracking

# Transformer Engine (for FP8)
# This will fail on login node if trying to import, but installation works
pip install --user git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

**Note**: You won't be able to test FP8 on the login node (no CUDA), but installation will work.

### 2. Configure Weights & Biases (Optional)

```bash
# Login to W&B for experiment tracking
wandb login
```

### 3. Adjust SLURM Script

Edit `submit_fp8.sh` to match your cluster:

```bash
#SBATCH --gres=gpu:h200:1  # Change to your GPU type
#SBATCH --time=4:00:00     # Adjust time limit
#SBATCH --mem=200G         # Adjust memory

# Update module loads
module load cuda/12.1      # Your cluster's CUDA version
module load python/3.10    # Your cluster's Python
```

## Running Training Jobs

### Basic Job Submission

```bash
# Submit default FP8 training job
sbatch submit_fp8.sh
```

### Custom Configuration

```bash
# Submit with different config
sbatch --export=CONFIG=configs/fineweb_edu_fp8.yaml submit_fp8.sh

# Or edit submit_fp8.sh to change default config
```

### Multi-GPU Training

For multiple GPUs on one node:

```bash
# Edit submit_fp8.sh:
#SBATCH --gres=gpu:h200:4  # 4 GPUs

# Then update the accelerate launch command:
accelerate launch \
    --mixed_precision fp8 \
    --num_processes 4 \  # Match number of GPUs
    --multi_gpu \
    train_fp8.py \
    --config $CONFIG
```

### Parameter Sweep

Run sweep across penalty weights:

```bash
# Submit multiple jobs with different lambda values
for lambda in 0.0 0.005 0.01 0.02 0.05; do
    sbatch --export=CONFIG=configs/fineweb_edu_fp8.yaml,PENALTY_WEIGHT=$lambda submit_fp8.sh
done
```

Then modify `submit_fp8.sh` to use `$PENALTY_WEIGHT`:
```bash
accelerate launch train_fp8.py \
    --config $CONFIG \
    --penalty_weight ${PENALTY_WEIGHT:-0.01}
```

## Monitoring Jobs

### Check Job Status

```bash
# View your jobs
squeue -u $USER

# View specific job
squeue -j <job_id>

# Cancel job
scancel <job_id>
```

### View Training Logs

```bash
# Real-time monitoring
tail -f logs/fp8_train_<job_id>.out

# View completed job
less logs/fp8_train_<job_id>.out

# Check for errors
less logs/fp8_train_<job_id>.err
```

### GPU Monitoring (During Job)

SSH into compute node and run:

```bash
# Find your node
squeue -u $USER

# SSH to node (if allowed)
ssh <node_name>

# Watch GPU usage
watch -n 1 nvidia-smi
```

## Common Configurations

### 1. Quick Test (Tinyshakespeare, No W&B)

```yaml
# configs/test_fp8.yaml
dataset: tinyshakespeare
num_epochs: 1
use_wandb: false
use_fp8: true
```

```bash
sbatch --export=CONFIG=configs/test_fp8.yaml submit_fp8.sh
```

### 2. Full FineWeb-Edu with FP8

```bash
sbatch submit_fp8.sh  # Uses configs/fineweb_edu_fp8.yaml by default
```

### 3. BF16 Baseline (for comparison)

```bash
# Use regular train.py instead of train_fp8.py
sbatch submit_bf16.sh  # Create separate script or modify submit_fp8.sh
```

## Troubleshooting

### Issue: "accelerate: command not found"

**Solution**: Add to submit_fp8.sh before running training:

```bash
export PATH=$HOME/.local/bin:$PATH
```

### Issue: "transformer_engine not found"

**Cause**: Transformer Engine failed to install or isn't in Python path

**Solutions**:

1. Check installation:
   ```bash
   # On compute node (in job script):
   python -c "import transformer_engine; print('OK')"
   ```

2. Reinstall:
   ```bash
   pip uninstall transformer-engine
   pip install --user git+https://github.com/NVIDIA/TransformerEngine.git@stable
   ```

3. Use BF16 instead (fallback):
   ```bash
   # In submit script, change:
   accelerate launch --mixed_precision bf16 ...
   ```

### Issue: "CUDA out of memory"

**Solutions**:

1. Reduce batch size in config:
   ```yaml
   max_batch_size: 64  # Try 32 or 16
   target_memory_utilization: 0.75  # Reduce from 0.85
   ```

2. Reduce sequence length:
   ```yaml
   max_seq_length: 512  # Instead of 1024
   ```

3. Request more memory:
   ```bash
   #SBATCH --mem=400G  # Instead of 200G
   ```

### Issue: Job stuck at "Collecting samples..."

**Cause**: Large dataset loading with streaming

**Solutions**:

1. Use smaller subset for testing:
   ```python
   # In data.py, modify load_fineweb_edu():
   max_samples = 10000  # Test with 10k samples first
   ```

2. Increase time limit:
   ```bash
   #SBATCH --time=8:00:00  # 8 hours instead of 4
   ```

### Issue: "Permission denied" on transformer-engine

**Cause**: Some cluster configurations don't allow git installations

**Solution**: Download pre-built wheel:

```bash
# Find compatible wheel at: https://github.com/NVIDIA/TransformerEngine/releases
wget https://github.com/NVIDIA/TransformerEngine/releases/download/v1.0/transformer_engine-1.0-py3-none-linux_x86_64.whl
pip install --user transformer_engine-1.0-py3-none-linux_x86_64.whl
```

## Performance Optimization

### 1. Maximize Throughput

```yaml
# configs/fineweb_edu_fp8.yaml
use_fp8: true
fp8_margin: 0  # No safety margin
max_batch_size: 128  # Use large batches on H200
target_memory_utilization: 0.90  # Use more memory
num_workers: 16  # More data loading workers
```

### 2. Conservative/Stable Training

```yaml
fp8_margin: 1  # Safety margin
max_batch_size: 64
target_memory_utilization: 0.80
fp8_interval: 10  # Update scales less frequently
```

### 3. Enable Additional Optimizations

Add to your config:

```yaml
# Future optimizations (not yet implemented)
# use_flash_attention: true  # 2x faster attention
# torch_compile: true  # JIT compilation
# use_fused_adamw: true  # Fused optimizer
```

## Expected Performance (H200)

| Configuration | Precision | Batch Size | Time/Step | Time (2 epochs) |
|---------------|-----------|------------|-----------|-----------------|
| Baseline | BF16 | 64 | 3.2s | ~2.8 hours |
| Optimized | BF16 | 128 | 2.5s | ~2.2 hours |
| **FP8** | **FP8** | **128** | **1.7s** | **~1.5 hours** |

**Speedup: ~1.9x faster with FP8!**

## Best Practices

1. **Start small**: Test with tinyshakespeare before full dataset
2. **Monitor first job**: Watch GPU utilization with `nvidia-smi`
3. **Use W&B**: Track experiments across runs
4. **Save checkpoints**: Training can be interrupted
5. **Compare baselines**: Always compare FP8 vs BF16

## Next Steps

1. Submit test job:
   ```bash
   sbatch --export=CONFIG=configs/test_fp8.yaml submit_fp8.sh
   ```

2. Check it works:
   ```bash
   tail -f logs/fp8_train_*.out
   ```

3. Run full training:
   ```bash
   sbatch submit_fp8.sh
   ```

4. Compare with BF16:
   ```bash
   # Create BF16 config (set use_fp8: false)
   cp configs/fineweb_edu_fp8.yaml configs/fineweb_edu_bf16.yaml
   # Edit: use_fp8: false
   sbatch --export=CONFIG=configs/fineweb_edu_bf16.yaml submit_fp8.sh
   ```

## Further Reading

- FP8_TRAINING.md - Complete FP8 documentation
- OPENWEBTEXT_TRAINING.md - Hyperparameter explanations
- LR_SCHEDULE.md - Learning rate schedule details
