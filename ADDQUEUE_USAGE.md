# Training with addqueue (Your Cluster System)

Quick guide for submitting FP8 training jobs using your cluster's `addqueue` system.

## Quick Start (3 Steps)

### Step 1: Install Dependencies (One-Time, Login Node)

```bash
pip install --user accelerate>=0.25.0
pip install --user transformers datasets wandb
pip install --user git+https://github.com/NVIDIA/TransformerEngine.git@stable
wandb login
```

### Step 2: Submit Job (Easiest)

```bash
# Using the submission script (easiest)
./submit_addqueue.sh

# Or with custom config
./submit_addqueue.sh configs/my_config.yaml
```

### Step 3: Monitor

```bash
# Check job status
qstat -u $USER

# View logs (adjust path for your cluster)
# Logs usually go to the submission directory or ~/logs/
tail -f *.out
```

## Manual Submission (More Control)

### Basic Submission

```bash
addqueue -q sondhigpu -s -n 30 -m 20 -c 'pt-fp8' \
    --gpus 1 --gputype h200with141gb \
    ./run_fp8_training.sh
```

### With Custom Config

```bash
addqueue -q sondhigpu -s -n 30 -m 20 -c 'pt-fp8' \
    --gpus 1 --gputype h200with141gb \
    ./run_fp8_training.sh configs/my_experiment.yaml
```

### Adjust Resources

```bash
# More time (48 hours)
addqueue -q sondhigpu -s -n 48 -m 20 -c 'pt-fp8' \
    --gpus 1 --gputype h200with141gb \
    ./run_fp8_training.sh

# More memory (40 GB)
addqueue -q sondhigpu -s -n 30 -m 40 -c 'pt-fp8' \
    --gpus 1 --gputype h200with141gb \
    ./run_fp8_training.sh

# Multiple GPUs (adjust train_fp8.py for multi-GPU)
addqueue -q sondhigpu -s -n 30 -m 20 -c 'pt-fp8' \
    --gpus 4 --gputype h200with141gb \
    ./run_fp8_training.sh
```

## Parameter Explanation

### addqueue Parameters

| Parameter | Example | Description |
|-----------|---------|-------------|
| `-q` | `sondhigpu` | Queue name (your GPU queue) |
| `-s` | - | Submit immediately |
| `-n` | `30` | Time limit in hours |
| `-m` | `20` | Memory in GB |
| `-c` | `'pt-fp8'` | Container/environment name |
| `--gpus` | `1` | Number of GPUs |
| `--gputype` | `h200with141gb` | GPU type (H200 with 141GB) |

### Script Arguments

```bash
./run_fp8_training.sh [config_file]

# Default (no argument): uses configs/fineweb_edu_fp8.yaml
./run_fp8_training.sh

# Custom config:
./run_fp8_training.sh configs/test_fp8.yaml
```

## Common Commands

### Submit Jobs

```bash
# Default training (FineWeb-Edu + FP8)
./submit_addqueue.sh

# Test run (Tinyshakespeare, fast)
./submit_addqueue.sh configs/test_auto_batch.yaml

# Custom experiment
./submit_addqueue.sh configs/my_experiment.yaml
```

### Monitor Jobs

```bash
# Check job status
qstat -u $USER

# Watch jobs
watch -n 5 'qstat -u $USER'

# View job details
qstat -j <job_id>

# Cancel job
qdel <job_id>
```

### View Logs

```bash
# Logs location depends on cluster config
# Common locations:
ls -lt *.out *.err          # Current directory
ls -lt ~/logs/*.out         # Home logs directory

# Watch logs in real-time
tail -f <job_name>.o<job_id>
tail -f <job_name>.e<job_id>
```

## Configuration Files

### Available Configs

| Config File | Dataset | FP8 | Batch | Time | Use Case |
|-------------|---------|-----|-------|------|----------|
| `fineweb_edu_fp8.yaml` | FineWeb-Edu | ✓ | 512 | ~1.5hr | **Production** |
| `openwebtext_adaptive_batch.yaml` | OpenWebText | ✗ | 512 | ~3hr | BF16 baseline |
| `test_auto_batch.yaml` | Tinyshake | ✓ | auto | ~5min | Quick test |

### Create Custom Config

```bash
# Copy and modify
cp configs/fineweb_edu_fp8.yaml configs/my_experiment.yaml

# Edit parameters
vim configs/my_experiment.yaml
# Change: penalty_weight, learning_rate, etc.

# Submit
./submit_addqueue.sh configs/my_experiment.yaml
```

## Parameter Sweep (Multiple Experiments)

Run multiple experiments with different lambda values:

```bash
# Create configs for different lambdas
for lambda in 0.0 0.005 0.01 0.02 0.05; do
    # Copy base config
    cp configs/fineweb_edu_fp8.yaml configs/fineweb_lambda${lambda}.yaml

    # Update penalty_weight
    sed -i "s/penalty_weight: 0.0/penalty_weight: ${lambda}/" configs/fineweb_lambda${lambda}.yaml

    # Submit job
    ./submit_addqueue.sh configs/fineweb_lambda${lambda}.yaml

    echo "Submitted job with lambda=$lambda"
    sleep 2  # Brief pause between submissions
done
```

## Troubleshooting

### Issue: "accelerate: command not found"

**Solution**: The wrapper script automatically adds `$HOME/.local/bin` to PATH. If still failing:

```bash
# Check if accelerate is installed
ls $HOME/.local/bin/accelerate

# If not found, install:
pip install --user accelerate>=0.25.0
```

### Issue: "Config file not found"

**Solution**: Use absolute or relative path from project root:

```bash
# Correct (from project root)
./submit_addqueue.sh configs/fineweb_edu_fp8.yaml

# Incorrect
./submit_addqueue.sh fineweb_edu_fp8.yaml
```

### Issue: Job runs but uses BF16 instead of FP8

**Cause**: Accelerate or Transformer Engine not installed/found

**What happens**: Wrapper automatically falls back to BF16 (still works, just 2x slower)

**Check logs**: Look for:
```
✗ Accelerate not found - will use BF16 instead of FP8
WARNING: Accelerate not available, falling back to standard training (BF16)
```

**Solution**: Install missing packages:
```bash
pip install --user accelerate>=0.25.0
pip install --user git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

### Issue: OOM (Out of Memory)

**Solution 1**: Reduce batch size in config:
```yaml
max_batch_size: 64  # Try 32 or 16
```

**Solution 2**: Request more memory:
```bash
./submit_addqueue.sh  # Edit to increase -m 20 to -m 40
```

### Issue: Job stuck at "Collecting samples..."

**Cause**: Large dataset loading

**Solution**: Increase time limit:
```bash
# Edit submit_addqueue.sh:
TIME_HOURS=48  # Instead of 30
```

## Files Overview

```
Your Project/
├── run_fp8_training.sh          ← Main wrapper (called by addqueue)
├── submit_addqueue.sh            ← Easy submission script
├── train_fp8.py                  ← FP8 training script
├── train.py                      ← Standard training (BF16)
│
├── configs/
│   ├── fineweb_edu_fp8.yaml     ← Main config (FP8 + FineWeb)
│   ├── test_auto_batch.yaml     ← Quick test config
│   └── *.yaml                    ← Your custom configs
│
└── logs/                         ← Job outputs (cluster-dependent)
```

## Expected Output

When job starts, you'll see in logs:

```
==========================================
FP8 Training Job - Starting
==========================================
Config file: configs/fineweb_edu_fp8.yaml
Working directory: /mnt/users/clin/workspace/superposition-pretraining
Hostname: gpu-node-123
Date: ...
==========================================

Setting up environment...

Verifying environment...
Python version: 3.10.x

GPU information:
NVIDIA H200, 141312 MiB, ...

PyTorch CUDA check:
  PyTorch version: 2.x.x
  CUDA available: True
  CUDA version: 12.1
  GPU count: 1

Checking required packages...
  transformers: 4.x.x
  datasets: 2.x.x
  accelerate: 0.x.x
  ✓ Accelerate available
  ✓ Transformer Engine available

==========================================
Starting Training
==========================================

Using accelerate for FP8 training

============================================================
FP8 Training with HuggingFace Accelerate
============================================================
Mixed precision: fp8
Device: cuda:0

🚀 FP8 Training Enabled!
  Format: hybrid
  Margin: 0

Batch configuration:
  Batch size per GPU: 128
  Gradient accumulation steps: 4
  Effective batch size: 512

Starting training...
Epoch 1/2: [training progress...]
```

## Performance Expectations

| Dataset | Precision | Time/Step | Total Time (2 epochs) |
|---------|-----------|-----------|----------------------|
| FineWeb-Edu | FP8 | 1.7s | ~1.5 hours |
| FineWeb-Edu | BF16 | 3.2s | ~3 hours |
| Tinyshake | FP8 | 0.3s | ~5 minutes |

## Next Steps

1. **Test first**:
   ```bash
   ./submit_addqueue.sh configs/test_auto_batch.yaml
   ```

2. **Run full training**:
   ```bash
   ./submit_addqueue.sh
   ```

3. **Monitor in W&B**:
   - Go to https://wandb.ai
   - Project: `gpt2-entropy-regularization`

4. **Run parameter sweep** (optional):
   - Use the loop above to test different lambda values

## Quick Reference

```bash
# SUBMIT JOB
./submit_addqueue.sh [config]

# CHECK STATUS
qstat -u $USER

# VIEW LOGS
tail -f *.out

# CANCEL JOB
qdel <job_id>

# EDIT CONFIG
vim configs/fineweb_edu_fp8.yaml
```

That's it! Your cluster is now configured for FP8 training. 🚀
