# Quick Start: FP8 Training on H200

**Goal**: Train GPT-2 on FineWeb-Edu with FP8 in ~1.5 hours on H200

## Step 1: Install (One-Time, 5 min)

```bash
# On login node
pip install --user accelerate>=0.25.0
pip install --user transformers datasets wandb
pip install --user git+https://github.com/NVIDIA/TransformerEngine.git@stable

# Login to W&B
wandb login
```

## Step 2: Configure SLURM Script (One-Time, 2 min)

Edit `submit_fp8.sh`:

```bash
# Adjust these lines for your cluster:
#SBATCH --gres=gpu:h200:1    # Your GPU type
module load cuda/12.1        # Your CUDA version
module load python/3.10      # Your Python version
```

## Step 3: Submit Job (30 seconds)

```bash
# Test first (5 min)
sbatch --export=CONFIG=configs/test_fp8.yaml submit_fp8.sh

# Full training (1.5 hr)
sbatch submit_fp8.sh

# Monitor
squeue -u $USER
tail -f logs/fp8_train_*.out
```

## That's It!

Your GPT-2 model will train on FineWeb-Edu with FP8 precision, achieving ~2x speedup over BF16.

---

## Next: Parameter Sweep

```bash
# Run lambda sweep (5 experiments)
wandb sweep sweep_lambda_openwebtext.yaml
wandb agent <sweep_id>
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "accelerate not found" | Add to submit_fp8.sh: `export PATH=$HOME/.local/bin:$PATH` |
| CUDA OOM | Edit config: `max_batch_size: 64` |
| NaN loss | Edit config: `fp8_margin: 1` |

**Full docs**: See SUMMARY_FP8_FINEWEB.md

---

**Expected Results**:
- Training time: ~1.5 hours (2 epochs)
- Final validation loss: ~2.5-3.5
- Speedup vs BF16: ~2x
