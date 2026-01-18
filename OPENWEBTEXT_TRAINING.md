# OpenWebText Training Guide

This guide explains the hyperparameter choices for training GPT-2 on the full OpenWebText dataset.

## Quick Start

```bash
# Single training run
python train.py --config configs/openwebtext_adaptive_batch.yaml

# Lambda sweep (5 experiments)
wandb sweep sweep_lambda_openwebtext.yaml
wandb agent <sweep_id>
```

## Dataset Statistics

- **OpenWebText**: ~9 billion tokens, ~40GB when tokenized
- **Sequences**: ~450,000 sequences at 1024 tokens each
- **Size**: Much larger than tinyshakespeare (1MB), similar scale to original GPT-2 pretraining data

## Hyperparameter Choices Explained

### 1. Effective Batch Size = 512

**Why 512?**
- This is the standard batch size used in GPT-2 training (from the original paper)
- Larger batch sizes provide:
  - More stable gradient estimates
  - Better utilization of GPU parallelism
  - Smoother convergence

**How it works with adaptive batch sizing:**
```
If GPU memory allows batch_size = 16:
  → gradient_accumulation_steps = 512 / 16 = 32
  → Effective batch = 16 * 32 = 512 ✓

If GPU memory allows batch_size = 32:
  → gradient_accumulation_steps = 512 / 32 = 16
  → Effective batch = 32 * 16 = 512 ✓
```

### 2. Learning Rate = 3e-3 (scaled from 6e-4)

**Why scale the learning rate?**

In machine learning, there's a well-established relationship between batch size and learning rate:

**Linear Scaling Rule**: LR_new = LR_old × (BatchSize_new / BatchSize_old)
- Old: LR = 6e-4, Batch = 32
- New: LR = 6e-4 × (512/32) = 6e-4 × 16 = 9.6e-3

**Square Root Scaling Rule** (more conservative): LR_new = LR_old × sqrt(BatchSize_new / BatchSize_old)
- New: LR = 6e-4 × sqrt(16) = 6e-4 × 4 = 2.4e-3

**Our choice: 3e-3** (between conservative and aggressive)

**Why does this matter?**
- Larger batches provide lower-variance gradient estimates
- Lower variance means you can take larger steps without overshooting
- If LR is too small with large batches, training will be unnecessarily slow
- If LR is too large, training becomes unstable

### 3. Learning Rate Schedule: Linear Warmup + Cosine Decay

**GPT-2 uses a specific LR schedule:**

1. **Linear warmup** (0 → 2000 steps):
   - LR increases linearly from 0 to max_lr (3e-3)
   - Prevents instability with random initialization

2. **Cosine decay** (2000 → end):
   - LR decreases smoothly following a cosine curve
   - Ends at min_lr = max_lr × 0.1 = 3e-4

**Visualization:**
```bash
# Visualize the schedule
python visualize_lr_schedule.py --warmup_steps 2000 --total_steps 1760 --max_lr 0.003

# For tinyshakespeare (smaller scale)
python visualize_lr_schedule.py --warmup_steps 500 --total_steps 5000 --max_lr 0.0006
```

**Why this schedule?**

**Linear warmup** (2000 steps):
- At the start of training, the model is randomly initialized
- Large learning rates + large batches + random weights = instability
- Warmup gradually increases LR from 0 to max over N steps
- Larger batches benefit from longer warmup periods

**Cosine decay**:
- Smoother decay than step/exponential schedules
- Prevents sudden drops in LR that can destabilize training
- Allows model to "settle" into local minima at end
- Standard in modern transformer training (GPT, BERT, etc.)

**Rule of thumb**: Warmup steps should be ~1-2% of total training steps
- With batch_size=512 and OpenWebText: ~880 steps/epoch
- 2 epochs = ~1760 steps total
- 2000 warmup steps ≈ 100% of one epoch (reasonable for this scale)

**Why decay to 10% (not 0)?**
- Complete decay to 0 can cause training to stall
- Small residual LR allows continued refinement
- 10% is a common choice (some use 1% or 5%)

### 4. Number of Epochs = 2 (not 10)

**Why fewer epochs?**

For large datasets, the relationship between dataset size and epochs is inverse:

| Dataset | Size | Typical Epochs |
|---------|------|----------------|
| Tiny Shakespeare | 1MB | 10-50 |
| Small dataset | 100MB | 5-10 |
| OpenWebText | 40GB | 1-3 |
| Full internet corpus | 100s of GB | <1 |

**Why?**
- Large datasets already contain massive variety
- Overfitting is less of a concern
- Each epoch is expensive (hours to days)
- Diminishing returns after 1-2 passes through data

**Original GPT-2 training**: Trained on 40GB of text for ~1 epoch (actually never saw all data)

### 5. Evaluation/Logging Intervals

**Adjusted for dataset size:**

```python
# Tinyshakespeare (1MB): ~100 batches/epoch → eval every 100 steps
# OpenWebText (40GB): ~880 batches/epoch → eval every 500 steps
```

**Reasoning:**
- You want to evaluate ~2-4 times per epoch
- Too frequent: wastes time on evaluation
- Too infrequent: miss important trends

## Expected Training Time

**On a single A100 (40GB):**
- Batch size per GPU: ~32-48
- Gradient accumulation: ~10-16 steps
- Time per step: ~2-3 seconds
- Steps per epoch: ~880
- Time per epoch: ~40-60 minutes
- Total time (2 epochs): **~2-3 hours**

**On a single RTX 3090/4090 (24GB):**
- Batch size per GPU: ~16-24
- Gradient accumulation: ~21-32 steps
- Time per step: ~3-5 seconds
- Time per epoch: ~60-90 minutes
- Total time (2 epochs): **~3-5 hours**

## Memory Estimation by GPU

| GPU | Memory | Estimated batch_size | grad_accum | Steps/sec |
|-----|--------|---------------------|------------|-----------|
| RTX 3060 | 12GB | 6-8 | 64-85 | 0.2-0.3 |
| RTX 3090 | 24GB | 16-24 | 21-32 | 0.3-0.4 |
| RTX 4090 | 24GB | 20-28 | 18-25 | 0.3-0.5 |
| A100 40GB | 40GB | 32-48 | 10-16 | 0.4-0.5 |
| A100 80GB | 80GB | 64 | 8 | 0.4-0.5 |

*Note: These are estimates. Actual values depend on model architecture and sequence length.*

## Monitoring Training

**Key metrics to watch:**

1. **Training loss**: Should decrease smoothly
   - Initial: ~10-11 (random)
   - After 1 epoch: ~3-4
   - After 2 epochs: ~2.5-3.5

2. **Validation loss**: Should track training loss
   - Gap > 1.0 suggests overfitting (unlikely with 2 epochs)

3. **Perplexity**: exp(loss)
   - After 2 epochs: expect ~12-30
   - Lower is better

4. **Learning rate**: Should follow warmup → cosine decay schedule

5. **Generated samples**: Quality should improve noticeably

## Troubleshooting

### OOM (Out of Memory) Errors

If you get OOM even with adaptive batch sizing:
1. Reduce `max_batch_size` to a lower value (e.g., 16 or 8)
2. Reduce `target_memory_utilization` to 0.75
3. Reduce `max_seq_length` to 512 (if acceptable)

### Training is too slow

- Check that `auto_batch_size` is enabled
- Verify actual batch size with the printed configuration
- Consider using multiple GPUs with distributed training (not yet implemented)

### Loss not decreasing

- Verify learning rate isn't too high (try 2e-3 instead of 3e-3)
- Check that penalty_weight isn't too large (start with 0.0)
- Increase warmup_steps to 5000

## Comparison: Batch Size Scaling Effects

| Effective Batch | LR | Warmup | Pros | Cons |
|-----------------|-----|--------|------|------|
| 32 | 6e-4 | 500 | Stable, works on small GPUs | Very slow on large datasets |
| 128 | 1.2e-3 | 1000 | Good balance | Still slow |
| 512 | 3e-3 | 2000 | **Fast, stable (recommended)** | Needs gradient accumulation |
| 2048 | 6e-3 | 5000 | Fastest convergence | Requires careful tuning |

## Further Reading

For more on batch size and learning rate scaling:
- **Linear Scaling Rule**: Goyal et al., "Accurate, Large Minibatch SGD" (2017)
- **GPT-2 Training**: Radford et al., "Language Models are Unsupervised Multitask Learners" (2019)
- **AdamW**: Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2019)
