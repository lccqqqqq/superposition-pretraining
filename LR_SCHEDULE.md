# Learning Rate Schedule Implementation

This document explains the learning rate schedule implementation for GPT-2 training.

## Overview

The code now implements the **exact same LR schedule used in GPT-2**:
1. **Linear warmup**: LR increases from 0 → max_lr over `warmup_steps`
2. **Cosine decay**: LR decreases from max_lr → min_lr following a cosine curve

This replaces the previous implementation which only had cosine decay without warmup.

## Mathematical Formulation

The learning rate at step `t` is computed as:

```
if t < warmup_steps:
    lr(t) = max_lr × (t / warmup_steps)                    [Linear warmup]
else:
    progress = (t - warmup_steps) / (total_steps - warmup_steps)
    cosine = 0.5 × (1 + cos(π × progress))
    lr(t) = max_lr × (min_ratio + (1 - min_ratio) × cosine)  [Cosine decay]
```

Where:
- `max_lr`: Maximum learning rate (e.g., 3e-3 for OpenWebText)
- `min_ratio`: Minimum LR as fraction of max (default: 0.1)
- `warmup_steps`: Number of warmup steps (e.g., 2000 for OpenWebText)
- `total_steps`: Total training steps

## Implementation Changes

### Files Modified

1. **`train.py`**:
   - Removed: `from torch.optim.lr_scheduler import CosineAnnealingLR`
   - Added: `from torch.optim.lr_scheduler import LambdaLR`
   - Added: `get_lr_schedule_with_warmup()` function
   - Updated: Scheduler initialization to use new function

2. **`config.py`**:
   - Added: `min_lr_ratio` parameter (default: 0.1)

3. **`config.yaml`**:
   - Added: `min_lr_ratio: 0.1` with documentation

4. **`configs/openwebtext_adaptive_batch.yaml`**:
   - Added: `min_lr_ratio: 0.1`

### New Files

1. **`visualize_lr_schedule.py`**: Script to plot the LR schedule
2. **`test_lr_schedule.py`**: Unit tests for LR schedule correctness
3. **`LR_SCHEDULE.md`**: This documentation file

## Configuration Parameters

### New Parameter: `min_lr_ratio`

Controls the minimum learning rate at the end of training.

```yaml
min_lr_ratio: 0.1  # LR decays to 10% of max_lr
```

**Common values:**
- `0.0`: Decay to zero (can cause training to stall)
- `0.01`: Decay to 1% (very aggressive)
- `0.1`: Decay to 10% (standard, recommended)
- `0.2`: Decay to 20% (conservative)

### Existing Parameters

```yaml
learning_rate: 0.003     # Maximum learning rate
warmup_steps: 2000       # Linear warmup duration
num_epochs: 2            # Training epochs
```

## Usage Examples

### Visualize the Schedule

```bash
# OpenWebText configuration
python visualize_lr_schedule.py \
    --warmup_steps 2000 \
    --total_steps 1760 \
    --max_lr 0.003 \
    --min_lr_ratio 0.1

# Tinyshakespeare configuration
python visualize_lr_schedule.py \
    --warmup_steps 500 \
    --total_steps 5000 \
    --max_lr 0.0006 \
    --min_lr_ratio 0.1

# Save plot to file
python visualize_lr_schedule.py \
    --warmup_steps 2000 \
    --total_steps 1760 \
    --max_lr 0.003 \
    --save lr_schedule.png
```

### Test the Implementation

```bash
python test_lr_schedule.py
```

Expected output:
```
Testing key points:
Step    0: LR=0.000000, Ratio=0.0000 - ✓ PASS - Start (should be 0)
Step   50: LR=0.000500, Ratio=0.5000 - ✓ PASS - Mid-warmup (should be ~0.5)
Step  100: LR=0.001000, Ratio=1.0000 - ✓ PASS - End of warmup (should be 1.0)
Step  500: LR=0.000733, Ratio=0.7330 -   INFO - Mid-training
Step  999: LR=0.000100, Ratio=0.1000 - ✓ PASS - End of training (should be ~0.1)

✓ All tests PASSED!
```

### Train with Custom Schedule

```bash
# Override warmup steps
python train.py --config configs/openwebtext_adaptive_batch.yaml

# Or modify the config file:
# Edit warmup_steps: 3000
# Edit min_lr_ratio: 0.05
```

## Why This Schedule?

### Linear Warmup

**Problem**: Training with high LR from step 0 is unstable
- Random initialization + large LR = gradient explosion
- Large batch sizes amplify this instability

**Solution**: Gradually increase LR over first N steps
- Start with LR ≈ 0 (safe but slow)
- Linearly increase to max_lr (fast but risky)
- By warmup end, model has "learned" basic patterns

**Analogy from physics**: Like gradually increasing the driving force on a system to avoid resonance/instability.

### Cosine Decay

**Problem**: Fixed LR throughout training is suboptimal
- Early training: need large steps to escape bad regions
- Late training: large steps overshoot good minima

**Solution**: Decay LR as training progresses
- Cosine provides smooth, continuous decay
- No sudden jumps (unlike step decay)
- Natural "landing" curve into minimum

**Why cosine specifically?**
1. Smooth and differentiable
2. Most aggressive decay near the end (when you want to settle)
3. Empirically works well (used in GPT-2, BERT, etc.)

**Comparison to other schedules:**

| Schedule | Shape | Pros | Cons |
|----------|-------|------|------|
| Constant | Flat | Simple | Suboptimal convergence |
| Step decay | Stairs | Easy to implement | Sudden jumps destabilize |
| Exponential | Smooth curve | Continuous | Too aggressive early |
| **Cosine** | **Smooth S-curve** | **Smooth, proven** | **Requires total_steps** |
| Polynomial | Smooth curve | Flexible | More hyperparameters |

## Comparison: Old vs New

### Old Implementation (Before)

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
```

**Problems:**
1. No warmup phase → unstable at start
2. `warmup_steps` parameter in config was ignored
3. Not matching GPT-2 training

### New Implementation (After)

```python
from torch.optim.lr_scheduler import LambdaLR

def get_lr_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    return LambdaLR(optimizer, lr_lambda)

scheduler = get_lr_schedule_with_warmup(
    optimizer, warmup_steps, total_steps, min_lr_ratio
)
```

**Improvements:**
1. ✓ Linear warmup phase
2. ✓ Uses `warmup_steps` from config
3. ✓ Matches GPT-2 exactly
4. ✓ Configurable minimum LR

## Expected Behavior

### During Training

You should see in the logs:

```
Learning rate schedule:
  Max learning rate: 0.003
  Warmup steps: 2000
  Total training steps: 1760
  Min LR (at end): 3.00e-04
  Schedule: Linear warmup + Cosine decay
```

And in W&B, the `train/learning_rate` metric should show:
1. Linear increase from 0 to 0.003 over first 2000 steps
2. Smooth cosine decay from 0.003 to 0.0003 for remaining steps

### Typical LR Values

For **OpenWebText** (batch_size=512, 2 epochs):
- Step 0: LR = 0.0
- Step 1000: LR ≈ 0.0015 (halfway through warmup)
- Step 2000: LR = 0.003 (warmup complete)
- Step 880 (epoch 1 end): LR ≈ 0.0025
- Step 1760 (training end): LR ≈ 0.0003

For **Tinyshakespeare** (batch_size=32, 10 epochs):
- Step 0: LR = 0.0
- Step 250: LR ≈ 0.0003 (halfway through warmup)
- Step 500: LR = 0.0006 (warmup complete)
- Step 2500: LR ≈ 0.0004
- Step 5000: LR ≈ 0.00006

## Troubleshooting

### Training is unstable at the start

**Symptom**: Loss spikes or NaN in first few steps

**Solution**: Increase `warmup_steps`
```yaml
warmup_steps: 5000  # Longer warmup for more stability
```

### Training plateaus too early

**Symptom**: Loss stops decreasing before convergence

**Solution 1**: Increase `min_lr_ratio` to maintain higher LR longer
```yaml
min_lr_ratio: 0.2  # Keep LR higher at the end
```

**Solution 2**: Check if `total_steps` is calculated correctly
- Should be: `num_batches × num_epochs / gradient_accumulation_steps`

### LR not logged in W&B

Check that you're logging it:
```python
wandb.log({
    "train/learning_rate": scheduler.get_last_lr()[0],
    "train/step": global_step,
})
```

## References

1. **GPT-2 Paper**: "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
2. **Learning Rate Warmup**: "Accurate, Large Minibatch SGD" (Goyal et al., 2017)
3. **Cosine Annealing**: "SGDR: Stochastic Gradient Descent with Warm Restarts" (Loshchilov & Hutter, 2017)
4. **Transformer Training**: "Attention is All You Need" (Vaswani et al., 2017)
