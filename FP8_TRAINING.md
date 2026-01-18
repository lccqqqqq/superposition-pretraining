# FP8 Training Guide for H200

This guide explains how to use FP8 (8-bit floating point) training on NVIDIA H200 GPUs to accelerate training with minimal precision loss.

## What is FP8?

FP8 is a new low-precision format supported by NVIDIA Hopper architecture (H100/H200):

### Precision Comparison

| Format | Bits | Exponent | Mantissa | Range | Precision | H200 Speed |
|--------|------|----------|----------|-------|-----------|------------|
| FP32 | 32 | 8 | 23 | Very large | Highest | 1x (67 TFLOPS) |
| TF32 | 19 | 8 | 10 | Large | High | ~3x (312 TFLOPS) |
| BF16 | 16 | 8 | 7 | Large | Medium | ~4x (989 TFLOPS) |
| **FP8-E4M3** | **8** | **4** | **3** | **Small** | **Low** | **~6x (1979 TFLOPS)** |
| **FP8-E5M2** | **8** | **5** | **2** | **Medium** | **Very Low** | **~6x (1979 TFLOPS)** |

**H200 FP8 Tensor Cores: 1979 TFLOPS** (vs 989 TFLOPS BF16)

### Two FP8 Formats

NVIDIA Hopper supports two FP8 formats:

1. **E4M3** (4-bit exponent, 3-bit mantissa):
   - Higher precision, smaller range
   - Used for **forward pass activations**
   - Range: ±448 (2^8.875)

2. **E5M2** (5-bit exponent, 2-bit mantissa):
   - Lower precision, larger range
   - Used for **gradients** (need larger range)
   - Range: ±57,344 (2^15.5)

## Why FP8 is Faster

**Throughput improvement:**
- **2x more FLOPs** than BF16 (1979 vs 989 TFLOPS)
- **2x less memory bandwidth** (8-bit vs 16-bit)
- **2x more activations in cache** (smaller footprint)

**For GPT-2 on H200:**
- BF16: ~3-4 seconds/step
- FP8: ~1.5-2 seconds/step (**~2x faster**)

## Trade-offs

**Pros:**
- ✓ ~2x faster training
- ✓ 2x less memory for activations
- ✓ Minimal accuracy loss (<1% typically)

**Cons:**
- ✗ Requires H100/H200 (Hopper architecture)
- ✗ More complex setup (Transformer Engine)
- ✗ Need careful monitoring for numerical issues
- ✗ Small risk of gradient underflow

## Implementation: Transformer Engine

NVIDIA's **Transformer Engine** provides the best FP8 support for transformers.

### Installation

```bash
# Install Transformer Engine
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable

# Or from PyPI (may be older)
pip install transformer-engine[pytorch]

# Verify installation
python -c "import transformer_engine.pytorch as te; print('Transformer Engine installed')"
```

### Option 1: Use HuggingFace Accelerate (Easiest)

HuggingFace `accelerate` library has built-in FP8 support via Transformer Engine.

**1. Install accelerate:**
```bash
pip install accelerate>=0.25.0
```

**2. Configure accelerate:**
```bash
accelerate config
```

Answer the prompts:
- Use FP8: **Yes**
- FP8 backend: **TE** (Transformer Engine)
- Mixed precision: **fp8**

**3. Modify training script:**

No code changes needed! Just launch with:
```bash
accelerate launch train.py --config configs/openwebtext_adaptive_batch.yaml
```

### Option 2: Direct Integration (More Control)

For more control, integrate Transformer Engine directly into the training code.

**Benefits:**
- Fine-grained control over FP8 behavior
- Better monitoring
- Can mix FP8 with other optimizations

See `train_fp8.py` for full implementation.

## Configuration

Add these parameters to your config:

```yaml
# FP8 Training Settings
use_fp8: true                    # Enable FP8 training
fp8_format: "hybrid"             # "hybrid" (E4M3+E5M2), "e4m3", "e5m2"
fp8_margin: 0                    # Safety margin for scaling (0-2)
fp8_interval: 1                  # Recalculate scaling factors every N steps
fp8_amax_history_len: 1024       # History length for amax tracking
fp8_amax_compute_algo: "max"     # "max" or "most_recent"
```

### Parameter Explanations

**`use_fp8`**: Enable/disable FP8 training
- `false`: Use BF16/FP16 (default)
- `true`: Use FP8 (requires H100/H200)

**`fp8_format`**:
- `"hybrid"`: E4M3 for forward, E5M2 for backward (recommended)
- `"e4m3"`: E4M3 only (higher precision, but gradient issues)
- `"e5m2"`: E5M2 only (lower precision)

**`fp8_margin`**: Safety margin for scaling
- `0`: No margin (maximum performance, slight risk)
- `1`: Small margin (recommended for stable training)
- `2`: Large margin (very safe, slightly slower)

**`fp8_interval`**: How often to update scaling factors
- `1`: Every step (most accurate, slight overhead)
- `10`: Every 10 steps (faster, less accurate)

## Usage Examples

### Quick Test (Tinyshakespeare + FP8)

```bash
# Create FP8 test config
python -c "
from config import TrainingConfig
c = TrainingConfig()
c.use_fp8 = True
c.fp8_format = 'hybrid'
c.dataset = 'tinyshakespeare'
c.num_epochs = 1
c.use_wandb = False
c.to_yaml('configs/test_fp8.yaml')
"

# Run with FP8
python train_fp8.py --config configs/test_fp8.yaml
```

### OpenWebText with FP8

```bash
# Edit configs/openwebtext_adaptive_batch.yaml
# Add: use_fp8: true

python train_fp8.py --config configs/openwebtext_adaptive_batch.yaml
```

### Compare FP8 vs BF16

Run two experiments to compare:

```bash
# BF16 baseline
python train.py --config configs/openwebtext_adaptive_batch.yaml \
    --wandb_name "openwebtext_bf16"

# FP8 accelerated
python train_fp8.py --config configs/openwebtext_adaptive_batch.yaml \
    --wandb_name "openwebtext_fp8"
```

Compare in W&B:
- Training speed (seconds/step)
- Final loss (should be within 1-2%)
- Memory usage

## Monitoring FP8 Training

**Critical metrics to watch:**

### 1. Gradient Scales

FP8 uses dynamic scaling to prevent underflow/overflow. Monitor:

```python
# In train_fp8.py, log these:
wandb.log({
    "fp8/fwd_scale": fwd_scale,  # Forward pass scale
    "fp8/bwd_scale": bwd_scale,  # Backward pass scale
    "fp8/amax_fwd": amax_fwd,    # Max activation magnitude
    "fp8/amax_bwd": amax_bwd,    # Max gradient magnitude
})
```

**Normal behavior:**
- Scales should stabilize after warmup
- Occasional jumps are OK
- Gradual drift is fine

**Warning signs:**
- Scales constantly changing
- Scales going to infinity or zero
- NaN in loss

### 2. Loss Comparison

Compare FP8 vs BF16 loss curves:

```python
# Expected:
# - FP8 loss slightly higher (~1-5%)
# - Similar convergence rate
# - No divergence
```

### 3. Gradient Norms

```python
wandb.log({
    "train/grad_norm": grad_norm,
    "train/grad_norm_clip_ratio": grad_norm / config.max_grad_norm,
})
```

**Normal:** Grad norm similar to BF16 training
**Problem:** Grad norms much larger or zero

### 4. Numerical Issues

Watch for:
- **NaN loss**: Reduce `fp8_margin`, increase `max_grad_norm`
- **Loss spikes**: Increase `fp8_margin` to 1 or 2
- **Slow convergence**: Check if scales are stable

## Troubleshooting

### Issue: NaN Loss

**Cause:** Gradient overflow in FP8

**Solutions:**
1. Increase safety margin:
   ```yaml
   fp8_margin: 2  # More conservative
   ```

2. Increase gradient clipping:
   ```yaml
   max_grad_norm: 0.5  # Stricter clipping
   ```

3. Use E5M2 for everything:
   ```yaml
   fp8_format: "e5m2"  # Larger range
   ```

### Issue: Training Slower than Expected

**Cause:** Scaling overhead or memory bottleneck

**Solutions:**
1. Reduce scaling frequency:
   ```yaml
   fp8_interval: 10  # Update scales every 10 steps
   ```

2. Increase batch size (more work per scaling):
   ```yaml
   max_batch_size: 128  # Bigger batches on H200
   ```

3. Check if you're memory-bound:
   ```bash
   nvidia-smi dmon -s u
   ```

### Issue: Loss Higher than BF16

**Cause:** Normal precision loss, or insufficient margin

**Solutions:**
1. Accept 1-5% higher loss (normal for FP8)

2. Increase margin slightly:
   ```yaml
   fp8_margin: 1
   ```

3. Use hybrid format:
   ```yaml
   fp8_format: "hybrid"  # Best balance
   ```

### Issue: "Transformer Engine not found"

**Solutions:**
```bash
# Reinstall
pip uninstall transformer-engine
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable

# Check CUDA version matches
python -c "import torch; print(torch.version.cuda)"
# Should be 12.x for H200
```

## Performance Benchmarks

Expected performance on **H200 (141GB)**:

### GPT-2 Small (124M params)

| Config | Precision | Batch Size | Time/Step | Throughput | Memory |
|--------|-----------|------------|-----------|------------|--------|
| Baseline | BF16 | 32 | 3.5s | 9.1 samples/s | 45GB |
| Optimized | BF16 | 64 | 2.8s | 22.9 samples/s | 68GB |
| **FP8** | **FP8** | **128** | **1.5s** | **85.3 samples/s** | **85GB** |

**Speedup: ~2.3x vs BF16 baseline, ~1.9x vs optimized BF16**

### OpenWebText (effective batch = 512)

| Precision | Per-GPU Batch | Grad Accum | Time/Step | Est. Time (2 epochs) |
|-----------|---------------|------------|-----------|----------------------|
| BF16 | 64 | 8 | 3.2s | ~2.8 hours |
| **FP8** | **128** | **4** | **1.7s** | **~1.5 hours** |

**Speedup: ~1.9x faster training**

## Best Practices

### 1. Start Conservative

```yaml
# First run: safe settings
use_fp8: true
fp8_format: "hybrid"
fp8_margin: 1
fp8_interval: 1
```

Monitor for issues, then optimize.

### 2. Gradually Optimize

Once stable, try:
```yaml
fp8_margin: 0          # Remove safety margin
fp8_interval: 10       # Less frequent updates
max_batch_size: 128    # Bigger batches
```

### 3. Always Compare

Run parallel experiments:
- FP8 vs BF16 on same config
- Monitor loss gap (<5% is acceptable)
- Check final metrics match

### 4. Use with Other Optimizations

FP8 combines well with:
- ✓ Gradient checkpointing
- ✓ Flash Attention
- ✓ Fused optimizers
- ✓ Compile mode (`torch.compile`)

Example:
```yaml
use_fp8: true
use_flash_attention: true  # 2x faster attention
torch_compile: true        # JIT optimization
```

**Combined speedup: ~3-4x over baseline!**

## When to Use FP8

**Use FP8 when:**
- ✓ You have H100/H200 GPU
- ✓ Training speed is critical
- ✓ 1-5% accuracy loss is acceptable
- ✓ You can monitor training closely

**Don't use FP8 when:**
- ✗ You don't have Hopper GPU (H100/H200)
- ✗ You need exact reproducibility
- ✗ Your model is numerically unstable
- ✗ You can't monitor for issues

## Physics Analogy

Think of precision like measurement resolution:

- **FP32**: Measuring with a micrometer (overkill for most tasks)
- **BF16**: Measuring with a caliper (good balance)
- **FP8**: Measuring with a ruler (faster but less precise)

For neural networks, gradient descent is robust to small errors, so FP8's lower precision usually doesn't hurt final results (just like you don't need micrometer precision to build most things).

The key is **dynamic scaling** - like using different units (mm vs cm vs m) depending on what you're measuring. FP8 automatically adjusts the scale to keep important values representable.

## References

1. **FP8 Formats for Deep Learning**: [NVIDIA White Paper](https://arxiv.org/abs/2209.05433)
2. **Transformer Engine**: [GitHub](https://github.com/NVIDIA/TransformerEngine)
3. **H200 Specs**: [NVIDIA Data Sheet](https://www.nvidia.com/en-us/data-center/h200/)
4. **FP8 Training**: [NVIDIA Blog](https://developer.nvidia.com/blog/nvidia-arm-and-intel-publish-fp8-specification-for-standardization-as-an-interchange-format-for-ai/)
