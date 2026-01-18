# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project trains GPT-2 from randomly initialized weights with a modified loss function that includes **entropy regularization** applied to the first N-2 tokens in a context window. The goal is to encourage sharper output distributions (more confident predictions) by penalizing entropy or diffuse probability distributions.

**Key modification**: The standard cross-entropy loss is augmented with a sharpness penalty term:
```
Loss = CE_loss + λ * sharpness_penalty
```

The sharpness penalty is applied to positions 0:N-2 (excluding the last position) to encourage confident predictions on all but the final token.

## Common Commands

### Training

```bash
# Quick test (20 steps, verifies everything works)
python quick_test.py

# Quick test with config file
python train.py --config configs/quick_test.yaml

# Baseline training (no penalty)
python train.py --config configs/baseline_no_penalty.yaml

# Training with specific penalty type
python train.py --config configs/non_max_sum_strong.yaml
python train.py --config configs/neg_entropy.yaml

# Full OpenWebText training
python train.py --config configs/openwebtext_full.yaml

# Override config parameters from command line
python train.py --config configs/quick_test.yaml --penalty_weight 0.05
python train.py --config configs/neg_entropy.yaml --no_wandb
python train.py --dataset tinyshakespeare --penalty_type non_max_sum
```

### Testing

```bash
# Run quick test script
python quick_test.py

# Run test suite
python -m pytest test_training.py
```

### Configuration

```bash
# Create a new config from existing one
cp configs/quick_test.yaml configs/my_experiment.yaml

# Generate config programmatically
python -c "from config import TrainingConfig; c = TrainingConfig(); c.penalty_weight = 0.05; c.to_yaml('configs/my_exp.yaml')"
```

## Code Architecture

### Core Components

**config.py** - `TrainingConfig` dataclass
- Centralized configuration using dataclass with YAML serialization
- Supports loading from YAML (`from_yaml()`), saving to YAML (`to_yaml()`), and command-line overrides
- Auto-detects device (CUDA/MPS/CPU) in `__post_init__()`
- Key parameters: `penalty_type`, `penalty_weight`, `dataset`, `batch_size`, `learning_rate`, `use_wandb`

**model.py** - Model and loss computation
- `create_gpt2_model()`: Loads GPT-2 architecture from HuggingFace and reinitializes weights randomly
- `compute_sharpness_penalty()`: Computes sharpness penalty on logits for positions 0:N-2
  - Supports 4 penalty types: `non_max_sum`, `neg_entropy`, `neg_max_prob`, `top_k_mass`
  - Takes logits of shape (batch, seq_length, vocab_size) and returns scalar penalty
- `compute_loss_with_penalty()`: Main loss function combining CE loss + λ * penalty
  - Returns total loss and dict of loss components for logging

**data.py** - Dataset loading and preprocessing
- `TextDataset`: PyTorch Dataset that chunks tokenized text into non-overlapping sequences
  - GPT-2 handles next-token prediction internally (labels = input_ids)
  - Pads last chunk with -100 for loss computation
- `load_tinyshakespeare()`: Downloads Shakespeare dataset for quick testing
- `load_openwebtext()`: Loads OpenWebText from HuggingFace (streaming for efficiency)
- `get_dataloaders()`: Factory function returning train and val dataloaders

**train.py** - Main training loop with W&B integration
- Entry point: CLI with argparse supporting config files and overrides
- Training loop features:
  - Gradient accumulation (effective batch size = batch_size * gradient_accumulation_steps)
  - Cosine annealing LR schedule
  - Gradient clipping
  - Regular logging, evaluation, text generation, and checkpointing
- W&B logging includes: total loss, CE loss, penalty, avg max prob, perplexity, learning rate, gradient norms, generated samples

### Key Design Patterns

**Loss Computation Flow**:
1. Forward pass through GPT-2 model → get logits and standard CE loss
2. Extract logits for positions 0:N-2 (exclude last token)
3. Compute softmax probabilities
4. Calculate sharpness penalty based on `penalty_type`
5. Combine: total_loss = ce_loss + penalty_weight * penalty

**Penalty Position Slicing**:
- The penalty is applied to positions 0:N-2 by slicing `logits[:, :-1, :]` before computing probabilities
- This is hardcoded in `compute_sharpness_penalty()` via the `positions_slice=(0, -1)` parameter
- Rationale: Encourage confident predictions throughout the sequence except the final position

**Dataset Chunking**:
- Text is tokenized into a flat list of token IDs
- `TextDataset` creates non-overlapping chunks of `max_seq_length` tokens
- Number of samples = `len(tokens) // max_seq_length`
- Each chunk becomes one training example where the model predicts token[i+1] from token[0:i]

### Sharpness Penalty Types

1. **non_max_sum**: `1 - max_prob` (encourages max probability to be high)
2. **neg_entropy**: Entropy itself (minimize entropy = sharper distribution)
3. **neg_max_prob**: `-max_prob` (maximize max probability)
4. **top_k_mass**: `1 - sum(top_k_probs)` (encourages top-k to capture most mass)

## Important Notes for Development

- **Gradient accumulation**: The loss in train.py:200 is normalized by `gradient_accumulation_steps` before `.backward()`. This is critical for correct gradient scaling.

- **Logging metrics**: Running averages are accumulated per batch but logged per global step. The division at train.py:228 accounts for `log_interval * gradient_accumulation_steps` batches.

- **Device handling**: The code auto-detects CUDA/MPS/CPU. MPS (Apple Silicon) disables `num_workers` and `pin_memory` to avoid compatibility issues.

- **Physics background consideration**: The user is trained as a physicist and may not be familiar with standard ML practices. When discussing modifications, explain ML conventions (e.g., why we normalize by gradient_accumulation_steps, why we use AdamW over SGD, etc.).

- **Position indexing**: Python's `-1` index refers to the last element. The slice `[:-1]` means "all except last", which is how we exclude position N-1 (the last position) from the penalty.

## Datasets

- **tinyshakespeare**: ~1MB text file, downloads from GitHub, good for quick testing
- **openwebtext**: Large dataset from HuggingFace, uses streaming, for full training runs

## Configuration Files

All YAML configs in `configs/` follow the `TrainingConfig` schema:
- `quick_test.yaml`: Small batch, few steps, no W&B
- `baseline_no_penalty.yaml`: Training without any penalty (λ=0)
- `non_max_sum_strong.yaml`: Strong penalty weight
- `neg_entropy.yaml`: Negative entropy penalty variant
- `openwebtext_full.yaml`: Full-scale training setup
