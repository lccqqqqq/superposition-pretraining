# GPT2 Training with Entropy Regularization

This project reproduces GPT2 training from randomly initialized weights with a modified loss function that includes entropy regularization to encourage sharper output distributions.

## Features

- **GPT2-small architecture** loaded from HuggingFace with random initialization
- **Configurable sharpness penalties** to encourage definite token predictions:
  - `non_max_sum`: Sum of non-max probabilities (1 - max_prob)
  - `neg_entropy`: Negative entropy to minimize entropy
  - `neg_max_prob`: Negative max probability
  - `top_k_mass`: 1 - sum of top-k probabilities
- **Dataset support**:
  - TinyShakespeare for quick testing
  - OpenWebText for full training
- **Comprehensive W&B integration** for tracking metrics and visualizations

## Installation

1. Activate your virtual environment:
```bash
source ../.venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start with YAML Configs (Recommended)

Use pre-made configuration files for easy experimentation:

```bash
# Quick test (small batch, no W&B)
python train.py --config configs/quick_test.yaml

# Baseline training without penalty
python train.py --config configs/baseline_no_penalty.yaml

# Strong penalty experiment
python train.py --config configs/non_max_sum_strong.yaml

# Full OpenWebText training
python train.py --config configs/openwebtext_full.yaml
```

Override specific parameters from command line:
```bash
python train.py --config configs/quick_test.yaml --penalty_weight 0.05
python train.py --config configs/neg_entropy.yaml --no_wandb
python train.py --dataset tinyshakespeare --penalty_type non_max_sum
```

### Create Your Own Config

1. Copy an existing config:
```bash
cp config.yaml my_experiment.yaml
```

2. Edit the YAML file with your parameters
3. Run training:
```bash
python train.py --config my_experiment.yaml
```

Or generate from Python:
```python
from config import TrainingConfig

config = TrainingConfig()
config.penalty_weight = 0.05
config.dataset = "openwebtext"
config.to_yaml("configs/my_experiment.yaml")
```

### Programmatic Usage

```python
from config import TrainingConfig
from train import train

# From YAML
config = TrainingConfig.from_yaml("configs/quick_test.yaml")

# Or create directly
config = TrainingConfig()
config.dataset = "tinyshakespeare"
config.penalty_type = "non_max_sum"
config.penalty_weight = 0.01

train(config)
```

### Configuration Options

See `config.yaml` or `config.py` for all available options. Key settings:

- `penalty_type`: Type of sharpness penalty ("non_max_sum", "neg_entropy", "neg_max_prob", "top_k_mass")
- `penalty_weight`: Weight λ for the penalty term (default: 0.01)
- `dataset`: "tinyshakespeare" or "openwebtext"
- `batch_size`: Training batch size
- `learning_rate`: AdamW learning rate
- `use_wandb`: Enable W&B logging

### Command-Line Arguments

```bash
python train.py --help
```

Available arguments:
- `--config`: Path to YAML config file
- `--penalty_weight`: Override penalty weight
- `--penalty_type`: Override penalty type
- `--dataset`: Override dataset
- `--no_wandb`: Disable W&B logging

## Project Structure

```
.
├── config.py              # Training configuration dataclass
├── config.yaml            # Default YAML configuration
├── model.py               # GPT2 model and loss functions
├── data.py                # Dataset loading and preprocessing
├── train.py               # Main training loop with CLI
├── quick_test.py          # Quick 20-step test script
├── scratch.py             # Interactive experimentation notebook
├── requirements.txt       # Python dependencies
└── configs/               # Example configuration files
    ├── quick_test.yaml
    ├── baseline_no_penalty.yaml
    ├── non_max_sum_strong.yaml
    ├── neg_entropy.yaml
    └── openwebtext_full.yaml
```

## W&B Metrics

The training script logs:
- Total loss, CE loss, and penalty term (separately)
- Average max probability across positions
- Perplexity
- Learning rate and gradient norms
- Generated text samples
- Validation metrics

## Loss Function

The total loss combines cross-entropy and sharpness penalty:

```
Loss = CE_loss + λ * sharpness_penalty
```

The sharpness penalty is applied to positions 0:N-2 (excluding the last position) to encourage the model to make confident predictions.
