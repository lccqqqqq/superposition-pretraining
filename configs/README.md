# Configuration Files

This directory contains example YAML configuration files for different training experiments.

## Available Configurations

### Quick Testing
- **quick_test.yaml**: Minimal config for fast testing and debugging
  - Small batch size, short sequences
  - Frequent logging, no W&B
  - Use: `python train.py --config configs/quick_test.yaml`

### Baseline
- **baseline_no_penalty.yaml**: Standard GPT2 training without sharpness penalty
  - Penalty weight = 0.0
  - Good for comparing against penalty experiments

### Penalty Type Experiments
- **non_max_sum_strong.yaml**: Strong non-max-sum penalty (λ=0.1)
  - Tests aggressive sharpness enforcement

- **neg_entropy.yaml**: Negative entropy penalty
  - Alternative approach to encourage sharp distributions

### Production Training
- **openwebtext_full.yaml**: Full-scale training on OpenWebText
  - Larger effective batch size (64)
  - More warmup steps
  - Appropriate logging intervals for large dataset

## Usage

### Use a config file
```bash
python train.py --config configs/quick_test.yaml
```

### Override specific parameters
You can modify config files directly or create your own:
```bash
cp configs/quick_test.yaml configs/my_experiment.yaml
# Edit my_experiment.yaml
python train.py --config configs/my_experiment.yaml
```

### Generate a config from code
```python
from config import TrainingConfig

config = TrainingConfig()
config.penalty_weight = 0.05
config.to_yaml("configs/my_custom_config.yaml")
```

## Creating Your Own Configs

1. Copy the default `config.yaml` or any example config
2. Modify the parameters you want to change
3. Run training with your config

You only need to specify parameters that differ from defaults. Partial configs are supported via `update_from_yaml()`.
