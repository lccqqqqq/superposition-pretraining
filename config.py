from dataclasses import dataclass, asdict
from typing import Literal
import yaml
from pathlib import Path

@dataclass
class TrainingConfig:
    # Model settings
    model_name: str = "gpt2"  # HuggingFace model identifier for GPT2-small

    # Dataset settings
    dataset: Literal["tinyshakespeare", "openwebtext"] = "tinyshakespeare"
    max_seq_length: int = 1024

    # Training hyperparameters
    batch_size: int = 8
    gradient_accumulation_steps: int = 4  # Effective batch size = 32
    num_epochs: int = 10
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_grad_norm: float = 1.0

    # Sharpness penalty settings
    penalty_type: Literal["non_max_sum", "neg_entropy", "neg_max_prob", "top_k_mass"] = "non_max_sum"
    penalty_weight: float = 0.01  # λ in loss = CE + λ * penalty
    top_k: int = 5  # Only used if penalty_type == "top_k_mass"
    apply_penalty_positions: str = "0:N-2"  # Positions to apply penalty (all except last)

    # Optimization settings
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8

    # Logging and checkpointing
    use_wandb: bool = True
    wandb_project: str = "gpt2-entropy-regularization"
    wandb_entity: str = None  # Set to your wandb username/team
    log_interval: int = 10  # Log every N steps
    eval_interval: int = 500  # Evaluate every N steps
    save_interval: int = 1000  # Save checkpoint every N steps
    generate_interval: int = 500  # Generate samples every N steps
    num_generate_samples: int = 3

    # System settings
    device: str = "cuda"  # Will auto-detect
    seed: int = 42
    num_workers: int = 4

    # Output settings
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"

    def __post_init__(self):
        # Auto-detect device
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        # Disable multiprocessing on MPS (macOS) to avoid issues
        if self.device == "mps":
            self.num_workers = 0

        # Adjust batch size for small dataset
        if self.dataset == "tinyshakespeare":
            self.eval_interval = 100
            self.save_interval = 500
            self.generate_interval = 100

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TrainingConfig':
        """
        Load configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file

        Returns:
            TrainingConfig instance with loaded parameters
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Convert betas from list to tuple if needed
        if 'betas' in config_dict and isinstance(config_dict['betas'], list):
            config_dict['betas'] = tuple(config_dict['betas'])

        return cls(**config_dict)

    def to_yaml(self, yaml_path: str):
        """
        Save configuration to a YAML file.

        Args:
            yaml_path: Path where to save the YAML configuration
        """
        config_dict = asdict(self)

        # Convert tuples to lists for YAML serialization
        if 'betas' in config_dict and isinstance(config_dict['betas'], tuple):
            config_dict['betas'] = list(config_dict['betas'])

        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        print(f"Configuration saved to {yaml_path}")

    def update_from_yaml(self, yaml_path: str):
        """
        Update current configuration with values from a YAML file.
        Only overwrites values that are present in the YAML file.

        Args:
            yaml_path: Path to the YAML configuration file
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Convert betas from list to tuple if needed
        if 'betas' in config_dict and isinstance(config_dict['betas'], list):
            config_dict['betas'] = tuple(config_dict['betas'])

        # Update attributes
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown config key '{key}' in {yaml_path}")

        # Re-run post-init logic
        self.__post_init__()
