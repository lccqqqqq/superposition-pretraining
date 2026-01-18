from dataclasses import dataclass, asdict
from typing import Literal
import yaml
from pathlib import Path
import torch


def detect_optimal_batch_size(
    model_name: str = "gpt2",
    sequence_length: int = 1024,
    target_memory_util: float = 0.85,
    max_batch_size: int = 32,
    device_index: int = 0
) -> int:
    """
    Detect optimal batch size based on available GPU memory.

    This function estimates the maximum batch size that can fit in GPU memory
    based on model size, sequence length, and available memory.

    Args:
        model_name: HuggingFace model name (used to estimate model size)
        sequence_length: Maximum sequence length for training
        target_memory_util: Target GPU memory utilization (0.0-1.0)
        max_batch_size: Maximum allowed batch size (safety limit)
        device_index: GPU device index

    Returns:
        Optimal batch size (integer >= 1)

    Note:
        This is a heuristic estimate. The actual optimal batch size may vary
        depending on the specific model architecture and training configuration.
    """
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, cannot detect optimal batch size")
        return 1

    # Get GPU properties
    props = torch.cuda.get_device_properties(device_index)
    total_memory_gb = props.total_memory / (1024**3)  # Convert to GB
    available_memory_gb = total_memory_gb * target_memory_util

    print(f"GPU {device_index}: {props.name}")
    print(f"Total memory: {total_memory_gb:.2f} GB")
    print(f"Target memory utilization: {target_memory_util*100:.0f}%")
    print(f"Available memory for training: {available_memory_gb:.2f} GB")

    # Estimate model size based on model name
    # GPT2 variants: gpt2 (124M), gpt2-medium (355M), gpt2-large (774M), gpt2-xl (1.5B)
    model_size_params = {
        "gpt2": 124_000_000,
        "gpt2-medium": 355_000_000,
        "gpt2-large": 774_000_000,
        "gpt2-xl": 1_500_000_000,
    }
    num_params = model_size_params.get(model_name, 124_000_000)

    # Estimate memory per sample (in GB)
    # Formula: (sequence_length * hidden_size * bytes_per_param * memory_multiplier)
    # - Model parameters: num_params * 4 bytes (fp32) or 2 bytes (fp16)
    # - Activations: sequence_length * hidden_size * batch_size * 4 bytes
    # - Gradients: same as parameters
    # - Optimizer states (AdamW): 2x parameters for momentum and variance
    # - Memory multiplier accounts for activations and temporary buffers (~3-4x)

    # Hidden size for GPT2 models
    hidden_sizes = {
        "gpt2": 768,
        "gpt2-medium": 1024,
        "gpt2-large": 1280,
        "gpt2-xl": 1600,
    }
    hidden_size = hidden_sizes.get(model_name, 768)

    # Model memory (parameters + gradients + optimizer states)
    # Parameters: num_params * 4 bytes (fp32)
    # Gradients: num_params * 4 bytes
    # Optimizer states: num_params * 8 bytes (2 states for AdamW)
    model_memory_gb = (num_params * 4 * 3) / (1024**3)  # 4 bytes * 3 (params + grads + 2 optimizer states simplified)

    # Activation memory per sample (estimated)
    # Rough estimate: sequence_length * hidden_size * num_layers * 4 bytes * activation_multiplier
    num_layers = {
        "gpt2": 12,
        "gpt2-medium": 24,
        "gpt2-large": 36,
        "gpt2-xl": 48,
    }.get(model_name, 12)

    # Activation memory includes attention scores, intermediate activations, etc.
    # Multiplier of ~8-10 accounts for all intermediate activations in transformer
    activation_multiplier = 10
    memory_per_sample_gb = (sequence_length * hidden_size * num_layers * 4 * activation_multiplier) / (1024**3)

    print(f"Estimated model memory: {model_memory_gb:.2f} GB")
    print(f"Estimated memory per sample: {memory_per_sample_gb*1024:.1f} MB")

    # Calculate max samples that can fit
    memory_for_samples = available_memory_gb - model_memory_gb
    if memory_for_samples <= 0:
        print(f"Warning: Insufficient GPU memory. Model alone requires {model_memory_gb:.2f} GB")
        return 1

    max_samples = int(memory_for_samples / memory_per_sample_gb)

    # Apply safety margin and constraints
    optimal_batch_size = max(1, min(max_samples, max_batch_size))

    print(f"Detected optimal batch size: {optimal_batch_size}")

    return optimal_batch_size


@dataclass
class TrainingConfig:
    # Model settings
    model_name: str = "gpt2"  # HuggingFace model identifier for GPT2-small
    reinitialize_weights: bool = True  # If False, load pretrained weights

    # Dataset settings
    dataset: Literal["tinyshakespeare", "openwebtext", "fineweb-edu"] = "tinyshakespeare"
    max_seq_length: int = 1024
    train_val_split: float = 0.9  # Train/val split ratio (for tinyshakespeare)
    openwebtext_val_samples: int = 1000  # Number of validation samples (for openwebtext and fineweb-edu)

    # Training hyperparameters
    batch_size: int = 8
    gradient_accumulation_steps: int = 4  # Effective batch size = 32
    num_epochs: int = 10
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    min_lr_ratio: float = 0.1  # Minimum LR as ratio of max LR (for cosine decay)

    # Adaptive batch size settings
    auto_batch_size: bool = False  # Enable automatic batch size detection
    target_effective_batch_size: int = 32  # Desired effective batch size (batch_size * grad_accum)
    max_batch_size: int = 32  # Maximum batch_size per GPU (safety limit)
    target_memory_utilization: float = 0.85  # Target GPU memory usage (0.0-1.0)

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

    # Text generation settings
    generate_max_length: int = 100  # Maximum length for generated text samples
    generate_temperature: float = 1.0  # Sampling temperature (higher = more random)
    generate_top_p: float = 0.9  # Nucleus sampling threshold

    # System settings
    device: str = "cuda"  # Will auto-detect
    seed: int = 42
    num_workers: int = 4

    # FP8 training settings (requires H100/H200 GPU)
    use_fp8: bool = False  # Enable FP8 training with Transformer Engine
    fp8_format: str = "hybrid"  # "hybrid" (E4M3+E5M2), "e4m3", or "e5m2"
    fp8_margin: int = 0  # Safety margin for FP8 scaling (0-2, higher = safer)
    fp8_interval: int = 1  # Recalculate FP8 scales every N steps
    fp8_amax_history_len: int = 1024  # History length for amax tracking
    fp8_amax_compute_algo: str = "max"  # "max" or "most_recent"

    # Output settings
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"

    def __post_init__(self):
        # Auto-detect device
        if self.device == "cuda" and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        # Disable multiprocessing on MPS (macOS) to avoid issues
        if self.device == "mps":
            self.num_workers = 0

        # Auto-detect optimal batch size if enabled
        if self.auto_batch_size and self.device == "cuda":
            print("\n" + "="*60)
            print("Auto-detecting optimal batch size...")
            print("="*60)

            detected_batch_size = detect_optimal_batch_size(
                model_name=self.model_name,
                sequence_length=self.max_seq_length,
                target_memory_util=self.target_memory_utilization,
                max_batch_size=self.max_batch_size,
                device_index=0
            )

            # Calculate gradient accumulation steps to maintain target effective batch size
            self.batch_size = detected_batch_size
            self.gradient_accumulation_steps = max(1, self.target_effective_batch_size // self.batch_size)

            # Calculate actual effective batch size
            actual_effective_batch = self.batch_size * self.gradient_accumulation_steps

            print("="*60)
            print(f"Batch configuration:")
            print(f"  Batch size per GPU: {self.batch_size}")
            print(f"  Gradient accumulation steps: {self.gradient_accumulation_steps}")
            print(f"  Effective batch size: {actual_effective_batch}")
            print(f"  Target effective batch size: {self.target_effective_batch_size}")
            if actual_effective_batch < self.target_effective_batch_size:
                print(f"  Note: Actual effective batch size is lower than target due to GPU memory constraints")
            print("="*60 + "\n")

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
