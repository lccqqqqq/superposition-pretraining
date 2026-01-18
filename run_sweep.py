#!/usr/bin/env python
"""
Launch W&B sweep for lambda (penalty_weight) grid search.

This script provides a programmatic way to launch W&B sweeps for hyperparameter
tuning. It's an alternative to the command-line `wandb sweep` approach.

Usage:
    # Launch sweep from YAML config
    python run_sweep.py --config sweep_lambda.yaml

    # Launch sweep and run N experiments
    python run_sweep.py --config sweep_lambda.yaml --count 5

    # Launch sweep with custom project name
    python run_sweep.py --config sweep_lambda.yaml --project my-project --count 5

For command-line approach (simpler):
    wandb sweep sweep_lambda.yaml
    wandb agent <sweep_id>
"""

import os
import argparse
import yaml
import wandb
from pathlib import Path

from train import train
from config import TrainingConfig


def sweep_train():
    """
    Training function called by W&B sweep agent.

    This function is executed for each run in the sweep. W&B automatically
    injects sweep parameters into wandb.config.
    """
    # Initialize wandb run (sweep agent handles this)
    with wandb.init() as run:
        # Get sweep config from W&B
        sweep_config = wandb.config

        # Load base config from file or use default
        if hasattr(sweep_config, 'config') and sweep_config.config:
            print(f"Loading base configuration from: {sweep_config.config}")
            config = TrainingConfig.from_yaml(sweep_config.config)
        else:
            print("Using default configuration")
            config = TrainingConfig()

        # Override with sweep parameters
        print("\nApplying sweep parameters:")

        if hasattr(sweep_config, 'penalty_weight'):
            config.penalty_weight = sweep_config.penalty_weight
            print(f"  penalty_weight = {config.penalty_weight}")

        if hasattr(sweep_config, 'dataset'):
            config.dataset = sweep_config.dataset
            print(f"  dataset = {config.dataset}")

        if hasattr(sweep_config, 'penalty_type'):
            config.penalty_type = sweep_config.penalty_type
            print(f"  penalty_type = {config.penalty_type}")

        if hasattr(sweep_config, 'num_epochs'):
            config.num_epochs = sweep_config.num_epochs
            print(f"  num_epochs = {config.num_epochs}")

        if hasattr(sweep_config, 'learning_rate'):
            config.learning_rate = sweep_config.learning_rate
            print(f"  learning_rate = {config.learning_rate}")

        if hasattr(sweep_config, 'batch_size'):
            config.batch_size = sweep_config.batch_size
            print(f"  batch_size = {config.batch_size}")

        if hasattr(sweep_config, 'auto_batch_size'):
            config.auto_batch_size = sweep_config.auto_batch_size
            print(f"  auto_batch_size = {config.auto_batch_size}")

        if hasattr(sweep_config, 'target_effective_batch_size'):
            config.target_effective_batch_size = sweep_config.target_effective_batch_size
            print(f"  target_effective_batch_size = {config.target_effective_batch_size}")

        print()

        # Re-run post-init to apply auto-adjustments (e.g., auto batch size)
        config.__post_init__()

        # W&B is already initialized by the sweep agent, so disable re-initialization
        config.use_wandb = True

        # Run training
        # Note: Don't call wandb.init() again since sweep agent already initialized it
        train(config)


def main():
    parser = argparse.ArgumentParser(
        description="Launch W&B sweep for GPT2 entropy regularization experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="sweep_lambda.yaml",
        help="Path to sweep configuration YAML file"
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="W&B project name (overrides config file)"
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="W&B entity/team name"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of runs to execute (default: run all sweep configs)"
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Only initialize sweep and print sweep ID (don't run agent)"
    )

    args = parser.parse_args()

    # Load sweep configuration
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Sweep config file not found: {args.config}")

    print(f"Loading sweep configuration from: {args.config}")
    with open(args.config, 'r') as f:
        sweep_config = yaml.safe_load(f)

    # Override project/entity if specified
    if args.project:
        print(f"Overriding project: {args.project}")

    if args.entity:
        print(f"Overriding entity: {args.entity}")

    # Initialize sweep
    print("\nInitializing W&B sweep...")
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=args.project or "gpt2-entropy-regularization",
        entity=args.entity
    )

    print(f"\n{'='*60}")
    print(f"Sweep initialized successfully!")
    print(f"Sweep ID: {sweep_id}")
    print(f"{'='*60}\n")

    if args.init_only:
        print("Initialization only mode. Run the following command to start the agent:")
        print(f"\n  wandb agent {sweep_id}\n")
        return

    # Run sweep agent
    print("Starting sweep agent...")
    print(f"Will run {args.count if args.count else 'all'} experiment(s)\n")

    try:
        wandb.agent(
            sweep_id,
            function=sweep_train,
            count=args.count
        )
        print("\n" + "="*60)
        print("Sweep completed successfully!")
        print("="*60)
    except KeyboardInterrupt:
        print("\n\nSweep interrupted by user")
        print(f"You can resume by running: wandb agent {sweep_id}")
    except Exception as e:
        print(f"\n\nError during sweep: {e}")
        print(f"You can retry by running: wandb agent {sweep_id}")
        raise


if __name__ == "__main__":
    main()
