"""
FP8 Training Script using HuggingFace Accelerate

This script enables FP8 training on H100/H200 GPUs using the Accelerate library,
which provides automatic integration with NVIDIA Transformer Engine.

Usage:
    # Setup accelerate first (one-time)
    accelerate config

    # Run training with FP8
    accelerate launch train_fp8.py --config configs/openwebtext_fp8.yaml

    # Or run directly (will use default accelerate config)
    python train_fp8.py --config configs/openwebtext_fp8.yaml

Requirements:
    pip install accelerate>=0.25.0
    pip install transformer-engine[pytorch]
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math
from transformers import GPT2TokenizerFast
from tqdm import tqdm
import random
import numpy as np
from pathlib import Path

# Try to import accelerate
try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed as accelerate_set_seed
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    print("Warning: accelerate not installed. FP8 training requires accelerate.")
    print("Install with: pip install accelerate>=0.25.0")

# Import from existing modules
from config import TrainingConfig
from model import create_gpt2_model, compute_loss_with_penalty
from data import get_dataloaders

# Import W&B if available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def get_lr_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.0):
    """Learning rate scheduler with linear warmup and cosine decay."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    return LambdaLR(optimizer, lr_lambda)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_with_accelerate(config: TrainingConfig):
    """Main training function using Accelerate for FP8."""

    if not ACCELERATE_AVAILABLE:
        raise ImportError(
            "accelerate is required for FP8 training. "
            "Install with: pip install accelerate>=0.25.0"
        )

    # Initialize Accelerator with FP8 settings
    accelerator = Accelerator(
        mixed_precision="fp8" if config.use_fp8 else "bf16",
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb" if config.use_wandb and WANDB_AVAILABLE else None,
        project_dir=config.output_dir,
    )

    # Set seed
    accelerate_set_seed(config.seed, device_specific=True)

    # Only print on main process
    if accelerator.is_main_process:
        print("=" * 60)
        print("FP8 Training with HuggingFace Accelerate")
        print("=" * 60)
        print(f"Mixed precision: {accelerator.mixed_precision}")
        print(f"Device: {accelerator.device}")
        print(f"Num processes: {accelerator.num_processes}")
        print(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")

        if config.use_fp8:
            print("\n🚀 FP8 Training Enabled!")
            print(f"  Format: {config.fp8_format}")
            print(f"  Margin: {config.fp8_margin}")
            print(f"  Update interval: {config.fp8_interval}")
        else:
            print("\n⚡ Using BF16 mixed precision")
        print("=" * 60)

    # Create output directories
    if accelerator.is_main_process:
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Initialize W&B
    if accelerator.is_main_process and config.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            config=vars(config),
            name=f"{config.dataset}_{config.penalty_type}_lambda{config.penalty_weight}_fp8" if config.use_fp8 else f"{config.dataset}_{config.penalty_type}_lambda{config.penalty_weight}",
        )

    # Load tokenizer
    if accelerator.is_main_process:
        print("\nLoading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Create model
    if accelerator.is_main_process:
        print("Creating model...")
    model = create_gpt2_model(config.model_name, reinitialize=config.reinitialize_weights)

    # Log model info
    if accelerator.is_main_process:
        num_params = count_parameters(model)
        print(f"Number of trainable parameters: {num_params:,}")

        effective_batch_size = config.batch_size * config.gradient_accumulation_steps * accelerator.num_processes
        print(f"\nBatch configuration:")
        print(f"  Batch size per GPU: {config.batch_size}")
        print(f"  Gradient accumulation steps: {config.gradient_accumulation_steps}")
        print(f"  Num GPUs: {accelerator.num_processes}")
        print(f"  Effective batch size: {effective_batch_size}")

    # Load data
    if accelerator.is_main_process:
        print("\nLoading data...")
    train_loader, val_loader = get_dataloaders(
        dataset_name=config.dataset,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=config.device,
        train_val_split=config.train_val_split,
        openwebtext_val_samples=config.openwebtext_val_samples,
    )

    if accelerator.is_main_process:
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )

    # Setup scheduler
    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    scheduler = get_lr_schedule_with_warmup(
        optimizer=optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=config.min_lr_ratio
    )

    if accelerator.is_main_process:
        print(f"\nLearning rate schedule:")
        print(f"  Max learning rate: {config.learning_rate}")
        print(f"  Warmup steps: {config.warmup_steps}")
        print(f"  Total training steps: {total_steps}")
        print(f"  Min LR (at end): {config.learning_rate * config.min_lr_ratio:.2e}")

    # Prepare everything with accelerator
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # Training loop
    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60)

    global_step = 0

    for epoch in range(config.num_epochs):
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        model.train()
        running_loss = 0
        running_ce_loss = 0
        running_penalty = 0
        running_max_prob = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", disable=not accelerator.is_main_process)

        for batch_idx, batch in enumerate(pbar):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                labels = batch["labels"]

                # Forward pass
                loss, loss_dict = compute_loss_with_penalty(
                    model=accelerator.unwrap_model(model),
                    input_ids=input_ids,
                    labels=labels,
                    penalty_weight=config.penalty_weight,
                    penalty_type=config.penalty_type,
                    top_k=config.top_k,
                )

                # Backward pass (accelerator handles gradient accumulation)
                accelerator.backward(loss)

                # Update running metrics
                running_loss += loss_dict["total_loss"]
                running_ce_loss += loss_dict["ce_loss"]
                running_penalty += loss_dict["penalty"]
                running_max_prob += loss_dict["avg_max_prob"]

                # Optimizer step (only when accumulated)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # Logging
                    if global_step % config.log_interval == 0:
                        num_batches = config.log_interval * config.gradient_accumulation_steps
                        avg_loss = running_loss / num_batches
                        avg_ce_loss = running_ce_loss / num_batches
                        avg_penalty = running_penalty / num_batches
                        avg_max_prob = running_max_prob / num_batches

                        if accelerator.is_main_process:
                            pbar.set_postfix({
                                "loss": f"{avg_loss:.4f}",
                                "ce": f"{avg_ce_loss:.4f}",
                                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                            })

                            if config.use_wandb and WANDB_AVAILABLE:
                                log_dict = {
                                    "train/loss": avg_loss,
                                    "train/ce_loss": avg_ce_loss,
                                    "train/penalty": avg_penalty,
                                    "train/avg_max_prob": avg_max_prob,
                                    "train/learning_rate": scheduler.get_last_lr()[0],
                                    "train/epoch": epoch,
                                    "train/step": global_step,
                                }
                                wandb.log(log_dict)

                        running_loss = 0
                        running_ce_loss = 0
                        running_penalty = 0
                        running_max_prob = 0

    # Finish
    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)

        if config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()

    accelerator.end_training()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train GPT2 with FP8 using Accelerate")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--penalty_weight", type=float, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--use_fp8", action="store_true", help="Enable FP8 training")
    parser.add_argument("--no_wandb", action="store_true")

    args = parser.parse_args()

    # Load config
    if args.config:
        config = TrainingConfig.from_yaml(args.config)
    else:
        config = TrainingConfig()

    # Apply overrides
    if args.penalty_weight is not None:
        config.penalty_weight = args.penalty_weight
    if args.dataset is not None:
        config.dataset = args.dataset
    if args.use_fp8:
        config.use_fp8 = True
    if args.no_wandb:
        config.use_wandb = False

    # Re-run post-init
    config.__post_init__()

    # Train
    train_with_accelerate(config)
