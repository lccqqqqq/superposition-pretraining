import os
# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math
from transformers import GPT2TokenizerFast
import wandb
from tqdm import tqdm
import random
import numpy as np
from pathlib import Path

from config import TrainingConfig
from model import create_gpt2_model, compute_loss_with_penalty
from data import get_dataloaders


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.0):
    """
    Create learning rate scheduler with linear warmup and cosine decay.

    This matches the schedule used in GPT-2 and many modern transformers:
    1. Linear warmup: LR increases from 0 to max_lr over warmup_steps
    2. Cosine decay: LR decreases from max_lr to min_lr following cosine curve

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of steps for linear warmup
        total_steps: Total number of training steps
        min_lr_ratio: Minimum LR as ratio of max LR (default: 0.0)

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step):
        # Linear warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # Cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Scale from min_lr_ratio to 1.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_samples(model, tokenizer, device, config):
    """Generate text samples from the model."""
    model.eval()
    samples = []

    prompts = [
        "The quick brown fox",
        "Once upon a time",
        "In the beginning",
    ][:config.num_generate_samples]

    with torch.no_grad():
        for prompt in prompts:
            # Encode prompt
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

            # Generate
            output = model.generate(
                input_ids,
                max_length=config.generate_max_length,
                temperature=config.generate_temperature,
                do_sample=True,
                top_p=config.generate_top_p,
                pad_token_id=tokenizer.eos_token_id,
            )

            # Decode
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            samples.append({"prompt": prompt, "generated": generated_text})

    model.train()
    return samples


def evaluate(model, val_loader, config, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_ce_loss = 0
    total_penalty = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            loss, loss_dict = compute_loss_with_penalty(
                model=model,
                input_ids=input_ids,
                labels=labels,
                penalty_weight=config.penalty_weight,
                penalty_type=config.penalty_type,
                top_k=config.top_k,
            )

            batch_size = input_ids.size(0)
            total_loss += loss_dict["total_loss"] * batch_size
            total_ce_loss += loss_dict["ce_loss"] * batch_size
            total_penalty += loss_dict["penalty"] * batch_size
            total_samples += batch_size

    model.train()

    return {
        "val_loss": total_loss / total_samples,
        "val_ce_loss": total_ce_loss / total_samples,
        "val_penalty": total_penalty / total_samples,
    }


def train(config: TrainingConfig):
    """Main training function."""
    # Set seed
    set_seed(config.seed)

    # Create output directories
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Initialize W&B
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            config=vars(config),
            name=f"{config.dataset}_{config.penalty_type}_lambda{config.penalty_weight}",
        )

        # Override config from W&B sweep if in sweep mode
        # W&B sweep agent will set wandb.config with sweep parameters
        if wandb.run.sweep_id is not None:
            print("\nRunning in W&B sweep mode, applying sweep parameters...")
            # Override parameters that are commonly swept
            if hasattr(wandb.config, 'penalty_weight'):
                config.penalty_weight = wandb.config.penalty_weight
                print(f"  Sweep override: penalty_weight = {config.penalty_weight}")
            if hasattr(wandb.config, 'penalty_type'):
                config.penalty_type = wandb.config.penalty_type
                print(f"  Sweep override: penalty_type = {config.penalty_type}")
            if hasattr(wandb.config, 'learning_rate'):
                config.learning_rate = wandb.config.learning_rate
                print(f"  Sweep override: learning_rate = {config.learning_rate}")
            if hasattr(wandb.config, 'batch_size'):
                config.batch_size = wandb.config.batch_size
                print(f"  Sweep override: batch_size = {config.batch_size}")
            print()

    # Setup device
    device = torch.device(config.device)
    print(f"Using device: {device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Create model
    print("Creating model...")
    model = create_gpt2_model(config.model_name, reinitialize=config.reinitialize_weights)
    model = model.to(device)

    # Log model info
    num_params = count_parameters(model)
    print(f"Number of trainable parameters: {num_params:,}")

    # Log batch configuration
    effective_batch_size = config.batch_size * config.gradient_accumulation_steps
    print(f"\nBatch configuration:")
    print(f"  Batch size per GPU: {config.batch_size}")
    print(f"  Gradient accumulation steps: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")

    if config.use_wandb:
        wandb.config.update({
            "num_parameters": num_params,
            "actual_batch_size": config.batch_size,
            "actual_gradient_accumulation_steps": config.gradient_accumulation_steps,
            "actual_effective_batch_size": effective_batch_size,
        })

    # Load data
    print("Loading data...")
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

    # Setup scheduler with linear warmup + cosine decay (matches GPT-2)
    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    scheduler = get_lr_schedule_with_warmup(
        optimizer=optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=config.min_lr_ratio
    )

    print(f"\nLearning rate schedule:")
    print(f"  Max learning rate: {config.learning_rate}")
    print(f"  Warmup steps: {config.warmup_steps}")
    print(f"  Total training steps: {total_steps}")
    print(f"  Min LR (at end): {config.learning_rate * config.min_lr_ratio:.2e}")
    print(f"  Schedule: Linear warmup + Cosine decay")

    # Training loop
    print("\nStarting training...")
    global_step = 0
    running_loss = 0
    running_ce_loss = 0
    running_penalty = 0
    running_max_prob = 0

    model.train()

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            loss, loss_dict = compute_loss_with_penalty(
                model=model,
                input_ids=input_ids,
                labels=labels,
                penalty_weight=config.penalty_weight,
                penalty_type=config.penalty_type,
                top_k=config.top_k,
            )

            # Normalize loss for gradient accumulation
            loss = loss / config.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Update running metrics
            running_loss += loss_dict["total_loss"]
            running_ce_loss += loss_dict["ce_loss"]
            running_penalty += loss_dict["penalty"]
            running_max_prob += loss_dict["avg_max_prob"]

            # Gradient accumulation
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Logging
                if global_step % config.log_interval == 0:
                    # We accumulate per batch, but log per global step
                    # Need to divide by (log_interval * gradient_accumulation_steps)
                    num_batches = config.log_interval * config.gradient_accumulation_steps
                    avg_loss = running_loss / num_batches
                    avg_ce_loss = running_ce_loss / num_batches
                    avg_penalty = running_penalty / num_batches
                    avg_max_prob = running_max_prob / num_batches

                    pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "ce": f"{avg_ce_loss:.4f}",
                        "pen": f"{avg_penalty:.4f}",
                        "max_p": f"{avg_max_prob:.4f}",
                    })

                    if config.use_wandb:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/ce_loss": avg_ce_loss,
                            "train/penalty": avg_penalty,
                            "train/avg_max_prob": avg_max_prob,
                            "train/perplexity": loss_dict["perplexity"],
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                            "train/step": global_step,
                        })

                    running_loss = 0
                    running_ce_loss = 0
                    running_penalty = 0
                    running_max_prob = 0

                # Evaluation
                if global_step % config.eval_interval == 0:
                    print("\nEvaluating...")
                    eval_metrics = evaluate(model, val_loader, config, device)

                    print(f"Validation - Loss: {eval_metrics['val_loss']:.4f}, "
                          f"CE: {eval_metrics['val_ce_loss']:.4f}, "
                          f"Penalty: {eval_metrics['val_penalty']:.4f}")

                    if config.use_wandb:
                        wandb.log({
                            "val/loss": eval_metrics["val_loss"],
                            "val/ce_loss": eval_metrics["val_ce_loss"],
                            "val/penalty": eval_metrics["val_penalty"],
                            "train/step": global_step,
                        })

                # Generate samples
                if global_step % config.generate_interval == 0:
                    print("\nGenerating samples...")
                    samples = generate_samples(
                        model, tokenizer, device, config
                    )

                    for i, sample in enumerate(samples):
                        print(f"\nSample {i + 1}:")
                        print(f"Prompt: {sample['prompt']}")
                        print(f"Generated: {sample['generated']}")

                    if config.use_wandb:
                        wandb.log({
                            "samples": wandb.Table(
                                columns=["prompt", "generated"],
                                data=[[s["prompt"], s["generated"]] for s in samples]
                            ),
                            "train/step": global_step,
                        })

                # Save checkpoint
                if global_step % config.save_interval == 0:
                    checkpoint_path = Path(config.checkpoint_dir) / f"checkpoint_step_{global_step}.pt"
                    print(f"\nSaving checkpoint to {checkpoint_path}")

                    torch.save({
                        "step": global_step,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "config": vars(config),
                    }, checkpoint_path)

                    if config.use_wandb:
                        wandb.save(str(checkpoint_path))

    # Final save
    final_path = Path(config.output_dir) / "final_model.pt"
    print(f"\nSaving final model to {final_path}")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": vars(config),
    }, final_path)

    if config.use_wandb:
        wandb.save(str(final_path))
        wandb.finish()

    print("\nTraining complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train GPT2 with entropy regularization")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (e.g., configs/quick_test.yaml)"
    )
    parser.add_argument(
        "--penalty_weight",
        type=float,
        default=None,
        help="Override penalty weight (lambda)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["tinyshakespeare", "openwebtext", "fineweb-edu"],
        help="Override dataset"
    )
    parser.add_argument(
        "--penalty_type",
        type=str,
        default=None,
        choices=["non_max_sum", "neg_entropy", "neg_max_prob", "top_k_mass"],
        help="Override penalty type"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--auto_batch_size",
        action="store_true",
        help="Enable automatic batch size detection based on GPU memory"
    )
    parser.add_argument(
        "--target_effective_batch_size",
        type=int,
        default=None,
        help="Target effective batch size for auto batch size detection"
    )

    args = parser.parse_args()

    # Load config
    if args.config:
        print(f"Loading configuration from {args.config}")
        config = TrainingConfig.from_yaml(args.config)
    else:
        print("Using default configuration")
        config = TrainingConfig()

    # Apply command-line overrides
    if args.penalty_weight is not None:
        config.penalty_weight = args.penalty_weight
        print(f"Override: penalty_weight = {args.penalty_weight}")

    if args.dataset is not None:
        config.dataset = args.dataset
        print(f"Override: dataset = {args.dataset}")

    if args.penalty_type is not None:
        config.penalty_type = args.penalty_type
        print(f"Override: penalty_type = {args.penalty_type}")

    if args.no_wandb:
        config.use_wandb = False
        print("Override: W&B logging disabled")

    if args.auto_batch_size:
        config.auto_batch_size = True
        print("Override: Auto batch size detection enabled")

    if args.target_effective_batch_size is not None:
        config.target_effective_batch_size = args.target_effective_batch_size
        print(f"Override: target_effective_batch_size = {args.target_effective_batch_size}")

    # Re-run post-init to apply any auto-adjustments
    config.__post_init__()

    # Train
    train(config)
