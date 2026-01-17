import os
# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_samples(model, tokenizer, device, num_samples=3, max_length=100, temperature=1.0):
    """Generate text samples from the model."""
    model.eval()
    samples = []

    prompts = [
        "The quick brown fox",
        "Once upon a time",
        "In the beginning",
    ][:num_samples]

    with torch.no_grad():
        for prompt in prompts:
            # Encode prompt
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

            # Generate
            output = model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
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

    # Setup device
    device = torch.device(config.device)
    print(f"Using device: {device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Create model
    print("Creating model...")
    model = create_gpt2_model(config.model_name, reinitialize=True)
    model = model.to(device)

    # Log model info
    num_params = count_parameters(model)
    print(f"Number of trainable parameters: {num_params:,}")
    if config.use_wandb:
        wandb.config.update({"num_parameters": num_params})

    # Load data
    print("Loading data...")
    train_loader, val_loader = get_dataloaders(
        dataset_name=config.dataset,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=config.device,
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

    # Setup scheduler
    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

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
                    avg_loss = running_loss / config.log_interval
                    avg_ce_loss = running_ce_loss / config.log_interval
                    avg_penalty = running_penalty / config.log_interval
                    avg_max_prob = running_max_prob / config.log_interval

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
                        model, tokenizer, device,
                        num_samples=config.num_generate_samples,
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
        choices=["tinyshakespeare", "openwebtext"],
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

    # Re-run post-init to apply any auto-adjustments
    config.__post_init__()

    # Train
    train(config)
