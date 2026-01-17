"""
Quick test to run a few training steps and verify everything works.
"""
import torch
from transformers import GPT2TokenizerFast
from tqdm import tqdm

from config import TrainingConfig
from model import create_gpt2_model, compute_loss_with_penalty
from data import get_dataloaders

print("=" * 60)
print("Quick Training Test - Running 20 steps")
print("=" * 60)

# Config
config = TrainingConfig()
config.dataset = "tinyshakespeare"
config.batch_size = 2
config.max_seq_length = 128
config.use_wandb = False
config.num_workers = 0

# Setup
device = torch.device(config.device)
print(f"\nDevice: {device}")

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = GPT2TokenizerFast.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

# Create model
print("Creating model...")
model = create_gpt2_model(config.model_name, reinitialize=True)
model = model.to(device)

# Load data
print("\nLoading data...")
train_loader, val_loader = get_dataloaders(
    dataset_name=config.dataset,
    tokenizer=tokenizer,
    max_seq_length=config.max_seq_length,
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    device=config.device,
)

# Setup optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    betas=config.betas,
    weight_decay=config.weight_decay,
)

# Training loop
print("\nRunning 20 training steps...")
print("-" * 60)

model.train()
step = 0
max_steps = 20

train_iter = iter(train_loader)

for step in tqdm(range(max_steps), desc="Training"):
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        batch = next(train_iter)

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

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    optimizer.step()

    # Log every 5 steps
    if (step + 1) % 5 == 0:
        print(f"\nStep {step + 1}:")
        print(f"  Total Loss: {loss_dict['total_loss']:.4f}")
        print(f"  CE Loss: {loss_dict['ce_loss']:.4f}")
        print(f"  Penalty: {loss_dict['penalty']:.4f}")
        print(f"  Avg Max Prob: {loss_dict['avg_max_prob']:.4f}")
        print(f"  Perplexity: {loss_dict['perplexity']:.2f}")

print("\n" + "=" * 60)
print("Test completed successfully!")
print("=" * 60)
print("\nThe training infrastructure is working correctly.")
print("To run full training, use: python train.py")
