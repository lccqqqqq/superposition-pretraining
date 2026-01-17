#%% Imports
import torch as t
from config import TrainingConfig
from model import create_gpt2_model, compute_loss_with_penalty
from data import get_dataloaders
from transformers import GPT2TokenizerFast

#%% Quick setup
# Create config for experimentation
config = TrainingConfig()
config.dataset = "tinyshakespeare"
config.penalty_type = "non_max_sum"  # Options: "non_max_sum", "neg_entropy", "neg_max_prob", "top_k_mass"
config.penalty_weight = 0.01
config.use_wandb = True  # Set to False for local testing

#%% Load model and tokenizer
device = t.device(config.device)
print(f"Using device: {device}")

tokenizer = GPT2TokenizerFast.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

model = create_gpt2_model(config.model_name, reinitialize=True)
model = model.to(device)

#%% Load data
train_loader, val_loader = get_dataloaders(
    dataset_name=config.dataset,
    tokenizer=tokenizer,
    max_seq_length=config.max_seq_length,
    batch_size=config.batch_size,
    num_workers=0,  # Set to 0 for debugging in notebooks
)

#%% Test forward pass
batch = next(iter(train_loader))
input_ids = batch["input_ids"].to(device)
labels = batch["labels"].to(device)

loss, loss_dict = compute_loss_with_penalty(
    model=model,
    input_ids=input_ids,
    labels=labels,
    penalty_weight=config.penalty_weight,
    penalty_type=config.penalty_type,
)

print("Loss components:")
for key, value in loss_dict.items():
    print(f"  {key}: {value:.4f}")

#%% Run full training
# Uncomment to start training
# from train import train
# train(config)