"""
Quick test script to verify training setup works.
"""
from config import TrainingConfig
from train import train

# Create minimal config for testing
config = TrainingConfig()
config.dataset = "tinyshakespeare"
config.num_epochs = 1
config.batch_size = 2
config.gradient_accumulation_steps = 1
config.max_seq_length = 128  # Shorter sequences for faster testing
config.log_interval = 5
config.eval_interval = 20
config.save_interval = 50
config.generate_interval = 20
config.use_wandb = False  # Disable W&B for testing
config.num_workers = 0  # Disable multiprocessing for macOS compatibility

print("=" * 50)
print("Running minimal training test...")
print("=" * 50)

train(config)

print("\n" + "=" * 50)
print("Test completed successfully!")
print("=" * 50)
