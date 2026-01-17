import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from typing import Literal
import numpy as np


class TextDataset(Dataset):
    """
    Dataset for language modeling that chunks tokenized text into sequences.
    """
    def __init__(
        self,
        tokenized_data: list,
        max_seq_length: int = 1024,
    ):
        """
        Args:
            tokenized_data: List of token ids
            max_seq_length: Maximum sequence length
        """
        self.tokens = torch.tensor(tokenized_data, dtype=torch.long)
        self.max_seq_length = max_seq_length

    def __len__(self):
        # Number of non-overlapping chunks
        # We need max_seq_length + 1 tokens per chunk (for input + target)
        return len(self.tokens) // (self.max_seq_length + 1)

    def __getitem__(self, idx):
        # Get non-overlapping chunk
        start_idx = idx * (self.max_seq_length + 1)
        end_idx = start_idx + self.max_seq_length + 1

        chunk = self.tokens[start_idx:end_idx]

        # If chunk is too short, pad it (only for last chunk)
        if len(chunk) < self.max_seq_length + 1:
            padding = torch.zeros(self.max_seq_length + 1 - len(chunk), dtype=torch.long)
            chunk = torch.cat([chunk, padding])

        # Input is all tokens except last, target is all tokens except first
        input_ids = chunk[:-1]
        labels = chunk[1:]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def load_tinyshakespeare(tokenizer: GPT2TokenizerFast, max_seq_length: int = 1024):
    """
    Load TinyShakespeare dataset for quick testing.

    Args:
        tokenizer: GPT2 tokenizer
        max_seq_length: Maximum sequence length

    Returns:
        train_dataset, val_dataset
    """
    print("Loading TinyShakespeare dataset...")

    # Download the raw text file
    import urllib.request
    import ssl

    # Create SSL context that doesn't verify certificates (for local testing)
    ssl_context = ssl._create_unverified_context()

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    try:
        with urllib.request.urlopen(url, context=ssl_context) as response:
            text = response.read().decode('utf-8')
    except Exception as e:
        print(f"Error downloading TinyShakespeare: {e}")
        print("Trying alternative source...")
        # Alternative URL
        url = "https://raw.githubusercontent.com/karpathy/ng-video-lecture/master/input.txt"
        with urllib.request.urlopen(url, context=ssl_context) as response:
            text = response.read().decode('utf-8')

    # Split into train and validation (90/10 split)
    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    print(f"Total characters: {len(text):,}")
    print(f"Train characters: {len(train_text):,}")
    print(f"Val characters: {len(val_text):,}")

    # Tokenize
    train_tokens = tokenizer(train_text, truncation=False, padding=False)["input_ids"]
    val_tokens = tokenizer(val_text, truncation=False, padding=False)["input_ids"]

    print(f"TinyShakespeare - Train tokens: {len(train_tokens):,}")
    print(f"TinyShakespeare - Val tokens: {len(val_tokens):,}")

    # Create datasets
    train_dataset = TextDataset(train_tokens, max_seq_length)
    val_dataset = TextDataset(val_tokens, max_seq_length)

    return train_dataset, val_dataset


def load_openwebtext(tokenizer: GPT2TokenizerFast, max_seq_length: int = 1024, max_samples: int = None):
    """
    Load OpenWebText dataset for full training.

    Args:
        tokenizer: GPT2 tokenizer
        max_seq_length: Maximum sequence length
        max_samples: Maximum number of samples to load (for debugging)

    Returns:
        train_dataset, val_dataset
    """
    print("Loading OpenWebText dataset...")

    # Load from HuggingFace datasets
    # Note: OpenWebText is quite large, so we'll use streaming for efficiency
    dataset = load_dataset("openwebtext", split="train", streaming=True)

    # Take a subset for validation (e.g., 1000 samples)
    val_samples = []
    train_samples = []

    for idx, sample in enumerate(dataset):
        if idx < 1000:
            val_samples.append(sample["text"])
        else:
            train_samples.append(sample["text"])

        if max_samples and idx >= max_samples + 1000:
            break

    print(f"Collected {len(train_samples):,} train samples")
    print(f"Collected {len(val_samples):,} validation samples")

    # Tokenize
    def tokenize_batch(texts):
        tokens = []
        for text in texts:
            encoded = tokenizer(text, truncation=False, padding=False)
            tokens.extend(encoded["input_ids"])
        return tokens

    train_tokens = tokenize_batch(train_samples)
    val_tokens = tokenize_batch(val_samples)

    print(f"OpenWebText - Train tokens: {len(train_tokens):,}")
    print(f"OpenWebText - Val tokens: {len(val_tokens):,}")

    # Create datasets
    train_dataset = TextDataset(train_tokens, max_seq_length)
    val_dataset = TextDataset(val_tokens, max_seq_length)

    return train_dataset, val_dataset


def get_dataloaders(
    dataset_name: Literal["tinyshakespeare", "openwebtext"],
    tokenizer: GPT2TokenizerFast,
    max_seq_length: int = 1024,
    batch_size: int = 8,
    num_workers: int = 4,
    max_samples: int = None,
    device: str = "cuda",
):
    """
    Create train and validation dataloaders.

    Args:
        dataset_name: Name of dataset to load
        tokenizer: GPT2 tokenizer
        max_seq_length: Maximum sequence length
        batch_size: Batch size
        num_workers: Number of dataloader workers
        max_samples: Maximum samples to load (for debugging)
        device: Device type (used to determine pin_memory setting)

    Returns:
        train_loader, val_loader
    """
    if dataset_name == "tinyshakespeare":
        train_dataset, val_dataset = load_tinyshakespeare(tokenizer, max_seq_length)
    elif dataset_name == "openwebtext":
        train_dataset, val_dataset = load_openwebtext(tokenizer, max_seq_length, max_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Disable pin_memory on MPS (not supported)
    use_pin_memory = device == "cuda"

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader
