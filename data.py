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
        # Each chunk is max_seq_length tokens (model handles next-token prediction internally)
        return len(self.tokens) // self.max_seq_length

    def __getitem__(self, idx):
        # Get non-overlapping chunk
        start_idx = idx * self.max_seq_length
        end_idx = start_idx + self.max_seq_length

        chunk = self.tokens[start_idx:end_idx]

        # If chunk is too short, pad it (only for last chunk)
        if len(chunk) < self.max_seq_length:
            # Pad with -100 to ignore in loss computation
            padding = torch.full((self.max_seq_length - len(chunk),), -100, dtype=torch.long)
            chunk = torch.cat([chunk, padding])

        # GPT2LMHeadModel shifts labels internally, so we pass the same sequence
        # The model will predict token[i+1] from token[0:i]
        input_ids = chunk
        labels = chunk

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def load_tinyshakespeare(tokenizer: GPT2TokenizerFast, max_seq_length: int = 1024, train_val_split: float = 0.9):
    """
    Load TinyShakespeare dataset for quick testing.

    Args:
        tokenizer: GPT2 tokenizer
        max_seq_length: Maximum sequence length
        train_val_split: Fraction of data to use for training (default: 0.9)

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

    # Split into train and validation
    split_idx = int(len(text) * train_val_split)
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


def load_openwebtext(tokenizer: GPT2TokenizerFast, max_seq_length: int = 1024, max_samples: int = None, val_samples_count: int = 1000):
    """
    Load OpenWebText dataset for full training.

    Args:
        tokenizer: GPT2 tokenizer
        max_seq_length: Maximum sequence length
        max_samples: Maximum number of samples to load (for debugging)
        val_samples_count: Number of samples to use for validation (default: 1000)

    Returns:
        train_dataset, val_dataset
    """
    print("Loading OpenWebText dataset...")

    # Load from HuggingFace datasets
    # Note: OpenWebText is quite large, so we'll use streaming for efficiency
    dataset = load_dataset("openwebtext", split="train", streaming=True)

    # Take a subset for validation
    val_samples = []
    train_samples = []

    for idx, sample in enumerate(dataset):
        if idx < val_samples_count:
            val_samples.append(sample["text"])
        else:
            train_samples.append(sample["text"])

        if max_samples and idx >= max_samples + val_samples_count:
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


def load_fineweb_edu(tokenizer: GPT2TokenizerFast, max_seq_length: int = 1024, max_samples: int = None, val_samples_count: int = 5000):
    """
    Load FineWeb-Edu dataset for high-quality training.

    FineWeb-Edu is a filtered version of FineWeb containing educational web content.
    It's cleaner and higher quality than OpenWebText.

    Args:
        tokenizer: GPT2 tokenizer
        max_seq_length: Maximum sequence length
        max_samples: Maximum number of samples to load (for debugging)
        val_samples_count: Number of samples to use for validation (default: 5000)

    Returns:
        train_dataset, val_dataset
    """
    print("Loading FineWeb-Edu dataset...")
    print("Note: This is a large dataset (~1.3T tokens). Loading may take a while...")

    # Load from HuggingFace datasets
    # FineWeb-Edu sample-10BT contains 10B tokens subset (good for testing)
    # Use "sample-10BT" for faster loading, or "default" for full dataset
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",  # 10B token subset
        split="train",
        streaming=True
    )

    # Take a subset for validation
    val_samples = []
    train_samples = []

    print(f"Collecting samples from FineWeb-Edu...")
    for idx, sample in enumerate(dataset):
        if idx < val_samples_count:
            val_samples.append(sample["text"])
        else:
            train_samples.append(sample["text"])

        # Progress indicator
        if idx % 10000 == 0 and idx > 0:
            print(f"  Processed {idx:,} samples...")

        if max_samples and idx >= max_samples + val_samples_count:
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

    print("Tokenizing train samples...")
    train_tokens = tokenize_batch(train_samples)
    print("Tokenizing validation samples...")
    val_tokens = tokenize_batch(val_samples)

    print(f"FineWeb-Edu - Train tokens: {len(train_tokens):,}")
    print(f"FineWeb-Edu - Val tokens: {len(val_tokens):,}")

    # Create datasets
    train_dataset = TextDataset(train_tokens, max_seq_length)
    val_dataset = TextDataset(val_tokens, max_seq_length)

    return train_dataset, val_dataset


def get_dataloaders(
    dataset_name: Literal["tinyshakespeare", "openwebtext", "fineweb-edu"],
    tokenizer: GPT2TokenizerFast,
    max_seq_length: int = 1024,
    batch_size: int = 8,
    num_workers: int = 4,
    max_samples: int = None,
    device: str = "cuda",
    train_val_split: float = 0.9,
    openwebtext_val_samples: int = 1000,
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
        train_val_split: Train/val split ratio (for tinyshakespeare)
        openwebtext_val_samples: Number of validation samples (for openwebtext)

    Returns:
        train_loader, val_loader
    """
    if dataset_name == "tinyshakespeare":
        train_dataset, val_dataset = load_tinyshakespeare(tokenizer, max_seq_length, train_val_split)
    elif dataset_name == "openwebtext":
        train_dataset, val_dataset = load_openwebtext(tokenizer, max_seq_length, max_samples, openwebtext_val_samples)
    elif dataset_name == "fineweb-edu":
        train_dataset, val_dataset = load_fineweb_edu(tokenizer, max_seq_length, max_samples, openwebtext_val_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: tinyshakespeare, openwebtext, fineweb-edu")

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
