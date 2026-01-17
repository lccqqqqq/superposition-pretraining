import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
from typing import Literal


def create_gpt2_model(model_name: str = "gpt2", reinitialize: bool = True) -> GPT2LMHeadModel:
    """
    Load GPT2 model from HuggingFace and optionally reinitialize weights.

    Args:
        model_name: HuggingFace model identifier (default: "gpt2" for GPT2-small)
        reinitialize: If True, reset all weights to random initialization

    Returns:
        GPT2LMHeadModel with randomly initialized weights
    """
    # Load the configuration
    config = GPT2Config.from_pretrained(model_name)

    if reinitialize:
        # Create model from config (random initialization)
        model = GPT2LMHeadModel(config)
        print(f"Created GPT2 model with random initialization")
        print(f"  Layers: {config.n_layer}")
        print(f"  Hidden size: {config.n_embd}")
        print(f"  Attention heads: {config.n_head}")
        print(f"  Vocab size: {config.vocab_size}")
    else:
        # Load pretrained weights
        model = GPT2LMHeadModel.from_pretrained(model_name)
        print(f"Loaded pretrained GPT2 model from {model_name}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model


def compute_sharpness_penalty(
    logits: torch.Tensor,
    penalty_type: Literal["non_max_sum", "neg_entropy", "neg_max_prob", "top_k_mass"] = "non_max_sum",
    top_k: int = 5,
    positions_slice: tuple = (0, -1),  # (start, end) - -1 means exclude last
) -> torch.Tensor:
    """
    Compute sharpness penalty to encourage confident predictions.

    Args:
        logits: Model logits of shape (batch_size, seq_length, vocab_size)
        penalty_type: Type of penalty to compute
            - "non_max_sum": Sum of all non-max probabilities (1 - max_prob)
            - "neg_entropy": Negative entropy (-Σ p log p)
            - "neg_max_prob": Negative of max probability
            - "top_k_mass": 1 - sum of top-k probabilities
        top_k: Number of top probabilities to consider (for top_k_mass)
        positions_slice: Tuple of (start, end) positions to apply penalty

    Returns:
        Scalar tensor with the penalty value
    """
    # Get the relevant positions (0:N-2 means exclude the last position)
    batch_size, seq_length, vocab_size = logits.shape
    start_pos, end_pos = positions_slice
    if end_pos == -1:
        end_pos = seq_length - 1

    # Slice the logits to get positions we care about
    logits_slice = logits[:, start_pos:end_pos, :]  # Shape: (batch, seq_len-1, vocab)

    # Compute probabilities
    probs = torch.softmax(logits_slice, dim=-1)  # Shape: (batch, seq_len-1, vocab)

    if penalty_type == "non_max_sum":
        # Sum of non-max probabilities = 1 - max_prob
        max_probs = torch.max(probs, dim=-1).values  # Shape: (batch, seq_len-1)
        penalty = (1.0 - max_probs).mean()

    elif penalty_type == "neg_entropy":
        # Negative entropy: -Σ p log p (minimizing this increases entropy)
        # We want to MINIMIZE entropy, so we use -(-Σ p log p) = Σ p log p
        # But conventionally, we want LOWER penalty = BETTER, so:
        # penalty = -entropy means lower penalty when entropy is high
        # Actually, we want to encourage LOW entropy (sharp distributions)
        # So penalty should be entropy itself (minimize entropy)
        eps = 1e-10
        entropy = -(probs * torch.log(probs + eps)).sum(dim=-1)  # Shape: (batch, seq_len-1)
        penalty = entropy.mean()

    elif penalty_type == "neg_max_prob":
        # Negative max probability (minimize this to maximize confidence)
        max_probs = torch.max(probs, dim=-1).values
        penalty = -max_probs.mean()

    elif penalty_type == "top_k_mass":
        # 1 - sum of top-k probabilities
        top_k_probs = torch.topk(probs, k=min(top_k, vocab_size), dim=-1).values
        top_k_sum = top_k_probs.sum(dim=-1)  # Shape: (batch, seq_len-1)
        penalty = (1.0 - top_k_sum).mean()

    else:
        raise ValueError(f"Unknown penalty type: {penalty_type}")

    return penalty


def compute_loss_with_penalty(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    penalty_weight: float = 0.01,
    penalty_type: str = "non_max_sum",
    top_k: int = 5,
) -> tuple[torch.Tensor, dict]:
    """
    Compute language modeling loss with sharpness penalty.

    Args:
        model: GPT2 model
        input_ids: Input token ids (batch_size, seq_length)
        labels: Target token ids (batch_size, seq_length)
        penalty_weight: Weight for the sharpness penalty (λ)
        penalty_type: Type of sharpness penalty
        top_k: Top-k parameter for top_k_mass penalty

    Returns:
        total_loss: Combined loss (CE + λ * penalty)
        loss_dict: Dictionary with individual loss components
    """
    # Forward pass
    outputs = model(input_ids=input_ids, labels=labels)
    ce_loss = outputs.loss  # Cross-entropy loss
    logits = outputs.logits  # Shape: (batch, seq_length, vocab)

    # Compute sharpness penalty for positions 0:N-2 (exclude last)
    penalty = compute_sharpness_penalty(
        logits=logits,
        penalty_type=penalty_type,
        top_k=top_k,
        positions_slice=(0, -1),
    )

    # Total loss
    total_loss = ce_loss + penalty_weight * penalty

    # Also compute some useful metrics
    with torch.no_grad():
        probs = torch.softmax(logits[:, :-1, :], dim=-1)  # Exclude last position
        max_probs = torch.max(probs, dim=-1).values
        avg_max_prob = max_probs.mean().item()

        # Compute perplexity from CE loss
        perplexity = torch.exp(ce_loss).item()

    loss_dict = {
        "total_loss": total_loss.item(),
        "ce_loss": ce_loss.item(),
        "penalty": penalty.item(),
        "avg_max_prob": avg_max_prob,
        "perplexity": perplexity,
    }

    return total_loss, loss_dict
