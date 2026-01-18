#!/usr/bin/env python
"""
Test the learning rate schedule implementation.

This script verifies that the LR schedule matches expected behavior.
"""

import torch
from torch.optim import AdamW
import math


def get_lr_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.0):
    """
    Create learning rate scheduler with linear warmup and cosine decay.
    (Copy of the function from train.py for testing)
    """
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def test_lr_schedule():
    """Test the learning rate schedule."""
    print("Testing Learning Rate Schedule")
    print("=" * 60)

    # Create dummy model and optimizer
    model = torch.nn.Linear(10, 10)
    base_lr = 1e-3
    optimizer = AdamW(model.parameters(), lr=base_lr)

    # Test configuration
    warmup_steps = 100
    total_steps = 1000
    min_lr_ratio = 0.1

    scheduler = get_lr_schedule_with_warmup(
        optimizer, warmup_steps, total_steps, min_lr_ratio
    )

    print(f"\nConfiguration:")
    print(f"  Base LR: {base_lr}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Total steps: {total_steps}")
    print(f"  Min LR ratio: {min_lr_ratio}")
    print(f"  Expected min LR: {base_lr * min_lr_ratio}")

    # Test key points
    test_points = [
        (0, 0.0, "Start (should be 0)"),
        (warmup_steps // 2, 0.5, "Mid-warmup (should be ~0.5)"),
        (warmup_steps, 1.0, "End of warmup (should be 1.0)"),
        (total_steps // 2, None, "Mid-training"),
        (total_steps - 1, min_lr_ratio, f"End of training (should be ~{min_lr_ratio})"),
    ]

    print("\n" + "=" * 60)
    print("Testing key points:")
    print("=" * 60)

    all_passed = True
    for step, expected_ratio, description in test_points:
        # Simulate steps
        for _ in range(step + 1):
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        current_ratio = current_lr / base_lr

        if expected_ratio is not None:
            # Check if within 5% tolerance
            tolerance = 0.05
            passed = abs(current_ratio - expected_ratio) < tolerance
            status = "✓ PASS" if passed else "✗ FAIL"
            all_passed = all_passed and passed
        else:
            status = "  INFO"

        print(f"Step {step:4d}: LR={current_lr:.6f}, Ratio={current_ratio:.4f} - {status} - {description}")

        # Reset for next test
        optimizer = AdamW(model.parameters(), lr=base_lr)
        scheduler = get_lr_schedule_with_warmup(
            optimizer, warmup_steps, total_steps, min_lr_ratio
        )

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests PASSED!")
    else:
        print("✗ Some tests FAILED!")
    print("=" * 60)

    # Generate a small LR curve for visual inspection
    print("\nLR curve (first 150 steps):")
    print("=" * 60)

    optimizer = AdamW(model.parameters(), lr=base_lr)
    scheduler = get_lr_schedule_with_warmup(
        optimizer, warmup_steps, total_steps, min_lr_ratio
    )

    for step in range(150):
        if step % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            bar_length = int(current_lr / base_lr * 50)
            bar = "█" * bar_length
            print(f"Step {step:3d}: {bar} {current_lr:.6f}")
        scheduler.step()

    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = test_lr_schedule()
    exit(0 if success else 1)
