#!/usr/bin/env python
"""
Visualize the learning rate schedule for GPT-2 training.

This script plots the learning rate schedule (linear warmup + cosine decay)
for different configurations to help understand the training dynamics.

Usage:
    python visualize_lr_schedule.py
    python visualize_lr_schedule.py --warmup_steps 2000 --total_steps 5000
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import math


def get_lr_at_step(step, warmup_steps, total_steps, max_lr=1.0, min_lr_ratio=0.1):
    """
    Calculate learning rate at a given step.

    Args:
        step: Current training step
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        max_lr: Maximum learning rate
        min_lr_ratio: Minimum LR as ratio of max LR

    Returns:
        Learning rate at the given step
    """
    # Linear warmup
    if step < warmup_steps:
        lr_multiplier = float(step) / float(max(1, warmup_steps))
    else:
        # Cosine decay
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr_multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return max_lr * lr_multiplier


def visualize_schedule(warmup_steps, total_steps, max_lr=1.0, min_lr_ratio=0.1, save_path=None):
    """
    Visualize the learning rate schedule.

    Args:
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        max_lr: Maximum learning rate
        min_lr_ratio: Minimum LR as ratio of max LR
        save_path: Path to save the plot (if None, display only)
    """
    steps = np.arange(0, total_steps + 1)
    lrs = [get_lr_at_step(s, warmup_steps, total_steps, max_lr, min_lr_ratio) for s in steps]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Full schedule
    ax1.plot(steps, lrs, linewidth=2, color='#2E86AB')
    ax1.axvline(warmup_steps, color='red', linestyle='--', alpha=0.7, label=f'Warmup end ({warmup_steps} steps)')
    ax1.axhline(max_lr, color='green', linestyle='--', alpha=0.5, label=f'Max LR ({max_lr})')
    ax1.axhline(max_lr * min_lr_ratio, color='orange', linestyle='--', alpha=0.5,
                label=f'Min LR ({max_lr * min_lr_ratio:.2e})')
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Learning Rate', fontsize=12)
    ax1.set_title('Learning Rate Schedule: Linear Warmup + Cosine Decay', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Zoomed warmup phase
    warmup_end_idx = min(warmup_steps * 2, total_steps)
    ax2.plot(steps[:warmup_end_idx], lrs[:warmup_end_idx], linewidth=2, color='#A23B72')
    ax2.axvline(warmup_steps, color='red', linestyle='--', alpha=0.7, label='Warmup end')
    ax2.axhline(max_lr, color='green', linestyle='--', alpha=0.5, label='Max LR')
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Warmup Phase (Zoomed)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    # Add info text
    info_text = f"""Schedule Parameters:
• Warmup steps: {warmup_steps:,} ({warmup_steps/total_steps*100:.1f}% of training)
• Total steps: {total_steps:,}
• Max LR: {max_lr}
• Min LR: {max_lr * min_lr_ratio:.2e} ({min_lr_ratio*100:.0f}% of max)
• Warmup type: Linear (0 → max LR)
• Decay type: Cosine (max LR → min LR)
"""
    fig.text(0.02, 0.02, info_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize learning rate schedule")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Number of warmup steps (default: 500)")
    parser.add_argument("--total_steps", type=int, default=5000,
                        help="Total training steps (default: 5000)")
    parser.add_argument("--max_lr", type=float, default=6e-4,
                        help="Maximum learning rate (default: 6e-4)")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1,
                        help="Minimum LR as ratio of max LR (default: 0.1)")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save plot (default: display only)")

    args = parser.parse_args()

    print("=" * 60)
    print("Learning Rate Schedule Visualization")
    print("=" * 60)
    print(f"Warmup steps: {args.warmup_steps:,}")
    print(f"Total steps: {args.total_steps:,}")
    print(f"Max LR: {args.max_lr}")
    print(f"Min LR: {args.max_lr * args.min_lr_ratio:.2e}")
    print(f"Warmup phase: {args.warmup_steps/args.total_steps*100:.1f}% of training")
    print("=" * 60)

    visualize_schedule(
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        max_lr=args.max_lr,
        min_lr_ratio=args.min_lr_ratio,
        save_path=args.save
    )


if __name__ == "__main__":
    main()
