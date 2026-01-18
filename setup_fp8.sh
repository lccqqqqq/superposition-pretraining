#!/bin/bash
# Setup script for FP8 training on H100/H200 GPUs
#
# This script installs required dependencies and configures accelerate for FP8 training.
#
# Usage:
#   bash setup_fp8.sh
#   # Or make it executable and run:
#   chmod +x setup_fp8.sh
#   ./setup_fp8.sh

set -e  # Exit on error

echo "========================================"
echo "Setting up FP8 Training Environment"
echo "========================================"

# Check GPU
echo ""
echo "Checking GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. NVIDIA GPU required for FP8 training."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "Detected GPU: $GPU_NAME"

if [[ "$GPU_NAME" == *"H100"* ]] || [[ "$GPU_NAME" == *"H200"* ]]; then
    echo "✓ GPU supports FP8 training (Hopper architecture)"
else
    echo "⚠ Warning: FP8 training requires H100 or H200 GPU"
    echo "  Your GPU: $GPU_NAME"
    echo "  FP8 will not work, but BF16 training will still be available."
fi

# Check CUDA version
echo ""
echo "Checking CUDA version..."
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
echo "CUDA Version: $CUDA_VERSION"

if [[ "$CUDA_VERSION" < "12.0" ]]; then
    echo "⚠ Warning: FP8 requires CUDA 12.0+. You have $CUDA_VERSION"
    echo "  Consider updating CUDA for optimal FP8 performance."
fi

# Install dependencies
echo ""
echo "Installing Python dependencies..."

# Install accelerate
echo "  Installing accelerate..."
pip install accelerate>=0.25.0 --quiet

# Install Transformer Engine
echo "  Installing Transformer Engine..."
if pip show transformer-engine &> /dev/null; then
    echo "  → Transformer Engine already installed"
else
    # Try to install from git (most up-to-date)
    echo "  → Installing from source (this may take a few minutes)..."
    pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable --quiet || {
        echo "  → Source install failed, trying PyPI..."
        pip install transformer-engine[pytorch] --quiet
    }
fi

# Verify installations
echo ""
echo "Verifying installations..."

python3 << EOF
import sys

# Check accelerate
try:
    import accelerate
    print(f"✓ accelerate {accelerate.__version__} installed")
except ImportError:
    print("✗ accelerate not found")
    sys.exit(1)

# Check Transformer Engine
try:
    import transformer_engine
    print(f"✓ Transformer Engine {transformer_engine.__version__} installed")
except ImportError:
    print("✗ Transformer Engine not found")
    print("  FP8 training will not work")
    sys.exit(1)

# Check for FP8 support
try:
    import torch
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        major, minor = capability
        if major >= 9:  # Hopper is compute capability 9.0
            print(f"✓ GPU supports FP8 (compute capability {major}.{minor})")
        else:
            print(f"⚠ GPU compute capability {major}.{minor} does not support FP8")
            print("  FP8 requires compute capability 9.0+ (H100/H200)")
    else:
        print("⚠ CUDA not available")
except Exception as e:
    print(f"⚠ Could not check GPU capability: {e}")
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "Installation verification failed!"
    exit 1
fi

# Configure accelerate
echo ""
echo "========================================"
echo "Configuring Accelerate for FP8"
echo "========================================"
echo ""
echo "This will create an accelerate config file for FP8 training."
echo "You can reconfigure later with: accelerate config"
echo ""

# Create accelerate config directory
mkdir -p ~/.cache/huggingface/accelerate

# Create FP8 config
cat > ~/.cache/huggingface/accelerate/default_config.yaml << 'EOFCONFIG'
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: fp8
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOFCONFIG

echo "✓ Accelerate configured for FP8 training"
echo ""
echo "Config saved to: ~/.cache/huggingface/accelerate/default_config.yaml"

# Test FP8 setup
echo ""
echo "========================================"
echo "Testing FP8 Setup"
echo "========================================"
echo ""

python3 << 'EOFTEST'
import torch
from accelerate import Accelerator

print("Creating Accelerator with FP8...")
try:
    accelerator = Accelerator(mixed_precision="fp8")
    print(f"✓ Accelerator created successfully")
    print(f"  Device: {accelerator.device}")
    print(f"  Mixed precision: {accelerator.mixed_precision}")
    print(f"  Num processes: {accelerator.num_processes}")

    # Try a simple FP8 operation
    print("\nTesting FP8 computation...")
    model = torch.nn.Linear(10, 10).to(accelerator.device)
    model = accelerator.prepare(model)
    x = torch.randn(4, 10).to(accelerator.device)
    y = model(x)
    print(f"✓ FP8 computation successful")
    print(f"  Input shape: {x.shape}, Output shape: {y.shape}")

except Exception as e:
    print(f"✗ FP8 test failed: {e}")
    print("\nThis might be OK if you don't have H100/H200 GPU.")
    print("You can still use BF16 training.")
EOFTEST

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Test FP8 training:"
echo "     accelerate launch train_fp8.py --config configs/openwebtext_fp8.yaml --no_wandb"
echo ""
echo "  2. Compare FP8 vs BF16:"
echo "     # BF16 baseline"
echo "     python train.py --config configs/openwebtext_adaptive_batch.yaml"
echo "     # FP8 accelerated"
echo "     accelerate launch train_fp8.py --config configs/openwebtext_fp8.yaml"
echo ""
echo "  3. Monitor training:"
echo "     nvidia-smi dmon -s um  # Watch GPU memory and utilization"
echo ""
echo "For more information, see FP8_TRAINING.md"
echo ""
