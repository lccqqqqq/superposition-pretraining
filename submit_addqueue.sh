#!/bin/bash
# Submit FP8 training job using addqueue
#
# This script submits a training job to the cluster using addqueue.
# Adjust the parameters below for your cluster and requirements.

# ============================================================================
# Configuration
# ============================================================================

# Queue and resource settings
QUEUE="sondhigpu"
TIME_HOURS=30          # Time limit in hours
MEMORY_GB=20           # Memory in GB per core
GPU_COUNT=1            # Number of GPUs
GPU_TYPE="h200with141gb"  # GPU type
COMMENT="pt-fp8"     # Container/environment name

# Training config file (can be overridden)
CONFIG=${1:-configs/fineweb_edu_fp8.yaml}

# ============================================================================
# Submit Job
# ============================================================================

echo "Submitting FP8 training job..."
echo "  Queue: $QUEUE"
echo "  Time: $TIME_HOURS hours"
echo "  Memory: $MEMORY_GB GB"
echo "  GPUs: $GPU_COUNT x $GPU_TYPE"
echo "  Comment: $COMMENT"
echo "  Config: $CONFIG"
echo ""

addqueue -q $QUEUE \
    -s \
    -n $TIME_HOURS \
    -m $MEMORY_GB \
    -c "$COMMENT" \
    --gpus $GPU_COUNT \
    --gputype $GPU_TYPE \
    ./run_fp8_training.sh $CONFIG

echo ""
echo "Job submitted!"
echo "Monitor with: qstat -u $USER"
