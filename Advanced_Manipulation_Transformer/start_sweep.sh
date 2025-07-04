#!/bin/bash
# Weights & Biases Sweep Runner Script

# Default values
PROJECT_NAME="advanced-manipulation-transformer"
NUM_RUNS=50

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --project)
            PROJECT_NAME="$2"
            shift 2
            ;;
        --num-runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        --sweep-id)
            SWEEP_ID="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --project NAME     W&B project name (default: advanced-manipulation-transformer)"
            echo "  --num-runs COUNT   Number of sweep runs (default: 50)"
            echo "  --sweep-id ID      Join existing sweep with this ID"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set DexYCB dataset path
export DEX_YCB_DIR="/home/n231/231nProjectV2/dex-ycb-toolkit/data"

# Make sure we're in the right directory
cd /home/n231/231nProjectV2/Advanced_Manipulation_Transformer

# Check if wandb is logged in
if ! wandb verify; then
    echo "Please log in to Weights & Biases first:"
    echo "wandb login"
    exit 1
fi

# Enable W&B in config if not already enabled
echo "Checking W&B configuration..."
if grep -q "use_wandb: false" configs/default_config.yaml; then
    echo "Enabling W&B in configuration..."
    sed -i 's/use_wandb: false/use_wandb: true/g' configs/default_config.yaml
fi

# Run the sweep
if [ -z "$SWEEP_ID" ]; then
    echo "Creating new sweep for project: $PROJECT_NAME"
    echo "Running $NUM_RUNS experiments..."
    python run_sweep.py --project "$PROJECT_NAME" --count "$NUM_RUNS"
else
    echo "Joining existing sweep: $SWEEP_ID"
    echo "Running $NUM_RUNS experiments..."
    python run_sweep.py --sweep_id "$SWEEP_ID" --project "$PROJECT_NAME" --count "$NUM_RUNS"
fi