#!/bin/bash
# Environment setup script for Advanced Manipulation Transformer

# Set DexYCB dataset path
export DEX_YCB_DIR="/home/n231/231nProjectV2/dex-ycb-toolkit/data"

# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:/home/n231/231nProjectV2/Advanced_Manipulation_Transformer"
export PYTHONPATH="${PYTHONPATH}:/home/n231/231nProjectV2/dex-ycb-toolkit"

echo "Environment variables set:"
echo "DEX_YCB_DIR=$DEX_YCB_DIR"
echo "PYTHONPATH=$PYTHONPATH"

# Optional: Set W&B project name
export WANDB_PROJECT="advanced-manipulation-transformer"

echo ""
echo "Environment ready! You can now run:"
echo "  python run_sweep.py --count 50"
echo "  ./start_sweep.sh"