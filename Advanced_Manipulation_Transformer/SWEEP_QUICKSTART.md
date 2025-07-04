# Quick Start Guide for W&B Sweeps

## Environment Setup

### 1. Set Environment Variables
```bash
# Option 1: Source the setup script
source setup_env.sh

# Option 2: Set manually
export DEX_YCB_DIR="/home/n231/231nProjectV2/dex-ycb-toolkit/data"
```

### 2. Activate Python Environment
Make sure you have activated the correct Python environment with PyTorch installed:
```bash
# Activate your conda/venv environment that has PyTorch
conda activate your_env_name
# or
source /path/to/venv/bin/activate
```

### 3. Install Missing Dependencies
```bash
pip install pyyaml wandb
```

## Running Sweeps

### Option 1: Using the Bash Script
```bash
# Run with defaults (50 runs)
./start_sweep.sh

# Custom number of runs
./start_sweep.sh --num-runs 100

# Join existing sweep
./start_sweep.sh --sweep-id YOUR_SWEEP_ID --num-runs 20
```

### Option 2: Using Python Directly
```bash
# Create new sweep and run 50 experiments
python run_sweep.py --project advanced-manipulation-transformer --count 50

# Join existing sweep
python run_sweep.py --sweep_id YOUR_SWEEP_ID --count 10
```

## Common Issues

### 1. DEX_YCB_DIR not set
The scripts automatically set this, but you can also:
```bash
export DEX_YCB_DIR="/home/n231/231nProjectV2/dex-ycb-toolkit/data"
```

### 2. Module not found errors
Make sure you're in the correct Python environment with PyTorch installed.

### 3. W&B not logged in
```bash
wandb login
```

### 4. Enable W&B in config
The bash script does this automatically, but you can manually edit:
```yaml
# In configs/default_config.yaml
training:
  use_wandb: true  # Change from false
```

## Monitoring Progress

1. **Terminal**: Watch real-time training logs
2. **W&B Dashboard**: https://wandb.ai/YOUR_USERNAME/advanced-manipulation-transformer/sweeps
3. **GPU Usage**: `nvidia-smi -l 1` in another terminal

## Expected Behavior

- Each sweep run will try different hyperparameters
- Bayesian optimization will focus on promising regions
- Early stopping (Hyperband) will kill bad runs early
- Best model checkpoints are saved automatically
- Target metric: Minimize validation hand MPJPE (< 100mm is good)

## Sweep Configuration

The sweep optimizes:
- Learning rate (1e-5 to 1e-2)
- Batch size (16, 32, 64)
- Model architecture (hidden dims, layers)
- Loss weights (hand, object, contact, etc.)
- Augmentation parameters
- Optimization features (attention fusion, sigma reparam)

See `configs/sweep_config.yaml` for full details.