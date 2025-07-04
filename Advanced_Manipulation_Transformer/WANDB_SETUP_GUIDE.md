# Weights & Biases (W&B) Setup Guide for Advanced Manipulation Transformer

## Current Status

W&B is **integrated** but currently **disabled** in the configuration. The sweep configuration has been verified to match the model structure.

## Quick Start

### 1. Enable W&B in Configuration

Edit `/home/n231/231nProjectV2/Advanced_Manipulation_Transformer/configs/default_config.yaml`:
```yaml
training:
  use_wandb: true  # Change from false to true
```

### 2. Set W&B API Key

```bash
# Option 1: Interactive login
wandb login

# Option 2: Set environment variable
export WANDB_API_KEY=your_api_key_here
```

### 3. Run a Sweep

```bash
# Create and run a new sweep with 50 runs
python run_sweep.py --project advanced-manipulation-transformer --count 50

# Or join an existing sweep
python run_sweep.py --sweep_id YOUR_SWEEP_ID --count 10
```

## Sweep Configuration Overview

The sweep configuration (`configs/sweep_config.yaml`) uses **Bayesian optimization** with **Hyperband early stopping** for efficient hyperparameter search.

### Key Parameters Being Optimized

1. **Model Architecture**
   - `hidden_dim`: [512, 768, 1024]
   - `contact_hidden_dim`: [256, 512, 768]
   - `dropout`: [0.1, 0.2, 0.3]
   - `num_refinement_steps`: [1, 2, 3]
   - `freeze_layers`: [8, 10, 12] (DINOv2 layers)

2. **Training Parameters**
   - `learning_rate`: 1e-5 to 1e-2 (log scale)
   - `batch_size`: [16, 32, 64]
   - `weight_decay`: 0.0001 to 0.1 (log scale)
   - `grad_clip`: 0.5 to 2.0

3. **Loss Weights** (all properly mapped)
   - `loss_weight_hand_coarse`: 0.5 to 2.0
   - `loss_weight_hand_refined`: 0.8 to 2.5
   - `loss_weight_object_position`: 0.5 to 2.0
   - `loss_weight_contact`: 0.1 to 0.5
   - And more...

4. **Optimization Features**
   - `use_attention_fusion`: [true, false]
   - `use_sigma_reparam`: [true, false]
   - `gradient_checkpointing`: [true, false]

## How W&B is Integrated

### 1. Training Notebook
In `notebooks/train_full_featured.ipynb`, W&B logging is integrated:
```python
if config.training.use_wandb:
    wandb.init(
        project="advanced-manipulation-transformer",
        name=config.experiment_name,
        config=OmegaConf.to_container(config)
    )
    wandb.watch(model, log_freq=100)
```

### 2. Trainer Class
The `ManipulationTrainer` class logs metrics to W&B:
- Training/validation losses
- Hand MPJPE metrics
- Learning rates
- GPU memory usage

### 3. Sweep Runner
`run_sweep.py` handles:
- Loading sweep configuration
- Updating model config with sweep parameters
- Running training with W&B logging
- Early stopping based on validation metrics

## Sweep Metrics

The sweep optimizes for **minimizing validation hand MPJPE** (Mean Per Joint Position Error in mm).

### Logged Metrics
- `train/loss`: Training loss
- `train/hand_mpjpe`: Training hand MPJPE (mm)
- `val/loss`: Validation loss
- `val/hand_mpjpe`: Validation hand MPJPE (mm) - **PRIMARY METRIC**
- `system/gpu_memory_gb`: GPU memory usage

## Advanced Usage

### Custom Sweep Configuration

Create your own sweep config:
```yaml
method: grid  # or random, bayes
metric:
  name: val/hand_mpjpe
  goal: minimize
parameters:
  learning_rate:
    values: [1e-4, 5e-4, 1e-3]
  batch_size:
    values: [32, 64]
```

### Parallel Sweep Agents

Run multiple agents in parallel:
```bash
# Terminal 1
python run_sweep.py --sweep_id SWEEP_ID --count 10

# Terminal 2 (same sweep)
python run_sweep.py --sweep_id SWEEP_ID --count 10
```

### Sweep Analysis

View results at: https://wandb.ai/YOUR_USERNAME/advanced-manipulation-transformer/sweeps

W&B provides:
- Parallel coordinates plot
- Hyperparameter importance
- Best run configurations
- Training curves comparison

## Troubleshooting

### GPU Memory Issues
The sweep uses GPU-cached datasets with reduced samples:
- Training: 10,000 samples max
- Validation: 2,000 samples max

Adjust in `run_sweep.py` if needed:
```python
gpu_config = {
    'gpu_max_samples': 10000,  # Increase if you have more GPU memory
    'gpu_max_samples_val': 2000,
}
```

### Early Stopping
Configured with Hyperband:
- `s`: 2 (speedup factor)
- `eta`: 3 (reduction factor)
- `max_iter`: 27 (max epochs)

### Loss Weight Mapping
The sweep uses `loss_weight_X` format which is correctly mapped to `config.loss.loss_weights.X` in the `update_config_from_sweep()` function.

## Best Practices

1. **Start Small**: Run a few test sweeps with `--count 2` to ensure everything works
2. **Monitor GPU**: The sweep logs GPU memory usage - watch for OOM errors
3. **Use Bayesian Optimization**: More efficient than random/grid search
4. **Enable Mixed Precision**: Already configured with BFloat16 for H200 GPUs
5. **Check Sweep Progress**: View live results on W&B dashboard

## Next Steps

1. Enable W&B in your config
2. Run `wandb login`
3. Start a sweep with `python run_sweep.py --count 50`
4. Monitor results on W&B dashboard
5. Use best hyperparameters for final training

The sweep is configured to find optimal hyperparameters for your hand manipulation model, focusing on minimizing hand pose prediction error while maintaining stable training.