# W&B Sweep Configuration Guide

This directory contains the Weights & Biases (W&B) sweep configuration for hyperparameter optimization of the Advanced Manipulation Transformer.

## Files

- `sweep_config.yaml`: Main sweep configuration file defining the hyperparameter search space
- `default_config.yaml`: Base configuration that gets modified by sweep parameters

## Quick Start

### 1. Create a New Sweep

```bash
# From the project root directory
python run_sweep.py --project my-sweep-project --count 100
```

This will:
- Create a new W&B sweep
- Print the sweep ID and URL
- Start running 100 sweep runs

### 2. Join an Existing Sweep

```bash
# Join a sweep on multiple machines/GPUs
python run_sweep.py --sweep_id SWEEP_ID --project my-sweep-project --count 10
```

### 3. Monitor Progress

Visit the W&B dashboard URL printed when creating the sweep to monitor:
- Real-time metrics
- Hyperparameter importance
- Best performing runs
- Parallel coordinates plot

## Hyperparameters Being Swept

### Model Architecture
- `hidden_dim`: [512, 768, 1024] - Main hidden dimension
- `contact_hidden_dim`: [256, 384, 512] - Contact decoder hidden dimension
- `dropout`: [0.1, 0.2, 0.3] - Dropout rate
- `num_refinement_steps`: [1, 2, 3] - Number of iterative refinement steps
- `freeze_layers`: [8, 10, 12] - Number of DINOv2 layers to freeze

### Training
- `learning_rate`: 1e-5 to 1e-2 (log scale) - Base learning rate
- `batch_size`: [16, 32, 64] - Training batch size
- `weight_decay`: 0.0001 to 0.1 (log scale) - Weight decay
- `grad_clip`: 0.5 to 2.0 - Gradient clipping value

### Loss Weights
- `loss_weight_hand_coarse`: 0.5 to 2.0 - Coarse hand prediction loss
- `loss_weight_hand_refined`: 0.8 to 2.5 - Refined hand prediction loss
- `loss_weight_object_position`: 0.5 to 2.0 - Object position loss
- `loss_weight_object_rotation`: 0.2 to 1.0 - Object rotation loss
- `loss_weight_contact`: 0.1 to 0.5 - Contact prediction loss
- `loss_weight_physics`: 0.05 to 0.2 - Physics-based loss
- `loss_weight_diversity`: 0.001 to 0.1 (log scale) - Diversity regularization
- `loss_weight_reprojection`: 0.2 to 1.0 - 2D reprojection loss
- `loss_weight_kl`: 0.0001 to 0.01 (log scale) - KL divergence for sigma reparam

### Data Augmentation
- `aug_rotation_range`: 10° to 30° - Random rotation range
- `aug_scale_min/max`: 0.7-0.9 to 1.1-1.3 - Random scale range
- `aug_translation_std`: 0.02 to 0.1 - Translation noise std
- `aug_color_jitter`: 0.1 to 0.3 - Color jitter strength
- `aug_joint_noise_std`: 0.002 to 0.01 - Joint position noise (2-10mm)

### Other Parameters
- Multi-rate learning multipliers for different parameter groups
- Scheduler settings (T_0, min_lr)
- EMA decay rate
- Memory optimization settings

## Sweep Configuration

The sweep uses **Bayesian optimization** (`method: bayes`) to efficiently explore the hyperparameter space. The optimization goal is to minimize `val/hand_mpjpe` (validation hand mean per-joint position error).

### Early Stopping

The sweep uses Hyperband early stopping with:
- `s`: 2 (max early stopping rate)
- `eta`: 3 (halving rate)
- `max_iter`: 27 (maximum iterations)

This helps terminate poorly performing runs early to save compute.

## Advanced Usage

### Modify Search Space

Edit `sweep_config.yaml` to:
- Add new parameters
- Change parameter ranges
- Switch between discrete values and distributions
- Change the optimization metric or method

### Custom Sweep Logic

Modify `run_sweep.py` to:
- Add custom preprocessing
- Implement different training strategies
- Add custom metrics
- Change data loading strategy

### Distributed Sweeps

Run the sweep agent on multiple GPUs/machines:

```bash
# Machine 1
CUDA_VISIBLE_DEVICES=0 python run_sweep.py --sweep_id SWEEP_ID --count 25

# Machine 2 
CUDA_VISIBLE_DEVICES=1 python run_sweep.py --sweep_id SWEEP_ID --count 25

# Continue for more GPUs...
```

## Tips for Effective Sweeps

1. **Start Small**: Run a small sweep (10-20 runs) first to verify everything works
2. **Monitor Early**: Check the sweep dashboard early to catch any issues
3. **Use Compute Efficiently**: 
   - Enable early stopping
   - Use smaller validation sets for faster evaluation
   - Consider reducing training epochs for initial exploration
4. **Analyze Results**: Use W&B's built-in analysis tools:
   - Parallel coordinates plot
   - Parameter importance
   - Correlation matrix
5. **Iterate**: Based on initial results, refine the search space and run focused sweeps

## Example Commands

```bash
# Quick test sweep (10 runs)
python run_sweep.py --count 10

# Production sweep (100 runs)
python run_sweep.py --project amt-hyperparam-search --count 100

# Join existing sweep on 4 GPUs
CUDA_VISIBLE_DEVICES=0 python run_sweep.py --sweep_id abc123 --count 25 &
CUDA_VISIBLE_DEVICES=1 python run_sweep.py --sweep_id abc123 --count 25 &
CUDA_VISIBLE_DEVICES=2 python run_sweep.py --sweep_id abc123 --count 25 &
CUDA_VISIBLE_DEVICES=3 python run_sweep.py --sweep_id abc123 --count 25 &
```

## Analyzing Results

After the sweep completes:

1. **Best Hyperparameters**: Check the sweep dashboard for the best performing run
2. **Download Config**: Use `wandb.api` to download the best configuration
3. **Retrain**: Train the final model with the best hyperparameters for more epochs
4. **Ablation Studies**: Run targeted sweeps on important parameters

## Troubleshooting

- **OOM Errors**: Reduce batch sizes in the sweep range or limit GPU cache size
- **Slow Runs**: Check if data loading is the bottleneck, consider pre-caching
- **Poor Results**: Expand search ranges or add more parameters to sweep
- **W&B Sync Issues**: Use `WANDB_MODE=offline` for offline logging, then sync later