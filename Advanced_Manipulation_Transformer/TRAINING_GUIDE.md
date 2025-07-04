# Training Guide

This guide provides detailed instructions for training the Advanced Manipulation Transformer model.

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA H200 (140GB) or similar high-memory GPU
- **RAM**: 64GB+ system memory
- **Storage**: 500GB+ for dataset and checkpoints

### Software Requirements
- Python 3.12+
- CUDA 12.4+
- PyTorch 2.5.0+cu124
- All packages in `requirements.txt`

### Dataset Setup
Ensure DexYCB dataset is properly set up:
```bash
# Dataset should be at: ../dex-ycb-toolkit/
# Update path in configs/default_config.yaml if different
```

## Configuration

### Key Parameters in `configs/default_config.yaml`

```yaml
# Model Architecture
model:
  hidden_dim: 1024          # Transformer width
  use_sigma_reparam: true   # CRITICAL: Prevents mode collapse
  num_refinement_steps: 2   # Pixel-aligned refinement iterations
  
# Training Settings  
training:
  batch_size: 32           # Adjust based on GPU memory
  learning_rate: 1e-3      # Base learning rate
  num_epochs: 100          # Total training epochs
  use_amp: true            # Mixed precision training
  use_bf16: true           # BFloat16 for H200 (better than FP16)
  
# Loss Weights (carefully tuned)
loss_weights:
  hand_pose: 1.0           # Primary objective
  hand_pose_refined: 1.2   # Higher weight for refined predictions
  diversity: 0.01          # CRITICAL: Prevents constant predictions
  hand_2d: 0.3            # 2D reprojection for 3D accuracy
```

### Memory Settings for Large Models

```yaml
optimizations:
  memory:
    gradient_checkpointing: true  # Trade compute for memory
    checkpoint_ratio: 0.5         # Checkpoint 50% of layers
    
training:
  accumulation_steps: 2          # Gradient accumulation
  batch_size: 16                 # Reduce if OOM
```

## Training Workflow

### 1. Initial Setup

```bash
# Navigate to project directory
cd Advanced_Manipulation_Transformer

# Verify GPU availability
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Expected output: GPU: NVIDIA H200
```

### 2. Using Jupyter Notebook (Recommended)

The notebook provides the best research experience:

```bash
jupyter notebook notebooks/train_advanced_manipulation.ipynb
```

**Notebook Structure**:
1. **Setup & Config**: Load libraries, set paths
2. **Data Exploration**: Visualize samples, check augmentation
3. **Model Initialization**: Build model, count parameters
4. **Training Loop**: Live metrics and visualization
5. **Evaluation**: Detailed analysis of results

**Key Notebook Features**:
- Live loss plotting
- Sample visualization every N steps
- Interactive debugging
- Easy hyperparameter tuning

### 3. Command Line Training

For long runs or cluster deployment:

```bash
# Basic training
python train.py --config configs/default_config.yaml

# Resume from checkpoint
python train.py --config configs/default_config.yaml \
                --resume checkpoints/latest.pth

# Custom experiment
python train.py --config configs/my_experiment.yaml
```

### 4. Monitoring Training

#### Metrics to Watch

1. **Loss Components**:
   ```
   hand_pose: Should decrease steadily
   diversity: Should stay positive (prevents collapse)
   hand_2d: Indicates 2D-3D consistency
   ```

2. **MPJPE Progress**:
   ```
   Epoch 1-10:   ~100mm (learning basic structure)
   Epoch 10-30:  ~50mm  (refining predictions)
   Epoch 30-50:  ~20mm  (fine details)
   Epoch 50+:    <15mm  (final refinement)
   ```

3. **Warning Signs**:
   - Diversity loss → 0: Mode collapse imminent
   - Hand pose std < 0.001: Constant predictions
   - NaN losses: Learning rate too high
   - No improvement: Check data loading

#### Using Weights & Biases (W&B)

Training automatically logs to W&B:
```python
# In trainer.py, logs are sent to:
wandb.init(project="231nProject_AMT")

# Key metrics logged:
- train/loss (every 100 steps)
- val/mpjpe (every 1000 steps)
- learning rates per component
- gradient norms
```

## Stage-by-Stage Training

### Stage 1: Component Pretraining (Optional)

Train individual components first:

```python
# In notebook or script
# 1. Train hand encoder only
model.freeze_all_except(['hand_encoder', 'hand_decoder'])
train_epochs(20, lr=1e-3)

# 2. Train object encoder
model.freeze_all_except(['object_encoder', 'object_decoder'])
train_epochs(20, lr=1e-3)

# 3. Unfreeze all for joint training
model.unfreeze_all()
```

### Stage 2: Joint Training (Main Phase)

This is where most learning happens:

```python
# Full model training with all losses
config['training']['num_epochs'] = 50
config['training']['learning_rate'] = 5e-4

# Key settings for this stage
config['loss_weights']['diversity'] = 0.02  # Higher to prevent collapse
config['loss_weights']['physics'] = 0.0    # Not yet, too constraining
```

### Stage 3: Fine-tuning

Polish the model with constraints:

```python
# Lower learning rate, add physics
config['training']['learning_rate'] = 1e-4
config['loss_weights']['physics_penetration'] = 0.1
config['loss_weights']['diversity'] = 0.005  # Can reduce now

# Train final 30 epochs
train_epochs(30)
```

## Troubleshooting

### Problem: Mode Collapse (std = 0.0003)

**Symptoms**: All predictions identical, very low standard deviation

**Solutions**:
1. Increase diversity loss weight:
   ```yaml
   loss_weights:
     diversity: 0.05  # Increase from 0.01
   ```

2. Check sigma reparameterization is active:
   ```python
   assert model.use_sigma_reparam == True
   ```

3. Add noise to inputs:
   ```python
   x = x + 0.01 * torch.randn_like(x)
   ```

### Problem: High MPJPE (>100mm)

**Symptoms**: Poor 3D accuracy despite good 2D projections

**Solutions**:
1. Increase refinement steps:
   ```yaml
   model:
     num_refinement_steps: 3  # From 2
   ```

2. Increase 2D reprojection weight:
   ```yaml
   loss_weights:
     hand_2d: 0.5  # From 0.3
   ```

3. Check camera parameters are correct

### Problem: Out of Memory (OOM)

**Solutions by effectiveness**:

1. **Reduce batch size**:
   ```yaml
   training:
     batch_size: 16  # From 32
     accumulation_steps: 4  # Compensate with accumulation
   ```

2. **Enable gradient checkpointing**:
   ```yaml
   optimizations:
     memory:
       gradient_checkpointing: true
   ```

3. **Use smaller model**:
   ```yaml
   model:
     hidden_dim: 768  # From 1024
   ```

4. **Mixed precision**:
   ```yaml
   training:
     use_amp: true
     use_bf16: true  # Better than fp16 for H200
   ```

### Problem: Slow Training

**Solutions**:

1. **Check data loading**:
   ```python
   # In dataset
   use_cache: true  # Cache preprocessed data
   num_workers: 8   # Parallel data loading
   ```

2. **Optimize model**:
   ```python
   # Compile model (PyTorch 2.0+)
   model = torch.compile(model, mode='reduce-overhead')
   ```

3. **Profile bottlenecks**:
   ```python
   with torch.profiler.profile() as prof:
       output = model(batch)
   print(prof.key_averages())
   ```

## Best Practices

### 1. Experiment Tracking

Always use descriptive experiment names:
```yaml
experiment_name: "amt_dinov2_multicoord_bf16_div0.02"
# Not: "test_1"
```

### 2. Regular Checkpointing

```yaml
training:
  save_freq: 10  # Save every 10 epochs
  val_freq: 1000  # Validate every 1000 steps
```

### 3. Learning Rate Scheduling

The model uses cosine annealing with warm restarts:
```yaml
scheduler: "cosine"
T_0: 10        # Restart every 10 epochs
T_mult: 2      # Double period each restart
```

### 4. Data Augmentation

Essential for generalization:
```yaml
data:
  augment: true
  # Includes: rotation, scaling, translation, color jitter
```

## Expected Timeline

On H200 GPU with default settings:

- **Epoch time**: ~30 minutes
- **Total training**: ~50 hours (100 epochs)
- **Validation**: ~5 minutes per run

To speed up:
- Reduce validation frequency
- Use larger batch size (if memory allows)
- Use torch.compile()

## Evaluation

### During Training

The model automatically computes:
- MPJPE (Mean Per Joint Position Error)
- PA-MPJPE (Procrustes Aligned)
- PCK@5px (Percentage of Correct Keypoints)

### Post-Training Evaluation

```python
# Load best checkpoint
model.load_state_dict(torch.load('checkpoints/best.pth'))

# Run comprehensive evaluation
from evaluation.metrics import evaluate_all
results = evaluate_all(model, test_loader)

print(f"MPJPE: {results['mpjpe']:.2f}mm")
print(f"PA-MPJPE: {results['pa_mpjpe']:.2f}mm")
print(f"PCK@5px: {results['pck_5']:.2%}")
```

## Tips for Success

1. **Start with smaller experiments**: Test on 10% of data first
2. **Monitor diversity loss**: Should never go to zero
3. **Visualize predictions**: Catch issues early
4. **Save everything**: Config, code version, random seeds
5. **Use EMA model for evaluation**: More stable than raw model

## Next Steps

After successful training:
1. Analyze failure cases
2. Try different augmentations
3. Experiment with loss weights
4. Add temporal modeling for video
5. Deploy for real-time inference