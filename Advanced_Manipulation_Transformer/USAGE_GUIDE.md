# Advanced Manipulation Transformer - Usage Guide

## Command Line Arguments

### Running train_advanced.py

The training script uses Hydra for configuration management, which provides powerful command-line override capabilities.

#### Basic Usage
```bash
python train_advanced.py
```

#### Configuration Overrides

You can override any configuration parameter from the command line:

```bash
# Change batch size and learning rate
python train_advanced.py training.batch_size=64 training.learning_rate=5e-4

# Enable specific optimizations
python train_advanced.py \
    optimizations.use_flash_attention=true \
    optimizations.use_fp8=true \
    optimizations.use_mode_collapse_prevention=true

# Change data paths
python train_advanced.py \
    data.root_dir=/path/to/dexycb \
    data.train_split=s1_train \
    data.val_split=s1_val

# Adjust model architecture
python train_advanced.py \
    model.hidden_dim=2048 \
    model.num_refinement_steps=3 \
    model.dropout=0.2

# Control training duration
python train_advanced.py \
    training.num_epochs=50 \
    training.val_freq=500 \
    training.save_freq=5

# Enable debugging
python train_advanced.py \
    debug.enabled=true \
    debug.log_gradient_norms=true \
    debug.save_attention_maps=true

# Resume from checkpoint
python train_advanced.py \
    checkpoint.resume_from=/path/to/checkpoint.pth

# Change experiment name and output directory
python train_advanced.py \
    experiment_name=my_experiment \
    output_dir=outputs/my_experiment
```

#### Multi-GPU Training

```bash
# Single node, multi-GPU with DDP
torchrun --nproc_per_node=4 train_advanced.py

# Multi-node training
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_backend=c10d \
    --rdzv_endpoint=hostname:port train_advanced.py

# Use FSDP for very large models
python train_advanced.py optimizations.use_fsdp=true
```

#### Advanced Configuration

```bash
# Disable specific components
python train_advanced.py \
    training.use_ema=false \
    training.mixed_precision=false \
    optimizations.use_memory_optimization=false

# Adjust loss weights dynamically
python train_advanced.py \
    loss.loss_weights.diversity=0.05 \
    loss.loss_weights.hand_refined=2.0 \
    loss.loss_weights.physics=0.2

# Configure data augmentation
python train_advanced.py \
    data.augmentation.rotation_range=30.0 \
    data.augmentation.scale_range=[0.7,1.3] \
    data.augmentation.joint_noise_std=0.01

# Set multi-rate learning
python train_advanced.py \
    training.multi_rate.pretrained=0.001 \
    training.multi_rate.new_encoders=0.5 \
    training.multi_rate.decoders=1.0

# Configure memory optimization
python train_advanced.py \
    optimizations.memory.gradient_checkpointing=true \
    optimizations.memory.checkpoint_ratio=0.75 \
    optimizations.memory.dynamic_batch_sizing=true \
    optimizations.memory.target_memory_usage=0.95
```

### Running inference.py

The inference script provides various options for real-time prediction:

```bash
# Basic single image inference
python inference.py checkpoint.pth --input image.jpg --output result.jpg

# Video processing
python inference.py checkpoint.pth --input video.mp4 --output output_video.mp4

# Disable optimizations for debugging
python inference.py checkpoint.pth --input image.jpg --no-fp16 --no-flash

# Use CPU instead of GPU
python inference.py checkpoint.pth --input image.jpg --device cpu

# Run performance benchmark
python inference.py checkpoint.pth --benchmark

# Specify config file
python inference.py checkpoint.pth --config path/to/config.yaml --input image.jpg
```

## Configuration File Structure

### default_config.yaml

The main configuration file contains all adjustable parameters:

```yaml
# Experiment settings
experiment_name: "advanced_manipulation_transformer"
output_dir: "outputs/${experiment_name}"
wandb_project: "231nProject_AMT"

# Data settings
data:
  root_dir: "../dex-ycb-toolkit"
  train_split: "s0_train"
  val_split: "s0_val"
  sequence_length: 1  # Set >1 for temporal modeling
  num_workers: 8
  prefetch_factor: 4
  persistent_workers: true
  
  # Augmentation settings
  augmentation:
    rotation_range: 15.0  # degrees
    scale_range: [0.8, 1.2]
    translation_std: 0.05
    color_jitter: 0.2
    joint_noise_std: 0.005  # 5mm

# Model architecture
model:
  # DINOv2 settings
  freeze_layers: 12
  
  # Hidden dimensions
  hidden_dim: 1024
  contact_hidden_dim: 512
  
  # Model components
  use_mano_vertices: true
  use_sigma_reparam: true
  use_attention_fusion: true
  
  # Refinement settings
  num_refinement_steps: 2
  
  # Object and contact settings
  max_objects: 10
  num_object_classes: 100
  num_contact_points: 10
  
  # Regularization
  dropout: 0.1

# Training settings
training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 1e-3
  weight_decay: 0.01
  
  # Mixed precision
  use_amp: true
  use_bf16: true
  mixed_precision: true
  
  # Optimization
  accumulation_steps: 2
  grad_clip: 1.0
  ema_decay: 0.999
  use_ema: true
  
  # Multi-rate learning
  multi_rate:
    pretrained: 0.01
    new_encoders: 0.5
    decoders: 1.0
  
  # Scheduler
  scheduler: "cosine"
  T_0: 10
  T_mult: 2
  min_lr: 1e-6
  
  # Logging
  log_freq: 100
  val_freq: 1000
  save_freq: 10
  use_wandb: true

# Loss settings
loss:
  loss_weights:
    hand_coarse: 1.0
    hand_refined: 1.2
    object_position: 1.0
    object_rotation: 0.5
    contact: 0.3
    physics: 0.1
    diversity: 0.01
    reprojection: 0.5
    kl: 0.001
  
  # Loss-specific settings
  diversity_margin: 0.01
  object_position_weight: 1.0
  per_joint_weighting: true
  fingertip_weight: 1.5

# Evaluation settings
evaluation:
  metrics: ["mpjpe", "pa_mpjpe", "pck_2d", "pck_3d"]
  pck_thresholds: [20, 30, 40, 50]
  save_visualizations: true

# Optimization settings
optimizations:
  # H200 GPU optimizations
  use_flash_attention: true
  use_fp8: false
  use_memory_optimization: true
  use_mode_collapse_prevention: true
  
  # Distributed training
  use_fsdp: false
  
  memory:
    gradient_checkpointing: true
    checkpoint_ratio: 0.5
    dynamic_batch_sizing: true
    target_memory_usage: 0.9

# Debugging settings
debug:
  enabled: false
  debug_initial_model: true
  debug_final_model: true
  save_attention_maps: false
  log_gradient_norms: true

# Checkpoint settings
checkpoint:
  resume_from: null
  checkpoint_dir: "${output_dir}/checkpoints"
```

## Environment Variables

Some features can be controlled via environment variables:

```bash
# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Enable TF32 for Ampere/Hopper GPUs
export TORCH_ALLOW_TF32=1

# Set NCCL backend for distributed training
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not available

# PyTorch settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Hydra settings
export HYDRA_FULL_ERROR=1  # Show full error traces
```

## Logging and Monitoring

### Weights & Biases Integration

When `training.use_wandb=true`, the following metrics are logged:
- Training/validation losses (all components)
- Learning rates
- Gradient norms
- GPU utilization and memory usage
- Sample predictions and visualizations

### TensorBoard

Even without W&B, metrics are logged to TensorBoard:
```bash
tensorboard --logdir outputs/experiment_name/
```

### Debug Outputs

When `debug.enabled=true`, additional files are saved:
- `debug_outputs/activation_distributions.png`
- `debug_outputs/gradient_flow.png`
- `debug_outputs/parameter_distributions.png`
- `debug_outputs/prediction_diversity.png`
- Attention maps (if enabled)

## Performance Tuning

### Memory Optimization

For large models or batch sizes:
```bash
python train_advanced.py \
    optimizations.memory.gradient_checkpointing=true \
    optimizations.memory.checkpoint_ratio=0.75 \
    training.batch_size=16 \
    training.accumulation_steps=4
```

### Speed Optimization

For maximum training speed:
```bash
python train_advanced.py \
    optimizations.use_flash_attention=true \
    optimizations.use_fp8=true \
    training.use_amp=true \
    data.num_workers=16 \
    data.prefetch_factor=8
```

### Debugging Mode

For easier debugging:
```bash
python train_advanced.py \
    debug.enabled=true \
    training.batch_size=4 \
    training.num_epochs=5 \
    optimizations.use_flash_attention=false \
    optimizations.use_memory_optimization=false \
    training.mixed_precision=false
```

## Common Workflows

### 1. Quick Testing
```bash
# Test on small subset with debugging
python train_advanced.py \
    training.num_epochs=3 \
    training.batch_size=8 \
    debug.enabled=true
```

### 2. Production Training
```bash
# Full training with all optimizations
torchrun --nproc_per_node=4 train_advanced.py \
    optimizations.use_flash_attention=true \
    optimizations.use_memory_optimization=true \
    training.batch_size=64 \
    training.num_epochs=100
```

### 3. Fine-tuning
```bash
# Resume from checkpoint with lower learning rate
python train_advanced.py \
    checkpoint.resume_from=checkpoints/best_model.pth \
    training.learning_rate=1e-4 \
    training.num_epochs=20
```

### 4. Ablation Studies
```bash
# Disable specific components
python train_advanced.py \
    experiment_name=no_refinement \
    model.num_refinement_steps=0

python train_advanced.py \
    experiment_name=no_diversity \
    loss.loss_weights.diversity=0.0
```

## Troubleshooting

### Out of Memory
- Reduce batch size
- Enable gradient checkpointing
- Use FSDP for model sharding
- Enable dynamic batch sizing

### Slow Training
- Enable FlashAttention
- Use more data workers
- Enable persistent workers
- Check GPU utilization

### Mode Collapse
- Increase diversity loss weight
- Enable mode collapse prevention
- Check sigma reparameterization
- Use larger batch size

### Poor Accuracy
- Increase refinement steps
- Check data augmentation
- Verify camera parameters
- Increase model capacity