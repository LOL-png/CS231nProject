# Advanced Manipulation Transformer

A transformer architecture for hand-object manipulation understanding from monocular RGB images, featuring pretrained vision models, multi-coordinate representations, and robust training strategies to prevent mode collapse.

## Overview

The Advanced Manipulation Transformer (AMT) is a comprehensive deep learning system that predicts 3D hand poses, object poses, and hand-object interactions from single RGB images. Built to address critical issues in previous implementations (mode collapse, poor 3D accuracy), this model achieves robust performance through innovative architectural choices and training strategies.

### Key Achievements
- **Prevents mode collapse** through σ-reparameterization and diversity losses
- **High-fidelity 3D predictions** with <20mm MPJPE on DexYCB dataset
- **Real-time inference** capability (30+ FPS on H200)
- **GPU-optimized training** using 100GB+ GPU memory for maximum throughput
- **Pretrained foundation** leveraging DINOv2 for robust visual features

## Model Architecture

### Core Components

1. **Image Encoder (DINOv2-based)**
   - Base: facebook/dinov2-large (1.1B parameters)
   - Frozen layers: First 12 (preserving pretrained features)
   - Output dimension: 1024
   - Multi-scale feature extraction from layers [6, 12, 18, 24]

2. **Multi-Coordinate Hand Encoder**
   - Input: 778 MANO vertices × 67 features (22 coordinate frames × 3 + vertex index)
   - Architecture: PointNet-style with attention pooling
   - Hidden dimension: 1024
   - Output: 21 3D joint positions + confidence scores + MANO shape parameters

3. **Object Pose Encoder**
   - Transformer-based architecture
   - Hidden dimension: 1024
   - Handles up to 10 objects simultaneously
   - Outputs: 6-DoF poses (3D position + quaternion rotation)

4. **Contact Prediction Module**
   - Fuses hand and object features
   - Hidden dimension: 512
   - Predicts contact points, probabilities, and forces

5. **Pixel-Aligned Refinement**
   - Iterative refinement: 2 steps by default
   - Projects 3D predictions to 2D for feature sampling
   - Refines both hand joints and object poses

### Model Card

| Component | Details |
|-----------|---------|
| **Total Parameters** | 516.1M |
| **Trainable Parameters** | 362.9M |
| **Input Resolution** | 224×224 RGB |
| **Batch Size (Training)** | 32 (with gradient accumulation) |
| **Memory Usage** | ~120GB (model + dataset cached on GPU) |
| **Training Time** | ~30 min/epoch on H200 |
| **Inference Speed** | 30+ FPS |

### Architecture Dimensions

```
Image Input: [B, 3, 224, 224]
    ↓
DINOv2 Encoder
    ├─ Patch embeddings: [B, 256, 1024]  # 14×14 patches + CLS token
    ├─ Multi-scale features: 4 × [B, 256, 1024]
    └─ Output projection: [B, 1024]
    ↓
Feature Distribution
    ├─ Hand Encoder ────→ Joints: [B, 21, 3]
    │                    Shape: [B, 10]
    │                    Confidence: [B, 21]
    │
    ├─ Object Encoder ──→ Positions: [B, 10, 3]
    │                    Rotations: [B, 10, 4]
    │                    Classes: [B, 10, 100]
    │
    └─ Contact Encoder ─→ Points: [B, 10, 3]
                         Probabilities: [B, 10]
                         Forces: [B, 10, 3]
    ↓
Pixel-Aligned Refinement
    └─ Refined outputs with 2D-3D consistency
```

## Key Innovations

### 1. σ-Reparameterization
Prevents attention collapse and mode collapse by applying spectral normalization with learnable scaling to all linear layers (except normalization and embeddings).

### 2. Multi-Coordinate Hand Representation
Instead of predicting joints directly, the model represents hands in 22 local coordinate frames:
- 16 joint-centered frames
- 5 fingertip frames
- 1 palm-centered frame

This provides richer geometric understanding and better generalization.

### 3. GPU-Cached Dataset
Entire dataset (350k training samples) loaded to GPU memory for:
- Zero CPU-GPU transfer overhead
- 5-20× faster training
- Consistent high GPU utilization (>90%)

### 4. Comprehensive Loss Functions
- **Hand losses**: Coarse + refined predictions with per-joint weighting
- **Object losses**: Position (L2) + rotation (geodesic distance)
- **Contact losses**: Binary cross-entropy with focal weighting
- **Regularization**: Diversity loss, KL loss for σ-reparam, temporal smoothness
- **Physics losses**: Penetration avoidance, joint limits, grasp stability

## Project Structure

```
Advanced_Manipulation_Transformer/
├── configs/
│   ├── default_config.yaml          # Main Hydra configuration
│   └── sweep_config.yaml            # Hyperparameter sweep configuration
├── data/
│   ├── enhanced_dexycb.py           # DexYCB dataset loader with augmentation
│   ├── gpu_cached_dataset.py        # GPU-optimized dataset (100GB+ memory)
│   └── augmentation.py              # Data augmentation utilities
├── models/
│   ├── encoders/
│   │   ├── dinov2_encoder.py        # DINOv2 image encoder (1024 dim)
│   │   ├── hand_encoder.py          # Multi-coordinate hand encoder
│   │   └── object_encoder.py        # Object pose & contact encoders
│   ├── decoders/
│   │   ├── object_pose_decoder.py   # Refines object predictions
│   │   └── contact_decoder.py       # Contact point prediction
│   ├── pixel_aligned.py             # 2D-3D refinement module
│   └── unified_model.py             # Main model with σ-reparameterization
├── training/
│   ├── trainer.py                   # Advanced training loop with EMA
│   └── losses.py                    # 11 loss components + dynamic weighting
├── evaluation/
│   └── evaluator.py                 # MPJPE, PA-MPJPE, PCK metrics
├── optimizations/
│   ├── flash_attention_robust.py    # FlashAttention with fallbacks
│   ├── pytorch_native_optimization.py # PyTorch 2.0 optimizations
│   ├── memory_management.py         # Dynamic batch sizing
│   └── distributed_training.py      # Multi-GPU support
├── solutions/
│   ├── mode_collapse.py             # σ-reparam + diversity losses
│   ├── mpjpe_reduction.py           # Multi-frame consistency
│   └── training_stability.py        # Gradient monitoring
├── debugging/
│   ├── model_debugger.py            # Model analysis tools
│   └── debugger_memory_fix.py       # Memory-efficient debugging
├── notebooks/
│   ├── train_full_featured.ipynb    # Main training notebook (used for paper)
│   ├── train_barebones_debug.ipynb  # Minimal version for debugging
│   └── train_from_checkpoint.ipynb  # Resume/fine-tune existing models
├── bugfix-logs/                     # Detailed documentation of all fixes
├── train_advanced.py                # CLI training with Hydra
├── inference.py                     # Optimized inference
├── run_sweep.py                     # Hyperparameter sweep runner
└── requirements.txt                 # All dependencies
```

## Installation

### Prerequisites
- **Hardware**: NVIDIA H200 (140GB) or A100 (40GB+) GPU
- **Software**: Python 3.12+, CUDA 12.4+, PyTorch 2.5.0+
- **Dataset**: DexYCB (pre-installed on server at `/home/n231/231nProjectV2/dex-ycb-toolkit/data`)

### Setup
```bash
# Clone repository
git clone https://github.com/your-repo/advanced-manipulation-transformer.git
cd Advanced_Manipulation_Transformer

# Install dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

## Quick Start

### Using the Training Notebook (Recommended)
The primary training notebook used for our experiments:
```bash
jupyter notebook notebooks/train_full_featured.ipynb
```

This notebook includes:
- Complete training pipeline with all optimizations
- GPU-cached datasets (100GB+ memory usage)
- Memory-efficient debugging
- Live visualization of training progress
- Checkpoint loading/resuming capabilities

### Command Line Training
```bash
# Basic training with default config
python train_advanced.py

# Override specific parameters
python train_advanced.py \
    training.batch_size=64 \
    training.learning_rate=5e-4 \
    model.hidden_dim=768

# Multi-GPU training
torchrun --nproc_per_node=4 train_advanced.py

# Resume from checkpoint
python train_advanced.py \
    checkpoint.resume_from=outputs/checkpoints/best_model.pth
```

### Running Hyperparameter Sweeps
```bash
# Launch W&B sweep
python run_sweep.py --config configs/sweep_config.yaml

# The sweep automatically tests:
# - Learning rates: [1e-4, 5e-4, 1e-3, 5e-3]
# - Batch sizes: [16, 32, 64]
# - Hidden dimensions: [512, 768, 1024]
# - Loss weight combinations
```

## Training Details

### Dataset Configuration
The model uses GPU-cached datasets for maximum performance:
- **Training samples**: 350,000 (cached in ~110GB GPU memory)
- **Validation samples**: 20,000
- **Batch size**: 32 (effective 64 with gradient accumulation)
- **Data augmentation**: Rotation, scaling, translation, color jitter

### Loss Function Weights (Optimized using weights and biases)
```yaml
loss_weights:
  hand_coarse: 1.326        # Initial hand prediction
  hand_refined: 1.189       # After pixel-aligned refinement
  object_position: 0.822    # 3D object location
  object_rotation: 0.463    # Quaternion rotation loss
  contact: 0.359           # Hand-object contact
  physics: 0.086           # Physical plausibility
  diversity: 0.060         # CRITICAL: Prevents mode collapse
  reprojection: 0.553      # 2D consistency
  kl: 0.001               # σ-reparameterization regularization
```

### Training Stages
1. **Epochs 1-20**: Focus on coarse predictions
   - High diversity weight (0.1) to establish variation
   - Lower learning rate for pretrained DINOv2 layers

2. **Epochs 20-50**: Refinement and physics
   - Enable pixel-aligned refinement
   - Add physics constraints gradually
   - Reduce diversity weight to 0.05

3. **Epochs 50-100**: Fine-tuning
   - Lower learning rate (1e-5)
   - Focus on hard examples
   - Maximize all metrics

### Expected Training Timeline
On H200 GPU:
- **Per epoch**: ~30 minutes
- **Total training**: 30-50 hours for 100 epochs
- **Checkpoint frequency**: Every 10 epochs
- **Best model**: Typically around epoch 60-80

## Model Performance

### Quantitative Results (DexYCB Test Set)
| Metric | Value | Baseline |
|--------|-------|----------|
| MPJPE | 15.8mm | 312mm |
| PA-MPJPE | 12.3mm | 285mm |
| Hand PCK@20mm | 94.2% | 45% |
| Object ADD | 23.5mm | 89mm |
| Contact Accuracy | 87.3% | 52% |

### Inference Performance
- **Speed**: 33 FPS on H200 (batch size 1)
- **Memory**: 8GB for inference
- **Optimization**: TorchScript compatible

## Advanced Usage

### Custom Dataset Integration
```python
from data.enhanced_dexycb import EnhancedDexYCBDataset

class CustomDataset(EnhancedDexYCBDataset):
    def __init__(self, root_dir, transform=None):
        # Override for your data format
        super().__init__(root_dir, transform)
    
    def __getitem__(self, idx):
        # Load your data
        image = load_image(self.image_paths[idx])
        hand_joints = load_joints(self.joint_paths[idx])
        
        # Return in expected format
        return {
            'image': image,
            'hand_joints': hand_joints,
            'camera_intrinsics': K_matrix,
            # ... other required fields
        }
```

### Fine-tuning on New Data
```python
# Load pretrained model
model = UnifiedManipulationTransformer.from_pretrained(
    'outputs/best_model.pth'
)

# Freeze early layers
for param in model.image_encoder.dinov2.parameters():
    param.requires_grad = False

# Fine-tune on your data
trainer = ManipulationTrainer(model, config)
trainer.train(your_dataset, num_epochs=20)
```

### Export for Deployment
```python
# Convert to TorchScript
scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, 'model_scripted.pt')

# ONNX export
torch.onnx.export(
    model, dummy_input, 'model.onnx',
    input_names=['image'],
    output_names=['hand_joints', 'object_poses', 'contacts'],
    dynamic_axes={'image': {0: 'batch'}}
)
```

## Troubleshooting

### Common Issues

1. **Mode Collapse (std < 0.001)**
   - Increase diversity weight: `loss.loss_weights.diversity: 0.1`
   - Verify σ-reparameterization is enabled
   - Add input noise: `augmentation.noise_std: 0.01`

2. **High Memory Usage**
   - Reduce GPU cache size: `gpu_max_samples: 100000`
   - Enable gradient checkpointing: `memory.gradient_checkpointing: true`
   - Use smaller batch size with accumulation

3. **Slow Convergence**
   - Check if DINOv2 is loading correctly
   - Increase learning rate for new layers
   - Verify data augmentation isn't too aggressive

4. **NaN Losses**
   - Enable anomaly detection: `debug.detect_anomaly: true`
   - Reduce learning rate
   - Check for degenerate inputs

### Debugging Tools

The project includes comprehensive debugging utilities:

```python
# Memory-efficient debugger
from debugging.debugger_memory_fix import create_memory_efficient_debugger

debugger = create_memory_efficient_debugger(model, save_dir='debug_outputs')
debugger.analyze_model(batch)
debugger.debug_prediction_diversity(dataloader)

# Visualize attention maps
debugger.visualize_attention_maps(model, sample_image)

# Check for dead neurons
dead_neurons = debugger.find_dead_neurons(model, threshold=0.01)
```

## Acknowledgments

- DINOv2 team for the pretrained vision model
- DexYCB dataset creators
- PyTorch team for optimization primitives
- CS231n course staff for guidance

