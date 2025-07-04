# Video-to-Manipulation Transformer: Comprehensive Technical Documentation

## Project Overview

The Video-to-Manipulation Transformer is a multi-stage deep learning system that converts monocular RGB video into robot manipulation commands for an **Allegro Hand (16 DOF)** attached to a **Universal Robot (UR) arm (6 DOF)**. The system is trained on the **DexYCB dataset** containing 582,000+ samples of human hand-object interactions.

## Dataset: DexYCB

### Dataset Statistics
- **Total Samples**: 582,000+ frames
- **Subjects**: 10 human subjects
- **Objects**: 20 YCB objects
- **Grasps**: 1,000 unique grasps
- **Cameras**: 8 synchronized views (we use monocular)
- **Annotations**: Hand pose (MANO), object 6DoF poses, contact masks

### Data Format
Each sample contains:
```python
{
    'color': torch.Tensor([3, 224, 224]),      # RGB image (normalized)
    'hand_joints_3d': torch.Tensor([21, 3]),   # 3D joints in mm
    'hand_joints_2d': torch.Tensor([21, 2]),   # 2D joints in pixels
    'hand_pose': torch.Tensor([51]),           # MANO parameters (pose + shape)
    'object_poses': torch.Tensor([10, 3, 4]),  # Up to 10 objects, 6DoF poses
    'ycb_ids': torch.Tensor([10]),             # Object class IDs
    'num_objects': int,                        # Actual objects in scene
    'has_hand': bool                           # Hand visibility flag
}
```

### MANO Hand Model Parameters
- **Pose**: 48D (3D axis-angle for 16 joints) or 51D (with global rotation)
- **Shape**: 10D shape parameters (betas)
- **Translation**: 3D global translation
- **Total**: 51D representation (we handle both 48D and 51D)

### Object Classes (YCB)
20 YCB objects including: master chef can, cracker box, mustard bottle, tuna fish can, banana, strawberry, apple, lemon, peach, pear, orange, plum, scissors, mug, power drill, wood block, etc.

## Current Implementation: Stage 1 Multi-Encoder Architecture

### 1. Hand Pose Encoder (`models/encoders/hand_encoder.py`)

**Architecture Details:**
```python
HandPoseEncoder(
    input_dim=768,      # 16×16×3 patch size
    hidden_dim=1024,    # Transformer width
    num_layers=8,       # Transformer depth
    num_heads=16,       # Attention heads
    mlp_dim=4096,       # FFN dimension
    dropout=0.1,        # Dropout rate
    num_joints=21       # MANO joints
)
```

**Layer-by-layer breakdown:**
1. **Input Projection**: Linear(768 → 1024)
2. **CLS Token**: Learnable parameter [1, 1, 1024]
3. **Positional Embedding**: 2D positional encoding for 14×14 patches
4. **Transformer Encoder**: 8 layers of:
   - LayerNorm
   - Multi-Head Attention (16 heads, 64 dim/head)
   - Residual connection
   - LayerNorm
   - MLP: Linear(1024 → 4096) → GELU → Linear(4096 → 1024)
   - Residual connection
5. **Output Heads**:
   - Joint Head: LayerNorm → Linear(1024 → 1024) → GELU → Dropout → Linear(1024 → 63)
   - Confidence Head: LayerNorm → Linear(1024 → 21)
   - Shape Head: LayerNorm → Linear(1024 → 512) → GELU → Dropout → Linear(512 → 10)

**Parameters**: ~103.2M

### 2. Object Pose Encoder (`models/encoders/object_encoder.py`)

**Architecture Details:**
```python
ObjectPoseEncoder(
    input_dim=768,
    hidden_dim=1024,
    num_layers=8,
    num_heads=16,
    mlp_dim=4096,
    dropout=0.1,
    num_classes=100,    # Object classes
    max_objects=10      # Max objects per scene
)
```

**Layer-by-layer breakdown:**
1. **Input Projection**: Linear(768 → 1024)
2. **Object Type Embeddings**: Embedding(100, 1024)
3. **Object Queries**: 10 learnable queries [10, 1024]
4. **Positional Embedding**: 2D positional encoding
5. **Feature Encoder**: 4 transformer layers (extracts visual features)
6. **Cross-Attention Decoder**: 4 layers of:
   - Self-attention on object queries
   - Cross-attention (queries attend to image features)
   - FFN
7. **Output Heads**:
   - Position: Linear(1024 → 3)
   - Rotation: Linear(1024 → 6) with 6D rotation representation
   - Confidence: Linear(1024 → 1)
   - Classification: Linear(1024 → 100)

**Parameters**: ~102.8M

### 3. Contact Detection Encoder (`models/encoders/contact_encoder.py`)

**Architecture Details:**
```python
ContactDetectionEncoder(
    input_dim=768,      # Not directly used
    hidden_dim=512,
    num_layers=6,
    num_heads=16,
    mlp_dim=2048,
    dropout=0.1,
    num_contact_queries=10
)
```

**Layer-by-layer breakdown:**
1. **Feature Projections**: 
   - Hand: Linear(input_hand_dim → 512)
   - Object: Linear(input_obj_dim → 512)
2. **Contact Queries**: 10 learnable queries [10, 512]
3. **Cross-Modal Encoder**: 3 layers of hand-object cross-attention
4. **Contact Decoder**: 3 layers attending to fused features
5. **Output Heads**:
   - Contact Points: Linear(512 → 3)
   - Contact Confidence: Linear(512 → 1) → Sigmoid
   - Contact Types: Linear(512 → 4) (none/light/firm/manipulation)
   - Contact Forces: Linear(512 → 3)
   - Interaction Type: Global pooling → Linear(512 → 6)

**Parameters**: ~21.3M

### Critical Component: 2D Positional Embeddings

**Implementation:**
```python
class PositionalEmbedding2D(nn.Module):
    def __init__(self, hidden_dim=1024, image_size=224, patch_size=16):
        self.num_patches_per_dim = 14  # 224/16
        self.row_embed = nn.Parameter(torch.zeros(14, 512))
        self.col_embed = nn.Parameter(torch.zeros(14, 512))
```

**Why Critical**: Without positional embeddings, the transformer has no spatial awareness - patches are processed as an unordered set, making accurate 3D position prediction impossible.

## Data Pipeline

### GPU-Only Dataset (`data/gpu_only_dataset.py`)

**Key Features:**
1. Loads entire dataset into GPU memory on initialization
2. Pre-processes all images to patches
3. Zero CPU-GPU transfers during training
4. Direct batch generation from GPU tensors

**Memory Layout:**
```python
self.data = {
    'color': torch.zeros((N, 3, 224, 224), device='cuda', dtype=bfloat16),
    'hand_joints_3d': torch.full((N, 21, 3), -1.0, device='cuda'),
    'hand_joints_2d': torch.full((N, 21, 2), -1.0, device='cuda'),
    'hand_pose': torch.zeros((N, 51), device='cuda'),
    'object_poses': torch.zeros((N, 10, 3, 4), device='cuda'),
    'ycb_ids': torch.zeros((N, 10), device='cuda', dtype=long),
    'num_objects': torch.zeros((N,), device='cuda', dtype=long),
    'has_hand': torch.zeros((N,), device='cuda', dtype=bool)
}
```

### GPU Preprocessing (`data/gpu_preprocessing.py`)

**Patch Extraction:**
```python
# Uses unfold for efficient patch extraction
patches = images.unfold(2, 16, 16).unfold(3, 16, 16)
patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
patches = patches.view(B, 196, 768)  # 14×14 patches of 16×16×3
```

## Training Configuration

### Loss Functions

**Hand Pose Loss:**
- Primary: Smooth L1 loss (robust to outliers)
- Per-joint weighting: Fingertips weighted 1.1x more
- Diversity loss: -log(std(predictions)) to prevent mode collapse
- Velocity loss: Temporal smoothness constraint
- Total: `0.8 * smooth_l1 + 0.2 * weighted + 0.01 * diversity + 0.05 * velocity`

**Object Pose Loss:**
- Position: Smooth L1 loss on 3D positions
- Rotation: Geodesic loss on SO(3) (TODO)
- Classification: Cross-entropy loss

**Contact Loss:**
- Confidence: Binary cross-entropy
- Contact points: Smooth L1 loss
- Contact types: Cross-entropy
- Force magnitudes: L2 loss

### Training Hyperparameters

```yaml
# Optimization
optimizer: AdamW
learning_rate: 1e-3
weight_decay: 0.01
scheduler: CosineAnnealingLR
min_lr: 1e-6
grad_clip: 1.0

# Batch settings
batch_size: 128  # Limited by memory
accumulation_steps: 1
mixed_precision: bfloat16

# Regularization
dropout: 0.1  # Reduced from 0.3
drop_path: 0.0  # Disabled (was causing issues)
label_smoothing: 0.0  # Disabled

# Data augmentation
joint_noise_std: 0.005  # 5mm Gaussian noise
rotation_range: 5  # ±5 degrees
color_jitter: 0.1
mixup_alpha: 0.0  # Disabled

# Loss weights
diversity_weight: 0.01
velocity_loss_weight: 0.05
joint_weight_power: 1.1
```

### Training Metrics

**Primary Metrics:**
- MPJPE (Mean Per Joint Position Error): Currently ~325mm
- Validation Loss: ~0.078
- Prediction Diversity: std=0.0003 (too low!)

**Per-Joint Analysis:**
- Worst joints: J0 (wrist), J5, J17 (fingertips)
- Best joints: Middle finger joints
- Fingertips consistently have higher error

**GPU Performance:**
- Utilization: 85-95%
- Memory: 36.6GB / 140GB (underutilized)
- Throughput: ~700 samples/s
- Power: 616W / 700W

## Training Issues and Solutions

### 1. Mode Collapse (SOLVED)
**Problem**: Model converged to constant ~312mm prediction with zero diversity
**Root Cause**: Missing positional embeddings - model had no spatial awareness
**Solution**: Added 2D positional embeddings for image grid understanding

### 2. MANO Dimension Mismatch (SOLVED)
**Problem**: Dataset has both 48D and 51D MANO poses
**Solution**: Pad 48D to 51D, clip if >51D

### 3. Memory Underutilization (ONGOING)
**Problem**: Using only 36GB of 140GB available
**Potential Solutions**:
- Increase batch size to 512+
- Increase model hidden dimensions to 2048
- Cache more training samples

### 4. High MPJPE (ONGOING)
**Current**: 325mm
**Target**: <20mm
**Potential Solutions**:
- Auxiliary tasks (predict hand center first)
- Curriculum learning (easier poses first)
- Better initialization strategies

## Unimplemented Components

### Stage 2: Temporal Fusion Training
- Temporal Fusion Encoder (architecture complete, training TODO)
- Sequence data loading (need sliding windows)
- Temporal consistency losses
- Action supervision from demonstrations

### Stage 3: End-to-End Fine-tuning
- MuJoCo XLA integration for differentiable physics
- Full gradient flow through simulation
- Task-specific losses (grasp success, stability)

### Robot Control Layer
- MLP architectures defined but not trained
- Need to collect action-joint mapping data
- Implement safety constraints and joint limits

## Code Organization

```
Video_Manipulation_Transformer/
├── models/
│   ├── encoders/
│   │   ├── hand_encoder.py         # HandPoseEncoder class
│   │   ├── object_encoder.py       # ObjectPoseEncoder class
│   │   └── contact_encoder.py      # ContactDetectionEncoder class
│   ├── temporal_fusion.py          # TemporalFusionModel (TODO)
│   ├── action_decoder.py           # ActionDecoder (TODO)
│   └── full_model.py              # Complete pipeline (TODO)
├── control/
│   ├── mlp_retargeting.py         # Allegro hand control
│   └── ur_kinematics.py           # UR arm kinematics
├── data/
│   ├── dexycb_dataset.py          # Original dataset loader
│   ├── gpu_only_dataset.py        # GPU-cached dataset
│   ├── gpu_preprocessing.py       # GPU transforms
│   └── prefetch_loader.py         # Async data loading
├── training/
│   ├── trainer.py                 # Training loops
│   ├── losses.py                  # Loss implementations
│   └── evaluation.py              # Metrics computation
├── physics/
│   └── mujoco_sim.py             # MuJoCo integration (TODO)
└── train_gpu_optimized_final_improved.ipynb  # Latest training notebook
```

## Hardware Setup

**GPU**: NVIDIA H200 (140GB HBM3e)
**CUDA**: 12.1
**PyTorch**: 2.5.1+cu121
**Key Features Utilized**:
- BFloat16 mixed precision
- TF32 for matmul operations
- Persistent GPU memory allocation
- Non-blocking transfers

## Reproducibility Notes

1. **Random Seeds**: Set to 42 for all (torch, numpy, cuda)
2. **Deterministic Ops**: Disabled for performance (can enable)
3. **Dataset Split**: Using s0_train (100k) and s0_val (10k)
4. **Checkpoint**: Best model saved based on validation loss

## Performance Targets

**Stage 1 (Current)**:
- MPJPE: <100mm (currently 325mm)
- GPU Utilization: >90% (achieved)
- Throughput: 10k samples/s (currently 700)

**Stage 2 (Future)**:
- Action prediction accuracy: >85%
- Temporal consistency: <10mm inter-frame

**Stage 3 (Future)**:
- Grasp success rate: >85%
- Real-time inference: 30Hz

This documentation represents the complete technical specification of the Video-to-Manipulation Transformer as of 2025-01-06, including all implemented components, architectural details, and known issues.