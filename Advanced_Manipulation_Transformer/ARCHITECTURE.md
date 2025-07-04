# Architecture Documentation

## Overview

The Advanced Manipulation Transformer addresses fundamental issues in previous approaches:
- **Mode collapse**: Previous models converged to constant predictions (std=0.0003)
- **High MPJPE**: 325mm error due to lack of spatial understanding
- **Poor generalization**: Single global representation couldn't capture hand complexity
- **Training instability**: Gradient explosions and vanishing gradients
- **Memory inefficiency**: Poor utilization of H200 GPU capabilities

## Core Innovations

### 1. Multi-Coordinate Hand Representation

Instead of a single global coordinate frame, we use 22 local coordinate frames:

```python
# 22 coordinate frames from hand structure
frames = {
    'fingertips': 5,      # One per fingertip
    'metacarpals': 5,     # One per finger base
    'finger_joints': 12   # 3 per finger (4 fingers, thumb has 2)
}
```

**Why this works**:
- Local representations are more invariant to global pose
- Each frame captures local geometry naturally
- Attention mechanism learns to combine frames

**Implementation**:
```python
# In hand_encoder.py
def get_coordinate_frames(mano_joints):
    """Create 22 local coordinate frames from hand joints"""
    frames = []
    
    # Fingertip frames
    for finger in range(5):
        tip_idx = 4 + finger * 4  # MANO joint indices
        frames.append(create_frame_from_joint(mano_joints, tip_idx))
    
    # ... (metacarpal and joint frames)
    
    return torch.stack(frames, dim=1)  # [B, 22, 4, 4]
```

### 2. DINOv2 Integration

We leverage DINOv2's pretrained features instead of training from scratch:

```python
# Multi-scale feature extraction
features = {
    'low': layer_6_output,   # Coarse features
    'mid': layer_9_output,   # Mid-level features
    'high': layer_12_output  # Fine-grained features
}
```

**Benefits**:
- Robust visual features from 142M image pretraining
- Multi-scale understanding for both global and local features
- Frozen early layers reduce overfitting

### 3. Advanced Optimizations

#### FlashAttention-3 Integration
- 1.5-2x speedup over standard attention
- Memory-efficient with O(N) instead of O(N²) memory
- Automatically enabled for compatible GPUs

#### FP8 Mixed Precision (H200)
- Leverages H200's FP8 Tensor Cores
- 2x throughput improvement over FP16
- Automatic loss scaling for stability

#### Memory-Efficient Training
- Selective gradient checkpointing
- Dynamic batch sizing based on available memory
- xFormers for memory-efficient attention
- Activation offloading for very large models

### 4. Mode Collapse Prevention

#### Sigma Reparameterization
```python
class SigmaReparam(nn.Module):
    def forward(self, x):
        # Predict log-variance
        log_var = self.variance_head(x)
        
        # Reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        
        # Add controlled noise
        output = x + std * eps
        
        # Minimize entropy of variance
        self.kl_loss = -0.5 * torch.mean(1 + log_var - log_var.exp())
        
        return output
```

#### Additional Techniques
- FeatureNoise injection at multiple layers
- DropPath regularization
- MixUp data augmentation
- Diversity loss to encourage varied predictions

### 5. Training Stabilization

#### Gradient Centralization
```python
# Center gradients for more stable optimization
for p in model.parameters():
    if p.grad is not None and len(p.shape) > 1:
        p.grad.data -= p.grad.data.mean(dim=0, keepdim=True)
```

#### Anomaly Detection
- Automatic NaN/Inf detection and handling
- Loss explosion detection with rollback
- Gradient norm monitoring

#### Robust Optimizer Settings
- Different learning rates for different parameter groups
- No weight decay for normalization and bias parameters
- Adaptive gradient clipping

```python
# Feature extraction hierarchy
dinov2_features = {
    'early': layers[0:4],    # Low-level, frozen
    'middle': layers[4:8],   # Mid-level, frozen  
    'late': layers[8:12],    # High-level, fine-tuned
    'head': layers[12:]      # Task-specific, fully trained
}
```

**Benefits**:
- Robust visual features from large-scale pretraining
- Multi-scale understanding through feature pyramid
- Transfer learning reduces data requirements

### 3. Pixel-Aligned Refinement

Critical for accurate 3D predictions:

```python
# Iterative refinement process
for step in range(num_refinement_steps):
    # 1. Project current 3D estimate to 2D
    joints_2d = project_3d_to_2d(joints_3d, camera_params)
    
    # 2. Sample image features at projected locations
    sampled_features = bilinear_sample(image_features, joints_2d)
    
    # 3. Predict 3D offset using sampled features
    offset_3d = offset_mlp(sampled_features)
    
    # 4. Update 3D positions
    joints_3d = joints_3d + step_size * offset_3d
    
    # Decrease step size for finer adjustments
    step_size *= 0.5
```

**Why it's essential**:
- Grounds 3D predictions in 2D image evidence
- Iterative process allows coarse-to-fine refinement
- Reduces 3D ambiguity using 2D constraints

### 4. Sigma Reparameterization

Prevents mode collapse by ensuring diverse predictions:

```python
class SigmaReparam(nn.Module):
    """Prevent mode collapse through variance regularization"""
    
    def forward(self, features):
        # Split features into mean and log_var
        mu, log_var = features.chunk(2, dim=-1)
        
        # Reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        
        # Sample with controlled variance
        output = mu + eps * std
        
        # Regularization loss encourages non-zero variance
        kl_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()
        
        return output, kl_loss
```

**Benefits**:
- Forces model to learn meaningful variance
- Prevents collapse to dataset mean
- Enables uncertainty estimation

## Component Details

### Hand Pose Pipeline

```
RGB Image → DINOv2 → Multi-Scale Features
                           ↓
              Multi-Coordinate Hand Encoder
                     ↓            ↓
              Coarse 3D Joints   Hand Features
                     ↓                 ↓
              Pixel-Aligned ← Image Features
                Refinement
                     ↓
              Refined 3D Joints
```

### Object Pose Pipeline

```
RGB Image → DINOv2 → Object Detection
                           ↓
                    Object Encoder
                     ↓         ↓
               6D Pose    Class Logits
              (SE3 repr)
```

### Contact Prediction Pipeline

```
Hand Features ⊕ Object Features → Contact Encoder
                                         ↓
                                  Contact Points
                                  + Confidence
                                  + Forces
```

## Loss Function Design

### Adaptive MPJPE Loss

Standard MPJPE treats all joints equally, but:
- Fingertips are harder to predict
- Root joints are more stable
- Some joints are occluded

Our solution:
```python
class AdaptiveMPJPELoss(nn.Module):
    def __init__(self):
        # Learnable per-joint weights
        self.joint_weights = nn.Parameter(torch.ones(21))
        
        # Prior importance (fingertips = 1.5x)
        self.importance = torch.ones(21)
        self.importance[[4,8,12,16,20]] = 1.5
```

### Dynamic Loss Weighting

Different losses need different schedules:
```python
def get_loss_weight(loss_name, epoch):
    if loss_name == 'diversity':
        # High early to prevent collapse
        return 2.0 - min(epoch / 30, 1.0)
    elif loss_name == 'refinement':
        # Low early, increase over time
        return min(epoch / 50, 1.0)
    elif loss_name == 'physics':
        # Only after good predictions
        return max(0, (epoch - 20) / 30)
```

## Memory Optimizations

### Gradient Checkpointing
```python
# Trade compute for memory
def forward(self, x):
    # Don't save intermediate activations
    x = checkpoint(self.layer1, x)
    x = checkpoint(self.layer2, x)
    return self.layer3(x)  # Only final layer saves activations
```

### Mixed Precision (BF16)
```python
# Use BFloat16 for H200 GPU
with autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)
```

### Efficient Attention
```python
# Use Flash Attention when available
if torch.cuda.is_available():
    torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_math=False,
        enable_mem_efficient=True
    )
```

## Training Strategy

### Three-Stage Approach

**Stage 1: Component Pretraining (20 epochs)**
- Train each encoder independently
- High learning rate (1e-3)
- Direct supervision only

**Stage 2: Joint Training (50 epochs)**
- All components together
- Medium learning rate (5e-4)
- Balance all losses

**Stage 3: Fine-tuning (30 epochs)**
- Low learning rate (1e-4)
- Focus on hard cases
- Physics constraints weighted higher

### Learning Rate Schedule

Different components need different rates:
```python
param_groups = [
    {'params': dinov2_params, 'lr': base_lr * 0.01},     # Pretrained
    {'params': new_encoder_params, 'lr': base_lr * 0.5}, # New but stable
    {'params': decoder_params, 'lr': base_lr},           # Full rate
]
```

## Debugging and Monitoring

### Key Metrics to Track

1. **Prediction Diversity**:
   ```python
   std = predictions.std(dim=0).mean()
   # Should be > 0.01, not 0.0003
   ```

2. **Attention Entropy**:
   ```python
   entropy = -(attn * torch.log(attn)).sum(dim=-1).mean()
   # Should be moderate, not too high or low
   ```

3. **Gradient Flow**:
   ```python
   for name, param in model.named_parameters():
       if param.grad is not None:
           grad_norm = param.grad.norm()
           # Log if too small or too large
   ```

## Common Patterns

### Attention Fusion
```python
# Combine multiple modalities
hand_feats = self.hand_encoder(x)
obj_feats = self.object_encoder(x)

# Cross-attention
attn_feats = self.cross_attention(
    query=hand_feats,
    key=obj_feats,
    value=obj_feats
)

# Residual connection
fused = hand_feats + self.dropout(attn_feats)
```

### Coarse-to-Fine Prediction
```python
# Initial coarse prediction
coarse = self.coarse_decoder(features)

# Iterative refinement
refined = coarse
for i in range(num_steps):
    refined = self.refine_step(refined, features)
    
# Use both in loss
loss = coarse_loss(coarse) + refined_loss(refined)
```

## Future Improvements

1. **Temporal Modeling**: Add sequence processing for video
2. **Physics Simulation**: Integrate differentiable physics
3. **Active Learning**: Focus on failure cases
4. **Real-time Optimization**: Optimize for deployment