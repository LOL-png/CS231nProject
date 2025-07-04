# Model Components Documentation

This document provides detailed explanations of each component in the Advanced Manipulation Transformer.

## Core Components Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  DINOv2 Encoder │────▶│ Feature Pyramid  │────▶│ Task Encoders   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                              ┌────────────────────────────┼────────────────────────────┐
                              │                            │                            │
                    ┌─────────▼─────────┐      ┌──────────▼──────────┐      ┌─────────▼─────────┐
                    │   Hand Encoder    │      │   Object Encoder    │      │ Contact Encoder   │
                    │ (Multi-Coordinate)│      │    (SE3 + Class)    │      │  (Physics-Aware)  │
                    └─────────┬─────────┘      └──────────┬──────────┘      └─────────┬─────────┘
                              │                            │                            │
                    ┌─────────▼─────────┐      ┌──────────▼──────────┐      ┌─────────▼─────────┐
                    │  Pixel-Aligned    │      │   Attention Fusion  │      │  Contact Points   │
                    │    Refinement     │      │                     │      │   + Confidence    │
                    └───────────────────┘      └─────────────────────┘      └───────────────────┘
```

## 1. DINOv2 Image Encoder

**File**: `models/encoders/dinov2_encoder.py`

### Purpose
Extracts robust visual features using Facebook's DINOv2 vision transformer, pretrained on massive datasets.

### Key Features
- Multi-scale feature extraction
- Frozen early layers for transfer learning
- Adaptive pooling for different resolutions

### Implementation Details

```python
class DINOv2ImageEncoder(nn.Module):
    def __init__(self, freeze_layers=12):
        # Load pretrained DINOv2
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        
        # Freeze early layers (general features)
        for i, layer in enumerate(self.dinov2.blocks):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
```

### Why DINOv2?
- **Self-supervised pretraining**: Learns without labels
- **Robust features**: Generalizes across domains
- **Multi-scale understanding**: Captures both local and global features

### Feature Pyramid
Extracts features at multiple scales:
```python
features = {
    'low': blocks[3].output,   # 1/4 resolution, local details
    'mid': blocks[7].output,   # 1/8 resolution, semantic features  
    'high': blocks[11].output, # 1/16 resolution, global context
    'final': final_output      # Full model output
}
```

## 2. Multi-Coordinate Hand Encoder

**File**: `models/encoders/hand_encoder.py`

### Purpose
Encodes hand pose using 22 local coordinate frames instead of a single global frame.

### Key Innovation
Each coordinate frame captures local geometry:
- **Fingertip frames (5)**: Capture grasp points
- **Metacarpal frames (5)**: Capture palm structure
- **Joint frames (12)**: Capture articulation

### Implementation

```python
class MultiCoordinateHandEncoder(nn.Module):
    def get_coordinate_frames(self, joints):
        """Create 22 local coordinate frames"""
        frames = []
        
        # Example: Fingertip frame
        for finger_idx in range(5):
            tip = joints[:, tip_indices[finger_idx]]
            mcp = joints[:, mcp_indices[finger_idx]]
            
            # Z-axis: finger direction
            z_axis = F.normalize(tip - mcp, dim=-1)
            
            # X-axis: perpendicular to palm
            x_axis = F.normalize(torch.cross(z_axis, palm_normal), dim=-1)
            
            # Y-axis: complete the frame
            y_axis = torch.cross(z_axis, x_axis)
            
            frame = torch.stack([x_axis, y_axis, z_axis], dim=-1)
            frames.append(frame)
```

### Vertex Processing
```python
def encode_vertices_in_frames(self, vertices, frames):
    """Transform vertices to each coordinate frame"""
    # vertices: [B, 778, 3] MANO vertices
    # frames: [B, 22, 3, 3] rotation matrices
    
    # Transform to each frame
    vertices_local = []
    for i in range(22):
        # Rotate vertices to local frame
        v_local = torch.matmul(vertices, frames[:, i].transpose(-1, -2))
        vertices_local.append(v_local)
    
    # Stack: [B, 22, 778, 3]
    return torch.stack(vertices_local, dim=1)
```

### Why Multi-Coordinate?
1. **Invariance**: Local frames are invariant to global pose
2. **Expressiveness**: Captures fine-grained geometry
3. **Robustness**: Multiple views prevent information loss

## 3. Pixel-Aligned Refinement Module

**File**: `models/pixel_aligned.py`

### Purpose
Refines 3D predictions by grounding them in 2D image evidence.

### Algorithm
```python
class PixelAlignedRefinement(nn.Module):
    def forward(self, joints_3d, image_features, camera_params):
        refined = joints_3d
        step_size = 1.0
        
        for step in range(self.num_steps):
            # 1. Project to 2D
            joints_2d = self.project_3d_to_2d(refined, camera_params)
            
            # 2. Sample features at projected locations
            sampled_feats = self.sample_features(image_features, joints_2d)
            
            # 3. Predict 3D offset
            offset_3d = self.offset_predictor(sampled_feats)
            
            # 4. Update with decreasing step size
            refined = refined + step_size * offset_3d
            step_size *= 0.5
            
        return refined
```

### Feature Sampling
Uses bilinear interpolation to sample features:
```python
def sample_features(self, features, locations_2d):
    """Sample features at 2D locations"""
    # Normalize coordinates to [-1, 1]
    locations_norm = 2 * locations_2d / image_size - 1
    
    # Bilinear sampling
    sampled = F.grid_sample(
        features, 
        locations_norm.unsqueeze(1),
        align_corners=True
    )
    
    return sampled
```

### Why It Works
- **2D-3D Consistency**: Ensures 3D predictions match 2D appearance
- **Iterative**: Coarse-to-fine refinement
- **Differentiable**: End-to-end trainable

## 4. Object Pose Encoder

**File**: `models/encoders/object_encoder.py`

### Purpose
Predicts 6-DoF object poses and classes from visual features.

### SE(3) Representation
Uses continuous 6D rotation representation:
```python
class ObjectPoseEncoder(nn.Module):
    def forward(self, features, object_queries):
        # Attend to object regions
        object_feats = self.cross_attention(
            query=object_queries,
            key=features,
            value=features
        )
        
        # Predict pose components
        positions = self.position_head(object_feats)  # [B, N, 3]
        rotations_6d = self.rotation_head(object_feats)  # [B, N, 6]
        
        # 6D to matrix conversion (continuous representation)
        rotation_matrices = six_d_to_matrix(rotations_6d)
        
        return {
            'positions': positions,
            'rotations': rotations_6d,
            'rotation_matrices': rotation_matrices,
            'class_logits': self.class_head(object_feats)
        }
```

### Why 6D Rotation?
- **Continuous**: No discontinuities like quaternions
- **Unique**: One-to-one mapping to SO(3)
- **Stable**: Better gradient flow

## 5. Contact Detection Encoder

**File**: `models/encoders/contact_encoder.py`

### Purpose
Predicts hand-object contact points and forces.

### Architecture
```python
class ContactEncoder(nn.Module):
    def forward(self, hand_features, object_features):
        # Compute pairwise interactions
        interaction_feats = self.interaction_module(
            hand_features, 
            object_features
        )
        
        # Predict contact locations (in 3D)
        contact_points = self.point_head(interaction_feats)
        
        # Predict contact confidence
        contact_conf = torch.sigmoid(self.conf_head(interaction_feats))
        
        # Predict contact forces (optional)
        contact_forces = self.force_head(interaction_feats)
        
        return {
            'contact_points': contact_points,
            'contact_confidence': contact_conf,
            'contact_forces': contact_forces
        }
```

### Physics Constraints
Enforces realistic contacts:
```python
def physics_constraints(contacts, hand_surface, object_surface):
    # Contacts should be on surfaces
    dist_to_hand = distance_to_surface(contacts, hand_surface)
    dist_to_obj = distance_to_surface(contacts, object_surface)
    
    # Penalize contacts far from both surfaces
    surface_loss = torch.min(dist_to_hand, dist_to_obj)
    
    return surface_loss
```

## 6. Decoders

### Hand Decoder
**File**: `models/decoders/hand_decoder.py`

Converts encoded features to hand predictions:
```python
class HandDecoder(nn.Module):
    def forward(self, features):
        # Global features for shape
        global_feat = features.mean(dim=1)  # Pool over positions
        shape_params = self.shape_mlp(global_feat)  # MANO β
        
        # Local features for pose
        pose_params = self.pose_mlp(features)  # MANO θ
        
        # Direct joint regression (backup)
        joints_3d = self.joint_mlp(features)
        
        return {
            'mano_shape': shape_params,  # [B, 10]
            'mano_pose': pose_params,    # [B, 45]
            'joints_3d': joints_3d       # [B, 21, 3]
        }
```

### Object Decoder
**File**: `models/decoders/object_decoder.py`

Refines object predictions:
```python
class ObjectDecoder(nn.Module):
    def forward(self, object_features, initial_poses):
        # Iterative refinement
        refined_poses = initial_poses
        
        for _ in range(self.num_iterations):
            # Compute pose-conditioned features
            pose_feats = self.pose_encoder(refined_poses)
            combined = torch.cat([object_features, pose_feats], dim=-1)
            
            # Predict residuals
            delta_pos = self.position_refiner(combined)
            delta_rot = self.rotation_refiner(combined)
            
            # Update poses
            refined_poses = self.update_poses(
                refined_poses, 
                delta_pos, 
                delta_rot
            )
            
        return refined_poses
```

## 7. Unified Model

**File**: `models/unified_model.py`

### Purpose
Combines all components into a single end-to-end model.

### Key Features
- **Sigma Reparameterization**: Prevents mode collapse
- **Feature Fusion**: Combines modalities effectively
- **End-to-end Training**: All components optimize together

### Forward Pass
```python
class UnifiedManipulationTransformer(nn.Module):
    def forward(self, images, mano_vertices=None, camera_params=None):
        # 1. Extract image features
        image_features = self.image_encoder(images)
        
        # 2. Encode hand with multi-coordinate representation
        hand_features = self.hand_encoder(
            image_features, 
            mano_vertices
        )
        
        # 3. Initial hand prediction
        hand_predictions = self.hand_decoder(hand_features)
        
        # 4. Pixel-aligned refinement
        if camera_params is not None:
            hand_predictions['joints_3d_refined'] = self.refiner(
                hand_predictions['joints_3d'],
                image_features,
                camera_params
            )
        
        # 5. Object detection and pose
        object_predictions = self.object_encoder(image_features)
        
        # 6. Contact prediction
        contact_predictions = self.contact_encoder(
            hand_features,
            object_predictions['features']
        )
        
        return {
            'hand': hand_predictions,
            'objects': object_predictions,
            'contacts': contact_predictions
        }
```

### Sigma Reparameterization
Critical for preventing mode collapse:
```python
class SigmaReparam(nn.Module):
    """Forces model to learn meaningful variance"""
    
    def forward(self, features):
        # Split features
        mu, log_sigma = features.chunk(2, dim=-1)
        
        # Ensure minimum variance
        log_sigma = torch.clamp(log_sigma, min=-4, max=2)
        sigma = torch.exp(log_sigma)
        
        # Reparameterization trick
        if self.training:
            eps = torch.randn_like(mu)
            output = mu + sigma * eps
        else:
            output = mu  # Use mean at test time
        
        # KL regularization
        kl_loss = -0.5 * (1 + 2*log_sigma - mu**2 - sigma**2).mean()
        
        return output, kl_loss
```

## Component Interactions

### Information Flow
1. **Image → DINOv2**: Extract visual features
2. **Features → Encoders**: Task-specific encoding
3. **Encoders → Fusion**: Combine modalities
4. **Fusion → Decoders**: Generate predictions
5. **Predictions → Refinement**: Iterative improvement

### Cross-Modal Attention
```python
def cross_modal_fusion(hand_feats, object_feats):
    # Hand attending to objects
    hand_to_obj = cross_attention(
        query=hand_feats,
        key=object_feats,
        value=object_feats
    )
    
    # Objects attending to hand
    obj_to_hand = cross_attention(
        query=object_feats,
        key=hand_feats,
        value=hand_feats
    )
    
    # Symmetric fusion
    fused_hand = hand_feats + hand_to_obj
    fused_obj = object_feats + obj_to_hand
    
    return fused_hand, fused_obj
```

## Best Practices

### 1. Component Testing
Test each component independently:
```python
# Test hand encoder
hand_enc = MultiCoordinateHandEncoder()
dummy_input = torch.randn(2, 512, 768)
output = hand_enc(dummy_input)
assert output.shape == (2, 22, 256)  # 22 frames
```

### 2. Gradient Flow
Monitor gradient flow through components:
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm()
        print(f"{name}: {grad_norm:.4f}")
```

### 3. Feature Visualization
Visualize intermediate features:
```python
# Visualize attention maps
attn_weights = model.hand_encoder.attention.get_attention_weights()
plot_attention_heatmap(attn_weights)

# Visualize coordinate frames
frames = model.hand_encoder.get_coordinate_frames(joints)
visualize_frames_3d(frames)
```

## Debugging Tips

1. **Mode Collapse**: Check sigma values in reparameterization
2. **Poor Refinement**: Verify camera parameters are correct
3. **Bad Contacts**: Visualize predicted contact points
4. **Training Instability**: Check gradient norms and loss components