# Transition Merger Model Guide

## Overview

This model merges two video sequences by learning smooth transitions using HOISDF outputs. It takes the outputs from HOISDF (MANO parameters, SDFs, contact points) and generates natural transitions between different tasks.

## Input Structure

The model takes **HOISDF outputs** from two separate videos:

### HOISDFOutputs Structure:
```python
@dataclass
class HOISDFOutputs:
    mano_params: torch.Tensor      # [T, 51] MANO hand parameters
    hand_sdf: torch.Tensor         # [T, D, H, W] Hand signed distance field
    object_sdf: torch.Tensor       # [T, D, H, W] Object signed distance field
    contact_points: torch.Tensor   # [T, N, 3] 3D contact locations
    contact_frames: torch.Tensor   # [T, N] Contact indicators
    hand_vertices: torch.Tensor    # [T, 778, 3] MANO mesh vertices
    object_center: torch.Tensor    # [T, 3] Object center position
```

### MANO Parameters (51 dimensions):
- **Translation** (3): Global hand position
- **Pose** (45): 15 joints × 3 rotation parameters
- **Shape** (3): Hand shape parameters

## Model Architecture

### 1. HOISDF Tokenizer
- Encodes MANO parameters, SDFs, and contact information
- Outputs 256-dimensional tokens for transformer input
- Processes each timestep independently

### 2. Transition Transformer
- Takes tokens from both videos
- Uses attention to understand task patterns
- Generates:
  - MANO parameter predictions for transition
  - Task boundary detection
  - Transition quality scores
  - Task embeddings for contrastive learning

### 3. Transition Diffuser
- Refines transformer predictions
- Ensures smooth, contact-aware transitions
- Uses iterative denoising process
- Conditioned on transformer states and contact information

## Loss Functions

### 1. MANO Reconstruction Loss
```python
L_mano = MSE(predicted_mano, ground_truth_mano)
```
Ensures accurate hand pose prediction.

### 2. Contact Consistency Loss
```python
L_contact = penalty for movement during contact frames
```
Maintains physical realism during object interaction.

### 3. Smoothness Loss
```python
L_smooth = ||velocity||² + ||acceleration||²
```
Penalizes jerky movements for natural motion.

### 4. Boundary Detection Loss
```python
L_boundary = BCE(predicted_boundaries, true_boundaries)
```
Identifies task transition points.

### 5. Contrastive Loss
```python
L_contrastive = similarity loss for task embeddings
```
Ensures consistent task representations.

### 6. Diffusion Loss
```python
L_diffusion = E[||ε - ε_θ(x_t, t, c)||²]
```
Trains the denoising network.

## Usage Example

```python
# Initialize model
model = TransitionMergerModel(config).to(device)

# Load HOISDF outputs from two videos
hoisdf_outputs1 = load_hoisdf_outputs(video1)  # First video
hoisdf_outputs2 = load_hoisdf_outputs(video2)  # Second video

# Generate transition
outputs = model(hoisdf_outputs1, hoisdf_outputs2, 
                transition_length=30, mode='inference')

# Extract refined MANO parameters for transition
transition_mano = outputs['transformer']['refined_mano']
```

## Key Features

1. **Contact-Aware Transitions**: Maintains contact constraints during transitions
2. **Task Understanding**: Learns semantic differences between tasks
3. **Smooth Motion**: Diffusion model ensures natural trajectories
4. **Flexible Length**: Can generate transitions of any length

## Training Process

1. **Stage 1**: Train on paired video sequences with known transitions
2. **Stage 2**: Fine-tune diffusion model for smoother results
3. **Stage 3**: End-to-end training with all losses

## Output

The model outputs:
- Smooth MANO parameter trajectory for transition
- Task boundary predictions
- Quality scores for generated transition
- Can be directly used to animate MANO hand model

## Files

- `transition_merger_model.py`: Complete implementation
- `transition_merger.ipynb`: Usage examples and visualization
- `TRANSITION_MERGER_GUIDE.md`: This documentation

## Next Steps

1. Load your trained HOISDF model
2. Extract HOISDF outputs from your video pairs
3. Train the transition merger on your dataset
4. Generate smooth transitions between any two videos