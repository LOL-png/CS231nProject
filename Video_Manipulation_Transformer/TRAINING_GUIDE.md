# Video-to-Manipulation Transformer Training Guide

## ⚠️ IMPORTANT: Jupyter Notebook Users

**If you get mask shape errors** like `"The shape of the mask [8, 2] at index 1 does not match..."`:

1. **Restart the kernel**: Kernel → Restart & Clear Output
2. **Run all cells in order**: Cell → Run All

This is required because Jupyter caches old function definitions. If you ran the notebook before the fixes were applied, you must restart to clear the cache.

## Quick Start

For a quick test of Stage 1 training without Jupyter complications:
```bash
python quick_start_stage1.py
```

## Stage 1: Encoder Pre-training

### Using Jupyter Notebook
1. Open `train_stage1_notebook.ipynb`
2. **Important**: If you get errors, restart kernel and run all cells in order
3. The notebook includes visualizations and detailed explanations

### Common Issues and Fixes

#### 1. Variable Object Counts Error
**Error**: `RuntimeError: stack expects each tensor to be equal size`
**Fix**: Already implemented - DexYCBDataset now pads to `max_objects=10`

#### 2. Contact Encoder Dimension Mismatch
**Error**: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x512 and 256x256)`
**Fix**: Already implemented - Contact encoder now expects 512-dim inputs from hand/object encoders

#### 3. Hand Validation Mask Shape
**Error**: `IndexError: The shape of the mask [2, 3] at index 1 does not match...`
**Fix**: Already implemented - Mask creation now uses proper dimension reduction:
```python
valid_hands = ~torch.all(hand_gt.view(hand_gt.shape[0], -1) == -1, dim=1)
```

#### 4. MSE Loss Shape Warning
**Warning**: `Using a target size (torch.Size([2, 1, 21, 3])) that is different to the input size`
**Fix**: Already implemented - Properly squeeze hand ground truth tensors

#### 5. Hand Region Extraction Errors
**Warning**: `Could not extract hand regions: Input and output sizes should be greater than 0, but got input (H: 0, W: 76)`
**Cause**: Hand joints near image edges result in invalid crop regions
**Fix**: 
- Updated `get_hand_region` to ensure minimum crop size
- Set `extract_hand_regions=False` in training (not critical for performance)
- The hand encoder still works well with full image patches

### Key Architecture Details

#### Hand Pose Encoder
- Input: RGB patches (768 dim = 16×16×3)
- Hidden: 512 dimensions
- Layers: 6 transformer layers
- Heads: 8 attention heads
- Output: 21 3D joint positions

#### Object Pose Encoder
- Input: RGB patches (768 dim)
- Hidden: 512 dimensions
- Layers: 6 transformer layers
- Heads: 8 attention heads
- Output: Up to 10 object 6-DoF poses

#### Contact Detection Encoder
- Input: Features from hand (512) and object (512) encoders
- Hidden: 256 dimensions
- Layers: 4 transformer layers
- Heads: 8 attention heads
- Output: Contact points and interaction types

### Data Format

DexYCB samples contain:
- `color`: RGB image [3, H, W]
- `depth`: Depth image [H, W]
- `segmentation`: Segmentation mask [H, W]
- `object_poses`: Object 6-DoF poses [max_objects, 3, 4]
- `hand_joints_3d`: 3D hand joints [1, 21, 3]
- `hand_joints_2d`: 2D hand joints [1, 21, 2]
- `ycb_ids`: Object IDs [max_objects]
- `mano_betas`: Hand shape parameters [10]

### Training Parameters

Default configuration:
```python
config = {
    'batch_size': 8,
    'sequence_length': 8,
    'patch_size': 16,
    'image_size': [224, 224],
    'learning_rate': 1e-4,
    'num_epochs': 5,
    'grad_clip': 1.0,
}
```

### Evaluation Metrics

- **MPJPE**: Mean Per Joint Position Error (in mm) for hand pose
- **ADD**: Average Distance metric for object pose
- **Contact F1**: Precision/recall for contact detection

### Checkpoints

Saved to `checkpoints/stage1/`:
- `hand_encoder.pth`
- `object_encoder.pth`
- `contact_encoder.pth`

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Training epoch
- Configuration

## Stage 2: Frozen Encoder Training

(Coming soon - trains temporal fusion and action decoder)

## Stage 3: End-to-End Fine-tuning

(Coming soon - includes physics simulation gradients)