# Bugfix Log: train_full_featured.ipynb Issues
**Date**: 2025-01-06  
**Fixed By**: Claude  
**Issue**: Multiple runtime errors preventing notebook execution

## Summary
Fixed critical issues in the Advanced Manipulation Transformer implementation that were preventing the `train_full_featured.ipynb` notebook from running. The main issue was that the model couldn't handle dictionary input from dataloaders, followed by several other compatibility and implementation bugs.

## Issues Fixed

### 1. Model Dictionary Input Handling
**Error**: `AttributeError: 'dict' object has no attribute 'shape'`  
**Location**: `models/unified_model.py`, line 192+  
**Root Cause**: Model expected tensor inputs but received dictionary from dataloader  
**Fix**: Modified `UnifiedManipulationTransformer.forward()` to handle both dictionary and tensor inputs:
```python
def forward(self, images=None, mano_vertices=None, camera_params=None, return_features=False, **kwargs):
    # Handle dictionary input (from data loader)
    if images is None and 'image' in kwargs:
        images = kwargs['image']
    elif isinstance(images, dict):
        batch_dict = images
        images = batch_dict.get('image')
        # Extract other fields...
```

### 2. Loss Function Key Mapping
**Error**: Loss function expected different key names than dataset provided  
**Location**: `training/losses.py`, multiple locations  
**Root Cause**: Dataset uses `hand_joints_3d` but loss expected `hand_joints`  
**Fix**: Added key mapping logic:
```python
# Handle both 'hand_joints' and 'hand_joints_3d' keys
hand_gt_key = 'hand_joints_3d' if 'hand_joints_3d' in targets else 'hand_joints'
```

### 3. Pixel Alignment Module Dimension Mismatch
**Error**: `RuntimeError: The size of tensor a (16) must match the size of tensor b (8) at non-singleton dimension 3`  
**Location**: `models/pixel_aligned.py`, line 89  
**Root Cause**: Incorrect skip connection implementation with pooling  
**Fix**: Proper residual connections:
```python
# Store input for skip connection
identity = feat_grid
# Apply conv, norm, activation
feat_grid = conv(feat_grid)
feat_grid = norm(feat_grid)
feat_grid = F.relu(feat_grid)
# Skip connection with matching dimensions
if i < len(self.feat_refiner) - 1:
    if identity.shape[1] == feat_grid.shape[1]:
        feat_grid = feat_grid + identity
```

### 4. Missing use_sigma_reparam Attribute
**Error**: `AttributeError: 'UnifiedManipulationTransformer' object has no attribute 'use_sigma_reparam'`  
**Location**: `models/unified_model.py`, line 117  
**Root Cause**: Attribute referenced before initialization  
**Fix**: Initialize attribute from config before use:
```python
self.use_sigma_reparam = config.get('use_sigma_reparam', True)
if self.use_sigma_reparam:
    self.apply_sigma_reparam()
```

### 5. SE3Loss Argument Mismatch
**Error**: `TypeError: SE3Loss.forward() takes 4 positional arguments but 5 were given`  
**Location**: `training/losses.py`, line 90  
**Root Cause**: SE3Loss expects 3 args but was passed 4  
**Fix**: Corrected call signature:
```python
losses['object_pose'] = self.object_pose_loss(
    obj_pred_pos,      # position
    obj_pred_rot,      # rotation  
    targets['object_pose']  # target pose matrix
)
```

### 6. Device Mismatch in HandJointLoss
**Error**: `RuntimeError: Expected all tensors to be on the same device`  
**Location**: `training/losses.py`, line 300  
**Root Cause**: Adaptive weights not moved to correct device  
**Fix**: Ensure weights on same device:
```python
# Ensure weights are on the same device as errors
adaptive_weights = adaptive_weights.to(joint_errors.device)
```

### 7. Contact Predictions Key Mismatch
**Error**: `TypeError: unsupported operand type(s) for *: 'NoneType' and 'Tensor'`  
**Location**: `training/losses.py`, line 435  
**Root Cause**: Model outputs `contact_confidence` but maps to `contact_probs` as None  
**Fix**: Correct key mapping in unified model:
```python
'contact_probs': contact_outputs.get('contact_confidence'),  # Map contact_confidence to contact_probs
```

### 8. Autocast API Deprecation
**Warning**: `FutureWarning: torch.cuda.amp.autocast(args...) is deprecated`  
**Location**: `training/trainer.py`, line 261  
**Root Cause**: Using old PyTorch autocast API  
**Fix**: Updated to new API:
```python
# Old: with autocast(device_type='cuda', dtype=torch.bfloat16):
# New:
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
```

## Testing Results
All fixes were validated with comprehensive tests:
- ✅ Model accepts dictionary input from dataloader
- ✅ Model accepts separate tensor arguments  
- ✅ Loss function handles dataset key names
- ✅ Training step completes without errors
- ✅ All loss components compute correctly

## Files Modified
1. `/models/unified_model.py` - Dictionary input handling, contact key mapping
2. `/training/losses.py` - Key mapping, device fixes, SE3Loss arguments
3. `/models/pixel_aligned.py` - Skip connection dimension fixes
4. `/models/encoders/hand_encoder.py` - Split joint encoders for different inputs
5. `/models/decoders/contact_decoder.py` - Fixed hardcoded dimensions
6. `/training/trainer.py` - Updated autocast API

## Verification
Run `test_notebook_fixes.py` to verify all fixes:
```bash
python test_notebook_fixes.py
```

## Example Usage
See `notebook_example.py` for a complete working example of how to use the fixed model in a notebook environment.