# Notebook Dictionary Input Fix

## Date: 2025-01-06

## Problem
The `train_full_featured.ipynb` notebook failed with:
```
AttributeError: 'dict' object has no attribute 'shape'
```
When the ModelDebugger passed a batch dictionary to the model.

## Root Causes
1. **Model Input Format**: `UnifiedManipulationTransformer.forward()` expected separate arguments (`images`, `mano_vertices`, etc.) but received a dictionary from the dataloader
2. **Loss Function Keys**: `ComprehensiveLoss` expected different key names than what the dataset provided
3. **Dimension Mismatch**: HandEncoder's joint_encoder expected 63 dimensions but received 256 when using image features
4. **Autocast API**: Using deprecated `device_type` parameter

## Solutions Applied

### 1. Model Dictionary Input Support
Modified `UnifiedManipulationTransformer.forward()` to handle both:
- Dictionary input from dataloaders
- Separate arguments for backward compatibility

```python
def forward(self, images=None, mano_vertices=None, camera_params=None, return_features=False, **kwargs):
    # Handle dictionary input
    if images is None and 'image' in kwargs:
        images = kwargs['image']
    elif isinstance(images, dict):
        batch_dict = images
        images = batch_dict.get('image')
        # Extract other fields...
```

### 2. Loss Function Key Mapping
Updated `ComprehensiveLoss` to handle dataset keys:
- `hand_joints_3d` (dataset) → `hand_joints` (loss)
- `object_pose` (singular) → `object_poses` (plural)
- `camera_intrinsics` → `intrinsics`

### 3. Model Output Format
Updated model outputs to match loss function expectations:
```python
outputs = {
    'hand_joints_coarse': hand_outputs['joints_3d'],
    'hand_joints': hand_outputs.get('joints_3d_refined', hand_outputs['joints_3d']),
    'object_poses': torch.cat([positions, rotations], dim=-1),
    # etc...
}
```

### 4. HandEncoder Dimension Fix
Split joint encoder into two paths:
- `joint_encoder_from_coords`: For 21*3 joint coordinates
- `joint_encoder_from_image`: For hidden_dim image features

### 5. Autocast API Update
Fixed deprecated autocast usage:
```python
# Old: with autocast(device_type='cuda', dtype=...)
# New: with autocast(enabled=..., dtype=...)
```

## Files Modified
- `models/unified_model.py` - Added dictionary input handling and fixed sigma reparam application
- `training/losses.py` - Updated key mapping and total loss calculation
- `training/trainer.py` - Fixed autocast API and train_step
- `models/encoders/hand_encoder.py` - Fixed dimension mismatch with separate encoders
- `models/decoders/contact_decoder.py` - Fixed hardcoded dimensions and added projection layers

## Testing
Run `python test_notebook_fixes.py` to verify all fixes.

## Impact
The notebook should now work correctly with:
- ModelDebugger analysis
- Training loop execution
- Loss computation with dataset batches