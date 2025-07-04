# Train Full Featured Notebook - Fix Summary

## Quick Fix
To make the `train_full_featured.ipynb` notebook work, apply these changes:

### 1. Model Dictionary Input Support
The model expects dictionary input from the dataloader.

**File**: `models/unified_model.py`
- Modified `forward()` to accept dictionary input
- Added extraction of images and other fields from batch dict

### 2. Loss Function Key Mapping  
The dataset provides different key names than the loss expects.

**File**: `training/losses.py`
- Maps `hand_joints_3d` → `hand_joints`
- Maps `object_pose` → `object_poses`
- Maps `camera_intrinsics` → `intrinsics`

### 3. Autocast API Fix
PyTorch 2.5 uses different autocast API.

**File**: `training/trainer.py`
- Changed from `autocast(device_type='cuda', ...)` to `autocast(enabled=..., dtype=...)`

### 4. Dimension Mismatches
Several hardcoded dimensions didn't match config values.

**Files Modified**:
- `models/encoders/hand_encoder.py` - Split joint encoder into two paths
- `models/decoders/contact_decoder.py` - Use config's hidden_dim instead of hardcoded 1024

### 5. SigmaReparam Compatibility
SigmaReparam wrapping breaks PyTorch's MultiheadAttention.

**File**: `models/unified_model.py`
- Skip attention modules when applying sigma reparameterization
- Or disable with `use_sigma_reparam: False` in config

## Testing
The notebook should now:
1. Load the model without errors
2. Pass batches through the model
3. Compute losses correctly
4. Run the training loop

## Known Issues
- Pixel alignment module has dimension issues (can be disabled if needed)
- Some warnings about deprecated APIs (harmless)

## Quick Test
```python
# In the notebook, after creating model:
sample_batch = next(iter(train_loader))
outputs = model(sample_batch)
losses = criterion(outputs, sample_batch)
print(f"Total loss: {losses['total'].item()}")
```