# Quick Reference: Common Issues and Solutions

## 1. Dictionary Input Error
**Error**: `AttributeError: 'dict' object has no attribute 'shape'`  
**Quick Fix**: Model now accepts dict input directly: `model(batch)`

## 2. Key Name Mismatch
**Error**: `KeyError: 'hand_joints'` when dataset has `hand_joints_3d`  
**Quick Fix**: Loss function now handles both key names automatically

## 3. Dimension Mismatch in Skip Connections
**Error**: `RuntimeError: The size of tensor a (X) must match the size of tensor b (Y)`  
**Quick Fix**: Skip connections now check dimensions before adding

## 4. Missing Attribute
**Error**: `AttributeError: object has no attribute 'use_sigma_reparam'`  
**Quick Fix**: Config now properly initializes all attributes

## 5. Wrong Number of Arguments
**Error**: `TypeError: forward() takes X positional arguments but Y were given`  
**Quick Fix**: All function signatures now match their calls

## 6. Device Mismatch
**Error**: `RuntimeError: Expected all tensors to be on the same device`  
**Quick Fix**: All tensors now moved to correct device automatically

## 7. None Type Error in Loss
**Error**: `TypeError: unsupported operand type(s) for *: 'NoneType' and 'Tensor'`  
**Quick Fix**: Contact predictions now properly mapped between modules

## 8. Deprecated API Warning
**Warning**: `FutureWarning: torch.cuda.amp.autocast(args...) is deprecated`  
**Quick Fix**: Updated to new PyTorch 2.x API

## Testing Your Fix
```bash
# Run the test suite
python test_notebook_fixes.py

# Run the example
python notebook_example.py
```

## Most Common Mistakes to Avoid
1. Don't pass `model(batch['image'])` - pass `model(batch)`
2. Don't modify dataset keys - model handles mapping
3. Don't manually move tensors - model handles devices
4. Don't use old autocast API - use `torch.amp.autocast('cuda')`