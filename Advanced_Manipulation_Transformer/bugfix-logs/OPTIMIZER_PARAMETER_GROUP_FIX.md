# Optimizer Parameter Group Fix

## Date: 2025-01-06

## Problem
**Error**: `ValueError: some parameters appear in more than one parameter group`
**File**: `notebooks/train_full_featured.ipynb` (cell 14)

## Root Cause
The parameter grouping logic was creating overlapping groups. The original code used list comprehensions that could include the same parameter in multiple groups:

```python
# Original problematic code
param_groups = [
    {
        'params': [p for n, p in model.named_parameters() if 'dinov2' in n],
        'lr': config.training.learning_rate * config.training.multi_rate.pretrained
    },
    {
        'params': [p for n, p in model.named_parameters() 
                  if 'encoder' in n and 'dinov2' not in n],  # This could still overlap!
        'lr': config.training.learning_rate * config.training.multi_rate.new_encoders
    },
    # ... more groups
]
```

The issue is that parameters with names like `base_model.encoder.dinov2...` would match both the 'dinov2' group and the 'encoder' group conditions.

## Solution
Implemented mutually exclusive parameter grouping:

```python
# Create mutually exclusive parameter groups
dinov2_params = []
encoder_params = []
decoder_params = []
other_params = []

for name, param in all_params.items():
    if 'dinov2' in name:
        dinov2_params.append(param)
    elif 'decoder' in name:
        decoder_params.append(param)
    elif 'encoder' in name:  # This catches encoder params that are NOT dinov2
        encoder_params.append(param)
    else:
        other_params.append(param)
```

The key is using `elif` statements to ensure each parameter goes into exactly one group.

## Additional Notes

### SE3Loss Issue
The cell also showed `TypeError: SE3Loss.__init__() got an unexpected keyword argument 'position_weight'`. This was already fixed in the `training/losses.py` file, but the notebook cell needed to be re-run after the fix.

### Parameter Group Summary
The fix now prints a summary of parameter groups:
```
Parameter groups:
  dinov2: X parameters, lr=1e-5
  encoders: Y parameters, lr=5e-4
  decoders: Z parameters, lr=1e-3
  other: W parameters, lr=1e-3
```

## Impact
This fix ensures:
1. Each parameter appears in exactly one group
2. Different parts of the model can have different learning rates
3. The optimizer can be created without errors
4. Multi-rate learning works as intended

## Testing
To verify the fix works:
1. Re-run the notebook cells in order
2. Check that cell 14 prints parameter group summary
3. Confirm no ValueError is raised
4. Training should proceed normally