# FlashAttention SigmaReparam Compatibility Fix

## Date: 2025-01-06

## Problem
**Error**: `AttributeError: 'SigmaReparam' object has no attribute 'weight'`
**File**: `optimizations/flash_attention.py`
**Context**: When applying FlashAttention to a model that uses SigmaReparam

## Root Cause
The `replace_with_flash_attention` function assumes that `out_proj` is a standard `nn.Linear` layer with direct access to `weight` and `bias`. However, in the UnifiedManipulationTransformer, some linear layers are wrapped with `SigmaReparam`, which stores the actual linear layer in a `linear` attribute.

Structure difference:
- Standard: `attention.out_proj.weight`
- With SigmaReparam: `attention.out_proj.linear.weight`

## Solution
Modified the weight copying logic to handle both cases:

```python
# Handle case where out_proj might be wrapped (e.g., SigmaReparam)
if hasattr(child.out_proj, 'weight'):
    # Standard linear layer
    flash_attn.out_proj.weight.copy_(child.out_proj.weight)
    if child.out_proj.bias is not None:
        flash_attn.out_proj.bias.copy_(child.out_proj.bias)
elif hasattr(child.out_proj, 'linear'):
    # Wrapped linear layer (like SigmaReparam)
    flash_attn.out_proj.weight.copy_(child.out_proj.linear.weight)
    if child.out_proj.linear.bias is not None:
        flash_attn.out_proj.bias.copy_(child.out_proj.linear.bias)
```

## Files Modified
- `/home/n231/231nProjectV2/Advanced_Manipulation_Transformer/optimizations/flash_attention.py`

## Testing
Run the test script to verify the fix:
```bash
python test_flash_sigma_fix.py
```

Expected output:
```
✅ Basic model works
✅ FlashAttention replacement succeeded
✅ Forward pass works
✅ FlashAttention applied to UnifiedManipulationTransformer
```

## Impact
This fix allows FlashAttention to be properly applied to models that use σ-reparameterization (SigmaReparam) for improved training stability. The fix maintains compatibility with both standard linear layers and wrapped variants.

## Additional Notes
- SigmaReparam is used in the model to implement σ-reparameterization for preventing mode collapse
- The fix checks for both standard and wrapped layer patterns
- FlashAttention will now work correctly with the UnifiedManipulationTransformer

## Related Issues
This is part of the series of fixes for the Advanced Manipulation Transformer, building on previous fixes for:
- TransformerEncoderLayer attribute access
- Mode collapse prevention module
- Parameter group optimization