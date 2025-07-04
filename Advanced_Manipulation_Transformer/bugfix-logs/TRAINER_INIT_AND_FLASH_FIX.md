# ManipulationTrainer Initialization and FlashAttention Fix

## Date: 2025-01-06

## Problem 1: ManipulationTrainer Initialization
**Error**: `TypeError: ManipulationTrainer.__init__() got an unexpected keyword argument 'criterion'`
**File**: `notebooks/train_full_featured.ipynb` (cell 14)

### Root Cause
The `ManipulationTrainer` class doesn't accept `criterion` or `optimizer` as constructor arguments. Instead, it creates its own optimizer and criterion internally based on the config.

### Solution
1. Create the trainer with only the required arguments
2. Replace the trainer's criterion and optimizer after creation if custom ones are needed

```python
# Create trainer first
trainer = ManipulationTrainer(
    model=model,
    config=config.training,
    device=device,
    distributed=False,
    local_rank=0
)

# Then replace criterion and optimizer
trainer.criterion = ComprehensiveLoss(config.loss)
trainer.optimizer = torch.optim.AdamW(param_groups)
trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(...)
```

## Problem 2: FlashAttention Not Detected
**Issue**: FlashAttention is installed but not being detected by the code

### Possible Causes
1. **Import path issues**: The module might be installed in a different Python environment
2. **Version mismatch**: Incompatible flash-attn version with PyTorch
3. **CUDA compatibility**: FlashAttention requires specific CUDA versions
4. **Import timing**: The check happens before the module is available

### Debugging Steps
1. Run the test script:
   ```bash
   python test_flash_installation.py
   ```

2. Check installation:
   ```bash
   pip show flash-attn
   ```

3. Verify in Python directly:
   ```python
   import flash_attn
   print(flash_attn.__file__)
   ```

### Common Solutions
1. **Reinstall in correct environment**:
   ```bash
   pip uninstall flash-attn
   pip install flash-attn --no-build-isolation
   ```

2. **Check CUDA compatibility**:
   - FlashAttention requires CUDA 11.6+
   - Must match PyTorch's CUDA version

3. **Import order**: Ensure flash_attn is imported after torch

## Updated Code Structure

The fixed cell 14 now:
1. Creates the trainer with basic arguments
2. Replaces the criterion with ComprehensiveLoss
3. Creates custom parameter groups for multi-rate learning
4. Replaces the optimizer and scheduler
5. Applies FP8 training if available

## Impact
- Training can now proceed with custom loss functions
- Multi-rate learning is properly configured
- FlashAttention can be debugged separately

## Next Steps
1. Run `python test_flash_installation.py` to debug FlashAttention
2. Re-run the notebook cells in order
3. Verify training starts successfully