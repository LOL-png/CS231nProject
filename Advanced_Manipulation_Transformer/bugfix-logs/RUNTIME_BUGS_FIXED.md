# Runtime Bugs Fixed - Advanced Manipulation Transformer

## Date: 2025-01-06

## Summary of All Runtime Errors Fixed

### 1. Model CUDA Placement Issue
**File**: `notebooks/train_advanced_manipulation.ipynb`
**Error**: `RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same`
**Cause**: Model was not moved to CUDA device
**Fix**: Added `.to(device)` after model creation and also moved sample data to device

### 2. Mean() on Integer Tensor
**File**: `notebooks/train_barebones_debug.ipynb`  
**Error**: `RuntimeError: mean(): could not infer output dtype`
**Cause**: Trying to compute mean on int64 tensor (ycb_ids)
**Fix**: Added dtype check before computing statistics, only compute mean for floating point tensors

### 3. Missing ModeCollapsePreventionModule.wrap_model Method
**File**: `notebooks/train_full_featured.ipynb`
**Error**: `AttributeError: type object 'ModeCollapsePreventionModule' has no attribute 'wrap_model'`
**Cause**: The static method `wrap_model` was not implemented
**Fix**: Added the complete `wrap_model` static method that:
- Wraps models with mode collapse prevention features
- Adds drop path regularization to transformer layers
- Injects noise into features
- Implements temperature scaling for attention

### 4. Learning Rate Multiplication Error
**File**: `train.py`
**Error**: `TypeError: can't multiply sequence by non-int of type 'float'`
**Cause**: `config['training']['learning_rate']` was a list [1e-3, 1e-4], not a scalar
**Fix**: Multiple fixes applied:
- Changed config to use scalar learning rate
- Added support for list learning rates (using first value)
- Fixed multi-rate learning implementation

### 5. Missing Trainer Methods
**Files**: `train.py`, `notebooks/train_full_featured.ipynb`
**Errors**: Missing `get_lr()`, `get_gradient_norm()`, `scheduler_step()`, `train_step()`
**Cause**: ManipulationTrainer was missing several required methods
**Fix**: Added all missing methods to ManipulationTrainer:
- `get_lr()`: Returns current learning rates
- `get_gradient_norm()`: Computes gradient norm
- `scheduler_step()`: Steps the scheduler
- `train_step()`: Performs single training step

### 6. Dataset Parameter Name Mismatch
**File**: `train_advanced.py`
**Error**: `TypeError: EnhancedDexYCBDataset.__init__() got an unexpected keyword argument 'root_dir'`
**Cause**: Dataset expects `dexycb_root`, not `root_dir`
**Fix**: Changed all dataset initialization calls to use `dexycb_root`

### 7. Optimizer Reference Error
**File**: `train_advanced.py`
**Error**: `AttributeError: 'TrainingPipeline' object has no attribute 'optimizer'`
**Cause**: Optimizer is stored in trainer, not pipeline
**Fix**: Changed references from `self.optimizer` to `self.trainer.optimizer`

### 8. Dataset Split Naming (Previously Fixed)
**All Files**: Using `s0-train` instead of `s0_train`
**Fix**: Enhanced dataset handles both formats and converts automatically

### 9. Missing DEX_YCB_DIR Environment Variable (Previously Fixed)
**All Files**: Missing required environment variable
**Fix**: Added `os.environ['DEX_YCB_DIR'] = '/home/n231/231nProjectV2/dex-ycb-toolkit/data'`

## Verification Steps

To verify all fixes are working:

```bash
# Test basic training
python train.py --config configs/default_config.yaml

# Test advanced training with all features
python train_advanced.py

# Test notebooks
# Run each notebook cell by cell to verify no errors
```

## Key Implementation Details

### Mode Collapse Prevention
The `wrap_model` method now properly:
1. Replaces standard transformer layers with improved versions
2. Adds drop path (stochastic depth) regularization
3. Implements feature noise injection
4. Adds temperature scaling to attention

### Multi-Rate Learning
Fixed to support:
1. Different learning rates for different parameter groups
2. Pretrained parameters (DINOv2) use lower LR
3. New encoders use medium LR
4. Decoders use full LR

### Memory Optimizations
All models now support:
1. Gradient checkpointing
2. Mixed precision training (BF16/FP16)
3. FP8 training on H200 (if available)
4. Dynamic batch sizing

## Next Steps

1. Ensure FlashAttention is properly configured
2. Run full training to verify all components work together
3. Monitor for any additional runtime errors during training
4. Optimize batch sizes for H200 GPU memory