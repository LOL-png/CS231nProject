# Complete Advanced Manipulation Transformer Bug Fixes

## Date: 2025-01-06

## Project Overview
The Advanced Manipulation Transformer is a sophisticated deep learning system for robotic manipulation using:
- DINOv2 pretrained backbone
- σ-reparameterization for mode collapse prevention
- FlashAttention-3 for efficient training
- FP8 mixed precision for H200 GPUs
- Multi-rate learning for different model components

## All Bug Fixes Applied

### 1. Import and Configuration Errors

#### Bug: UnifiedManipulationModel Import Error
- **File**: `train.py`, `train_advanced.py`
- **Fix**: Changed import from `models.advanced_model` to `models.unified_model`
```python
from models.unified_model import UnifiedManipulationTransformer
```

#### Bug: AdvancedTrainer Import Error
- **File**: `train_advanced.py`
- **Fix**: Changed to ManipulationTrainer
```python
from training.trainer import ManipulationTrainer
```

#### Bug: Dataset Root Configuration
- **File**: Multiple training scripts
- **Fix**: Changed from `config['dexycb_root']` to `config['data']['dexycb_root']`

#### Bug: Batch Size Configuration
- **File**: Multiple training scripts
- **Fix**: Changed from `config['batch_size']` to `config['training']['batch_size']`

### 2. Dataset Issues

#### Bug: Dataset Split Naming
- **File**: `data/enhanced_dexycb.py`
- **Issue**: 's0-test' doesn't exist in DexYCB
- **Fix**: Maps to 's0_val' with warning
```python
if split == 's0-test':
    print("Warning: 's0-test' not found, using 's0_val' instead")
    split = 's0_val'
```

#### Bug: DEX_YCB_DIR Environment Variable
- **File**: Multiple scripts
- **Fix**: Set environment variable before imports
```python
import os
os.environ['DEX_YCB_DIR'] = '/home/n231/231nProjectV2/dex-ycb-toolkit/data'
```

### 3. Model Architecture Bugs

#### Bug: SigmaReparam Dimension Mismatch
- **File**: `models/unified_model.py`
- **Issue**: RuntimeError when input/output dimensions differ
- **Fix**: Added dimension checking in SigmaReparam
```python
if in_features != out_features:
    self.w_scale = nn.Parameter(torch.ones(out_features))
else:
    self.register_parameter('w_scale', None)
```

#### Bug: SimpleLoss UnboundLocalError
- **File**: `training/losses.py`
- **Issue**: `pred_joints` referenced before assignment
- **Fix**: Initialize at start of forward()
```python
def forward(self, predictions, targets):
    pred_joints = None  # Initialize
    total_loss = 0.0
```

### 4. Training Infrastructure Bugs

#### Bug: Learning Rate Type Error
- **File**: `training/trainer.py`
- **Issue**: Can't multiply sequence by float
- **Fix**: Handle both scalar and list learning rates
```python
if isinstance(self.config.learning_rate, (list, tuple)):
    lr = self.config.learning_rate[0]
else:
    lr = self.config.learning_rate
```

#### Bug: Missing Trainer Methods
- **File**: `training/trainer.py`
- **Fix**: Added missing methods:
  - `get_lr()`
  - `get_gradient_norm()`
  - `scheduler_step()`
  - `train_step()`

### 5. Mode Collapse Prevention Bugs

#### Bug: TransformerEncoderLayer Attribute Access
- **File**: `solutions/mode_collapse.py`
- **Issue**: 'd_model' attribute doesn't exist
- **Fix**: Extract from internal layers
```python
d_model = module.self_attn.embed_dim
nhead = module.self_attn.num_heads
```

#### Bug: ImprovedTransformerLayer Signature
- **File**: `solutions/mode_collapse.py`
- **Issue**: Forward method signature mismatch
- **Fix**: Updated to match PyTorch's TransformerEncoderLayer
```python
def forward(self, src, src_mask=None, src_key_padding_mask=None):
    # Removed is_causal parameter
```

### 6. Loss Function Bugs

#### Bug: SE3Loss Missing Parameter
- **File**: `training/losses.py`
- **Issue**: Missing position_weight parameter
- **Fix**: Added __init__ method
```python
class SE3Loss(nn.Module):
    def __init__(self, position_weight: float = 1.0):
        super().__init__()
        self.position_weight = position_weight
```

### 7. Optimizer Configuration Bugs

#### Bug: Parameter Group Overlap
- **File**: `notebooks/train_full_featured.ipynb`
- **Issue**: "parameters appear in multiple groups"
- **Fix**: Use mutually exclusive parameter sets
```python
# Collect all special parameters
special_params = set()
special_params.update(dinov2_params)
special_params.update(head_params)
special_params.update(sigma_params)

# Other params are everything else
other_params = [p for p in model.parameters() if p not in special_params]
```

#### Bug: ManipulationTrainer Initialization
- **File**: `notebooks/train_full_featured.ipynb`
- **Issue**: Unexpected keyword arguments
- **Fix**: Create trainer first, then replace components
```python
trainer = ManipulationTrainer(model=model, config=config.training, device=device)
trainer.criterion = ComprehensiveLoss(config.loss)
trainer.optimizer = torch.optim.AdamW(param_groups)
```

### 8. FlashAttention Issues

#### Bug: FlashAttention Not Detected
- **Issue**: Package installed but import fails
- **Solution**: Created robust import handler
- **Files**: 
  - `optimizations/flash_attention_robust.py`
  - `debug_flash_import.py`
  - `check_flash_versions.py`

#### Bug: FlashAttention SigmaReparam Compatibility
- **File**: `optimizations/flash_attention.py`
- **Issue**: AttributeError: 'SigmaReparam' object has no attribute 'weight'
- **Fix**: Handle wrapped linear layers
```python
if hasattr(child.out_proj, 'weight'):
    flash_attn.out_proj.weight.copy_(child.out_proj.weight)
elif hasattr(child.out_proj, 'linear'):
    # Handle wrapped linear layers (like SigmaReparam)
    flash_attn.out_proj.weight.copy_(child.out_proj.linear.weight)
```

## Testing and Verification

### Test Scripts Created
1. `test_training_pipeline.py` - Tests complete training pipeline
2. `test_mode_collapse_fix.py` - Verifies mode collapse prevention
3. `test_flash_sigma_fix.py` - Tests FlashAttention with SigmaReparam
4. `debug_flash_import.py` - Debugs FlashAttention imports
5. `check_flash_versions.py` - Checks different import patterns

### Notebooks Fixed
1. `train_barebones_debug.ipynb` - Basic training loop
2. `train_advanced_manipulation.ipynb` - Advanced features
3. `train_full_featured.ipynb` - Complete system with all optimizations

## Key Learnings

1. **Configuration Structure**: Always verify nested configuration paths
2. **Import Names**: Package names (pip) often differ from import names
3. **Model Wrapping**: When using wrappers like SigmaReparam, handle attribute access carefully
4. **Parameter Groups**: Ensure parameters don't appear in multiple optimizer groups
5. **PyTorch Internals**: TransformerEncoderLayer stores attributes in sub-modules
6. **Environment Setup**: Set environment variables before any imports

## Current Status
✅ All identified bugs have been fixed
✅ Training scripts are functional
✅ FlashAttention is properly integrated
✅ Mode collapse prevention is working
✅ All test scripts pass successfully

## Next Steps
1. Run full training with fixed code
2. Monitor for mode collapse using provided metrics
3. Tune hyperparameters based on H200 performance
4. Implement Stage 2 temporal fusion if needed