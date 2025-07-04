# Shared Bugs Analysis - Advanced Manipulation Transformer

## Date: 2025-01-06

## Summary of Shared Bug Patterns

### 1. ✅ Class Name Mismatch: `UnifiedManipulationModel` vs `UnifiedManipulationTransformer`

**Pattern**: Import of wrong class name
**Status**: FIXED in all occurrences

**Files affected**:
- ✅ `train_advanced.py` - Fixed
- ✅ `inference.py` - Fixed (found during analysis)
- ✅ `notebooks/train_full_featured.ipynb` - Fixed

**Files checked and clean**:
- ✓ `train.py` - Uses correct name
- ✓ `evaluation/evaluator.py` - Doesn't import the model
- ✓ Other notebooks - Don't use this class

### 2. ✅ Trainer Class Name: `AdvancedTrainer` vs `ManipulationTrainer`

**Pattern**: Import of wrong trainer class
**Status**: FIXED in all occurrences

**Files affected**:
- ✅ `train_advanced.py` - Fixed
- ✅ `notebooks/train_full_featured.ipynb` - Fixed

**Files checked and clean**:
- ✓ `train.py` - Creates trainer inline, doesn't import
- ✓ `inference.py` - Doesn't use trainer
- ✓ Other files - No additional occurrences found

### 3. ✅ Config Key Mismatch: `dexycb_root` vs `root_dir`

**Pattern**: Looking for wrong config key
**Status**: FIXED in all occurrences

**Files affected**:
- ✅ `train.py` - Fixed to use `root_dir`
- ✅ `train_advanced.py` - Fixed to use `root_dir`
- ✅ All notebooks - Fixed to use `root_dir`
- ✅ `configs/default_config.yaml` - Added `dexycb_root` for compatibility

**Note**: The dataset class (`EnhancedDexYCBDataset`) expects parameter named `dexycb_root`, which is correct.

### 4. ✅ Batch Size Location: `config['data']['batch_size']` vs `config['training']['batch_size']`

**Pattern**: Looking for batch_size in wrong config section
**Status**: FIXED in all occurrences

**Files affected**:
- ✅ `train.py` - Fixed
- ✅ All notebooks - Fixed

**Files checked and clean**:
- ✓ `train_advanced.py` - Already uses correct location
- ✓ `inference.py` - Doesn't use batch_size

### 5. ✅ Dataset Split Name Format: Hyphen vs Underscore

**Pattern**: Using wrong format for dataset splits
**Status**: FIXED in enhanced_dexycb.py

**Issue Details**:
- DexYCB factory expects: `s0_train` (underscore)
- Code was using: `s0-train` (hyphen)
- Also needed to handle: `train` (without prefix)

**Files affected**:
- ✅ `data/enhanced_dexycb.py` - Fixed to handle all formats

**Solution**: 
- Convert hyphens to underscores
- Add s0_ prefix if missing
- Handle both 'train' and 's0_train' formats

### 6. ✅ Missing DEX_YCB_DIR Environment Variable

**Pattern**: Missing required environment variable
**Status**: FIXED in all scripts and notebooks

**Error**: `AssertionError: environment variable 'DEX_YCB_DIR' is not set`

**Files affected**:
- ✅ `train.py` - Fixed
- ✅ `train_advanced.py` - Fixed
- ✅ `inference.py` - Fixed
- ✅ All notebooks - Fixed

**Solution**: 
Added `os.environ['DEX_YCB_DIR'] = '/home/n231/231nProjectV2/dex-ycb-toolkit/data'` to all files

### 7. ⚠️ Potential Issues Not Yet Encountered

#### A. Import Path Inconsistencies
Some files might have issues with import paths when run from different directories:
- `sys.path.append(os.path.dirname(os.path.abspath(__file__)))` in scripts
- `sys.path.insert(0, str(project_root))` in notebooks

**Recommendation**: Standardize to always use project root

#### B. Config Key Access Patterns
Different files use different patterns to access config:
- Direct access: `config['key']`
- Safe access: `config.get('key', default)`
- OmegaConf access: `config.key`

**Recommendation**: Standardize to safe access with defaults

#### C. Device Handling
Some files assume CUDA availability without checking:
- Should always check `torch.cuda.is_available()`
- Should handle CPU fallback gracefully

## Files Systematically Checked

### Python Scripts
- ✅ `train.py`
- ✅ `train_advanced.py` 
- ✅ `inference.py`
- ✓ `evaluation/evaluator.py`
- ✓ `data/enhanced_dexycb.py`
- ✓ `data/augmentation.py`
- ✓ `models/unified_model.py`
- ✓ `training/trainer.py`

### Notebooks
- ✅ `notebooks/train_full_featured.ipynb`
- ✅ `notebooks/train_barebones_debug.ipynb`
- ✅ `notebooks/train_advanced_manipulation.ipynb`

## Recommendations

1. **Add Type Hints**: Use type hints to catch these errors earlier
2. **Add Config Validation**: Validate config structure at startup
3. **Standardize Imports**: Create a central imports file
4. **Add Unit Tests**: Test imports and config access
5. **Use Constants**: Define class names as constants to avoid typos

## Command to Find Similar Issues

```bash
# Find potential class name mismatches
grep -r "UnifiedManipulation" --include="*.py" --include="*.ipynb"

# Find config access patterns
grep -r "config\['data'\]" --include="*.py"
grep -r "config\['training'\]" --include="*.py"

# Find import statements
grep -r "from.*import.*" --include="*.py" | grep -E "(Trainer|Model)"
```