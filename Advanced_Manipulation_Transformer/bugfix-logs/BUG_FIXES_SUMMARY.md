# Bug Fixes Summary - Advanced Manipulation Transformer

## Date: 2025-01-06

### Issues Fixed

1. **Import Error in train_advanced.py**
   - **Error**: `ImportError: cannot import name 'UnifiedManipulationModel' from 'models.unified_model'`
   - **Fix**: Changed to correct class name `UnifiedManipulationTransformer`

2. **Import Error for AdvancedTrainer**
   - **Error**: `ImportError: cannot import name 'AdvancedTrainer' from 'training.trainer'`
   - **Fix**: Changed to correct class name `ManipulationTrainer`

3. **KeyError: 'dexycb_root'**
   - **Error**: Config file uses `root_dir` but code expects `dexycb_root`
   - **Fix**: Updated all references from `config['data']['dexycb_root']` to `config['data']['root_dir']`
   - **Files affected**: train.py, train_advanced.py, all notebooks

4. **KeyError: 'batch_size'**
   - **Error**: Code looks for `config['data']['batch_size']` but it's in `config['training']['batch_size']`
   - **Fix**: Updated all references to use `config['training']['batch_size']`

5. **ManipulationTrainer Initialization**
   - **Error**: Wrong parameters passed to ManipulationTrainer constructor
   - **Fix**: Updated to pass only required parameters: `model`, `config`, `device`, `distributed`, `local_rank`
   - **Removed**: External creation of optimizer and criterion (trainer creates them internally)

6. **train_epoch and validate Methods**
   - **Error**: TrainingPipeline was trying to implement its own training logic
   - **Fix**: Delegated to ManipulationTrainer's built-in methods: `train_epoch()` and `validate()`

7. **EnhancedDexYCBDataset Parameter**
   - **Error**: `TypeError: EnhancedDexYCBDataset.__init__() got an unexpected keyword argument 'root_dir'`
   - **Fix**: Changed parameter name from `root_dir` to `dexycb_root` in dataset initialization

### Files Modified

1. `/train_advanced.py`
   - Fixed imports
   - Fixed trainer initialization
   - Delegated training/validation to ManipulationTrainer

2. `/train.py`
   - Fixed dataset initialization parameters
   - Fixed batch_size references

3. `/configs/default_config.yaml`
   - Added `dexycb_root` field for backward compatibility

4. `/notebooks/train_full_featured.ipynb`
   - Fixed imports
   - Fixed dataset initialization
   - Fixed model class name
   - Fixed trainer initialization

5. `/notebooks/train_barebones_debug.ipynb`
   - Fixed dataset initialization parameters

6. `/notebooks/train_advanced_manipulation.ipynb`
   - Fixed config key references
   - Fixed batch_size location in config

### How to Run

```bash
# Command line training
cd Advanced_Manipulation_Transformer
python train_advanced.py

# Or with custom settings
python train_advanced.py \
    training.batch_size=64 \
    training.learning_rate=5e-4 \
    optimizations.use_flash_attention=true

# Notebook training
# Open any of the notebooks in Jupyter and run all cells
```

### Expected Behavior

All scripts and notebooks should now:
1. Load configuration correctly
2. Initialize datasets without errors
3. Create models with proper class names
4. Initialize trainers with correct parameters
5. Run training loops using ManipulationTrainer's methods
6. Save checkpoints and log metrics properly

### Notes

- The ManipulationTrainer class handles its own optimizer and loss function creation
- Always use `config['data']['root_dir']` for the DexYCB dataset path
- Batch size is located in `config['training']['batch_size']`
- The correct model class is `UnifiedManipulationTransformer`
- The correct trainer class is `ManipulationTrainer`