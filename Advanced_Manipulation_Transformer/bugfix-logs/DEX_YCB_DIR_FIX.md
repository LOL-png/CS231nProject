# DEX_YCB_DIR Environment Variable Fix

## Problem
All scripts and notebooks were failing with:
```
AssertionError: environment variable 'DEX_YCB_DIR' is not set
```

## Cause
The DexYCB toolkit requires the `DEX_YCB_DIR` environment variable to be set to locate the dataset files. This is checked in `dex_ycb_toolkit/dex_ycb.py`.

## Solution
Added the following line to all scripts and notebooks before any DexYCB imports:
```python
os.environ['DEX_YCB_DIR'] = '/home/n231/231nProjectV2/dex-ycb-toolkit/data'
```

## Files Updated
1. **Python Scripts**:
   - `train.py`
   - `train_advanced.py`
   - `inference.py`

2. **Notebooks**:
   - `notebooks/train_full_featured.ipynb`
   - `notebooks/train_barebones_debug.ipynb`
   - `notebooks/train_advanced_manipulation.ipynb`

## Where to Add
The environment variable must be set:
- **In scripts**: After initial imports but before importing any DexYCB-related modules
- **In notebooks**: In the first cell with imports, before adding project to path

## Example
```python
import os
import sys

# Set DEX_YCB_DIR environment variable
os.environ['DEX_YCB_DIR'] = '/home/n231/231nProjectV2/dex-ycb-toolkit/data'

# Now safe to import DexYCB-related modules
from data.enhanced_dexycb import EnhancedDexYCBDataset
```

## Alternative Approaches
You could also:
1. Set it in your shell before running: `export DEX_YCB_DIR=/path/to/dex-ycb-toolkit/data`
2. Add it to your `.bashrc` or `.zshrc` for permanent setting
3. Use a `.env` file with python-dotenv

But setting it in code ensures it works for everyone without manual setup.