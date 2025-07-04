# FlashAttention Import Debugging Guide

## Date: 2025-01-06

## Issue
FlashAttention is installed but not being detected by the code.

## Common Import Patterns

The package name and import name often differ:
- **Install name**: `pip install flash-attn` (with hyphen)
- **Import name**: `from flash_attn import ...` (with underscore)

However, this can vary between versions:

### FlashAttention v1:
```python
from flash_attn.flash_attention import FlashAttention
```

### FlashAttention v2:
```python
from flash_attn import flash_attn_func
# or
from flash_attn.flash_attn_interface import flash_attn_func
```

### Some installations:
```python
import flash_attention  # Different module name entirely
```

## Debugging Steps

1. **Run the debug script**:
   ```bash
   python debug_flash_import.py
   ```

2. **Check different import patterns**:
   ```bash
   python check_flash_versions.py
   ```

3. **Use the robust version**:
   If the standard `flash_attention.py` doesn't work, try:
   ```python
   # Replace in your code
   from optimizations.flash_attention_robust import replace_with_flash_attention
   ```

## Quick Checks

1. **Verify installation**:
   ```bash
   pip show flash-attn
   conda list flash
   ```

2. **Check Python path**:
   ```python
   import sys
   print(sys.path)
   ```

3. **Find the module**:
   ```python
   import importlib
   spec = importlib.util.find_spec('flash_attn')
   print(spec.origin if spec else "Not found")
   ```

## Common Solutions

1. **Wrong environment**:
   ```bash
   conda activate your_env
   which python
   which pip
   ```

2. **Reinstall with correct CUDA**:
   ```bash
   pip uninstall flash-attn
   pip install flash-attn --no-build-isolation
   ```

3. **Build from source**:
   ```bash
   git clone https://github.com/Dao-AILab/flash-attention.git
   cd flash-attention
   python setup.py install
   ```

4. **Use the robust import**:
   The `flash_attention_robust.py` file tries multiple import patterns automatically.

## Environment Variables

Sometimes needed:
```bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## Fallback

If FlashAttention cannot be imported, the code will automatically fall back to standard PyTorch attention. This is slower but functionally identical.

## Testing

After fixing, test with:
```python
from optimizations.flash_attention import FLASH_AVAILABLE
print(f"FlashAttention available: {FLASH_AVAILABLE}")
```

The training will work regardless - FlashAttention is an optimization, not a requirement.