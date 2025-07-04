# Fix for CUDA Fork Error

## Problem
When running the notebook with `num_workers > 0`, you get:
```
RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
```

## Solution

The fix has already been applied in the notebook:

1. **Set multiprocessing start method to 'spawn'** at the beginning:
```python
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
```

2. **Set num_workers to 0** in the configuration:
```python
config = {
    'num_workers': 0,  # CRITICAL: Set to 0 to avoid CUDA fork error
    ...
}
```

## Why This Happens

- PyTorch's DataLoader uses multiprocessing for parallel data loading
- CUDA doesn't support the default 'fork' method on Linux
- Setting num_workers=0 disables multiprocessing entirely
- This is fine for H200 because the GPU is so fast that CPU data loading isn't the bottleneck

## Alternative Solutions

If you need parallel data loading:

1. **Use spawn method** (already done in notebook)
2. **Use GPU-cached dataset** (see cells about GPU-only dataset)
3. **Pre-process data** and save to disk in the format needed

## Performance Impact

With the H200 GPU and our optimizations:
- GPU preprocessing eliminates most CPU work
- FastDataCollator moves data directly to GPU
- The impact of num_workers=0 is minimal
- For maximum performance, use the GPU-only dataset approach