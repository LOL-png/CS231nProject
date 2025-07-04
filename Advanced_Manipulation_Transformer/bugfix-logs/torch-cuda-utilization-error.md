# Bug Fix: torch.cuda.utilization() Error

## Date: 2025-01-06
## File: `notebooks/train_full_featured.ipynb`
## Cell: 23, Line: ~91

## Problem
The training loop in cell 23 contained a call to `torch.cuda.utilization()` which is not a valid PyTorch function:

```python
wandb.log({
    'train/loss': loss_value,
    'train/hand_mpjpe': mpjpe_value,
    'train/lr': manipulation_trainer.optimizer.param_groups[0]['lr'],
    'train/grad_norm': history['gradient_norms'][-1] if history['gradient_norms'] else 0,
    'system/gpu_memory_gb': torch.cuda.memory_allocated() / 1e9,
    'system/gpu_utilization': torch.cuda.utilization()  # ERROR: This function doesn't exist
})
```

## Root Cause
PyTorch does not have a built-in `torch.cuda.utilization()` function. GPU utilization monitoring requires external libraries like `pynvml` (NVIDIA Management Library).

## Solution
Commented out the GPU utilization logging line:

```python
wandb.log({
    'train/loss': loss_value,
    'train/hand_mpjpe': mpjpe_value,
    'train/lr': manipulation_trainer.optimizer.param_groups[0]['lr'],
    'train/grad_norm': history['gradient_norms'][-1] if history['gradient_norms'] else 0,
    'system/gpu_memory_gb': torch.cuda.memory_allocated() / 1e9,
    # GPU utilization requires pynvml library - commented out
    # 'system/gpu_utilization': get_gpu_utilization() 
})
```

## Alternative Solutions
If GPU utilization monitoring is needed, you can:

1. Install `pynvml` and use it:
```python
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
```

2. Use the `gpustat` package:
```python
import gpustat
gpu_stats = gpustat.new_query()
utilization = gpu_stats[0].utilization
```

3. Monitor externally with tools like `nvidia-smi` or `nvitop`

## Impact
- This was preventing the training loop from running
- GPU memory tracking still works (using `torch.cuda.memory_allocated()`)
- All other metrics are still logged properly

## Testing
The fix has been tested and the training loop should now run without errors. GPU memory usage is still tracked and displayed in the progress bar and logs.