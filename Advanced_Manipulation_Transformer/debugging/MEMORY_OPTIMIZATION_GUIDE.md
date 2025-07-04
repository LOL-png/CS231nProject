# Memory Optimization Guide for Model Debugging

## Common Memory Issues and Solutions

### Issue 1: OSError: [Errno 12] Cannot allocate memory
**Cause**: Creating multiprocessing queues when system is low on memory
**Solution**: Use simple iteration without DataLoader:

```python
# BAD: Uses multiprocessing which requires memory for queues
for batch in dataloader:
    process(batch)

# GOOD: Simple iteration without multiprocessing
val_iter = iter(dataset)
for i in range(num_batches):
    batch_samples = []
    for j in range(batch_size):
        sample = next(val_iter)
        batch_samples.append(sample)
    # Manually collate batch
    batch = collate_samples(batch_samples)
    process(batch)
```

### Issue 2: 100GB+ CPU Memory Usage
**Cause**: Storing all activations on CPU without limits
**Solution**: Use MemoryEfficientDebugger:

```python
from debugging.debugger_memory_fix import create_memory_efficient_debugger

# Automatically limits storage and uses GPU for computation
debugger = create_memory_efficient_debugger(
    model, 
    save_dir="debug_outputs",
    max_stored_activations=10,  # Limit activations
    max_stored_batches=5        # Limit batches for diversity check
)
```

### Issue 3: Gradient Accumulation During Debugging
**Cause**: Not using torch.no_grad()
**Solution**: Always wrap debugging code:

```python
with torch.no_grad():
    debugger.analyze_model(sample_batch)
    debugger.debug_prediction_diversity(dataloader)
```

### Issue 4: Large Tensor Storage
**Cause**: Storing full tensors when only statistics needed
**Solution**: Store summaries for large tensors:

```python
# For tensors > 1M elements, store only statistics
if tensor.numel() > 1e6:
    stats = {
        'mean': tensor.mean().item(),
        'std': tensor.std().item(),
        'norm': tensor.norm().item()
    }
    store_stats(stats)
else:
    store_tensor(tensor.detach())
```

### Issue 5: Memory Fragmentation
**Cause**: Not cleaning up between operations
**Solution**: Explicit cleanup:

```python
# After each major operation
del large_tensors
gc.collect()
torch.cuda.empty_cache()
```

## Best Practices

### 1. Move Data to GPU Immediately
```python
def move_batch_to_device(batch, device):
    gpu_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            gpu_batch[key] = value.to(device, non_blocking=True)
            # Delete CPU reference if it was on CPU
            if value.device.type == 'cpu':
                del value
        else:
            gpu_batch[key] = value
    gc.collect()
    return gpu_batch
```

### 2. Compute Statistics on GPU
```python
def compute_tensor_stats_gpu(tensor):
    with torch.no_grad():
        # All computation on GPU
        stats = {
            'mean': tensor.mean().item(),      # Only scalar to CPU
            'std': tensor.std().item(),
            'min': tensor.min().item(),
            'max': tensor.max().item(),
            'norm': tensor.norm().item()
        }
    return stats
```

### 3. Limited Hook Registration
```python
# Only hook important layers, not all
target_layers = ['encoder', 'decoder', 'attention']
for name, module in model.named_modules():
    if any(target in name for target in target_layers):
        register_hook(module)
```

### 4. Batch Processing for Large Models
```python
# Process in smaller chunks
chunk_size = 10
for i in range(0, len(data), chunk_size):
    chunk = data[i:i+chunk_size]
    process_chunk(chunk)
    # Clean up after each chunk
    gc.collect()
```

## Memory Profiling

### Check GPU Memory Usage
```python
def get_gpu_memory_info():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return {
            'allocated': allocated,
            'reserved': reserved,
            'free': torch.cuda.get_device_properties(0).total_memory / 1024**3 - reserved
        }
    return None
```

### Track Memory During Debugging
```python
class MemoryTracker:
    def __init__(self):
        self.initial = torch.cuda.memory_allocated()
    
    def checkpoint(self, name):
        current = torch.cuda.memory_allocated()
        delta = (current - self.initial) / 1024**3
        print(f"{name}: {delta:.2f} GB increase")
```

## Quick Fixes for Common Scenarios

### Notebook Running Out of Memory
```python
# Add at the start of each cell that might use memory
import gc
gc.collect()
torch.cuda.empty_cache()

# Run memory-intensive operations
# ...

# Clean up at the end
del large_variables
gc.collect()
torch.cuda.empty_cache()
```

### DataLoader Memory Issues
```python
# Reduce workers and disable persistent workers
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=0,  # No multiprocessing
    persistent_workers=False,
    pin_memory=False  # Don't pin memory
)
```

### Model Too Large for Debugging
```python
# Use gradient checkpointing during debugging
model.gradient_checkpointing_enable()

# Or process layer by layer
for name, module in model.named_modules():
    if should_analyze(name):
        analyze_module(module)
        # Clean up after each module
        gc.collect()
```

## When to Use What

- **Small models (<1B params)**: Original ModelDebugger is fine
- **Large models (>1B params)**: Always use MemoryEfficientDebugger
- **Low system memory**: Disable multiprocessing, use simple iteration
- **Production debugging**: Use summary statistics only, no full tensor storage

## Emergency Recovery

If notebook is stuck with high memory usage:
1. Interrupt kernel (multiple times if needed)
2. Restart kernel
3. Clear all outputs
4. Use memory-efficient approaches from the start