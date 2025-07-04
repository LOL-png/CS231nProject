# FlashAttention Setup Guide for Advanced Manipulation Transformer

## Overview
FlashAttention-3 provides 1.5-2x speedup over standard attention and is optimized for H200 GPUs. This guide ensures proper setup and usage.

## Current Status
- **Implementation**: ✅ Complete (`optimizations/flash_attention.py`)
- **Fallback**: ✅ Automatic fallback to standard attention if not available
- **Integration**: ✅ Can replace all MultiheadAttention modules automatically

## Installation

### Option 1: Install FlashAttention-3 (Recommended for H200)
```bash
# For CUDA 12.x and H200
pip install flash-attn --no-build-isolation

# Alternative: Build from source for latest optimizations
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install -e .
```

### Option 2: Install FlashAttention-2 (More stable)
```bash
# Stable version
pip install flash-attn==2.5.0
```

### Troubleshooting Installation
If installation fails:
```bash
# Check CUDA version
nvcc --version

# Install specific CUDA toolkit version
conda install -c nvidia cuda-toolkit=12.4

# Install with specific torch version
pip install flash-attn --no-build-isolation --no-deps
```

## Verification

### 1. Check if FlashAttention is Available
```python
# In Python or Jupyter notebook
from optimizations.flash_attention import FLASH_AVAILABLE
print(f"FlashAttention available: {FLASH_AVAILABLE}")

# Check which backend is being used
import torch
model = UnifiedManipulationTransformer(config)
model = replace_with_flash_attention(model)
# Will print warnings if FlashAttention not available
```

### 2. Performance Test
```python
import time
import torch
from optimizations.flash_attention import FlashAttention

# Create test tensors
batch_size, seq_len, embed_dim = 32, 196, 1024
x = torch.randn(batch_size, seq_len, embed_dim, device='cuda', dtype=torch.bfloat16)

# Test FlashAttention
flash_attn = FlashAttention(embed_dim, num_heads=16).cuda()
flash_attn = flash_attn.bfloat16()

# Warmup
for _ in range(10):
    _ = flash_attn(x)
torch.cuda.synchronize()

# Benchmark
start = time.time()
for _ in range(100):
    _ = flash_attn(x)
torch.cuda.synchronize()
flash_time = time.time() - start

print(f"FlashAttention time: {flash_time:.3f}s")
```

## Usage in Training

### Automatic Integration
FlashAttention is automatically used when:
1. Model has `use_flash_attention: true` in config
2. FlashAttention is installed
3. Using CUDA with float16 or bfloat16

```python
# In train.py or notebooks
if config.optimizations.use_flash_attention:
    model = replace_with_flash_attention(model)
```

### Manual Integration
```python
from optimizations.flash_attention import FlashAttention

# Replace specific attention module
model.hand_encoder.transformer.layers[0].self_attn = FlashAttention(
    embed_dim=1024,
    num_heads=16,
    dropout=0.1
)
```

## Performance Optimization Tips

### 1. Use BFloat16 (Recommended for H200)
```python
# Enable BFloat16
model = model.bfloat16()

# Or use automatic mixed precision
from torch.cuda.amp import autocast
with autocast(dtype=torch.bfloat16):
    output = model(input)
```

### 2. Optimize Sequence Length
- FlashAttention is most efficient when sequence length is divisible by 64
- Pad sequences if necessary:
```python
def pad_sequence_length(x, divisor=64):
    B, N, C = x.shape
    pad_len = (divisor - N % divisor) % divisor
    if pad_len > 0:
        x = F.pad(x, (0, 0, 0, pad_len))
    return x
```

### 3. Memory Settings
```python
# For H200 with 140GB memory
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
```

## Fallback Behavior

If FlashAttention is not available, the system automatically falls back to standard attention with these optimizations:
1. Efficient memory layout
2. Fused operations where possible
3. Proper scaling and numerical stability

## Expected Performance Gains

### With FlashAttention-3 on H200:
- **Attention Computation**: 1.5-2x faster
- **Memory Usage**: 10-20% reduction
- **Overall Training**: 20-30% faster
- **Larger Batch Sizes**: Can use 2-3x larger batches

### Benchmark Results (Expected on H200):
| Operation | Standard Attention | FlashAttention-3 | Speedup |
|-----------|-------------------|------------------|---------|
| Forward Pass | 100ms | 55ms | 1.8x |
| Backward Pass | 200ms | 95ms | 2.1x |
| Memory (32 batch) | 45GB | 38GB | 1.2x |

## Monitoring FlashAttention Usage

### During Training
```python
# The implementation logs when FlashAttention is used
# Look for messages like:
# "Replaced self_attn with FlashAttention"
# "FlashAttention not available, using standard attention"
```

### Profile Performance
```python
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    output = model(input)

print(prof.key_averages().table(sort_by="cuda_time_total"))
# Look for flash_attn kernels in the output
```

## Common Issues and Solutions

### Issue 1: ImportError
```
ImportError: cannot import name 'flash_attn_func'
```
**Solution**: Install FlashAttention with correct CUDA version

### Issue 2: Unsupported GPU
```
RuntimeError: FlashAttention only supports Ampere GPUs or newer
```
**Solution**: FlashAttention requires compute capability >= 8.0 (A100, H100, H200)

### Issue 3: Dtype Mismatch
```
RuntimeError: FlashAttention only supports fp16 and bf16
```
**Solution**: Use model.half() or model.bfloat16()

## Testing FlashAttention

Run this test script to verify FlashAttention is working:

```python
# test_flash_attention.py
import torch
from optimizations.flash_attention import FLASH_AVAILABLE, replace_with_flash_attention
from models.unified_model import UnifiedManipulationTransformer

print(f"FlashAttention available: {FLASH_AVAILABLE}")

if FLASH_AVAILABLE:
    # Create dummy model
    config = {'hidden_dim': 1024}
    model = UnifiedManipulationTransformer(config).cuda()
    
    # Replace with FlashAttention
    model = replace_with_flash_attention(model)
    
    # Test forward pass
    dummy_input = {
        'image': torch.randn(4, 3, 224, 224, device='cuda', dtype=torch.bfloat16)
    }
    
    model = model.bfloat16()
    with torch.no_grad():
        output = model(dummy_input)
    
    print("✅ FlashAttention test passed!")
else:
    print("⚠️ FlashAttention not installed, using standard attention")
```

## Next Steps

1. **Install FlashAttention**: Follow installation instructions above
2. **Verify Installation**: Run the test script
3. **Enable in Config**: Set `use_flash_attention: true` in your config
4. **Monitor Performance**: Check training logs for speedup
5. **Optimize Further**: Tune sequence lengths and batch sizes

## Additional Resources
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention)
- [H200 Optimization Guide](https://docs.nvidia.com/deeplearning/performance/index.html)