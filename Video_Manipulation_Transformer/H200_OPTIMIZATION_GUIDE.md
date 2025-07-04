# H200 GPU Optimization Guide

## Current Underutilization Issues
- GPU Utilization: 20% (should be 90%+)
- Power: 172W/700W (25% of capacity)
- Memory: 2.5GB/140GB (1.8% usage!)

## Optimization Strategy

### 1. **Batch Size Scaling** (Most Important)
- Current: 8 samples
- Optimized: 256+ samples
- With gradient accumulation: 1024 effective batch size

### 2. **Mixed Precision Training**
- Use BFloat16 (better for H200 than FP16)
- Automatic Mixed Precision (AMP) with GradScaler
- 2x memory efficiency, ~2x speed

### 3. **Model Scaling**
- Increase hidden dimensions: 512 → 1024/2048
- More layers: 6 → 12 
- More attention heads: 8 → 16
- Total params: ~20M → ~200M+

### 4. **Data Loading Optimization**
```python
DataLoader(
    num_workers=16,          # Parallel data loading
    pin_memory=True,         # Faster GPU transfer
    prefetch_factor=4,       # Prefetch batches
    persistent_workers=True, # Keep workers alive
    drop_last=True          # Consistent batch sizes
)
```

### 5. **CUDA Optimizations**
```python
# Enable TF32 for H100/H200
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Compile models (PyTorch 2.0+)
model = torch.compile(model, mode='max-autotune')
```

### 6. **Gradient Accumulation**
- Simulate larger batches: 256 × 4 = 1024
- Better convergence with large batches
- Allows going beyond memory limits

### 7. **Memory Usage Targets**
- Current: 2.5GB
- Target: 100GB+ (still conservative for 140GB)
- Achievable with:
  - Batch size 256+
  - Scaled models
  - Sequence length increase

### 8. **Power/Utilization Targets**
- GPU Utilization: 90%+
- Power: 600W+ (85%+ of 700W)
- Memory bandwidth: Near 4.8TB/s limit

## Quick Start

Run the optimized training:
```bash
python train_stage1_optimized.py
```

## Monitoring Performance

```python
# In training loop
print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")
print(f"GPU Utilization: Check nvidia-smi")
```

## Expected Improvements
- Training speed: 10-20x faster
- GPU utilization: 20% → 90%+
- Memory usage: 2.5GB → 100GB+
- Power: 172W → 600W+
- Throughput: ~100 samples/sec → 2000+ samples/sec

## Further Optimizations
1. **Multi-GPU Training**: Use all available H200s
2. **Larger Models**: Vision Transformers with 1B+ params
3. **Longer Sequences**: Process 32-64 frames
4. **Higher Resolution**: 224×224 → 512×512
5. **FlashAttention**: For longer sequences

## Troubleshooting

### Out of Memory
- Reduce batch size gradually: 256 → 128 → 64
- Enable gradient checkpointing
- Use gradient accumulation

### Low GPU Utilization
- Increase num_workers
- Check data loading bottleneck
- Ensure no CPU preprocessing bottleneck

### Slow Training
- Check if using mixed precision
- Verify torch.compile is working
- Monitor data loading time vs compute time