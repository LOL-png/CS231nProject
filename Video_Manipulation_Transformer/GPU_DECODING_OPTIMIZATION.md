# GPU Decoding Optimization Guide

## Current Bottleneck

In the current implementation, image loading and preprocessing happens on CPU during dataset creation:

```python
# Current CPU-based loading in GPUOnlyDataset._build_dataset()
# Line ~100-106 in the embedded dataset class
img = cv2.imread(sample['color_file'])      # CPU: Disk I/O + JPEG decode
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # CPU: Color conversion
img = cv2.resize(img, self.image_size)      # CPU: Resize operation
img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
self.data['color'][i] = img_tensor.to(self.device, dtype=self.dtype)  # CPU→GPU transfer
```

**Bottleneck Impact**: 
- Loading 50,000 images: ~3-4 minutes
- Loading 200,000 images: ~15-20 minutes
- This only happens once due to caching, but could be optimized

## GPU Decoding Options

### 1. NVIDIA DALI (Recommended)
**Data Loading Library** - Purpose-built for GPU-accelerated data pipelines

**Pros:**
- Mature, production-ready
- Supports JPEG GPU decoding via nvJPEG
- Full pipeline on GPU (decode → resize → normalize → augment)
- Officially supported by NVIDIA
- Used in production by Meta, NVIDIA

**Cons:**
- Additional dependency
- Learning curve for pipeline API
- May need to restructure data loading

**Installation:**
```bash
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda120
```

### 2. TorchVision GPU Decoding
**Built into PyTorch ecosystem**

**Pros:**
- No extra dependencies
- Familiar API
- Improving rapidly

**Cons:**
- Still experimental
- Limited format support
- Not all operations GPU-accelerated

### 3. CuPy + nvJPEG
**Lower-level approach**

**Pros:**
- Fine-grained control
- Can optimize specific bottlenecks

**Cons:**
- More complex implementation
- Manual memory management

## Implementation Plan

### Where to Integrate in Current Setup

Replace the CPU loading section in `GPUOnlyDataset._build_dataset()` with:

```python
# Option 1: DALI Pipeline
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

@pipeline_def
def create_dali_pipeline(image_files, device_id=0):
    images = fn.readers.file(files=image_files, name="Reader")
    images = fn.decoders.image(images, device="mixed")  # CPU read, GPU decode
    images = fn.resize(images, size=(224, 224), device="gpu")
    images = fn.crop_mirror_normalize(images,
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225],
                                     device="gpu")
    return images

# In _build_dataset():
def _build_dataset_gpu_decode(self):
    """Build dataset with GPU decoding"""
    # Collect all image paths
    image_files = []
    for i in range(self.num_samples):
        sample = self.dex_dataset[i]
        image_files.append(sample['color_file'])
    
    # Create DALI pipeline
    pipe = create_dali_pipeline(image_files, device_id=0)
    pipe.build()
    
    # Process in batches
    batch_size = 256
    for batch_idx in range(0, self.num_samples, batch_size):
        pipe_out = pipe.run()
        gpu_images = pipe_out[0].as_tensor()  # Already on GPU!
        
        # Store in pre-allocated tensor
        end_idx = min(batch_idx + batch_size, self.num_samples)
        self.data['color'][batch_idx:end_idx] = gpu_images[:end_idx-batch_idx]
```

### Required Changes

1. **Dataset Class Modification**:
   - Add parameter: `use_gpu_decode=True`
   - Conditional path: GPU decode vs CPU decode
   - Batch processing for DALI (it works on batches)

2. **Dependencies to Add**:
   ```python
   # In requirements.txt or environment.yml
   nvidia-dali-cuda120>=1.32.0
   ```

3. **Memory Considerations**:
   - DALI uses additional GPU memory for decode buffers
   - Estimate: ~2-4GB overhead
   - May need to reduce max_samples slightly

4. **File Format Handling**:
   ```python
   # DALI supports: JPEG, PNG, BMP, TIFF
   # For other formats, fall back to CPU
   if sample['color_file'].endswith(('.jpg', '.jpeg', '.png')):
       # Use GPU decode
   else:
       # Fall back to CPU decode
   ```

## Performance Expectations

**Current CPU Approach**:
- 200,000 images: ~15 minutes
- Bottleneck: CPU decode + resize
- ~230 images/second

**With GPU Decoding**:
- 200,000 images: ~2-3 minutes (5-7x faster)
- Bottleneck: Disk I/O
- ~1,500+ images/second

**Note**: Actual speedup depends on:
- Storage speed (NVMe vs SSD vs HDD)
- Image sizes
- CPU vs GPU relative performance

## Future Implementation Checklist

When implementing GPU decoding:

- [ ] Install DALI: `pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda120`
- [ ] Verify JPEG format for all images (DALI requirement)
- [ ] Add `use_gpu_decode` parameter to GPUOnlyDataset
- [ ] Implement DALI pipeline for batch processing
- [ ] Add fallback for non-JPEG formats
- [ ] Test memory usage (may need to adjust batch size)
- [ ] Benchmark: time the loading phase before/after
- [ ] Handle edge cases (corrupted images, etc.)

## Alternative: Hybrid Approach

For immediate improvement without full GPU decode:

```python
# Use torch.cuda.Stream for overlapped transfers
stream = torch.cuda.Stream()

# CPU loads next batch while GPU processes current
with torch.cuda.stream(stream):
    # Transfer batch N+1 while processing batch N
    next_batch.to(device, non_blocking=True)
```

This won't decode on GPU but can hide some transfer latency.

## References

1. [NVIDIA DALI Documentation](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html)
2. [DALI GPU Decoding Examples](https://github.com/NVIDIA/DALI/tree/main/docs/examples)
3. [TorchVision GPU Transforms RFC](https://github.com/pytorch/vision/issues/4146)
4. [nvJPEG Documentation](https://docs.nvidia.com/cuda/nvjpeg/index.html)

## Notes for Implementation

- Start with a small subset (1000 images) to test
- Monitor GPU memory carefully during decode
- DALI can also do augmentations on GPU (rotations, color jitter, etc.)
- Consider keeping both CPU and GPU paths for flexibility
- GPU decode is most beneficial when not using caching (streaming data)