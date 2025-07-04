"""
Notebook cells for GPU-optimized training with debugging support
Copy these cells into your notebook to fix the issues
"""

# Cell 1: Disable torch.compile for debugging
# Add this as the FIRST cell after imports
"""
# Disable torch.compile to prevent graph break warnings during debugging
import sys
sys.path.append('..')
from disable_compile_for_debug import disable_torch_compile

# Disable compilation for easier debugging
disable_torch_compile()
print("✓ Debugging mode enabled - no more graph break warnings!")
"""

# Cell 2: Replace standard dataloaders with GPU-cached versions
"""
# Replace the standard dataset loading with GPU-cached version
from data.gpu_cached_dataset import create_gpu_cached_dataloaders, GPUCachedDataset, GPUDataLoader

# Update config for GPU caching
config.update({
    'gpu_max_samples': 100000,  # Load 100k samples to GPU (adjust based on your GPU memory)
    'gpu_max_samples_val': 10000,  # Smaller for validation
    'gpu_cache_path': './gpu_cache',  # Where to save preprocessed data
    'use_bfloat16': True,  # Use bfloat16 to fit more data
    'preload_dinov2': False,  # Set True to pre-extract DINOv2 features
})

# Create GPU-cached dataloaders
print("Creating GPU-cached datasets...")
print("This will load the entire dataset into GPU memory for maximum performance")
print("First run will be slow (preprocessing), subsequent runs will be instant")

train_loader, val_loader = create_gpu_cached_dataloaders(config)

print(f"\\n✓ Dataloaders ready!")
print(f"  Batch size: {config['batch_size']}")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")
"""

# Cell 3: Alternative - Manual GPU dataset creation for more control
"""
# Alternative: Create GPU datasets manually for more control
from data.gpu_cached_dataset import GPUCachedDataset, GPUDataLoader

# Create train dataset - adjust max_samples based on your GPU memory
# Rule of thumb: ~1GB per 1000 samples
train_dataset = GPUCachedDataset(
    split='train',
    max_samples=100000,  # 100k samples ≈ 100GB GPU memory
    image_size=(224, 224),
    device='cuda',
    dtype=torch.bfloat16,  # Use bfloat16 to fit more data
    cache_path='./gpu_cache',
    normalize=True,
    load_dinov2_features=False  # Set True if using DINOv2 model
)

# Create smaller val dataset
val_dataset = GPUCachedDataset(
    split='val',
    max_samples=10000,  # 10k samples ≈ 10GB GPU memory
    image_size=(224, 224),
    device='cuda',
    dtype=torch.bfloat16,
    cache_path='./gpu_cache',
    normalize=True,
    load_dinov2_features=False
)

# Create loaders
train_loader = GPUDataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True)
val_loader = GPUDataLoader(val_dataset, batch_size=256, shuffle=False, drop_last=False)

print(f"GPU Memory Usage:")
print(f"  Train dataset: {train_dataset._get_memory_usage():.1f} GB")
print(f"  Val dataset: {val_dataset._get_memory_usage():.1f} GB")
print(f"  Total: {train_dataset._get_memory_usage() + val_dataset._get_memory_usage():.1f} GB")
"""

# Cell 4: Monitor GPU memory usage during training
"""
# Add this to your training loop to monitor GPU memory
import torch

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

# In your training loop:
for epoch in range(num_epochs):
    print(f"\\nEpoch {epoch+1}")
    print_gpu_memory()
    
    # Your training code here...
"""

# Cell 5: Optimize batch size for maximum GPU utilization
"""
# Find optimal batch size for your GPU
def find_optimal_batch_size(model, train_dataset, initial_batch_size=32, max_batch_size=1024):
    print("Finding optimal batch size...")
    
    batch_size = initial_batch_size
    best_batch_size = batch_size
    
    while batch_size <= max_batch_size:
        try:
            print(f"\\nTesting batch size: {batch_size}")
            
            # Create loader with test batch size
            loader = GPUDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Test forward and backward pass
            batch = next(iter(loader))
            outputs = model(batch)
            
            # Simulate loss computation
            if 'hand_joints' in outputs:
                loss = outputs['hand_joints'].mean()
                loss.backward()
            
            # Clear gradients
            model.zero_grad()
            torch.cuda.empty_cache()
            
            # If successful, this is our new best
            best_batch_size = batch_size
            print(f"  ✓ Batch size {batch_size} works!")
            print(f"  GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
            
            # Try larger batch size
            batch_size *= 2
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  ✗ Batch size {batch_size} causes OOM")
                break
            else:
                raise e
    
    print(f"\\nOptimal batch size: {best_batch_size}")
    return best_batch_size

# Find optimal batch size
optimal_batch_size = find_optimal_batch_size(model, train_dataset)
config['batch_size'] = optimal_batch_size

# Recreate loaders with optimal batch size
train_loader = GPUDataLoader(train_dataset, batch_size=optimal_batch_size, shuffle=True)
val_loader = GPUDataLoader(val_dataset, batch_size=optimal_batch_size, shuffle=False)
"""

# Cell 6: Performance comparison
"""
# Compare performance: Standard vs GPU-cached dataset
import time

def benchmark_dataloader(loader, num_batches=50):
    start_time = time.time()
    
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        # Simulate some computation
        _ = batch['image'].mean()
        torch.cuda.synchronize()
    
    elapsed = time.time() - start_time
    samples_per_sec = (num_batches * loader.batch_size) / elapsed
    
    return elapsed, samples_per_sec

# Benchmark GPU-cached loader
print("Benchmarking GPU-cached dataloader...")
elapsed, samples_per_sec = benchmark_dataloader(train_loader)
print(f"  Time for 50 batches: {elapsed:.2f}s")
print(f"  Throughput: {samples_per_sec:.0f} samples/sec")
print(f"  Batches/sec: {50/elapsed:.1f}")

# For comparison with standard dataloader (if you have one):
# print("\\nBenchmarking standard dataloader...")
# elapsed_std, samples_per_sec_std = benchmark_dataloader(standard_loader)
# print(f"  Speedup: {elapsed_std/elapsed:.1f}x faster with GPU caching")
"""

# Cell 7: Memory-efficient model initialization
"""
# If running out of memory, use this approach to initialize model more efficiently
import gc

# Clear any existing models
if 'model' in globals():
    del model
    gc.collect()
    torch.cuda.empty_cache()

# Initialize model with memory-efficient settings
if config.get('use_dinov2', True):
    # For DINOv2-based model, load with reduced precision
    from models.unified_model import UnifiedManipulationModel
    model = UnifiedManipulationModel(
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        num_refinement_steps=config['num_refinement_steps'],
        use_sigma_reparam=False,  # Disable for debugging
        freeze_dinov2_layers=12,  # Freeze more layers to save memory
        dinov2_model="facebook/dinov2-base"  # Use smaller model if needed
    )
else:
    # Use simple model for debugging
    from notebooks.cells_gpu_optimized import SimpleManipulationModel
    model = SimpleManipulationModel(config)

# Move to GPU with specific dtype
model = model.to(device, dtype=torch.bfloat16 if config.get('use_bfloat16', True) else torch.float32)

# Enable gradient checkpointing to save memory
if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable()
    print("✓ Gradient checkpointing enabled")

print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
print_gpu_memory()
"""

# Usage instructions
USAGE = """
GPU-Optimized Training Setup Instructions:

1. Add Cell 1 at the beginning of your notebook to disable torch.compile
2. Replace your data loading section with Cell 2 or Cell 3
3. Add Cell 4 to your training loop for memory monitoring
4. Use Cell 5 to find the optimal batch size for your GPU
5. Use Cell 6 to verify performance improvements
6. Use Cell 7 if you need more memory-efficient model initialization

Expected improvements:
- GPU memory usage: 15GB → 100GB+
- Training speed: 5-20x faster
- No more CPU bottlenecks
- No more graph break warnings

Tips:
- Start with max_samples=50000 and increase gradually
- Use bfloat16 to fit more data
- First run will be slow (caching), subsequent runs instant
- Monitor GPU memory to find your limit
"""

if __name__ == "__main__":
    print(USAGE)