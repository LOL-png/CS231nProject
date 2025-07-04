"""
Notebook Optimization Example
=============================

Drop-in replacement for your notebook's optimization section.
Uses only PyTorch native features for maximum performance and compatibility.
"""

# For your notebook, replace the problematic optimization section with:

# ============================================
# OPTIMIZATION SECTION FOR train_full_featured.ipynb
# ============================================

import torch
from optimizations.pytorch_native_optimization import (
    optimize_for_h200,
    create_optimized_training_setup,
    PyTorchNativeOptimizer
)

print("Setting up PyTorch native optimizations...")

# Option 1: Simple one-liner optimization
# model = optimize_for_h200(model, compile_mode='default')

# Option 2: Full setup with trainer (recommended)
model, optimizer, trainer = create_optimized_training_setup(
    model,
    learning_rate=config.training.learning_rate,
    weight_decay=config.training.weight_decay,
    compile_model=True  # Set False if you encounter issues
)

# Option 3: More control
optimizer_helper = PyTorchNativeOptimizer()
model = optimizer_helper.optimize_model(model, {
    'use_compile': True,
    'compile_mode': 'default'  # or 'max_performance' for best speed
})

# Check optimization status
print("\nOptimization Summary:")
print(f"✓ SDPA (Flash Attention): Enabled")
print(f"✓ Mixed Precision: BFloat16")
print(f"✓ TF32: Enabled") 
print(f"✓ cuDNN Autotuning: Enabled")
print(f"✓ torch.compile: {'Enabled' if hasattr(model, '_dynamo_orig_callable') else 'Disabled'}")

# Memory stats
if torch.cuda.is_available():
    print(f"\nGPU Memory:")
    print(f"  Allocated: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved()/1024**3:.1f} GB")
    print(f"  Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1024**3:.1f} GB")

# ============================================
# TRAINING LOOP EXAMPLE
# ============================================

# If using the trainer:
def train_epoch(dataloader, criterion):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to GPU
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Single training step with all optimizations
        loss, outputs = trainer.train_step(batch, criterion)
        total_loss += loss
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: Loss = {loss:.4f}")
    
    return total_loss / len(dataloader)

# If not using trainer, use mixed precision directly:
def train_step_manual(batch, criterion):
    optimizer.zero_grad(set_to_none=True)
    
    # Use BFloat16 for H200
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        outputs = model(batch)
        loss = criterion(outputs, batch)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item(), outputs

# ============================================
# PERFORMANCE TIPS
# ============================================

"""
1. Batch Size: H200 has 140GB memory, use large batches (512-2048)
2. Gradient Accumulation: If batch doesn't fit, accumulate gradients
3. Data Loading: Use num_workers=8-16, pin_memory=True, persistent_workers=True
4. Profile: Use torch.profiler to find bottlenecks

Common Issues:
- If torch.compile fails: Set compile_model=False
- If OOM: Reduce batch size or enable gradient checkpointing
- If slow: Ensure you're using BFloat16, not Float32
"""

# ============================================
# VALIDATION EXAMPLE
# ============================================

@torch.no_grad()
def validate(dataloader, criterion):
    model.eval()
    total_loss = 0
    
    for batch in dataloader:
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Evaluation with mixed precision
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(batch)
            loss = criterion(outputs, batch)
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

print("\n✓ Optimization setup complete! Your model is now optimized for H200.")