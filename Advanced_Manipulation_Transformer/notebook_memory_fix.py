"""
Memory optimization fix for train_full_featured.ipynb

Use this instead of the problematic memory optimization code.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

def setup_model_with_safe_optimizations(model, config):
    """
    Setup model with safe memory optimizations that avoid compatibility issues
    
    Args:
        model: The UnifiedManipulationTransformer model
        config: Training configuration
    
    Returns:
        Optimized model
    """
    # 1. Enable gradient checkpointing selectively
    # This saves memory without compatibility issues
    def enable_gradient_checkpointing(module):
        if hasattr(module, 'gradient_checkpointing_enable'):
            module.gradient_checkpointing_enable()
            logger.info(f"Enabled gradient checkpointing for {module.__class__.__name__}")
    
    # Apply to image encoder (DINOv2)
    if hasattr(model, 'image_encoder'):
        enable_gradient_checkpointing(model.image_encoder)
    
    # 2. Set memory-efficient CUDA options
    if torch.cuda.is_available():
        # Enable TF32 for A100/H100/H200 GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("Enabled TF32 for faster computation")
        
        # Enable cudnn benchmarking for optimized kernels
        torch.backends.cudnn.benchmark = True
        logger.info("Enabled cuDNN benchmarking")
        
        # Set memory fraction to prevent OOM
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        # Clear cache before starting
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # 3. Use PyTorch 2.0+ optimizations if available
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        logger.info("PyTorch 2.0+ detected, SDPA will be used automatically for attention")
        # This happens automatically in PyTorch 2.0+
    
    # 4. Model-specific optimizations
    # Set model to use memory-efficient configurations
    if hasattr(model, 'config'):
        # Reduce intermediate activation memory
        if 'hidden_dim' in model.config and model.config['hidden_dim'] > 1024:
            logger.warning("Large hidden_dim detected. Consider reducing for memory efficiency.")
    
    # 5. Don't use XFormers - it has compatibility issues
    # Don't use experimental features that might break
    
    logger.info("Model optimized with safe memory settings")
    return model


def get_memory_stats():
    """Get current GPU memory statistics"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
            'free': (torch.cuda.get_device_properties(0).total_memory - 
                    torch.cuda.memory_allocated()) / 1024**3       # GB
        }
    return None


def optimize_dataloader_memory(dataloader_config):
    """
    Optimize DataLoader configuration for memory efficiency
    
    Args:
        dataloader_config: Dictionary with DataLoader settings
    
    Returns:
        Optimized configuration
    """
    optimized = dataloader_config.copy()
    
    # Key optimizations
    optimized.update({
        'pin_memory': True,          # Faster GPU transfer
        'persistent_workers': True,   # Avoid worker recreation
        'prefetch_factor': 2,        # Moderate prefetching
        'num_workers': min(8, dataloader_config.get('num_workers', 4))  # Cap workers
    })
    
    return optimized


# Example usage in notebook:
if __name__ == "__main__":
    # Example configuration
    print("Example usage:")
    print("""
# In your notebook, replace the memory optimization section with:

from notebook_memory_fix import setup_model_with_safe_optimizations, get_memory_stats

# Setup model with safe optimizations
model = setup_model_with_safe_optimizations(model, config)

# Check memory usage
mem_stats = get_memory_stats()
if mem_stats:
    print(f"GPU Memory - Allocated: {mem_stats['allocated']:.1f}GB, Free: {mem_stats['free']:.1f}GB")

# Continue with training...
""")
    
    # Show current memory stats
    stats = get_memory_stats()
    if stats:
        print(f"\nCurrent GPU memory:")
        print(f"  Allocated: {stats['allocated']:.2f} GB")
        print(f"  Reserved: {stats['reserved']:.2f} GB")
        print(f"  Free: {stats['free']:.2f} GB")