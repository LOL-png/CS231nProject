"""
Fixed imports for train_full_featured.ipynb

Replace the problematic import with this code in your notebook:
"""

# Standard imports
import os
import sys
import torch
import numpy as np
from pathlib import Path
import wandb
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Core model imports
from models.unified_model import UnifiedManipulationTransformer
from training.trainer import ManipulationTrainer
from training.losses import ComprehensiveLoss
from evaluation.metrics import ManipulationMetrics
from data.dexycb_dataset import DexYCBDataset

# Optimization imports with graceful fallback
from optimizations.memory_management import MemoryOptimizer
from optimizations.flash_attention import replace_with_flash_attention
from optimizations.data_loading import OptimizedDataLoader

# Mixed precision with fallback - this handles FP8 gracefully
from optimizations.mixed_precision_fallback import (
    enable_mixed_precision_training,
    check_mixed_precision_support,
    MixedPrecisionTrainer
)

# Print system info
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CUDA version: {torch.version.cuda}")
    
    # Check mixed precision support
    mp_info = check_mixed_precision_support()
    print(f"\nMixed Precision Support:")
    print(f"  BFloat16: {'✓' if mp_info['bfloat16_support'] else '✗'}")
    print(f"  Float16: {'✓' if mp_info['float16_support'] else '✗'}")
    print(f"  FP8: {'✓' if mp_info['fp8_support'] else '✗'} (hardware)")
    print(f"  FP8 Available: {'✓' if mp_info.get('fp8_available', False) else '✗'} (software)")

print("\nAll components imported successfully!")

# Example of how to use mixed precision in the notebook:
def setup_mixed_precision(model, optimizer, config):
    """
    Setup mixed precision training with automatic fallback
    
    Args:
        model: The model to train
        optimizer: The optimizer
        config: Training configuration
    
    Returns:
        model, optimizer, scaler (scaler is None for bfloat16/fp8)
    """
    use_fp8 = config.get('use_fp8', True)  # Try FP8 by default
    fallback_dtype = torch.bfloat16  # Best for H200
    
    model, optimizer, scaler = enable_mixed_precision_training(
        model, 
        optimizer,
        use_fp8=use_fp8,
        fallback_dtype=fallback_dtype
    )
    
    return model, optimizer, scaler

# Example usage in training loop:
def training_step_example(model, optimizer, scaler, batch, criterion):
    """Example of a training step with mixed precision"""
    optimizer.zero_grad()
    
    # Use the appropriate dtype based on what's available
    dtype = torch.bfloat16  # or get from config
    
    with torch.amp.autocast('cuda', dtype=dtype):
        outputs = model(batch)
        loss = criterion(outputs, batch)
    
    if scaler is not None:
        # Float16 path
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        # BFloat16 or FP8 path
        loss.backward()
        optimizer.step()
    
    return loss.item(), outputs

print("\nNote: FP8 requires specific NVIDIA libraries. Using BFloat16 as fallback.")
print("BFloat16 provides excellent training stability on H200 GPUs.")