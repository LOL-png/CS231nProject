#!/usr/bin/env python3
"""
Quick test to verify DexYCB dataset loading works
"""

import os
import sys

# Set environment variable if not set
if 'DEX_YCB_DIR' not in os.environ:
    os.environ['DEX_YCB_DIR'] = '/home/n231/231nProjectV2/dex-ycb-toolkit/data'

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dex-ycb-toolkit'))

print(f"DEX_YCB_DIR: {os.environ['DEX_YCB_DIR']}")
print(f"Python paths: {sys.path[-2:]}")

try:
    from dex_ycb_toolkit.factory import get_dataset
    print("✓ Successfully imported dex_ycb_toolkit")
    
    # Try to load dataset
    dataset = get_dataset('s0_train')
    print(f"✓ Successfully loaded dataset with {len(dataset)} samples")
    
    # Try to get a sample
    sample = dataset[0]
    print(f"✓ Successfully loaded sample 0")
    print(f"  Keys: {list(sample.keys())}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Now test GPU cached dataset
print("\nTesting GPU cached dataset...")
try:
    from data.gpu_cached_dataset import create_gpu_cached_dataloaders
    
    config = {
        'gpu_max_samples': 100,  # Small for testing
        'gpu_max_samples_val': 10,
        'gpu_cache_path': './test_cache',
        'batch_size': 4,
        'use_bfloat16': True,
        'preload_dinov2': False
    }
    
    train_loader, val_loader = create_gpu_cached_dataloaders(config)
    print(f"✓ Successfully created dataloaders")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
except Exception as e:
    print(f"✗ Error creating dataloaders: {e}")
    import traceback
    traceback.print_exc()

print("\nDataset test complete!")