#!/usr/bin/env python3
"""
Test the specific configuration that failed in the sweep
"""

import os
import sys
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from pathlib import Path

# Setup paths
os.environ['DEX_YCB_DIR'] = '/home/n231/231nProjectV2/dex-ycb-toolkit/data'
project_root = Path('.').absolute()
sys.path.insert(0, str(project_root))

# Import the compile fix
from fixes.torch_compile_fix import optimize_for_h200_with_compile_fix, create_compile_config, safe_compile_model

def test_failing_config():
    """Test the specific parameter configuration that failed"""
    
    # Load default config
    config = OmegaConf.load('configs/default_config.yaml')
    
    # Apply the failing parameters
    failing_params = {
        'aug_color_jitter': 0.2769351500778673,
        'aug_joint_noise_std': 0.005099076477518796,
        'aug_rotation_range': 21.35586881545446,
        'aug_scale_max': 1.1268385807384125,
        'aug_scale_min': 0.7210495002720914,
        'aug_translation_std': 0.09960400314873824,
        'batch_size': 256,
        'diversity_margin': 0.019205821528771892,
        'dropout': 0.25,
        'fingertip_weight': 1.4075833071262758,
        'learning_rate': 0.0020758257142178805,
        'loss_weight_contact': 0.3583734664506522,
        'loss_weight_diversity': 0.029075928243702778,
        'loss_weight_hand_coarse': 0.9669472208242692,
        'loss_weight_hand_refined': 1.212172215748071,
        'loss_weight_object_position': 0.969000102984706,
        'loss_weight_object_rotation': 0.48479648604961056,
        'loss_weight_physics': 0.08685002608255316,
        'loss_weight_reprojection': 0.5840843889284524,
        'per_joint_weighting': True,
        'scheduler_type': 'cosine_warmup',
    }
    
    # Apply parameters to config
    config.training.batch_size = failing_params['batch_size']
    config.model.dropout = failing_params['dropout']
    config.loss.loss_weights.contact = failing_params['loss_weight_contact']
    
    print("Testing configuration that failed in sweep...")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Dropout: {config.model.dropout}")
    
    # Import model
    from models.unified_model import UnifiedManipulationTransformer
    
    # Create model
    model = UnifiedManipulationTransformer(config.model)
    
    print(f"\nModel created successfully")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Test 1: Without compilation
    print("\n1. Testing forward pass without compilation...")
    model = model.cuda()
    
    # Create dummy batch
    batch = {
        'image': torch.randn(config.training.batch_size, 3, 224, 224, device='cuda'),
        'joints_3d': torch.randn(config.training.batch_size, 21, 3, device='cuda'),
        'mano_vertices': torch.randn(config.training.batch_size, 778, 3, device='cuda'),
    }
    
    try:
        with torch.no_grad():
            outputs = model(batch)
        print("✓ Forward pass without compilation successful")
    except Exception as e:
        print(f"✗ Forward pass without compilation failed: {e}")
    
    # Test 2: With standard torch.compile
    print("\n2. Testing with standard torch.compile...")
    try:
        compiled_model = torch.compile(model, mode='reduce-overhead', fullgraph=True)
        with torch.no_grad():
            outputs = compiled_model(batch)
        print("✓ Standard torch.compile successful")
    except Exception as e:
        print(f"✗ Standard torch.compile failed: {e}")
        print(f"   Error type: {type(e).__name__}")
    
    # Test 3: With our safe compile
    print("\n3. Testing with safe compile fix...")
    model_fresh = UnifiedManipulationTransformer(config.model).cuda()
    
    compile_config = create_compile_config(config.training.batch_size, config.model)
    print(f"   Compile mode: {compile_config['compile_mode']}")
    
    try:
        compiled_model = safe_compile_model(model_fresh, compile_config)
        with torch.no_grad():
            outputs = compiled_model(batch)
        print("✓ Safe compile successful")
    except Exception as e:
        print(f"✗ Safe compile failed: {e}")
    
    # Test 4: With full optimization pipeline
    print("\n4. Testing with full optimization pipeline...")
    model_fresh2 = UnifiedManipulationTransformer(config.model)
    
    try:
        optimized_model = optimize_for_h200_with_compile_fix(model_fresh2, compile_config)
        optimized_model = optimized_model.cuda()
        
        with torch.no_grad():
            outputs = optimized_model(batch)
        print("✓ Full optimization pipeline successful")
    except Exception as e:
        print(f"✗ Full optimization pipeline failed: {e}")
    
    # Test 5: Test with different batch sizes
    print("\n5. Testing different batch sizes...")
    for bs in [32, 64, 128, 256, 512]:
        print(f"\n   Batch size {bs}:")
        
        # Create model
        model_test = UnifiedManipulationTransformer(config.model)
        compile_config = create_compile_config(bs, config.model)
        
        try:
            compiled_model = safe_compile_model(model_test, compile_config)
            compiled_model = compiled_model.cuda()
            
            # Test batch
            test_batch = {
                'image': torch.randn(bs, 3, 224, 224, device='cuda'),
                'joints_3d': torch.randn(bs, 21, 3, device='cuda'),
                'mano_vertices': torch.randn(bs, 778, 3, device='cuda'),
            }
            
            with torch.no_grad():
                outputs = compiled_model(test_batch)
            print(f"   ✓ Batch size {bs} works with compile mode: {compile_config['compile_mode']}")
        except Exception as e:
            print(f"   ✗ Batch size {bs} failed: {e}")
    
    print("\n" + "="*60)
    print("Test complete!")


if __name__ == "__main__":
    test_failing_config()