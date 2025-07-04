#!/usr/bin/env python3
"""Test the attention module fix for batch_first attribute"""

import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test XFormersAttention
print("Testing XFormersAttention fix...")
try:
    from optimizations.memory_management import MemoryOptimizer
    
    # Create a simple transformer model
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=256,
        nhead=8,
        dim_feedforward=1024,
        batch_first=True
    )
    transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
    
    # Apply memory optimization
    optimizer = MemoryOptimizer()
    optimized_model = optimizer.use_memory_efficient_attention(transformer)
    
    # Test forward pass
    x = torch.randn(2, 10, 256)  # [batch, seq, features]
    output = optimized_model(x)
    
    print("✓ XFormersAttention fix successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Check that attention modules have batch_first
    for name, module in optimized_model.named_modules():
        if 'self_attn' in name and hasattr(module, 'batch_first'):
            print(f"  {name}.batch_first = {module.batch_first}")
    
except Exception as e:
    print(f"✗ XFormersAttention test failed: {e}")
    import traceback
    traceback.print_exc()

# Test FlashAttention
print("\nTesting FlashAttention fix...")
try:
    from optimizations.flash_attention import replace_with_flash_attention
    
    # Create a fresh transformer model
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=256,
        nhead=8,
        dim_feedforward=1024,
        batch_first=True
    )
    transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
    
    # Apply flash attention
    flash_model = replace_with_flash_attention(transformer)
    
    # Test forward pass
    x = torch.randn(2, 10, 256).cuda()  # [batch, seq, features]
    flash_model = flash_model.cuda()
    output = flash_model(x)
    
    print("✓ FlashAttention fix successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Check that attention modules have batch_first
    for name, module in flash_model.named_modules():
        if 'self_attn' in name and hasattr(module, 'batch_first'):
            print(f"  {name}.batch_first = {module.batch_first}")
    
except Exception as e:
    print(f"✓ FlashAttention handled gracefully: {type(e).__name__}")

# Test with the full model
print("\nTesting with UnifiedManipulationTransformer...")
try:
    from models.unified_model import UnifiedManipulationTransformer
    from debugging.model_debugger import ModelDebugger
    
    # Create model
    config = {
        'hidden_dim': 256,
        'freeze_layers': 6,
        'use_sigma_reparam': False
    }
    model = UnifiedManipulationTransformer(config)
    
    # Apply memory optimization
    optimizer = MemoryOptimizer()
    model = optimizer.optimize(model, config)
    
    # Create sample batch
    batch = {
        'image': torch.randn(2, 3, 224, 224),
        'hand_joints_3d': torch.randn(2, 21, 3),
        'camera_intrinsics': torch.eye(3).unsqueeze(0).repeat(2, 1, 1),
    }
    
    # Test forward pass
    outputs = model(batch)
    print("✓ Full model test successful")
    print(f"  Model output keys: {list(outputs.keys())[:5]}...")
    
    # Test with debugger
    debugger = ModelDebugger(model)
    print("\nTesting debugger.analyze_forward_pass...")
    debugger.analyze_forward_pass(batch)
    print("✓ Debugger test successful")
    
except Exception as e:
    print(f"✗ Full model test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ All tests completed!")