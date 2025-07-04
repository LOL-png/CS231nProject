#!/usr/bin/env python3
"""
Test the FlashAttention fix for SigmaReparam
"""

import os
import sys
import torch
import torch.nn as nn

# Set environment
os.environ['DEX_YCB_DIR'] = '/home/n231/231nProjectV2/dex-ycb-toolkit/data'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_flash_attention_with_sigma():
    """Test that FlashAttention works with SigmaReparam wrapped layers"""
    print("Testing FlashAttention with SigmaReparam fix...")
    
    try:
        from models.unified_model import UnifiedManipulationTransformer, SigmaReparam
        from optimizations.flash_attention import replace_with_flash_attention, FLASH_AVAILABLE
        
        print(f"\nFlashAttention available: {FLASH_AVAILABLE}")
        
        # Create a simple test model with MultiheadAttention
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = nn.MultiheadAttention(256, 8, batch_first=True)
                # Wrap output projection with SigmaReparam
                self.attention.out_proj = SigmaReparam(self.attention.out_proj)
                
            def forward(self, x):
                return self.attention(x, x, x)[0]
        
        # Test basic model
        print("\n1. Testing basic model with SigmaReparam...")
        model = TestModel()
        x = torch.randn(2, 10, 256)
        output = model(x)
        print(f"✅ Basic model works: input {x.shape} -> output {output.shape}")
        
        # Test FlashAttention replacement
        print("\n2. Testing FlashAttention replacement...")
        model = replace_with_flash_attention(model)
        print("✅ FlashAttention replacement succeeded")
        
        # Test forward pass after replacement
        print("\n3. Testing forward pass after replacement...")
        output = model(x)
        print(f"✅ Forward pass works: input {x.shape} -> output {output.shape}")
        
        # Test with real UnifiedManipulationTransformer
        print("\n4. Testing with UnifiedManipulationTransformer...")
        config = {'hidden_dim': 256, 'freeze_layers': 6}
        real_model = UnifiedManipulationTransformer(config)
        
        # Count attention modules before
        attn_count = sum(1 for _ in real_model.modules() if isinstance(_, nn.MultiheadAttention))
        print(f"   MultiheadAttention modules before: {attn_count}")
        
        # Apply FlashAttention
        real_model = replace_with_flash_attention(real_model)
        print("✅ FlashAttention applied to UnifiedManipulationTransformer")
        
        # Count attention modules after
        from optimizations.flash_attention import FlashAttention
        flash_count = sum(1 for _ in real_model.modules() if isinstance(_, FlashAttention))
        attn_count_after = sum(1 for _ in real_model.modules() if isinstance(_, nn.MultiheadAttention))
        print(f"   FlashAttention modules after: {flash_count}")
        print(f"   MultiheadAttention modules after: {attn_count_after}")
        
        # Test forward pass
        dummy_batch = {
            'image': torch.randn(2, 3, 224, 224),
            'camera_intrinsics': torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
        }
        
        real_model.eval()
        with torch.no_grad():
            outputs = real_model(**dummy_batch)
        
        print("✅ Forward pass with real model successful")
        print(f"   Output keys: {list(outputs.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing FlashAttention SigmaReparam Fix")
    print("=" * 60)
    
    success = test_flash_attention_with_sigma()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ FlashAttention SigmaReparam fix is working correctly!")
    else:
        print("❌ Test failed - please check the error above")
    print("=" * 60)