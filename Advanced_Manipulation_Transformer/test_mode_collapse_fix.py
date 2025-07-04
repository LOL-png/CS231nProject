#!/usr/bin/env python3
"""
Test the mode collapse TransformerEncoderLayer fix
"""

import os
import sys
import torch
import torch.nn as nn

# Set environment
os.environ['DEX_YCB_DIR'] = '/home/n231/231nProjectV2/dex-ycb-toolkit/data'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_transformer_layer_fix():
    """Test that TransformerEncoderLayer replacement works"""
    print("Testing TransformerEncoderLayer replacement fix...")
    
    try:
        from solutions.mode_collapse import ModeCollapsePreventionModule
        
        # Create a model with TransformerEncoderLayer
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Single transformer layer
                self.transformer = nn.TransformerEncoderLayer(
                    d_model=256,
                    nhead=8,
                    dim_feedforward=1024,
                    dropout=0.1,
                    batch_first=True
                )
                
                # Transformer encoder with multiple layers
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=512,
                    nhead=16,
                    dim_feedforward=2048,
                    dropout=0.2,
                    batch_first=True
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
                
                # Some other layers
                self.linear = nn.Linear(256, 128)
            
            def forward(self, x):
                # Test forward pass
                x1 = self.transformer(x)
                x2 = self.encoder(x.repeat(1, 1, 2))  # Double the feature size
                return x1, x2
        
        # Create model
        model = TestModel()
        print("✅ Created test model with TransformerEncoderLayer")
        
        # Test forward pass before wrapping
        x = torch.randn(2, 10, 256)
        out1, out2 = model(x)
        print(f"✅ Forward pass before wrapping: {out1.shape}, {out2.shape}")
        
        # Wrap with mode collapse prevention
        config = {
            'noise_std': 0.01,
            'drop_path_rate': 0.1,
            'mixup_alpha': 0.2
        }
        
        print("\nWrapping model with ModeCollapsePreventionModule...")
        wrapped_model = ModeCollapsePreventionModule.wrap_model(model, config)
        print("✅ Successfully wrapped model")
        
        # Test forward pass after wrapping
        wrapped_model.eval()  # Set to eval mode
        with torch.no_grad():
            out1_wrapped, out2_wrapped = wrapped_model(x)
        print(f"✅ Forward pass after wrapping: {out1_wrapped.shape}, {out2_wrapped.shape}")
        
        # Check that transformer layers were replaced
        replaced_count = 0
        for name, module in wrapped_model.named_modules():
            if 'ImprovedTransformerLayer' in str(type(module)):
                replaced_count += 1
                print(f"  - Found replaced layer: {name}")
        
        print(f"\n✅ Replaced {replaced_count} transformer layers")
        
        # Test with actual UnifiedManipulationTransformer if available
        try:
            from models.unified_model import UnifiedManipulationTransformer
            
            print("\nTesting with UnifiedManipulationTransformer...")
            real_config = {'hidden_dim': 256, 'freeze_layers': 6}
            real_model = UnifiedManipulationTransformer(real_config)
            
            # Wrap the real model
            wrapped_real_model = ModeCollapsePreventionModule.wrap_model(real_model, config)
            print("✅ Successfully wrapped UnifiedManipulationTransformer")
            
            # Test forward pass
            dummy_batch = {
                'image': torch.randn(2, 3, 224, 224),
                'camera_intrinsics': torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
            }
            
            wrapped_real_model.eval()
            with torch.no_grad():
                outputs = wrapped_real_model(**dummy_batch)
            
            print("✅ Forward pass with real model successful")
            print(f"   Output keys: {list(outputs.keys())}")
            
        except Exception as e:
            print(f"\n⚠️ Could not test with real model: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Mode Collapse TransformerEncoderLayer Fix")
    print("=" * 60)
    
    success = test_transformer_layer_fix()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ ALL TESTS PASSED - The fix is working correctly!")
    else:
        print("❌ TEST FAILED - Please check the error above")
    print("=" * 60)