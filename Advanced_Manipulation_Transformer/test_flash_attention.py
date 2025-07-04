#!/usr/bin/env python3
"""
Test script to verify FlashAttention installation and performance
"""

import os
import sys
import torch
import time
import numpy as np

# Set DEX_YCB_DIR environment variable
os.environ['DEX_YCB_DIR'] = '/home/n231/231nProjectV2/dex-ycb-toolkit/data'

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_flash_attention():
    """Test if FlashAttention is available and working"""
    print("=" * 60)
    print("FlashAttention Test Script")
    print("=" * 60)
    
    # Check PyTorch and CUDA
    print(f"\nSystem Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Compute Capability: {torch.cuda.get_device_capability()}")
    
    # Try to import FlashAttention
    print(f"\nChecking FlashAttention installation...")
    try:
        from flash_attn import flash_attn_func
        print("✅ FlashAttention imported successfully!")
        FLASH_AVAILABLE = True
        
        # Check version if available
        try:
            import flash_attn
            if hasattr(flash_attn, '__version__'):
                print(f"FlashAttention version: {flash_attn.__version__}")
        except:
            pass
            
    except ImportError as e:
        print(f"❌ FlashAttention not available: {e}")
        FLASH_AVAILABLE = False
    
    # Test our implementation
    print(f"\nTesting our FlashAttention wrapper...")
    try:
        from optimizations.flash_attention import FLASH_AVAILABLE as FA_AVAILABLE
        from optimizations.flash_attention import FlashAttention, replace_with_flash_attention
        
        print(f"Our wrapper reports FlashAttention available: {FA_AVAILABLE}")
        
        if torch.cuda.is_available():
            # Create test module
            print(f"\nCreating test attention module...")
            batch_size, seq_len, embed_dim = 8, 196, 1024
            num_heads = 16
            
            # Create FlashAttention module
            flash_module = FlashAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=0.1
            ).cuda()
            
            # Test with different dtypes
            for dtype in [torch.float16, torch.bfloat16]:
                print(f"\nTesting with dtype: {dtype}")
                
                # Create test input
                x = torch.randn(batch_size, seq_len, embed_dim, device='cuda', dtype=dtype)
                flash_module = flash_module.to(dtype)
                
                # Forward pass
                try:
                    with torch.no_grad():
                        output, _ = flash_module(x)
                    print(f"✅ Forward pass successful with {dtype}")
                    print(f"   Input shape: {x.shape}")
                    print(f"   Output shape: {output.shape}")
                except Exception as e:
                    print(f"❌ Forward pass failed: {e}")
            
            # Benchmark if available
            if FLASH_AVAILABLE:
                print(f"\nBenchmarking FlashAttention vs Standard Attention...")
                
                # Create standard attention for comparison
                standard_attn = torch.nn.MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=0.1,
                    batch_first=True
                ).cuda().bfloat16()
                
                x = torch.randn(batch_size, seq_len, embed_dim, device='cuda', dtype=torch.bfloat16)
                
                # Warmup
                for _ in range(10):
                    flash_module(x)
                    standard_attn(x, x, x)
                torch.cuda.synchronize()
                
                # Benchmark FlashAttention
                start = time.time()
                for _ in range(100):
                    flash_module(x)
                torch.cuda.synchronize()
                flash_time = time.time() - start
                
                # Benchmark Standard Attention
                start = time.time()
                for _ in range(100):
                    standard_attn(x, x, x)
                torch.cuda.synchronize()
                standard_time = time.time() - start
                
                print(f"\nBenchmark Results (100 iterations):")
                print(f"FlashAttention: {flash_time:.3f}s ({flash_time/100*1000:.1f}ms per forward)")
                print(f"Standard Attention: {standard_time:.3f}s ({standard_time/100*1000:.1f}ms per forward)")
                print(f"Speedup: {standard_time/flash_time:.2f}x")
        
    except Exception as e:
        print(f"❌ Error testing FlashAttention wrapper: {e}")
        import traceback
        traceback.print_exc()
    
    # Test model integration
    print(f"\n\nTesting model integration...")
    try:
        from models.unified_model import UnifiedManipulationTransformer
        
        # Create small model for testing
        config = {
            'hidden_dim': 512,  # Smaller for testing
            'freeze_layers': 6
        }
        
        model = UnifiedManipulationTransformer(config)
        
        # Count attention modules before
        attn_count_before = sum(1 for _ in model.modules() if isinstance(_, torch.nn.MultiheadAttention))
        print(f"MultiheadAttention modules before: {attn_count_before}")
        
        # Replace with FlashAttention
        model = replace_with_flash_attention(model)
        
        # Count attention modules after
        attn_count_after = sum(1 for _ in model.modules() if isinstance(_, torch.nn.MultiheadAttention))
        flash_count = sum(1 for _ in model.modules() if isinstance(_, FlashAttention))
        
        print(f"MultiheadAttention modules after: {attn_count_after}")
        print(f"FlashAttention modules after: {flash_count}")
        
        if flash_count > 0:
            print(f"✅ Successfully replaced {flash_count} attention modules with FlashAttention!")
        else:
            print(f"⚠️ No modules were replaced (FlashAttention may not be available)")
        
        # Test forward pass
        if torch.cuda.is_available():
            model = model.cuda().bfloat16()
            dummy_batch = {
                'image': torch.randn(2, 3, 224, 224, device='cuda', dtype=torch.bfloat16),
                'camera_intrinsics': torch.eye(3, device='cuda', dtype=torch.bfloat16).unsqueeze(0).repeat(2, 1, 1)
            }
            
            with torch.no_grad():
                output = model(dummy_batch)
            
            print(f"\n✅ Model forward pass successful!")
            print(f"Output keys: {list(output.keys())}")
            
    except Exception as e:
        print(f"❌ Error testing model integration: {e}")
        import traceback
        traceback.print_exc()
    
    # Installation instructions
    print(f"\n\n" + "=" * 60)
    print("Installation Instructions")
    print("=" * 60)
    
    if not FLASH_AVAILABLE:
        print("\nTo install FlashAttention, run one of the following:")
        print("\n1. For stable version (recommended):")
        print("   pip install flash-attn==2.5.0")
        print("\n2. For latest version:")
        print("   pip install flash-attn --no-build-isolation")
        print("\n3. If you encounter issues, try:")
        print("   pip install ninja")
        print("   pip install flash-attn --no-build-isolation --no-cache-dir")
        print("\nNote: FlashAttention requires:")
        print("- CUDA 11.6 or higher")
        print("- GPU with compute capability 8.0+ (A100, H100, H200)")
        print("- PyTorch compiled with the same CUDA version")
    else:
        print("\n✅ FlashAttention is properly installed and working!")
        print("\nTo use in training:")
        print("1. Set 'use_flash_attention: true' in your config")
        print("2. Use bfloat16 or float16 precision")
        print("3. Monitor logs for 'Replaced X with FlashAttention' messages")

if __name__ == "__main__":
    test_flash_attention()