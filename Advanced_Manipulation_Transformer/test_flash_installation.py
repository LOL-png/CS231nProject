#!/usr/bin/env python3
"""
Test FlashAttention installation
"""

import sys
import torch

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))

print("\nTrying to import flash_attn...")
try:
    import flash_attn
    print("✅ flash_attn module imported successfully")
    print(f"flash_attn location: {flash_attn.__file__}")
    if hasattr(flash_attn, '__version__'):
        print(f"flash_attn version: {flash_attn.__version__}")
    
    print("\nTrying to import flash_attn functions...")
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    print("✅ flash_attn_func imported successfully")
    print("✅ flash_attn_varlen_func imported successfully")
    
    # Test a simple forward pass
    print("\nTesting flash attention forward pass...")
    batch_size = 2
    seq_len = 128
    n_heads = 8
    head_dim = 64
    
    q = torch.randn(batch_size, seq_len, n_heads, head_dim, device='cuda', dtype=torch.float16)
    k = torch.randn(batch_size, seq_len, n_heads, head_dim, device='cuda', dtype=torch.float16)
    v = torch.randn(batch_size, seq_len, n_heads, head_dim, device='cuda', dtype=torch.float16)
    
    output = flash_attn_func(q, k, v)
    print(f"✅ Flash attention forward pass successful! Output shape: {output.shape}")
    
except ImportError as e:
    print(f"❌ Failed to import flash_attn: {e}")
    print("\nPython path:")
    for p in sys.path[:5]:
        print(f"  {p}")
    
    # Check if it's installed
    print("\nChecking pip list...")
    import subprocess
    result = subprocess.run(['pip', 'list', '|', 'grep', 'flash'], shell=True, capture_output=True, text=True)
    print(result.stdout)
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()

# Also check our wrapper
print("\n" + "="*60)
print("Testing our flash_attention module...")
try:
    from optimizations.flash_attention import FLASH_AVAILABLE, replace_with_flash_attention
    print(f"FLASH_AVAILABLE = {FLASH_AVAILABLE}")
    
    if FLASH_AVAILABLE:
        print("✅ Our module reports FlashAttention is available")
    else:
        print("❌ Our module reports FlashAttention is NOT available")
        
except Exception as e:
    print(f"❌ Error importing our module: {e}")