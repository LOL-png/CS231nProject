#!/usr/bin/env python3
"""
Check different flash attention versions and import patterns
"""

print("Checking various flash attention import patterns...")

# Pattern 1: Current pattern (flash_attn)
print("\n1. Standard pattern - flash_attn:")
try:
    from flash_attn import flash_attn_func
    print("✅ from flash_attn import flash_attn_func - SUCCESS")
except ImportError as e:
    print(f"❌ from flash_attn import flash_attn_func - FAILED: {e}")

# Pattern 2: Different function names
print("\n2. Alternative function names:")
try:
    import flash_attn
    print(f"✅ import flash_attn - SUCCESS")
    print(f"   Available functions: {[x for x in dir(flash_attn) if not x.startswith('_')]}")
except ImportError as e:
    print(f"❌ import flash_attn - FAILED: {e}")

# Pattern 3: Flash attention 2 style
print("\n3. Flash Attention 2 pattern:")
try:
    from flash_attn.flash_attn_interface import flash_attn_func
    print("✅ from flash_attn.flash_attn_interface import flash_attn_func - SUCCESS")
except ImportError as e:
    print(f"❌ Flash Attention 2 pattern - FAILED: {e}")

# Pattern 4: Older patterns
print("\n4. Older import patterns:")
try:
    from flash_attn.flash_attention import FlashAttention
    print("✅ from flash_attn.flash_attention import FlashAttention - SUCCESS")
except ImportError as e:
    print(f"❌ Older pattern - FAILED: {e}")

# Pattern 5: Check if it's installed but under different name
print("\n5. Checking if package is importable at all:")
try:
    import importlib
    spec = importlib.util.find_spec('flash_attn')
    if spec is None:
        print("❌ 'flash_attn' module not found in Python path")
    else:
        print(f"✅ 'flash_attn' found at: {spec.origin}")
except Exception as e:
    print(f"❌ Error finding module: {e}")

# Pattern 6: Try flash_attention (with underscore between flash and attention)
print("\n6. Trying flash_attention (different underscore pattern):")
try:
    import flash_attention
    print("✅ import flash_attention - SUCCESS")
except ImportError as e:
    print(f"❌ import flash_attention - FAILED: {e}")

# Pattern 7: Environmental check
print("\n7. Environment check:")
import os
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')[:100]}...")
print(f"Current working directory: {os.getcwd()}")

# Pattern 8: Check conda list
print("\n8. Checking conda environment:")
import subprocess
import sys
try:
    result = subprocess.run(['conda', 'list', 'flash'], capture_output=True, text=True)
    if result.stdout:
        print("Conda packages with 'flash':")
        print(result.stdout)
    else:
        print("No conda packages found with 'flash'")
except:
    print("Could not run conda list")

print("\n" + "="*60)
print("SOLUTION:")
print("Based on the results above, update the import statement in")
print("optimizations/flash_attention.py to use the working pattern.")