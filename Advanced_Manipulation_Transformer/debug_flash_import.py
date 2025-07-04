#!/usr/bin/env python3
"""
Debug flash attention import issue
"""

import sys
import subprocess

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("\nPython path:")
for i, p in enumerate(sys.path[:10]):
    print(f"  {i}: {p}")

print("\n" + "="*60)
print("Checking installed packages with 'flash' in name:")
result = subprocess.run([sys.executable, '-m', 'pip', 'list'], capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if 'flash' in line.lower():
        print(f"  {line}")

print("\n" + "="*60)
print("Trying different import variations:")

# Try 1: Standard import
print("\n1. Trying: from flash_attn import flash_attn_func")
try:
    from flash_attn import flash_attn_func
    print("   ✅ Success!")
except ImportError as e:
    print(f"   ❌ Failed: {e}")

# Try 2: Just import the module
print("\n2. Trying: import flash_attn")
try:
    import flash_attn
    print("   ✅ Success!")
    print(f"   Module location: {flash_attn.__file__}")
    print(f"   Module attributes: {dir(flash_attn)[:10]}...")
except ImportError as e:
    print(f"   ❌ Failed: {e}")

# Try 3: Check if it's under a different name
print("\n3. Checking sys.modules for anything with 'flash':")
for name in sys.modules:
    if 'flash' in name.lower():
        print(f"   Found: {name}")

# Try 4: Import with full error traceback
print("\n4. Full error traceback:")
try:
    from flash_attn import flash_attn_func
except Exception as e:
    import traceback
    traceback.print_exc()

# Try 5: Check site-packages
print("\n5. Checking site-packages for flash directories:")
import os
import site
for site_dir in site.getsitepackages():
    if os.path.exists(site_dir):
        for item in os.listdir(site_dir):
            if 'flash' in item.lower():
                print(f"   Found in {site_dir}: {item}")

print("\n" + "="*60)
print("Recommendations:")
print("1. Make sure you're in the correct conda environment")
print("2. Try: pip show flash-attn")
print("3. Try: pip install flash-attn --force-reinstall --no-build-isolation")
print("4. Check if you need to set PYTHONPATH or activate environment")