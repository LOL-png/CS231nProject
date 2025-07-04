#!/usr/bin/env python3
"""Check cuDNN installation and library locations"""

import os
import subprocess
import torch

print("PyTorch cuDNN version:", torch.backends.cudnn.version())
print("PyTorch cuDNN enabled:", torch.backends.cudnn.enabled)
print()

# Check for cuDNN libraries
lib_names = [
    "libcudnn.so",
    "libcudnn.so.9", 
    "libcudnn_adv.so.9",
    "libcudnn_cnn.so.9",
    "libcudnn_ops.so.9"
]

search_paths = [
    "/usr/local/cuda/lib64",
    "/usr/lib/x86_64-linux-gnu",
    "/opt/nvidia/cudnn/lib64",
    os.path.expanduser("~/miniconda3/envs/env2.0/lib"),
    "/usr/local/lib",
    "/usr/lib"
]

print("Searching for cuDNN libraries...")
found_libs = {}

for path in search_paths:
    if os.path.exists(path):
        try:
            files = os.listdir(path)
            for lib in lib_names:
                if any(f.startswith(lib) for f in files):
                    matching_files = [f for f in files if f.startswith(lib)]
                    if lib not in found_libs:
                        found_libs[lib] = []
                    found_libs[lib].extend([(path, f) for f in matching_files])
        except PermissionError:
            pass

print("\nFound libraries:")
for lib, locations in found_libs.items():
    print(f"\n{lib}:")
    for path, filename in locations:
        full_path = os.path.join(path, filename)
        print(f"  {full_path}")
        # Check if it's a symlink
        if os.path.islink(full_path):
            target = os.readlink(full_path)
            print(f"    -> {target}")

# Check ldconfig
print("\n\nChecking ldconfig for cuDNN:")
try:
    result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True)
    cudnn_libs = [line for line in result.stdout.split('\n') if 'cudnn' in line]
    for lib in cudnn_libs:
        print(f"  {lib.strip()}")
except Exception as e:
    print(f"Could not run ldconfig: {e}")

# Check environment
print("\n\nEnvironment variables:")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
print(f"CUDNN_PATH: {os.environ.get('CUDNN_PATH', 'Not set')}")