"""Setup cuDNN library path for Transformer Engine"""

import os
import sys
import site

def setup_cudnn_path():
    """Add cuDNN library path to LD_LIBRARY_PATH"""
    
    # Find the nvidia cudnn package location
    site_packages = site.getsitepackages()
    cudnn_path = None
    
    for sp in site_packages:
        potential_path = os.path.join(sp, 'nvidia', 'cudnn', 'lib')
        if os.path.exists(potential_path):
            cudnn_path = potential_path
            break
    
    # Also check in current environment
    if not cudnn_path:
        env_path = os.path.join(sys.prefix, 'lib', 'python3.12', 'site-packages', 'nvidia', 'cudnn', 'lib')
        if os.path.exists(env_path):
            cudnn_path = env_path
    
    if cudnn_path:
        # Add to LD_LIBRARY_PATH
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        if cudnn_path not in current_ld_path:
            os.environ['LD_LIBRARY_PATH'] = f"{cudnn_path}:{current_ld_path}"
            print(f"Added cuDNN path to LD_LIBRARY_PATH: {cudnn_path}")
        return True
    else:
        print("Could not find cuDNN libraries")
        return False

# Run setup when imported
setup_cudnn_path()