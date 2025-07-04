"""
Simple utility to disable torch.compile during debugging
This prevents graph break warnings and makes debugging easier
"""

import torch
import logging

def disable_torch_compile():
    """
    Disable torch.compile globally for debugging
    This prevents graph break warnings and allows for easier debugging
    """
    # Disable torch.compile
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.verbose = False
    torch._dynamo.disable()
    
    # Also disable inductor
    torch._inductor.config.disable_progress = True
    
    print("✓ torch.compile disabled for debugging")
    print("  - No more graph break warnings")
    print("  - Easier breakpoint debugging")
    print("  - Slightly slower training (but worth it for debugging)")
    
def enable_torch_compile():
    """
    Re-enable torch.compile after debugging
    """
    torch._dynamo.enable()
    torch._dynamo.config.suppress_errors = False
    print("✓ torch.compile re-enabled")

# Simple usage in notebook cell:
if __name__ == "__main__":
    print("Usage in notebook:")
    print("```python")
    print("from disable_compile_for_debug import disable_torch_compile")
    print("disable_torch_compile()  # Call this at the start of your notebook")
    print("```")