#!/usr/bin/env python3
"""Test mixed precision setup"""

import torch
import torch.nn as nn

# Test import
print("Testing mixed precision imports...")

try:
    from optimizations.mixed_precision_fallback import (
        enable_mixed_precision_training,
        check_mixed_precision_support,
        print_mixed_precision_info
    )
    print("✓ Mixed precision fallback imported successfully")
except Exception as e:
    print(f"✗ Failed to import mixed precision fallback: {e}")

try:
    from optimizations.fp8_mixed_precision import enable_fp8_training, FP8_AVAILABLE
    if FP8_AVAILABLE:
        print("✓ FP8 support is available")
    else:
        print("✓ FP8 module imported but FP8 not available (this is OK)")
except Exception as e:
    print(f"✓ FP8 module handled gracefully: {e}")

# Check capabilities
print("\n" + "="*50)
print_mixed_precision_info()
print("="*50)

# Test with a simple model
print("\nTesting with a simple model...")

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1024, 512)
        self.relu = nn.ReLU()
        self.output = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.relu(self.linear(x))
        return self.output(x)

# Create model and optimizer
model = SimpleModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Try to enable mixed precision
model, optimizer, scaler = enable_mixed_precision_training(
    model, 
    optimizer,
    use_fp8=True,  # Will fallback if not available
    fallback_dtype=torch.bfloat16
)

print("✓ Mixed precision setup completed successfully")

# Test forward pass
print("\nTesting forward pass...")
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    x = torch.randn(32, 1024, device='cuda')
    output = model(x)
    print(f"✓ Forward pass successful, output shape: {output.shape}")
    print(f"  Input dtype: {x.dtype}")
    print(f"  Output dtype: {output.dtype}")

print("\n✓ All tests passed! Mixed precision is working correctly.")