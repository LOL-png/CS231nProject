"""
Fix XFormers compatibility while maintaining performance

Two approaches:
1. Quick fix: Make XFormers compatible with TransformerEncoder
2. Better fix: Use PyTorch's native SDPA which is just as fast
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

def fix_xformers_modules(model: nn.Module) -> nn.Module:
    """
    Fix existing XFormers modules to be compatible with TransformerEncoder
    
    This maintains XFormers performance while fixing the compatibility issue.
    """
    fixed_count = 0
    
    for name, module in model.named_modules():
        if type(module).__name__ == 'XFormersAttention':
            # Add missing attributes without replacing the module
            if not hasattr(module, 'batch_first'):
                module.batch_first = True
                fixed_count += 1
                logger.info(f"Fixed {name} - added batch_first attribute")
            
            # Patch the forward method to handle TransformerEncoder calls
            original_forward = module.forward
            
            def compatible_forward(query, key=None, value=None, key_padding_mask=None,
                                 need_weights=True, attn_mask=None, average_attn_weights=True,
                                 is_causal=False):
                # XFormers expects a single input for self-attention
                if key is None and value is None:
                    # Self-attention case - call original forward
                    output = original_forward(query)
                    # Return output and None for weights (XFormers doesn't compute them)
                    return (output, None) if need_weights else output
                else:
                    # Cross-attention - not supported by simple XFormers
                    raise NotImplementedError("Cross-attention not supported in XFormersAttention")
            
            # Replace forward method
            import types
            module.forward = types.MethodType(compatible_forward, module)
    
    if fixed_count > 0:
        logger.info(f"Fixed {fixed_count} XFormersAttention modules")
    
    return model

def check_and_fix_model(model: nn.Module) -> nn.Module:
    """
    Check if model has XFormers issues and fix them
    
    This is the safest approach - fixes existing modules without replacement.
    """
    has_xformers = any(
        type(module).__name__ == 'XFormersAttention' 
        for _, module in model.named_modules()
    )
    
    if has_xformers:
        logger.info("Found XFormers modules, applying compatibility fixes...")
        model = fix_xformers_modules(model)
        logger.info("XFormers compatibility fixes applied")
    
    return model

# For notebook - simple one-liner fix
def notebook_fix_model(model):
    """One-line fix for notebook users"""
    return check_and_fix_model(model)

# Performance comparison
def compare_attention_performance():
    """Compare performance of different attention implementations"""
    
    if not torch.cuda.is_available():
        print("CUDA required for performance comparison")
        return
    
    import time
    
    # Test configuration
    batch = 16
    seq_len = 512
    dim = 1024
    heads = 16
    
    print(f"Comparing attention performance (B={batch}, L={seq_len}, D={dim}, H={heads})")
    print("-" * 60)
    
    # Create input
    x = torch.randn(batch, seq_len, dim, device='cuda', dtype=torch.float16)
    
    # 1. Standard MultiheadAttention (PyTorch 2.0 uses SDPA automatically)
    standard_attn = nn.MultiheadAttention(dim, heads, batch_first=True).cuda().half()
    
    # Warmup and benchmark
    for _ in range(10):
        with torch.no_grad():
            _ = standard_attn(x, x, x)
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(100):
        with torch.no_grad():
            _ = standard_attn(x, x, x)
    
    torch.cuda.synchronize()
    standard_time = time.time() - start
    
    print(f"PyTorch MultiheadAttention: {standard_time*10:.2f}ms per forward")
    
    # Check if SDPA was used
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        print("  → Using PyTorch 2.0+ SDPA (optimized)")
    
    # Memory usage
    print(f"\nMemory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    return standard_time

if __name__ == "__main__":
    print("XFormers Compatibility Fix")
    print("=" * 60)
    
    # Test the fix
    print("\nTesting fix on a transformer model...")
    
    # Create a simple model
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=256,
        nhead=8,
        batch_first=True
    )
    model = nn.TransformerEncoder(encoder_layer, num_layers=2)
    
    # Simulate XFormers optimization
    class DummyXFormersAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = 256
            self.num_heads = 8
            
        def forward(self, x):
            return x
    
    # Replace attention with dummy XFormers
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            parent = model
            for part in name.split('.')[:-1]:
                parent = getattr(parent, part)
            setattr(parent, name.split('.')[-1], DummyXFormersAttention())
    
    # This would normally fail
    print("Before fix: Model has XFormers without batch_first")
    
    # Apply fix
    model = check_and_fix_model(model)
    
    # Now it should work
    try:
        x = torch.randn(2, 10, 256)
        output = model(x)
        print("After fix: ✓ Model works correctly")
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"After fix: ✗ Still failed: {e}")
    
    # Performance comparison
    if torch.cuda.is_available():
        print("\n" + "=" * 60)
        compare_attention_performance()