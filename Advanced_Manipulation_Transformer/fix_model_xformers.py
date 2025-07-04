#!/usr/bin/env python3
"""
Fix models that already have XFormers applied
============================================

This script removes XFormers and restores standard PyTorch attention.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

def remove_xformers_from_model(model: nn.Module) -> nn.Module:
    """
    Remove XFormersAttention modules and replace with standard MultiheadAttention
    
    This fixes models that already have XFormers optimization applied.
    """
    replacements = []
    
    # Find all XFormersAttention modules
    for name, module in model.named_modules():
        if type(module).__name__ == 'XFormersAttention':
            replacements.append(name)
    
    if not replacements:
        logger.info("No XFormersAttention modules found in model")
        return model
    
    logger.info(f"Found {len(replacements)} XFormersAttention modules to replace")
    
    # Replace each XFormersAttention with MultiheadAttention
    for module_path in replacements:
        # Navigate to parent module
        parts = module_path.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        # Get the XFormers module
        xformers_module = getattr(parent, parts[-1])
        
        # Create replacement MultiheadAttention
        replacement = nn.MultiheadAttention(
            embed_dim=xformers_module.embed_dim,
            num_heads=xformers_module.num_heads,
            dropout=xformers_module.dropout,
            batch_first=getattr(xformers_module, 'batch_first', True)
        )
        
        # Copy weights if possible
        try:
            with torch.no_grad():
                # XFormers uses combined QKV weight
                if hasattr(xformers_module, 'qkv') and hasattr(xformers_module.qkv, 'weight'):
                    qkv_weight = xformers_module.qkv.weight
                    embed_dim = xformers_module.embed_dim
                    
                    # Split QKV weight
                    q_weight = qkv_weight[:embed_dim]
                    k_weight = qkv_weight[embed_dim:2*embed_dim]
                    v_weight = qkv_weight[2*embed_dim:]
                    
                    # Set in_proj_weight
                    replacement.in_proj_weight = nn.Parameter(
                        torch.cat([q_weight, k_weight, v_weight], dim=0)
                    )
                    
                    # Copy bias if exists
                    if hasattr(xformers_module.qkv, 'bias') and xformers_module.qkv.bias is not None:
                        qkv_bias = xformers_module.qkv.bias
                        q_bias = qkv_bias[:embed_dim]
                        k_bias = qkv_bias[embed_dim:2*embed_dim]
                        v_bias = qkv_bias[2*embed_dim:]
                        replacement.in_proj_bias = nn.Parameter(
                            torch.cat([q_bias, k_bias, v_bias], dim=0)
                        )
                
                # Copy output projection
                if hasattr(xformers_module, 'out_proj'):
                    replacement.out_proj.weight.copy_(xformers_module.out_proj.weight)
                    if hasattr(xformers_module.out_proj, 'bias') and xformers_module.out_proj.bias is not None:
                        replacement.out_proj.bias.copy_(xformers_module.out_proj.bias)
                        
            logger.info(f"Successfully copied weights for {module_path}")
                        
        except Exception as e:
            logger.warning(f"Could not copy weights for {module_path}: {e}")
            logger.warning("Module will be randomly initialized")
        
        # Replace the module
        setattr(parent, parts[-1], replacement)
        logger.info(f"Replaced {module_path} with MultiheadAttention")
    
    logger.info(f"Successfully replaced all {len(replacements)} XFormersAttention modules")
    return model


def clean_and_optimize_model(model: nn.Module, apply_native_opt: bool = True) -> nn.Module:
    """
    Complete cleanup and optimization pipeline
    
    1. Remove XFormers modules
    2. Apply PyTorch native optimizations
    """
    # Step 1: Remove XFormers
    model = remove_xformers_from_model(model)
    
    # Step 2: Apply native optimizations
    if apply_native_opt:
        try:
            from optimizations.pytorch_native_optimization import optimize_for_h200
            model = optimize_for_h200(model)
            logger.info("Applied PyTorch native optimizations")
        except ImportError:
            logger.warning("Could not import pytorch_native_optimization")
    
    return model


def check_model_for_xformers(model: nn.Module) -> bool:
    """Check if model contains XFormers modules"""
    for name, module in model.named_modules():
        if type(module).__name__ == 'XFormersAttention':
            return True
    return False


def diagnose_model(model: nn.Module):
    """Diagnose model for optimization issues"""
    print("\n=== Model Diagnosis ===")
    
    # Check for XFormers
    has_xformers = check_model_for_xformers(model)
    print(f"Contains XFormersAttention: {'Yes ⚠️' if has_xformers else 'No ✓'}")
    
    if has_xformers:
        count = sum(1 for _, m in model.named_modules() if type(m).__name__ == 'XFormersAttention')
        print(f"  Number of XFormersAttention modules: {count}")
        print("  ⚠️  This will cause compatibility issues!")
        print("  Fix: Run clean_and_optimize_model(model)")
    
    # Check for other optimizations
    multihead_count = sum(1 for _, m in model.named_modules() if isinstance(m, nn.MultiheadAttention))
    print(f"Standard MultiheadAttention modules: {multihead_count}")
    
    # Check if compiled
    is_compiled = hasattr(model, '_dynamo_orig_callable')
    print(f"torch.compile applied: {'Yes ✓' if is_compiled else 'No'}")
    
    # Check PyTorch version
    print(f"\nPyTorch version: {torch.__version__}")
    pytorch_good = torch.__version__.startswith('2.')
    print(f"PyTorch 2.0+ (has SDPA): {'Yes ✓' if pytorch_good else 'No ⚠️'}")
    
    print("\n" + "="*30)


# For notebook users - simple fix
def fix_model_for_notebook(model):
    """
    One-line fix for notebook users
    
    Removes XFormers and applies native optimizations.
    """
    return clean_and_optimize_model(model, apply_native_opt=True)


if __name__ == "__main__":
    print("XFormers Removal Tool")
    print("=" * 50)
    
    # Example usage
    print("\nExample usage:")
    print("```python")
    print("from fix_model_xformers import fix_model_for_notebook")
    print("model = fix_model_for_notebook(model)")
    print("```")
    
    # Test with dummy model
    print("\nTesting with dummy model...")
    
    # Create dummy model with fake XFormers
    class DummyXFormersAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = 512
            self.num_heads = 8
            self.dropout = 0.1
            self.batch_first = True
            self.qkv = nn.Linear(512, 1536)
            self.out_proj = nn.Linear(512, 512)
    
    model = nn.ModuleDict({
        'encoder': DummyXFormersAttention(),
        'decoder': nn.Linear(512, 256)
    })
    
    print("\nBefore fix:")
    diagnose_model(model)
    
    print("\nApplying fix...")
    model = clean_and_optimize_model(model, apply_native_opt=False)
    
    print("\nAfter fix:")
    diagnose_model(model)
    
    print("\n✓ Fix completed successfully!")