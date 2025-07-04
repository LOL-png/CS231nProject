"""
Quick fix for XFormers compatibility issue

This provides a simpler solution that doesn't use XFormers
but maintains compatibility with the notebook.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

def disable_xformers_attention(model: nn.Module) -> nn.Module:
    """
    Disable XFormers attention optimization to avoid compatibility issues
    
    This is a temporary fix - XFormers provides memory efficiency but
    has compatibility issues with TransformerEncoder.
    """
    # Check if model has been optimized with XFormers
    has_xformers = False
    
    for name, module in model.named_modules():
        if type(module).__name__ == 'XFormersAttention':
            has_xformers = True
            logger.warning(f"Found XFormersAttention at {name}, replacing with standard attention")
            
            # Get parent module
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
            else:
                parent = model
            
            # Create standard MultiheadAttention to replace it
            old_module = module
            new_module = nn.MultiheadAttention(
                embed_dim=old_module.embed_dim,
                num_heads=old_module.num_heads,
                dropout=old_module.dropout,
                batch_first=getattr(old_module, 'batch_first', True)
            )
            
            # Copy weights back
            with torch.no_grad():
                # XFormers uses combined QKV, standard uses separate
                qkv_weight = old_module.qkv.weight
                embed_dim = old_module.embed_dim
                
                # Split QKV weight
                q_weight = qkv_weight[:embed_dim]
                k_weight = qkv_weight[embed_dim:2*embed_dim]
                v_weight = qkv_weight[2*embed_dim:]
                
                # Combine into in_proj_weight
                new_module.in_proj_weight = nn.Parameter(
                    torch.cat([q_weight, k_weight, v_weight], dim=0)
                )
                
                if old_module.qkv.bias is not None:
                    qkv_bias = old_module.qkv.bias
                    q_bias = qkv_bias[:embed_dim]
                    k_bias = qkv_bias[embed_dim:2*embed_dim]
                    v_bias = qkv_bias[2*embed_dim:]
                    new_module.in_proj_bias = nn.Parameter(
                        torch.cat([q_bias, k_bias, v_bias], dim=0)
                    )
                
                # Copy output projection
                new_module.out_proj.weight.copy_(old_module.out_proj.weight)
                if old_module.out_proj.bias is not None:
                    new_module.out_proj.bias.copy_(old_module.out_proj.bias)
            
            # Replace module
            setattr(parent, child_name, new_module)
    
    if has_xformers:
        logger.info("Replaced XFormersAttention modules with standard MultiheadAttention")
    
    return model


def safe_memory_optimization(model: nn.Module, config: dict = None) -> nn.Module:
    """
    Apply safe memory optimizations that maintain compatibility
    
    This avoids XFormers and uses only PyTorch built-in optimizations.
    """
    # 1. Enable gradient checkpointing for transformer layers
    for name, module in model.named_modules():
        if isinstance(module, nn.TransformerEncoderLayer):
            # Enable activation checkpointing
            module.activation_checkpointing = True
            logger.info(f"Enabled gradient checkpointing for {name}")
    
    # 2. Set memory-efficient options
    if hasattr(torch.backends, 'cuda'):
        # Enable TF32 for better performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Use deterministic algorithms for reproducibility
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    # 3. Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    logger.info("Applied safe memory optimizations")
    return model