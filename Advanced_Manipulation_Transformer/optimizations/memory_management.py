import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential, checkpoint
from typing import Dict, List, Optional, Callable
import gc
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """
    Advanced memory optimization for H200 GPU
    """
    
    @staticmethod
    def optimize_model_for_h200(model: nn.Module, config: Dict) -> nn.Module:
        """
        Apply all memory optimizations to model
        """
        # 1. Enable gradient checkpointing
        model = MemoryOptimizer.enable_selective_checkpointing(model)
        
        # 2. Convert to memory-efficient attention
        model = MemoryOptimizer.use_memory_efficient_attention(model)
        
        # 3. Enable activation offloading
        model = MemoryOptimizer.setup_activation_offloading(model)
        
        # 4. Optimize buffer allocation
        MemoryOptimizer.optimize_cuda_allocation()
        
        return model
    
    @staticmethod
    def enable_selective_checkpointing(model: nn.Module, checkpoint_ratio: float = 0.5) -> nn.Module:
        """
        Enable gradient checkpointing for selected layers
        """
        
        class CheckpointedModule(nn.Module):
            def __init__(self, module: nn.Module, use_checkpoint: bool = True):
                super().__init__()
                self.module = module
                self.use_checkpoint = use_checkpoint
            
            def forward(self, *args, **kwargs):
                if self.use_checkpoint and self.training:
                    # Use checkpoint with use_reentrant=False for better performance
                    return checkpoint(
                        self.module,
                        *args,
                        use_reentrant=False,
                        **kwargs
                    )
                else:
                    return self.module(*args, **kwargs)
        
        # Identify expensive layers
        transformer_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.TransformerEncoderLayer) or \
               isinstance(module, nn.TransformerDecoderLayer):
                transformer_layers.append((name, module))
        
        # Checkpoint every other layer
        num_checkpoint = int(len(transformer_layers) * checkpoint_ratio)
        checkpoint_indices = set(
            np.linspace(0, len(transformer_layers)-1, num_checkpoint, dtype=int)
        )
        
        # Replace layers with checkpointed versions
        for idx, (name, layer) in enumerate(transformer_layers):
            if idx in checkpoint_indices:
                # Navigate to parent and replace
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                
                setattr(parent, child_name, CheckpointedModule(layer))
                logger.info(f"Checkpointed layer: {name}")
        
        return model
    
    @staticmethod
    def use_memory_efficient_attention(model: nn.Module) -> nn.Module:
        """
        Replace standard attention with memory-efficient versions
        """
        
        try:
            from xformers import ops as xops
            
            class XFormersAttention(nn.Module):
                def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, batch_first: bool = True):
                    super().__init__()
                    self.embed_dim = embed_dim
                    self.num_heads = num_heads
                    self.dropout = dropout
                    self.batch_first = batch_first  # Add this attribute for compatibility
                    
                    self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
                    self.out_proj = nn.Linear(embed_dim, embed_dim)
                
                def forward(self, query, key=None, value=None, key_padding_mask=None,
                           need_weights=True, attn_mask=None, average_attn_weights=True,
                           is_causal=False):
                    # Handle different input formats
                    if key is None and value is None:
                        # Self-attention case
                        x = query
                    else:
                        # Cross-attention not implemented for xformers
                        raise NotImplementedError("Cross-attention not supported in XFormersAttention")
                    
                    B, N, C = x.shape
                    
                    # Compute QKV
                    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
                    q, k, v = qkv.unbind(2)
                    
                    # Use xFormers memory-efficient attention
                    out = xops.memory_efficient_attention(
                        q, k, v,
                        attn_bias=None,
                        p=self.dropout if self.training else 0.0
                    )
                    
                    out = out.reshape(B, N, C)
                    out = self.out_proj(out)
                    
                    # Return output and None for attention weights (not computed)
                    if need_weights:
                        return out, None
                    else:
                        return out
            
            # Replace attention modules
            for name, module in model.named_modules():
                if isinstance(module, nn.MultiheadAttention):
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    
                    if parent_name:
                        parent = model.get_submodule(parent_name)
                    else:
                        parent = model
                    
                    new_attn = XFormersAttention(
                        module.embed_dim,
                        module.num_heads,
                        module.dropout,
                        batch_first=getattr(module, 'batch_first', True)
                    )
                    
                    # Copy weights
                    with torch.no_grad():
                        new_attn.qkv.weight.copy_(module.in_proj_weight)
                        if module.in_proj_bias is not None:
                            new_attn.qkv.bias.copy_(module.in_proj_bias)
                        new_attn.out_proj.weight.copy_(module.out_proj.weight)
                        if module.out_proj.bias is not None:
                            new_attn.out_proj.bias.copy_(module.out_proj.bias)
                    
                    setattr(parent, child_name, new_attn)
            
            logger.info("Enabled xFormers memory-efficient attention")
            
        except ImportError:
            logger.warning("xFormers not available, using standard attention")
        
        return model
    
    @staticmethod
    def setup_activation_offloading(model: nn.Module) -> nn.Module:
        """
        Setup activation offloading for very large models
        """
        
        class OffloadedActivation(nn.Module):
            def __init__(self, module: nn.Module, offload_device: str = 'cpu'):
                super().__init__()
                self.module = module
                self.offload_device = offload_device
                self.buffer = None
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Store input on CPU if needed
                if self.training and x.requires_grad:
                    self.buffer = x.detach().to(self.offload_device)
                
                return self.module(x)
            
            def backward_hook(self, grad):
                # Restore from CPU for backward
                if self.buffer is not None:
                    return self.buffer.to(grad.device).requires_grad_()
                return grad
        
        # Apply to large layers only
        # (This is a simplified example - in practice, be more selective)
        
        return model
    
    @staticmethod
    def optimize_cuda_allocation():
        """
        Optimize CUDA memory allocation settings
        """
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of available memory
        
        # Enable caching allocator
        torch.cuda.empty_cache()
        
        # Set allocation strategy
        if hasattr(torch.cuda, 'set_allocator_settings'):
            torch.cuda.set_allocator_settings({
                'max_split_size_mb': 128,
                'roundup_power2_divisions': 4,
                'garbage_collection_threshold': 0.8
            })
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info("Optimized CUDA memory allocation settings")
    
    @staticmethod
    def dynamic_batch_sizing(
        model: nn.Module,
        base_batch_size: int,
        sequence_length: int = 1,
        target_memory_usage: float = 0.9
    ) -> int:
        """
        Dynamically adjust batch size based on available memory
        """
        # Get current memory usage
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated()
        available_memory = total_memory * target_memory_usage - allocated_memory
        
        # Estimate memory per sample
        dummy_batch = {
            'image': torch.randn(1, 3, 224, 224, device='cuda')
        }
        
        # Measure memory for single sample
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = model(**dummy_batch)
        
        torch.cuda.synchronize()
        memory_per_sample = torch.cuda.max_memory_allocated() - allocated_memory
        
        # Calculate optimal batch size
        optimal_batch_size = int(available_memory / (memory_per_sample * sequence_length))
        
        # Round down to multiple of 8 for efficiency
        optimal_batch_size = (optimal_batch_size // 8) * 8
        
        # Ensure at least base batch size
        optimal_batch_size = max(optimal_batch_size, base_batch_size)
        
        logger.info(f"Dynamic batch size: {optimal_batch_size} " +
                   f"(memory per sample: {memory_per_sample/1024**2:.1f}MB)")
        
        return optimal_batch_size