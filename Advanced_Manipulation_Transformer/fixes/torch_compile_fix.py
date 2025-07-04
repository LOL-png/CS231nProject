"""
Fix for torch.compile issues with shape inference
=================================================

This module provides fixes for torch.compile errors that occur with certain
parameter combinations, particularly in the contact decoder.

The error occurs when torch.compile tries to trace through complex tensor
operations with dynamic shapes.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def safe_compile_model(model: nn.Module, config: Optional[Dict[str, Any]] = None) -> nn.Module:
    """
    Safely compile model with fallbacks for problematic modules
    
    Args:
        model: Model to compile
        config: Compilation configuration
        
    Returns:
        Compiled model (or original if compilation fails)
    """
    config = config or {}
    compile_mode = config.get('compile_mode', 'default')
    
    # Check if torch.compile is available
    if not hasattr(torch, 'compile'):
        logger.warning("torch.compile not available, skipping compilation")
        return model
    
    try:
        # First, try to identify modules that might cause issues
        problematic_modules = identify_problematic_modules(model)
        
        if problematic_modules:
            logger.info(f"Found {len(problematic_modules)} potentially problematic modules for torch.compile")
            
            # Apply workarounds to problematic modules
            for module_name in problematic_modules:
                apply_compile_workaround(model, module_name)
        
        # Try different compilation strategies
        if compile_mode == 'max_performance':
            # For maximum performance, try fullgraph first
            try:
                compiled_model = torch.compile(
                    model,
                    mode='max-autotune',
                    fullgraph=True,
                    dynamic=False
                )
                logger.info("✓ Model compiled with max-autotune mode (fullgraph)")
                return compiled_model
            except Exception as e:
                logger.warning(f"Fullgraph compilation failed: {e}")
                # Fall back to partial graph
                try:
                    compiled_model = torch.compile(
                        model,
                        mode='max-autotune',
                        fullgraph=False,
                        dynamic=True
                    )
                    logger.info("✓ Model compiled with max-autotune mode (partial graph)")
                    return compiled_model
                except Exception as e2:
                    logger.warning(f"Partial graph compilation also failed: {e2}")
        
        elif compile_mode == 'reduce-overhead':
            # For reduce-overhead, use safer settings
            try:
                compiled_model = torch.compile(
                    model,
                    mode='reduce-overhead',
                    fullgraph=False,  # Don't use fullgraph for reduce-overhead
                    dynamic=True      # Allow dynamic shapes
                )
                logger.info("✓ Model compiled with reduce-overhead mode")
                return compiled_model
            except Exception as e:
                logger.warning(f"Reduce-overhead compilation failed: {e}")
        
        else:
            # Default mode - most compatible
            try:
                compiled_model = torch.compile(
                    model,
                    fullgraph=False,
                    dynamic=True,
                    backend='inductor'
                )
                logger.info("✓ Model compiled with default mode")
                return compiled_model
            except Exception as e:
                logger.warning(f"Default compilation failed: {e}")
        
        # If all compilation attempts fail, return original model
        logger.warning("All compilation attempts failed, using eager mode")
        return model
        
    except Exception as e:
        logger.error(f"Unexpected error during model compilation: {e}")
        return model


def identify_problematic_modules(model: nn.Module) -> list:
    """
    Identify modules that might cause torch.compile issues
    
    Returns:
        List of module names that might be problematic
    """
    problematic = []
    
    for name, module in model.named_modules():
        # Check for modules with dynamic shapes
        if isinstance(module, nn.MultiheadAttention):
            # MultiheadAttention with certain configurations can cause issues
            if hasattr(module, 'batch_first') and module.batch_first:
                problematic.append(name)
        
        # Check for custom modules with complex tensor operations
        if 'ContactDecoder' in module.__class__.__name__:
            problematic.append(name)
        
        # Check for modules with parameter-dependent reshaping
        if 'decoder' in name.lower() and hasattr(module, 'forward'):
            # This is a heuristic - decoders often have complex reshaping
            problematic.append(name)
    
    return problematic


def apply_compile_workaround(model: nn.Module, module_name: str):
    """
    Apply workarounds to specific modules to make them compile-friendly
    
    Args:
        model: The model containing the module
        module_name: Name of the module to fix
    """
    try:
        # Get the module
        module = model
        for part in module_name.split('.'):
            module = getattr(module, part)
        
        # Apply specific workarounds based on module type
        if isinstance(module, nn.MultiheadAttention):
            # For MultiheadAttention, ensure it uses the right backend
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                # Prefer flash attention which compiles better
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(False)  # Disable math backend
        
        elif 'ContactDecoder' in module.__class__.__name__:
            # For ContactDecoder, we might need to modify the forward method
            # to be more compile-friendly
            _make_contact_decoder_compile_friendly(module)
        
        logger.info(f"Applied compile workaround to {module_name}")
        
    except Exception as e:
        logger.warning(f"Failed to apply workaround to {module_name}: {e}")


def _make_contact_decoder_compile_friendly(decoder: nn.Module):
    """
    Make ContactDecoder more compile-friendly by modifying problematic operations
    """
    # Store original forward
    original_forward = decoder.forward
    
    def compile_friendly_forward(self, global_features, hand_joints, object_positions):
        # Use torch.jit.annotate for better type inference
        B = global_features.shape[0]
        
        # Ensure consistent tensor shapes throughout
        with torch.jit.annotate(int, B):
            # Project features with explicit shape handling
            hand_feat = self.hand_proj(hand_joints.reshape(B, -1))
            obj_feat = self.obj_proj(object_positions.reshape(B, -1))
            global_feat = self.global_proj(global_features)
            
            # Combine features with explicit shape
            combined_feat = hand_feat + obj_feat + global_feat
            combined_feat = combined_feat.unsqueeze(1)
            
            # Expand contact queries with explicit size
            queries = self.contact_queries.unsqueeze(0).expand(B, -1, -1)
            
            # Use contiguous tensors for attention
            hand_feat_seq = self.hand_point_proj(hand_joints).contiguous()
            obj_feat_seq = self.obj_point_proj(object_positions).contiguous()
            
            # Cross-attention with explicit contiguous calls
            for i, attn in enumerate(self.cross_attention):
                if i == 0:
                    queries, _ = attn(
                        query=queries.contiguous(),
                        key=hand_feat_seq,
                        value=hand_feat_seq
                    )
                else:
                    queries, _ = attn(
                        query=queries.contiguous(),
                        key=obj_feat_seq,
                        value=obj_feat_seq
                    )
            
            # Ensure queries are contiguous before decoder
            queries = queries.contiguous()
            
            # Decode contacts with explicit tensor operations
            memory = torch.cat([combined_feat, queries], dim=1).contiguous()
            decoded = self.decoder(
                tgt=queries,
                memory=memory
            )
            
            # Predict outputs with explicit shapes
            contact_points = self.contact_point_head(decoded)
            contact_confidence = torch.sigmoid(self.contact_confidence_head(decoded)).squeeze(-1)
            contact_types = self.contact_type_head(decoded)
            contact_forces = self.contact_force_head(decoded)
            
            # Global interaction type
            pooled = decoded.mean(dim=1)
            interaction_type = self.interaction_type_head(pooled)
        
        return {
            'contact_points': contact_points,
            'contact_confidence': contact_confidence,
            'contact_types': contact_types,
            'contact_forces': contact_forces,
            'interaction_type': interaction_type,
            'features': decoded
        }
    
    # Replace forward method
    decoder.forward = compile_friendly_forward.__get__(decoder, decoder.__class__)


def create_compile_config(batch_size: int, model_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create optimal compile configuration based on batch size and model config
    
    Args:
        batch_size: Training batch size
        model_config: Model configuration
        
    Returns:
        Compile configuration dict
    """
    # Determine optimal compile mode based on batch size
    if batch_size >= 256:
        # Large batch sizes might have issues with fullgraph
        compile_mode = 'reduce-overhead'
    elif batch_size <= 32:
        # Small batch sizes can use more aggressive optimization
        compile_mode = 'max_performance'
    else:
        # Medium batch sizes use default
        compile_mode = 'default'
    
    # Check for specific model features that affect compilation
    if model_config.get('use_attention_fusion', True):
        # Attention fusion can cause shape inference issues
        compile_mode = 'default'
    
    if model_config.get('num_contact_points', 10) > 20:
        # Many contact points might cause compilation issues
        compile_mode = 'default'
    
    return {
        'compile_mode': compile_mode,
        'use_compile': True
    }


# Integration with existing optimization pipeline
def optimize_for_h200_with_compile_fix(model: nn.Module, config: Optional[Dict[str, Any]] = None) -> nn.Module:
    """
    Optimize model for H200 with torch.compile fixes
    
    Args:
        model: Model to optimize
        config: Optimization configuration
        
    Returns:
        Optimized model
    """
    from ..optimizations.pytorch_native_optimization import PyTorchNativeOptimizer
    
    config = config or {}
    
    # First apply standard optimizations
    optimizer = PyTorchNativeOptimizer()
    
    # Apply SDPA and CUDA optimizations
    model = optimizer._enable_sdpa_optimizations(model)
    optimizer._configure_cuda_settings()
    
    # Apply memory optimizations
    optimizer._enable_memory_optimizations(model)
    
    # Use our safe compile function instead of the standard one
    if config.get('use_compile', True):
        model = safe_compile_model(model, config)
    
    logger.info("✓ Model optimized with torch.compile fixes")
    return model


# Monkey patch for sweep scripts
def patch_sweep_optimization():
    """
    Patch the optimization function used in sweep scripts to use our fixed version
    """
    import sys
    if 'optimizations.pytorch_native_optimization' in sys.modules:
        pytorch_opt = sys.modules['optimizations.pytorch_native_optimization']
        # Replace the optimize_for_h200 function
        pytorch_opt.optimize_for_h200 = optimize_for_h200_with_compile_fix
        logger.info("✓ Patched pytorch_native_optimization with compile fixes")


if __name__ == "__main__":
    # Test the compile fix
    import torch.nn as nn
    
    print("Testing torch.compile fixes...")
    
    # Create a simple model with ContactDecoder-like structure
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(63, 512)  # 21 * 3
            self.attn = nn.MultiheadAttention(512, 8, batch_first=True)
            
        def forward(self, x):
            B = x.shape[0]
            x_flat = x.reshape(B, -1)
            x_proj = self.proj(x_flat)
            x_seq = x_proj.unsqueeze(1)
            out, _ = self.attn(x_seq, x_seq, x_seq)
            return out
    
    model = TestModel().cuda()
    
    # Test compilation with different batch sizes
    for batch_size in [32, 128, 256]:
        print(f"\nTesting batch size {batch_size}...")
        
        config = create_compile_config(batch_size, {'hidden_dim': 512})
        compiled_model = safe_compile_model(model, config)
        
        # Test forward pass
        x = torch.randn(batch_size, 21, 3, device='cuda')
        try:
            with torch.no_grad():
                out = compiled_model(x)
            print(f"✓ Batch size {batch_size} works correctly")
        except Exception as e:
            print(f"✗ Batch size {batch_size} failed: {e}")