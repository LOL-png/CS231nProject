import torch
import torch.nn as nn
from typing import Tuple
import logging
import os
import sys
import site

logger = logging.getLogger(__name__)

# Setup cuDNN path before trying to import transformer_engine
def _setup_cudnn_path():
    """Add cuDNN library path to LD_LIBRARY_PATH"""
    # Find the nvidia cudnn package location
    cudnn_path = os.path.join(sys.prefix, 'lib', 'python3.12', 'site-packages', 'nvidia', 'cudnn', 'lib')
    
    if os.path.exists(cudnn_path):
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        if cudnn_path not in current_ld_path:
            os.environ['LD_LIBRARY_PATH'] = f"{cudnn_path}:{current_ld_path}"
            logger.info(f"Added cuDNN path to LD_LIBRARY_PATH: {cudnn_path}")

_setup_cudnn_path()

# Try importing transformer engine
FP8_AVAILABLE = False
te = None
recipe = None

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    FP8_AVAILABLE = True
    logger.info("TransformerEngine loaded successfully")
except (ImportError, OSError) as e:
    logger.warning(f"TransformerEngine not available: {e}")
    logger.warning("FP8 training disabled. Model will use standard precision.")
    logger.info("This is normal if you don't have an H100/H200 GPU or proper drivers.")
    
    # Define dummy classes for compatibility
    class DummyTE:
        class Linear(nn.Linear):
            pass
        class TransformerLayer(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
            def forward(self, x):
                return x
        @staticmethod
        def fp8_autocast(*args, **kwargs):
            # Use bfloat16 as fallback
            import contextlib
            @contextlib.contextmanager
            def dummy_context():
                yield
            return dummy_context()
        class optimizers:
            FusedAdam = torch.optim.AdamW
    te = DummyTE()
    
    class DummyRecipe:
        class Format:
            E4M3 = None
            E5M2 = None
        @staticmethod
        def DelayedScaling(*args, **kwargs):
            return None
    recipe = DummyRecipe()

class FP8Module(nn.Module):
    """
    Wrapper for FP8 mixed precision on H200
    """
    
    def __init__(self, module: nn.Module, fp8_format='e4m3', amax_history_len=16):
        super().__init__()
        
        if not FP8_AVAILABLE:
            self.module = module
            self.use_fp8 = False
            return
        
        # Configure FP8 recipe
        self.fp8_recipe = recipe.DelayedScaling(
            margin=0,
            fp8_format=recipe.Format.E4M3 if fp8_format == 'e4m3' else recipe.Format.E5M2,
            amax_history_len=amax_history_len,
            amax_compute_algo='most_recent'
        )
        
        # Convert module to FP8
        self.module = self._convert_to_fp8(module)
        self.use_fp8 = True
    
    def _convert_to_fp8(self, module: nn.Module) -> nn.Module:
        """
        Convert linear and attention layers to FP8
        """
        fp8_module = module
        
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Replace with FP8 linear
                fp8_linear = te.Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None
                )
                
                # Copy weights
                with torch.no_grad():
                    fp8_linear.weight.copy_(child.weight)
                    if child.bias is not None:
                        fp8_linear.bias.copy_(child.bias)
                
                setattr(fp8_module, name, fp8_linear)
                
            elif isinstance(child, nn.TransformerEncoderLayer):
                # Replace with FP8 transformer layer
                fp8_layer = te.TransformerLayer(
                    hidden_size=child.self_attn.embed_dim,
                    ffn_hidden_size=child.linear1.out_features,
                    num_attention_heads=child.self_attn.num_heads,
                    layernorm_epsilon=child.norm1.eps,
                    hidden_dropout=child.dropout.p,
                    attention_dropout=child.self_attn.dropout
                )
                
                setattr(fp8_module, name, fp8_layer)
            else:
                # Recursively convert children
                setattr(fp8_module, name, self._convert_to_fp8(child))
        
        return fp8_module
    
    def forward(self, *args, **kwargs):
        if self.use_fp8:
            with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
                return self.module(*args, **kwargs)
        else:
            return self.module(*args, **kwargs)

def enable_fp8_training(model: nn.Module, optimizer: torch.optim.Optimizer) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """
    Enable FP8 training for the entire model
    Returns original model and optimizer if FP8 is not available
    """
    if not FP8_AVAILABLE:
        logger.info("FP8 not available, using standard precision training")
        return model, optimizer
    
    try:
        # Wrap model with FP8
        fp8_model = FP8Module(model)
        
        # Create FP8-aware optimizer
        fp8_optimizer = te.optimizers.FusedAdam(
            fp8_model.parameters(),
            lr=optimizer.defaults['lr'],
            betas=optimizer.defaults.get('betas', (0.9, 0.999)),
            eps=optimizer.defaults.get('eps', 1e-8),
            weight_decay=optimizer.defaults.get('weight_decay', 0.01)
        )
        
        logger.info("Enabled FP8 mixed precision training")
        return fp8_model, fp8_optimizer
        
    except Exception as e:
        logger.warning(f"Failed to enable FP8: {e}")
        logger.info("Falling back to standard precision training")
        return model, optimizer