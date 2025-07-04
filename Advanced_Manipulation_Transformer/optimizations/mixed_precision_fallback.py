"""
Mixed precision training fallback when FP8 is not available
Uses BFloat16 which is well-supported on H200
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def enable_mixed_precision_training(
    model: nn.Module, 
    optimizer: torch.optim.Optimizer,
    use_fp8: bool = True,
    fallback_dtype: torch.dtype = torch.bfloat16
) -> Tuple[nn.Module, torch.optim.Optimizer, Optional[torch.cuda.amp.GradScaler]]:
    """
    Enable mixed precision training with fallback options
    
    Args:
        model: The model to train
        optimizer: The optimizer
        use_fp8: Whether to attempt FP8 (will fallback if not available)
        fallback_dtype: Dtype to use if FP8 not available (bfloat16 or float16)
    
    Returns:
        model, optimizer, scaler (scaler is None for bfloat16)
    """
    
    # First try FP8 if requested
    if use_fp8:
        try:
            from .fp8_mixed_precision import enable_fp8_training, FP8_AVAILABLE
            if FP8_AVAILABLE:
                fp8_model, fp8_optimizer = enable_fp8_training(model, optimizer)
                logger.info("Using FP8 mixed precision training")
                return fp8_model, fp8_optimizer, None
        except Exception as e:
            logger.warning(f"Could not enable FP8: {e}")
    
    # Fallback to BFloat16 or Float16
    if fallback_dtype == torch.bfloat16:
        logger.info("Using BFloat16 mixed precision training (recommended for H200)")
        # BFloat16 doesn't need GradScaler
        return model, optimizer, None
    elif fallback_dtype == torch.float16:
        logger.info("Using Float16 mixed precision training")
        # Float16 needs GradScaler
        scaler = torch.cuda.amp.GradScaler()
        return model, optimizer, scaler
    else:
        logger.info("Using full precision training")
        return model, optimizer, None


class MixedPrecisionTrainer:
    """Helper class for mixed precision training"""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        use_fp8: bool = True,
        fallback_dtype: torch.dtype = torch.bfloat16
    ):
        self.model, self.optimizer, self.scaler = enable_mixed_precision_training(
            model, optimizer, use_fp8, fallback_dtype
        )
        self.dtype = fallback_dtype
        self.use_scaler = self.scaler is not None
        
    def train_step(self, batch, loss_fn):
        """Execute a training step with mixed precision"""
        self.optimizer.zero_grad()
        
        # Forward pass with autocast
        with torch.amp.autocast('cuda', dtype=self.dtype):
            outputs = self.model(batch)
            loss = loss_fn(outputs, batch)
        
        # Backward pass
        if self.use_scaler:
            # Float16 path with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # BFloat16 or FP8 path - no scaling needed
            loss.backward()
            self.optimizer.step()
        
        return loss.item(), outputs
    
    def evaluate_step(self, batch, loss_fn):
        """Execute an evaluation step with mixed precision"""
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=self.dtype):
                outputs = self.model(batch)
                loss = loss_fn(outputs, batch)
        
        return loss.item(), outputs


# Utility function to check mixed precision support
def check_mixed_precision_support():
    """Check what mixed precision options are available"""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'cuda_capability': torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        'bfloat16_support': False,
        'float16_support': False,
        'fp8_support': False,
        'tf32_enabled': False
    }
    
    if torch.cuda.is_available():
        # Check BFloat16 support (requires compute capability >= 8.0)
        major, minor = torch.cuda.get_device_capability(0)
        info['bfloat16_support'] = major >= 8
        info['float16_support'] = major >= 7  # Tensor cores on V100+
        info['fp8_support'] = major >= 9  # H100/H200
        info['tf32_enabled'] = torch.backends.cuda.matmul.allow_tf32
    
    # Try to import transformer engine for FP8
    try:
        from .fp8_mixed_precision import FP8_AVAILABLE
        info['fp8_available'] = FP8_AVAILABLE
    except:
        info['fp8_available'] = False
    
    return info


def print_mixed_precision_info():
    """Print mixed precision capabilities"""
    info = check_mixed_precision_support()
    
    print("Mixed Precision Support:")
    print(f"  GPU: {info['gpu_name']}")
    print(f"  CUDA Capability: {info['cuda_capability']}")
    print(f"  BFloat16: {'✓' if info['bfloat16_support'] else '✗'}")
    print(f"  Float16: {'✓' if info['float16_support'] else '✗'}")
    print(f"  FP8: {'✓' if info['fp8_support'] else '✗'} (HW support)")
    print(f"  FP8 Available: {'✓' if info.get('fp8_available', False) else '✗'} (SW support)")
    print(f"  TF32: {'✓' if info['tf32_enabled'] else '✗'}")
    
    # Recommendation
    if info['fp8_available']:
        print("\nRecommendation: Use FP8 training for best performance")
    elif info['bfloat16_support']:
        print("\nRecommendation: Use BFloat16 training for best stability")
    elif info['float16_support']:
        print("\nRecommendation: Use Float16 training with loss scaling")
    else:
        print("\nRecommendation: Use full precision training")