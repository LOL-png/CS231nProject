"""
BFloat16 Compatibility Fix for DINOv2 and other pretrained models

This module provides utilities to handle BFloat16 compatibility issues when using
pretrained models that expect Float32 inputs.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class BFloat16CompatibilityWrapper(nn.Module):
    """
    Wrapper that ensures inputs are in the correct dtype for models that don't support BFloat16.
    This is particularly important for pretrained models like DINOv2.
    """
    
    def __init__(self, model: nn.Module, convert_inputs: bool = True, convert_outputs: bool = False):
        """
        Args:
            model: The model to wrap
            convert_inputs: Whether to convert BFloat16 inputs to Float32
            convert_outputs: Whether to convert Float32 outputs back to BFloat16
        """
        super().__init__()
        self.model = model
        self.convert_inputs = convert_inputs
        self.convert_outputs = convert_outputs
        
    def forward(self, *args, **kwargs):
        # Convert inputs from BFloat16 to Float32 if needed
        if self.convert_inputs:
            args = [self._convert_to_float32(arg) for arg in args]
            kwargs = {k: self._convert_to_float32(v) for k, v in kwargs.items()}
        
        # Forward pass
        output = self.model(*args, **kwargs)
        
        # Convert outputs back to BFloat16 if needed
        if self.convert_outputs:
            output = self._convert_to_bfloat16(output)
            
        return output
    
    def _convert_to_float32(self, x):
        """Convert tensor or dict of tensors to float32"""
        if isinstance(x, torch.Tensor) and x.dtype == torch.bfloat16:
            return x.float()
        elif isinstance(x, dict):
            return {k: self._convert_to_float32(v) for k, v in x.items()}
        elif isinstance(x, (list, tuple)):
            return type(x)(self._convert_to_float32(v) for v in x)
        return x
    
    def _convert_to_bfloat16(self, x):
        """Convert tensor or dict of tensors to bfloat16"""
        if isinstance(x, torch.Tensor) and x.dtype == torch.float32:
            return x.bfloat16()
        elif isinstance(x, dict):
            return {k: self._convert_to_bfloat16(v) for k, v in x.items()}
        elif isinstance(x, (list, tuple)):
            return type(x)(self._convert_to_bfloat16(v) for v in x)
        return x


def fix_batch_dtype_for_dinov2(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fix batch data types for DINOv2 compatibility.
    Converts BFloat16 images to Float32 which DINOv2 expects.
    
    Args:
        batch: Dictionary containing batch data
        
    Returns:
        Batch with corrected dtypes
    """
    batch_fixed = batch.copy()
    
    # Convert image tensor if it's BFloat16
    if 'image' in batch and isinstance(batch['image'], torch.Tensor):
        if batch['image'].dtype == torch.bfloat16:
            batch_fixed['image'] = batch['image'].float()
    
    # Also handle 'images' key (some datasets use plural)
    if 'images' in batch and isinstance(batch['images'], torch.Tensor):
        if batch['images'].dtype == torch.bfloat16:
            batch_fixed['images'] = batch['images'].float()
            
    return batch_fixed


def create_mixed_precision_model(model: nn.Module, use_bf16: bool = True) -> nn.Module:
    """
    Create a model that handles mixed precision training with BFloat16.
    Automatically wraps components that need Float32 inputs.
    
    Args:
        model: The model to prepare for mixed precision
        use_bf16: Whether to use BFloat16 (if False, returns model as-is)
        
    Returns:
        Model ready for mixed precision training
    """
    if not use_bf16:
        return model
        
    # Check if model has DINOv2 encoder
    if hasattr(model, 'image_encoder') and hasattr(model.image_encoder, 'dinov2'):
        # Wrap the image encoder to handle dtype conversion
        model.image_encoder = BFloat16CompatibilityWrapper(
            model.image_encoder,
            convert_inputs=True,
            convert_outputs=True  # Convert back to BF16 for rest of model
        )
        print("✓ Wrapped DINOv2 encoder for BFloat16 compatibility")
        
    return model


class MixedPrecisionTrainingLoop:
    """
    Helper class for training loops with proper BFloat16 handling
    """
    
    @staticmethod
    def process_batch(batch: Dict[str, Any], model: nn.Module, use_bf16: bool = True) -> Dict[str, Any]:
        """
        Process a batch for training with proper dtype handling.
        
        Args:
            batch: The input batch
            model: The model (used to check for specific components)
            use_bf16: Whether BFloat16 training is enabled
            
        Returns:
            Processed batch ready for model input
        """
        if not use_bf16:
            return batch
            
        # Fix dtype issues for DINOv2
        if hasattr(model, 'image_encoder') and hasattr(model.image_encoder, 'dinov2'):
            batch = fix_batch_dtype_for_dinov2(batch)
            
        return batch


# Example usage in training loop:
"""
from fixes.bfloat16_compatibility import create_mixed_precision_model, MixedPrecisionTrainingLoop

# Wrap model for BFloat16 compatibility
model = create_mixed_precision_model(model, use_bf16=config.training.use_bf16)

# In training loop:
for batch in train_loader:
    # Fix batch dtypes if needed
    batch = MixedPrecisionTrainingLoop.process_batch(batch, model, config.training.use_bf16)
    
    # Now safe to use with mixed precision
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        outputs = model(batch)
"""