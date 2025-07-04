"""
Fix for BFloat16/Float32 mixing issues with camera parameters
This module provides utilities to ensure camera parameters match model dtype
"""

import torch
from typing import Dict, Any, Union


def convert_camera_params_dtype(
    camera_params: Dict[str, torch.Tensor], 
    target_dtype: torch.dtype
) -> Dict[str, torch.Tensor]:
    """
    Convert all camera parameters to target dtype
    
    Args:
        camera_params: Dictionary containing camera intrinsics/extrinsics
        target_dtype: Target dtype (e.g., torch.bfloat16 or torch.float32)
    
    Returns:
        Dictionary with all tensors converted to target dtype
    """
    converted = {}
    for key, value in camera_params.items():
        if isinstance(value, torch.Tensor):
            converted[key] = value.to(target_dtype)
        else:
            converted[key] = value
    return converted


def fix_batch_dtypes(batch: Dict[str, Any], use_bfloat16: bool = True) -> Dict[str, Any]:
    """
    Fix dtype inconsistencies in a batch, particularly for camera parameters
    
    Args:
        batch: Batch dictionary from dataloader
        use_bfloat16: Whether to use BFloat16 precision
    
    Returns:
        Batch with consistent dtypes
    """
    target_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
    
    # Fix camera intrinsics if present
    if 'camera_intrinsics' in batch and isinstance(batch['camera_intrinsics'], torch.Tensor):
        batch['camera_intrinsics'] = batch['camera_intrinsics'].to(target_dtype)
    
    # Fix camera params dictionary if present
    if 'camera_params' in batch and isinstance(batch['camera_params'], dict):
        batch['camera_params'] = convert_camera_params_dtype(batch['camera_params'], target_dtype)
    
    # Fix any other camera-related fields
    camera_fields = ['intrinsics', 'extrinsics', 'camera_K', 'camera_R', 'camera_t']
    for field in camera_fields:
        if field in batch and isinstance(batch[field], torch.Tensor):
            batch[field] = batch[field].to(target_dtype)
    
    return batch


class BFloat16DataWrapper:
    """
    Wrapper for dataloaders that ensures consistent BFloat16 dtypes
    Particularly important for camera parameters
    """
    
    def __init__(self, dataloader, use_bfloat16: bool = True):
        self.dataloader = dataloader
        self.use_bfloat16 = use_bfloat16
        self.target_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        
    def __iter__(self):
        for batch in self.dataloader:
            # Convert image to float32 for DINOv2
            if 'image' in batch and batch['image'].dtype == torch.bfloat16:
                batch['image'] = batch['image'].float()
            
            # Fix camera parameters and other fields
            batch = fix_batch_dtypes(batch, self.use_bfloat16)
            
            # Ensure camera_params dictionary exists and has proper dtype
            if 'camera_intrinsics' in batch and 'camera_params' not in batch:
                batch['camera_params'] = {
                    'intrinsics': batch['camera_intrinsics'].to(self.target_dtype)
                }
            
            yield batch
    
    def __len__(self):
        return len(self.dataloader)
    
    @property
    def batch_size(self):
        return getattr(self.dataloader, 'batch_size', None)


def prepare_camera_params_for_model(batch: Dict[str, Any], model_dtype: torch.dtype) -> Dict[str, Any]:
    """
    Prepare camera parameters in batch for model forward pass
    
    Args:
        batch: Input batch
        model_dtype: Model's dtype (from any model parameter)
    
    Returns:
        Batch with camera parameters matching model dtype
    """
    # Create camera_params dict if needed
    if 'camera_params' not in batch and 'camera_intrinsics' in batch:
        batch['camera_params'] = {'intrinsics': batch['camera_intrinsics']}
    
    # Convert camera params to model dtype
    if 'camera_params' in batch:
        batch['camera_params'] = convert_camera_params_dtype(batch['camera_params'], model_dtype)
    
    return batch


# Example usage in notebook:
"""
# In your training loop or debug cell:
from fixes.bfloat16_camera_params_fix import BFloat16DataWrapper, prepare_camera_params_for_model

# Wrap your dataloader
train_loader_wrapped = BFloat16DataWrapper(train_loader, use_bfloat16=config.training.use_bf16)
val_loader_wrapped = BFloat16DataWrapper(val_loader, use_bfloat16=config.training.use_bf16)

# In forward pass:
model_dtype = next(model.parameters()).dtype
batch = prepare_camera_params_for_model(batch, model_dtype)
outputs = model(batch)
"""