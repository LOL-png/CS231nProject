"""
GPU-accelerated preprocessing to eliminate CPU bottlenecks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GPUVideoPreprocessor(nn.Module):
    """
    GPU-accelerated video preprocessing
    Moves all preprocessing operations to GPU to eliminate CPU bottleneck
    """
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (224, 224),
                 patch_size: int = 16,
                 normalize: bool = True,
                 device: str = 'cuda'):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.normalize = normalize
        self.device = device
        
        # ImageNet normalization parameters as buffers (on GPU)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    @torch.no_grad()
    def preprocess_batch(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Preprocess a batch of frames on GPU
        Args:
            frames: [B, C, H, W] tensor already on GPU
        Returns:
            Preprocessed frames [B, C, H', W']
        """
        # Ensure we're on GPU
        if frames.device.type != 'cuda':
            frames = frames.to(self.device)
            
        # Resize all frames at once on GPU
        if frames.shape[-2:] != self.image_size:
            frames = F.interpolate(
                frames, 
                size=self.image_size, 
                mode='bilinear', 
                align_corners=False,
                antialias=True
            )
        
        # Normalize on GPU
        if self.normalize:
            frames = (frames - self.mean) / self.std
            
        return frames
    
    @torch.no_grad()
    def create_patches_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Convert batch of images to patches on GPU
        Args:
            images: [B, C, H, W] tensor on GPU
        Returns:
            patches: [B, num_patches, patch_dim] tensor
        """
        B, C, H, W = images.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        
        # Use unfold for efficient patch extraction on GPU
        patches = images.unfold(2, self.patch_size, self.patch_size)
        patches = patches.unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        
        # Flatten patches
        num_patches = patches.shape[1] * patches.shape[2]
        patch_dim = C * self.patch_size * self.patch_size
        patches = patches.view(B, num_patches, patch_dim)
        
        return patches
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Full preprocessing pipeline on GPU
        """
        frames = self.preprocess_batch(frames)
        patches = self.create_patches_batch(frames)
        return patches


class FastDataCollator:
    """
    Fast collation function that minimizes CPU operations
    """
    def __init__(self, device='cuda'):
        self.device = device
        
    def __call__(self, batch):
        """
        Collate batch and move to GPU in one operation
        """
        # Pre-allocate tensors on GPU when possible
        collated = {}
        
        # Stack tensors directly to GPU
        for key in batch[0].keys():
            if key in ['color', 'depth', 'segmentation', 'object_poses', 
                      'hand_pose', 'hand_joints_3d', 'hand_joints_2d', 
                      'ycb_ids', 'mano_betas', 'num_objects']:
                # Stack and move to GPU in one operation
                if isinstance(batch[0][key], torch.Tensor):
                    # Use non_blocking for async transfer
                    collated[key] = torch.stack([item[key] for item in batch]).to(
                        self.device, non_blocking=True
                    )
            else:
                # Keep other data as lists
                collated[key] = [item[key] for item in batch]
                
        return collated