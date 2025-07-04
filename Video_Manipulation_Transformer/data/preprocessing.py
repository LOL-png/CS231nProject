"""
Preprocessing utilities for 2D detection and data augmentation
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, List, Tuple, Optional


class VideoPreprocessor:
    """
    Preprocesses video frames for the transformer model
    Includes 2D keypoint detection, object detection, and patch embedding
    """
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (224, 224),
                 patch_size: int = 16,
                 normalize: bool = True):
        self.image_size = image_size
        self.patch_size = patch_size
        self.normalize = normalize
        
        # Image normalization (ImageNet stats)
        self.normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
    def preprocess_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Preprocess a single frame
        Args:
            frame: [H, W, 3] or [3, H, W] tensor
        Returns:
            Preprocessed frame [3, H', W']
        """
        # Ensure channel-first format
        if frame.dim() == 3 and frame.shape[-1] == 3:
            frame = frame.permute(2, 0, 1)
            
        # Resize
        frame = F.interpolate(
            frame.unsqueeze(0), 
            size=self.image_size, 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        # Normalize
        if self.normalize:
            frame = self.normalize_transform(frame)
            
        return frame
    
    def create_patches(self, image: torch.Tensor) -> torch.Tensor:
        """
        Convert image to patches for transformer input
        Args:
            image: [B, C, H, W] tensor
        Returns:
            patches: [B, num_patches, patch_dim] tensor
        """
        B, C, H, W = image.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        
        # Create patches
        patches = image.unfold(2, self.patch_size, self.patch_size)
        patches = patches.unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        
        # Flatten patches
        num_patches = patches.shape[1] * patches.shape[2]
        patch_dim = C * self.patch_size * self.patch_size
        patches = patches.view(B, num_patches, patch_dim)
        
        return patches
    
    def get_hand_region(self, 
                       image: torch.Tensor, 
                       hand_joints_2d: torch.Tensor,
                       expand_ratio: float = 1.5) -> torch.Tensor:
        """
        Extract hand region from image based on 2D joint positions
        """
        # Get bounding box from joints
        min_x = hand_joints_2d[:, :, 0].min(dim=1)[0]
        max_x = hand_joints_2d[:, :, 0].max(dim=1)[0]
        min_y = hand_joints_2d[:, :, 1].min(dim=1)[0]
        max_y = hand_joints_2d[:, :, 1].max(dim=1)[0]
        
        # Expand bounding box
        width = max_x - min_x
        height = max_y - min_y
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        new_width = width * expand_ratio
        new_height = height * expand_ratio
        
        # Ensure minimum size
        min_size = 32  # Minimum size to avoid zero-sized crops
        new_width = torch.clamp(new_width, min=min_size)
        new_height = torch.clamp(new_height, min=min_size)
        
        # Crop regions
        crops = []
        for i in range(image.shape[0]):
            x1 = int(max(0, center_x[i] - new_width[i]/2))
            x2 = int(min(image.shape[-1], center_x[i] + new_width[i]/2))
            y1 = int(max(0, center_y[i] - new_height[i]/2))
            y2 = int(min(image.shape[-2], center_y[i] + new_height[i]/2))
            
            # Ensure we have a valid crop region
            if x2 <= x1:
                x2 = x1 + min_size
            if y2 <= y1:
                y2 = y1 + min_size
                
            # Clip to image bounds again after adjustment
            x2 = min(x2, image.shape[-1])
            y2 = min(y2, image.shape[-2])
            
            # If still invalid, use a default region
            if x2 <= x1 or y2 <= y1:
                # Use center of image as fallback
                h, w = image.shape[-2], image.shape[-1]
                x1, x2 = w//4, 3*w//4
                y1, y2 = h//4, 3*h//4
            
            crop = image[i:i+1, :, y1:y2, x1:x2]
            # Resize to fixed size
            crop = F.interpolate(crop, size=(224, 224), mode='bilinear', align_corners=False)
            crops.append(crop)
            
        return torch.cat(crops, dim=0)
    
    def get_object_regions(self,
                          image: torch.Tensor,
                          segmentation: torch.Tensor,
                          ycb_ids: List[int]) -> Dict[int, torch.Tensor]:
        """
        Extract object regions based on segmentation masks
        """
        object_regions = {}
        
        for obj_idx, obj_id in enumerate(ycb_ids):
            # Object segmentation ID is obj_idx + 1 (0 is background)
            mask = (segmentation == obj_idx + 1).float()
            
            if mask.sum() > 0:
                # Get bounding box
                coords = torch.nonzero(mask[0])
                if len(coords) > 0:
                    y_min, x_min = coords.min(dim=0)[0]
                    y_max, x_max = coords.max(dim=0)[0]
                    
                    # Crop and resize
                    crop = image[:, :, y_min:y_max+1, x_min:x_max+1]
                    crop = F.interpolate(crop, size=(224, 224), mode='bilinear', align_corners=False)
                    
                    object_regions[obj_id] = crop
                    
        return object_regions
    
    def augment_sequence(self, sequence: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply data augmentation to a sequence
        """
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            sequence['color'] = torch.flip(sequence['color'], dims=[-1])
            if 'hand_joints_2d' in sequence:
                sequence['hand_joints_2d'][..., 0] = sequence['color'].shape[-1] - sequence['hand_joints_2d'][..., 0]
                
        # Color jittering
        color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        for t in range(sequence['color'].shape[0]):
            sequence['color'][t] = color_jitter(sequence['color'][t])
            
        return sequence


class PositionalEncoding:
    """
    Positional encoding for transformer inputs
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        self.d_model = d_model
        self.max_len = max_len
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
    def encode_spatial(self, num_patches_h: int, num_patches_w: int) -> torch.Tensor:
        """
        Create 2D spatial positional encoding
        """
        h_pos = torch.arange(num_patches_h).unsqueeze(1).repeat(1, num_patches_w)
        w_pos = torch.arange(num_patches_w).unsqueeze(0).repeat(num_patches_h, 1)
        
        h_encoding = self.pe[:, :num_patches_h, :self.d_model//2]
        w_encoding = self.pe[:, :num_patches_w, self.d_model//2:]
        
        spatial_encoding = torch.zeros(1, num_patches_h * num_patches_w, self.d_model)
        
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                idx = i * num_patches_w + j
                spatial_encoding[0, idx, :self.d_model//2] = h_encoding[0, i]
                spatial_encoding[0, idx, self.d_model//2:] = w_encoding[0, j]
                
        return spatial_encoding
    
    def encode_temporal(self, sequence_length: int) -> torch.Tensor:
        """
        Create temporal positional encoding
        """
        return self.pe[:, :sequence_length, :]