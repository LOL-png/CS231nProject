import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Dict, Optional, Tuple

class PixelAlignedRefinement(nn.Module):
    """
    Project 3D predictions back to 2D for feature refinement
    Critical for reducing MPJPE from 325mm to <100mm
    """
    
    def __init__(
        self,
        image_feat_dim: int = 1024,
        point_feat_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_refinement_steps: int = 2
    ):
        super().__init__()
        
        self.num_refinement_steps = num_refinement_steps
        
        # Feature refinement network with FPN-style architecture
        self.feat_refiner = nn.ModuleList([
            nn.Conv2d(image_feat_dim if i == 0 else 256, 256, 3, padding=1)
            for i in range(3)
        ])
        
        self.feat_norm = nn.ModuleList([
            nn.BatchNorm2d(256) for _ in range(3)
        ])
        
        # Final feature projection
        self.final_feat_proj = nn.Conv2d(256, point_feat_dim, 1)
        
        # Point feature encoder with positional encoding
        self.point_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, point_feat_dim)
        )
        
        # Iterative refinement modules
        self.refinement_modules = nn.ModuleList([
            nn.Sequential(
                nn.Linear(point_feat_dim * 2 + 3, hidden_dim),  # +3 for current position
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3)  # 3D offset prediction
            )
            for _ in range(num_refinement_steps)
        ])
        
        # Confidence prediction for each refinement step
        self.confidence_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1)
            for _ in range(num_refinement_steps)
        ])
    
    def project_3d_to_2d(
        self,
        points_3d: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: Optional[torch.Tensor] = None,
        image_size: Tuple[int, int] = (224, 224)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project 3D points to 2D image coordinates
        Args:
            points_3d: [B, N, 3] 3D points in camera/world coordinates
            intrinsics: [B, 3, 3] camera intrinsics
            extrinsics: [B, 4, 4] camera extrinsics (optional)
            image_size: (H, W) image dimensions for normalization
        Returns:
            points_2d_norm: [B, N, 2] 2D points in normalized [-1, 1] coordinates
            valid_mask: [B, N] boolean mask for points in front of camera
        """
        B, N, _ = points_3d.shape
        
        # Apply extrinsics if provided (world to camera)
        if extrinsics is not None:
            points_homo = torch.cat([
                points_3d, 
                torch.ones(B, N, 1, device=points_3d.device, dtype=points_3d.dtype)
            ], dim=-1)  # [B, N, 4]
            
            # Transform points - ensure same dtype
            points_cam = torch.matmul(
                points_homo, 
                extrinsics.to(points_homo.dtype).transpose(-1, -2)
            )[..., :3]  # [B, N, 3]
        else:
            points_cam = points_3d
        
        # Check if points are in front of camera
        valid_mask = points_cam[..., 2] > 0.1  # Z > 0.1
        
        # Project with intrinsics - ensure same dtype
        points_2d_homo = torch.matmul(points_cam, intrinsics.to(points_cam.dtype).transpose(-1, -2))
        
        # Normalize by depth
        depth = points_2d_homo[..., 2:3].clamp(min=0.1)
        points_2d = points_2d_homo[..., :2] / depth
        
        # Normalize to [-1, 1] for grid_sample
        H, W = image_size
        points_2d_norm = torch.stack([
            2.0 * points_2d[..., 0] / W - 1.0,
            2.0 * points_2d[..., 1] / H - 1.0
        ], dim=-1)
        
        # Clamp to valid range
        points_2d_norm = torch.clamp(points_2d_norm, -1.0, 1.0)
        
        return points_2d_norm, valid_mask
    
    def sample_image_features(
        self,
        image_features: torch.Tensor,
        points_2d: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample features from image feature map at 2D points
        Args:
            image_features: [B, C, H, W] feature map
            points_2d: [B, N, 2] normalized 2D coordinates in [-1, 1]
            valid_mask: [B, N] boolean mask for valid points
        Returns:
            [B, N, C] sampled features
        """
        B, C, H, W = image_features.shape
        N = points_2d.shape[1]
        
        # Reshape for grid_sample (requires 4D grid)
        points_2d = points_2d.unsqueeze(1)  # [B, 1, N, 2]
        
        # Sample features using bilinear interpolation
        sampled = F.grid_sample(
            image_features,
            points_2d,
            mode='bilinear',
            padding_mode='zeros',  # Use zeros for out-of-bounds
            align_corners=True
        )  # [B, C, 1, N]
        
        # Reshape to [B, N, C]
        sampled = sampled.squeeze(2).transpose(1, 2)
        
        # Zero out invalid points if mask provided
        if valid_mask is not None:
            sampled = sampled * valid_mask.unsqueeze(-1)
        
        return sampled
    
    def refine_image_features(self, feat_grid: torch.Tensor) -> torch.Tensor:
        """Apply feature pyramid refinement to image features"""
        # Progressive refinement through conv layers
        for i, (conv, norm) in enumerate(zip(self.feat_refiner, self.feat_norm)):
            # Store input for skip connection
            identity = feat_grid
            
            # Apply conv, norm, activation
            feat_grid = conv(feat_grid)
            feat_grid = norm(feat_grid)
            feat_grid = F.relu(feat_grid)
            
            # Skip connection with matching dimensions
            if i < len(self.feat_refiner) - 1:
                # Option 1: Simple residual connection (if dims match)
                if identity.shape[1] == feat_grid.shape[1]:
                    feat_grid = feat_grid + identity
                # Option 2: Use 1x1 conv to match channels if needed
                elif i == 0:  # First layer may have different input channels
                    # Skip the residual for first layer since channels don't match
                    pass
        
        # Final projection
        feat_grid = self.final_feat_proj(feat_grid)
        
        return feat_grid
    
    def forward(
        self,
        coarse_points: torch.Tensor,
        image_features: Dict[str, torch.Tensor],
        camera_params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Refine 3D points using pixel-aligned features
        Args:
            coarse_points: [B, N, 3] initial 3D predictions
            image_features: Dict with 'patch_grid' [B, H, W, C]
            camera_params: Dict with 'intrinsics' and optionally 'extrinsics'
        Returns:
            Dictionary with:
                - refined_points: [B, N, 3] refined 3D points
                - confidence: [B, N] confidence scores
                - intermediate_points: List of points at each refinement step
        """
        B, N, _ = coarse_points.shape
        
        # Get image feature grid and refine it
        feat_grid = image_features['patch_grid']  # [B, H, W, C]
        feat_grid = rearrange(feat_grid, 'b h w c -> b c h w')
        refined_features = self.refine_image_features(feat_grid)  # [B, C', H, W]
        
        # Initialize points and outputs
        current_points = coarse_points
        intermediate_points = [current_points]
        all_confidences = []
        
        # Iterative refinement
        for step in range(self.num_refinement_steps):
            # Project current 3D points to 2D
            points_2d, valid_mask = self.project_3d_to_2d(
                current_points,
                camera_params['intrinsics'],
                camera_params.get('extrinsics', None),
                image_size=(refined_features.shape[2], refined_features.shape[3])
            )
            
            # Sample pixel-aligned features
            aligned_features = self.sample_image_features(
                refined_features, 
                points_2d, 
                valid_mask
            )
            
            # Encode current 3D point positions
            point_features = self.point_encoder(current_points)
            
            # Concatenate all features
            combined = torch.cat([
                point_features, 
                aligned_features,
                current_points  # Include position for residual learning
            ], dim=-1)
            
            # Predict refinement
            refinement_features = self.refinement_modules[step](combined)
            
            # Extract offset (last layer already outputs 3D offset)
            offset = refinement_features
            
            # Predict confidence for this refinement
            if step < len(self.confidence_heads):
                # Pass through intermediate features for confidence
                intermediate_feat = self.refinement_modules[step][:-1](combined)  # All but last layer
                confidence = torch.sigmoid(
                    self.confidence_heads[step](intermediate_feat).squeeze(-1)
                )
                all_confidences.append(confidence)
            
            # Apply residual refinement with decreasing step size
            step_weight = 0.5 ** step  # Exponentially decreasing steps
            current_points = current_points + step_weight * offset
            intermediate_points.append(current_points)
        
        # Aggregate confidences
        if all_confidences:
            final_confidence = torch.stack(all_confidences, dim=-1).mean(dim=-1)
        else:
            final_confidence = torch.ones(B, N, device=current_points.device)
        
        return {
            'refined_points': current_points,
            'confidence': final_confidence,
            'intermediate_points': intermediate_points
        }