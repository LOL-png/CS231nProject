"""
Hand Pose Encoder for Video-to-Manipulation Transformer
Specialized encoder for extracting 3D hand joint positions from RGB patches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class HandPoseEncoder(nn.Module):
    """
    Transformer encoder specialized for hand pose estimation
    Architecture: 6 layers, 512 dim, 8 attention heads
    Input: RGB patches around detected hands
    Output: 3D hand joint positions (21 keypoints)
    """
    
    def __init__(self,
                 input_dim: int = 768,  # 16x16x3 patches
                 hidden_dim: int = 512,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 mlp_dim: int = 2048,
                 dropout: float = 0.1,
                 num_joints: int = 21):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_joints = num_joints
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # CLS token for global hand representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Output heads
        self.joint_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_joints * 3)  # 21 joints x 3 coordinates
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_joints)  # Confidence per joint
        )
        
        # Hand shape regression (MANO betas)
        self.shape_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 10)  # 10 MANO shape parameters
        )
        
    def forward(self, 
                hand_patches: torch.Tensor,
                spatial_encoding: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            hand_patches: [B, num_patches, patch_dim] hand region patches
            spatial_encoding: [B, num_patches, hidden_dim] positional encoding
            
        Returns:
            Dictionary containing:
                - joints_3d: [B, 21, 3] predicted 3D joint positions
                - confidence: [B, 21] joint confidence scores
                - shape_params: [B, 10] MANO shape parameters
                - features: [B, hidden_dim] hand features for fusion
        """
        B = hand_patches.shape[0]
        
        # Project patches to hidden dimension
        x = self.input_projection(hand_patches)  # [B, num_patches, hidden_dim]
        
        # Add positional encoding if provided
        if spatial_encoding is not None:
            x = x + spatial_encoding
            
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1 + num_patches, hidden_dim]
        
        # Apply transformer
        x = self.transformer(x)
        
        # Extract CLS token output
        cls_output = x[:, 0]  # [B, hidden_dim]
        
        # Predict joints
        joints_flat = self.joint_head(cls_output)  # [B, 63]
        joints_3d = joints_flat.view(B, self.num_joints, 3)  # [B, 21, 3]
        
        # Predict confidence
        confidence = torch.sigmoid(self.confidence_head(cls_output))  # [B, 21]
        
        # Predict shape parameters
        shape_params = self.shape_head(cls_output)  # [B, 10]
        
        return {
            'joints_3d': joints_3d,
            'confidence': confidence,
            'shape_params': shape_params,
            'features': cls_output  # For fusion with other encoders
        }
    
    def compute_loss(self, 
                    predictions: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute hand pose losses
        """
        losses = {}
        
        # Joint position loss (L2)
        joint_loss = F.mse_loss(
            predictions['joints_3d'], 
            targets['hand_joints_3d'].squeeze(1)  # Remove extra dimension
        )
        losses['joint_loss'] = joint_loss
        
        # Weighted by confidence
        if 'confidence' in predictions:
            weighted_joint_loss = (
                predictions['confidence'].unsqueeze(-1) * 
                (predictions['joints_3d'] - targets['hand_joints_3d'].squeeze(1)) ** 2
            ).mean()
            losses['weighted_joint_loss'] = weighted_joint_loss
            
        # Shape parameter loss
        if 'mano_betas' in targets and 'shape_params' in predictions:
            shape_loss = F.mse_loss(
                predictions['shape_params'],
                targets['mano_betas']
            )
            losses['shape_loss'] = shape_loss
            
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses


class HandFeatureExtractor(nn.Module):
    """
    Lightweight CNN for initial hand feature extraction before transformer
    """
    
    def __init__(self, in_channels: int = 3, out_dim: int = 768):
        super().__init__()
        
        self.backbone = nn.Sequential(
            # Initial conv
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Conv blocks
            self._make_block(64, 128, stride=2),
            self._make_block(128, 256, stride=2),
            self._make_block(256, 512, stride=2),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            # Projection
            nn.Linear(512, out_dim)
        )
        
    def _make_block(self, in_channels: int, out_channels: int, stride: int = 1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)