"""
Contact Detection Encoder for Video-to-Manipulation Transformer
Specialized encoder for detecting hand-object contact points and interaction types
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class ContactDetectionEncoder(nn.Module):
    """
    Transformer encoder specialized for contact detection
    Architecture: 4 layers, 256 dim, 8 attention heads
    Input: Hand-object interaction regions
    Output: Contact points and interaction types
    """
    
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 mlp_dim: int = 1024,
                 dropout: float = 0.1,
                 num_contact_points: int = 10):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_contact_points = num_contact_points
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Separate projections for hand and object features
        # These receive features from other encoders, which are 512-dim (from hand/object encoders)
        self.hand_projection = nn.Linear(512, hidden_dim)
        self.object_projection = nn.Linear(512, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Cross-attention for hand-object interaction
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Contact point queries
        self.contact_queries = nn.Parameter(torch.zeros(num_contact_points, hidden_dim))
        nn.init.normal_(self.contact_queries, std=0.02)
        
        # Output heads
        self.contact_location_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # 3D contact location
        )
        
        self.contact_confidence_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)  # Contact presence confidence
        )
        
        self.contact_type_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 4)  # Contact types: no contact, light touch, firm grasp, manipulation
        )
        
        self.force_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # 3D force vector
        )
        
        # Global interaction classifier
        self.interaction_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 6)  # Interaction types: idle, approach, grasp, manipulate, release, retract
        )
        
    def forward(self,
                hand_features: torch.Tensor,
                object_features: torch.Tensor,
                interaction_patches: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            hand_features: [B, num_hand_patches, patch_dim] or [B, hidden_dim] hand features
            object_features: [B, num_obj_patches, patch_dim] or [B, num_objects, hidden_dim] object features
            interaction_patches: [B, num_patches, patch_dim] optional interaction region patches
            
        Returns:
            Dictionary containing:
                - contact_points: [B, num_contact_points, 3] 3D contact locations
                - contact_confidence: [B, num_contact_points] contact presence confidence
                - contact_types: [B, num_contact_points, 4] contact type probabilities
                - contact_forces: [B, num_contact_points, 3] predicted contact forces
                - interaction_type: [B, 6] global interaction type probabilities
                - features: [B, hidden_dim] contact features for fusion
        """
        B = hand_features.shape[0]
        
        # Project hand and object features if needed
        if hand_features.dim() == 3:
            hand_features = self.hand_projection(hand_features)
            hand_features = hand_features.mean(dim=1)  # Global pooling
        elif hand_features.dim() == 2 and hand_features.shape[-1] != self.hidden_dim:
            hand_features = self.hand_projection(hand_features.unsqueeze(1)).squeeze(1)
            
        if object_features.dim() == 3:
            object_features = self.object_projection(object_features)
            if object_features.shape[1] > 1:
                object_features = object_features.mean(dim=1)  # Global pooling
        
        # Prepare features for transformer
        features = []
        
        # Add hand and object features
        if hand_features.dim() == 2:
            features.append(hand_features.unsqueeze(1))
        else:
            features.append(hand_features)
            
        if object_features.dim() == 2:
            features.append(object_features.unsqueeze(1))
        else:
            features.append(object_features)
            
        # Add interaction patches if provided
        if interaction_patches is not None:
            interaction_features = self.input_projection(interaction_patches)
            features.append(interaction_features)
            
        # Concatenate all features
        x = torch.cat(features, dim=1)  # [B, num_features, hidden_dim]
        
        # Add contact queries
        contact_queries = self.contact_queries.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([contact_queries, x], dim=1)  # [B, num_contact_points + num_features, hidden_dim]
        
        # Apply transformer
        x = self.transformer(x)
        
        # Extract contact features
        contact_features = x[:, :self.num_contact_points]  # [B, num_contact_points, hidden_dim]
        
        # Cross-attention between contacts and hand-object features
        context_features = x[:, self.num_contact_points:]
        contact_features_refined, _ = self.cross_attention(
            query=contact_features,
            key=context_features,
            value=context_features
        )
        
        # Predict contact properties
        contact_points = self.contact_location_head(contact_features_refined)  # [B, num_contact_points, 3]
        contact_confidence = torch.sigmoid(self.contact_confidence_head(contact_features_refined)).squeeze(-1)  # [B, num_contact_points]
        contact_types = self.contact_type_head(contact_features_refined)  # [B, num_contact_points, 4]
        contact_forces = self.force_head(contact_features_refined)  # [B, num_contact_points, 3]
        
        # Global interaction type from aggregated features
        global_features = x.mean(dim=1)  # [B, hidden_dim]
        interaction_type = self.interaction_head(global_features)  # [B, 6]
        
        return {
            'contact_points': contact_points,
            'contact_confidence': contact_confidence,
            'contact_types': F.softmax(contact_types, dim=-1),
            'contact_forces': contact_forces,
            'interaction_type': F.softmax(interaction_type, dim=-1),
            'features': global_features
        }
    
    def compute_loss(self,
                    predictions: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute contact detection losses
        """
        losses = {}
        
        # Contact point loss (only for confident predictions)
        if 'contact_points_gt' in targets:
            # Mask by confidence
            confidence_mask = predictions['contact_confidence'] > 0.5
            if confidence_mask.any():
                pred_points = predictions['contact_points'][confidence_mask]
                target_points = targets['contact_points_gt'][confidence_mask]
                contact_point_loss = F.mse_loss(pred_points, target_points)
                losses['contact_point_loss'] = contact_point_loss
                
        # Contact confidence loss (binary cross entropy)
        if 'contact_mask_gt' in targets:
            confidence_loss = F.binary_cross_entropy(
                predictions['contact_confidence'],
                targets['contact_mask_gt'].float()
            )
            losses['confidence_loss'] = confidence_loss
            
        # Contact type loss
        if 'contact_types_gt' in targets:
            type_loss = F.cross_entropy(
                predictions['contact_types'].reshape(-1, 4),
                targets['contact_types_gt'].reshape(-1)
            )
            losses['contact_type_loss'] = type_loss
            
        # Force regression loss
        if 'contact_forces_gt' in targets:
            force_mask = targets['contact_mask_gt']
            if force_mask.any():
                pred_forces = predictions['contact_forces'][force_mask]
                target_forces = targets['contact_forces_gt'][force_mask]
                force_loss = F.mse_loss(pred_forces, target_forces)
                losses['force_loss'] = force_loss
                
        # Interaction type loss
        if 'interaction_type_gt' in targets:
            interaction_loss = F.cross_entropy(
                predictions['interaction_type'],
                targets['interaction_type_gt']
            )
            losses['interaction_loss'] = interaction_loss
            
        # Total loss
        if losses:
            losses['total'] = sum(losses.values())
        else:
            losses['total'] = torch.tensor(0.0, device=predictions['contact_points'].device)
            
        return losses


class ContactFeatureFusion(nn.Module):
    """
    Module for fusing hand, object, and contact features
    """
    
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        
        # Feature projections
        self.hand_proj = nn.Linear(hidden_dim, hidden_dim)
        self.object_proj = nn.Linear(hidden_dim, hidden_dim)
        self.contact_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self,
                hand_features: torch.Tensor,
                object_features: torch.Tensor,
                contact_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse features from all three encoders
        """
        # Project features
        h = self.hand_proj(hand_features)
        o = self.object_proj(object_features)
        c = self.contact_proj(contact_features)
        
        # Concatenate and fuse
        combined = torch.cat([h, o, c], dim=-1)
        fused = self.fusion(combined)
        
        return fused