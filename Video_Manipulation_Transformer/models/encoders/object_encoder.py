"""
Object Pose Encoder for Video-to-Manipulation Transformer
Specialized encoder for 6-DoF object pose estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class ObjectPoseEncoder(nn.Module):
    """
    Transformer encoder specialized for object pose estimation
    Architecture: 6 layers, 512 dim, 8 attention heads
    Input: Object regions from detection
    Output: 6-DoF object poses (position + rotation)
    """
    
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = 512,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 mlp_dim: int = 2048,
                 dropout: float = 0.1,
                 max_objects: int = 10):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_objects = max_objects
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Object type embedding (for different YCB objects)
        self.object_type_embedding = nn.Embedding(100, hidden_dim)  # Support up to 100 object types
        
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
        
        # Object queries for detecting multiple objects
        self.object_queries = nn.Parameter(torch.zeros(max_objects, hidden_dim))
        nn.init.normal_(self.object_queries, std=0.02)
        
        # Output heads
        self.position_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # 3D position
        )
        
        self.rotation_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 6)  # 6D rotation representation
        )
        
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)  # Object presence confidence
        )
        
        self.class_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 100)  # Object class logits
        )
        
    def forward(self,
                object_patches: torch.Tensor,
                object_ids: Optional[List[int]] = None,
                spatial_encoding: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            object_patches: [B, num_patches, patch_dim] object region patches
            object_ids: List of YCB object IDs (optional)
            spatial_encoding: [B, num_patches, hidden_dim] positional encoding
            
        Returns:
            Dictionary containing:
                - positions: [B, max_objects, 3] object positions
                - rotations: [B, max_objects, 6] object rotations (6D representation)
                - confidence: [B, max_objects] object presence confidence
                - class_logits: [B, max_objects, 100] object class predictions
                - features: [B, max_objects, hidden_dim] object features for fusion
        """
        B = object_patches.shape[0]
        
        # Project patches
        x = self.input_projection(object_patches)  # [B, num_patches, hidden_dim]
        
        # Add positional encoding
        if spatial_encoding is not None:
            x = x + spatial_encoding
            
        # Add object type embeddings if IDs provided
        if object_ids is not None:
            # Handle both tensor and list inputs
            if torch.is_tensor(object_ids):
                # object_ids is a tensor [B, max_objects]
                obj_tokens = []
                for batch_idx in range(B):
                    # Get valid object IDs (not -1)
                    ids = object_ids[batch_idx]
                    valid_mask = ids != -1
                    valid_ids = ids[valid_mask]
                    
                    if valid_ids.numel() > 0:
                        tokens = self.object_type_embedding(valid_ids)
                    else:
                        # Use learnable queries if no valid IDs
                        tokens = self.object_queries[:1]
                    obj_tokens.append(tokens)
            else:
                # object_ids is a list
                obj_tokens = []
                for batch_idx in range(B):
                    if batch_idx < len(object_ids) and len(object_ids[batch_idx]) > 0:
                        # Get embeddings for detected objects
                        ids = torch.tensor(object_ids[batch_idx], device=x.device)
                        tokens = self.object_type_embedding(ids)
                    else:
                        # Use learnable queries if no IDs provided
                        tokens = self.object_queries[:1]  # At least one object
                    obj_tokens.append(tokens)
                
            # Pad to max_objects
            obj_tokens_padded = []
            for tokens in obj_tokens:
                num_objs = tokens.shape[0]
                if num_objs < self.max_objects:
                    padding = self.object_queries[num_objs:self.max_objects]
                    tokens = torch.cat([tokens, padding], dim=0)
                else:
                    tokens = tokens[:self.max_objects]
                obj_tokens_padded.append(tokens)
                
            obj_tokens = torch.stack(obj_tokens_padded, dim=0)  # [B, max_objects, hidden_dim]
        else:
            # Use learnable object queries
            obj_tokens = self.object_queries.unsqueeze(0).expand(B, -1, -1)
            
        # Concatenate object queries with patches
        x = torch.cat([obj_tokens, x], dim=1)  # [B, max_objects + num_patches, hidden_dim]
        
        # Apply transformer
        x = self.transformer(x)
        
        # Extract object features
        object_features = x[:, :self.max_objects]  # [B, max_objects, hidden_dim]
        
        # Predict poses
        positions = self.position_head(object_features)  # [B, max_objects, 3]
        rotations = self.rotation_head(object_features)  # [B, max_objects, 6]
        
        # Normalize 6D rotation representation
        rotations = self.normalize_6d_rotation(rotations)
        
        # Predict confidence and class
        confidence = torch.sigmoid(self.confidence_head(object_features)).squeeze(-1)  # [B, max_objects]
        class_logits = self.class_head(object_features)  # [B, max_objects, 100]
        
        return {
            'positions': positions,
            'rotations': rotations,
            'confidence': confidence,
            'class_logits': class_logits,
            'features': object_features
        }
    
    def normalize_6d_rotation(self, rot6d: torch.Tensor) -> torch.Tensor:
        """
        Normalize 6D rotation representation to ensure valid rotation
        """
        # Reshape to [..., 2, 3]
        shape = rot6d.shape[:-1]
        rot6d = rot6d.view(*shape, 2, 3)
        
        # Normalize first vector
        a1 = rot6d[..., 0, :]
        a1 = F.normalize(a1, dim=-1)
        
        # Make second vector orthogonal to first
        a2 = rot6d[..., 1, :]
        a2 = a2 - (a1 * a2).sum(dim=-1, keepdim=True) * a1
        a2 = F.normalize(a2, dim=-1)
        
        # Reshape back
        rot6d_normalized = torch.stack([a1, a2], dim=-2)
        rot6d_normalized = rot6d_normalized.view(*shape, 6)
        
        return rot6d_normalized
    
    def rotation_6d_to_matrix(self, rot6d: torch.Tensor) -> torch.Tensor:
        """
        Convert 6D rotation representation to rotation matrix
        """
        shape = rot6d.shape[:-1]
        rot6d = rot6d.view(*shape, 2, 3)
        
        a1 = rot6d[..., 0, :]
        a2 = rot6d[..., 1, :]
        
        # Compute third axis
        a3 = torch.cross(a1, a2, dim=-1)
        
        # Stack to form rotation matrix
        rot_mat = torch.stack([a1, a2, a3], dim=-1)  # [..., 3, 3]
        
        return rot_mat
    
    def compute_loss(self,
                    predictions: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute object pose losses
        """
        losses = {}
        
        # Position loss
        if 'object_poses' in targets:
            # Extract target positions from pose matrices
            target_positions = targets['object_poses'][..., :3, 3]  # [B, num_obj, 3]
            
            # Match predictions to targets (simplified - assumes ordering)
            num_targets = target_positions.shape[1]
            pred_positions = predictions['positions'][:, :num_targets]
            
            position_loss = F.mse_loss(pred_positions, target_positions)
            losses['position_loss'] = position_loss
            
            # Rotation loss (using 6D representation)
            target_rotations = targets['object_poses'][..., :3, :3]  # [B, num_obj, 3, 3]
            # Convert to 6D
            target_rot6d = torch.cat([
                target_rotations[..., :, 0],
                target_rotations[..., :, 1]
            ], dim=-1)  # [B, num_obj, 6]
            
            pred_rotations = predictions['rotations'][:, :num_targets]
            rotation_loss = F.mse_loss(pred_rotations, target_rot6d)
            losses['rotation_loss'] = rotation_loss
            
        # Classification loss
        if 'ycb_ids' in targets and 'class_logits' in predictions:
            # Create target labels
            B = predictions['class_logits'].shape[0]
            target_classes = torch.zeros(B, self.max_objects, dtype=torch.long, 
                                       device=predictions['class_logits'].device)
            
            for b in range(B):
                if b < len(targets['ycb_ids']):
                    num_objs = min(len(targets['ycb_ids'][b]), self.max_objects)
                    target_classes[b, :num_objs] = torch.tensor(targets['ycb_ids'][b][:num_objs])
                    
            class_loss = F.cross_entropy(
                predictions['class_logits'].reshape(-1, 100),
                target_classes.reshape(-1),
                ignore_index=0  # Ignore padding
            )
            losses['class_loss'] = class_loss
            
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses