import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np
from einops import rearrange, repeat
import logging

logger = logging.getLogger(__name__)

class MultiCoordinateHandEncoder(nn.Module):
    """
    HORT-style hand encoder using multiple coordinate systems
    This provides much richer geometric features than simple joint positions
    """
    
    def __init__(
        self,
        input_dim: int = 778 * 67,  # 778 vertices * (22 coords * 3 + 1 index)
        hidden_dim: int = 1024,
        num_layers: int = 5,
        dropout: float = 0.1,
        use_mano_vertices: bool = True,
        vertex_subset: Optional[int] = None  # Use subset of vertices for efficiency
    ):
        super().__init__()
        self.use_mano_vertices = use_mano_vertices
        self.hidden_dim = hidden_dim
        self.vertex_subset = vertex_subset
        
        if use_mano_vertices:
            # PointNet-style encoder for vertices
            vertex_feat_dim = 22 * 3 + 1  # 22 coordinate systems + vertex index
            
            self.vertex_encoder = nn.Sequential(
                nn.Linear(vertex_feat_dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(dropout),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(dropout),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(dropout),
                nn.Linear(512, hidden_dim)
            )
            
            # Attention-based global pooling
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            
            # Global feature refinement
            self.global_refiner = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # Joint-based encoder (fallback or additional)
        # Can accept either joint coordinates (21*3) or image features (hidden_dim)
        self.joint_encoder_from_coords = nn.Sequential(
            nn.Linear(21 * 3, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Image feature encoder (when no joints available)
        self.joint_encoder_from_image = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output heads with better initialization
        self.joint_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 21 * 3)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 21)
        )
        
        self.shape_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 10)
        )
        
        # Diversity-promoting components
        self.diversity_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )
        
        # Initialize output heads properly
        self._init_output_heads()
    
    def _init_output_heads(self):
        """Initialize output heads for stable training"""
        # Initialize joint prediction near zero (assume normalized coordinates)
        nn.init.normal_(self.joint_head[-1].weight, mean=0, std=0.01)
        nn.init.zeros_(self.joint_head[-1].bias)
        
        # Initialize confidence to high values
        nn.init.zeros_(self.confidence_head[-1].weight)
        nn.init.constant_(self.confidence_head[-1].bias, 2.0)  # Sigmoid(2) ≈ 0.88
        
        # Initialize shape parameters near zero
        nn.init.normal_(self.shape_head[-1].weight, mean=0, std=0.01)
        nn.init.zeros_(self.shape_head[-1].bias)
    
    def get_coordinate_frames(self, joints: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get 22 coordinate frames from hand joints
        Returns: 
            - origins: [B, 22, 3] frame origins
            - rotations: [B, 22, 3, 3] frame rotations
        """
        B = joints.shape[0]
        device = joints.device
        
        origins = torch.zeros(B, 22, 3, device=device, dtype=joints.dtype)
        rotations = torch.zeros(B, 22, 3, 3, device=device, dtype=joints.dtype)
        
        # 16 joint frames (first 16 joints)
        origins[:, :16] = joints[:, :16]
        
        # 5 fingertip frames (joints 4, 8, 12, 16, 20 in 21-joint format)
        fingertip_indices = [4, 8, 12, 16, 20]
        for i, idx in enumerate(fingertip_indices):
            origins[:, 16 + i] = joints[:, idx]
        
        # 1 palm frame (center of palm)
        palm_indices = [0, 1, 5, 9, 13, 17]  # Wrist and finger bases
        origins[:, 21] = joints[:, palm_indices].mean(dim=1)
        
        # Compute orientations based on hand structure
        # For now, using coordinate system aligned with bone directions
        for i in range(22):
            if i < 21:  # Joint frames
                # Simple heuristic: z-axis points to next joint in kinematic chain
                if i % 4 != 0 and i < 20:  # Not a fingertip
                    z_axis = joints[:, i + 1] - joints[:, i]
                else:
                    z_axis = joints[:, i] - joints[:, max(0, i - 1)]
                
                z_axis = F.normalize(z_axis, dim=-1)
                
                # x-axis perpendicular to z and global y
                y_global = torch.tensor([0, 1, 0], device=device, dtype=z_axis.dtype).expand_as(z_axis)
                x_axis = torch.cross(y_global, z_axis, dim=-1)
                x_axis = F.normalize(x_axis, dim=-1)
                
                # y-axis completes the frame
                y_axis = torch.cross(z_axis, x_axis, dim=-1)
                
                rotations[:, i, :, 0] = x_axis
                rotations[:, i, :, 1] = y_axis
                rotations[:, i, :, 2] = z_axis
            else:  # Palm frame
                # Use average orientation of finger base frames
                rotations[:, 21] = rotations[:, [1, 5, 9, 13, 17]].mean(dim=1)
                # Re-orthogonalize
                rotations[:, 21] = self._orthogonalize_rotation(rotations[:, 21])
        
        return origins, rotations
    
    def _orthogonalize_rotation(self, R: torch.Tensor) -> torch.Tensor:
        """Orthogonalize rotation matrix using SVD"""
        U, _, V = torch.svd(R)
        return torch.matmul(U, V.transpose(-1, -2))
    
    def transform_vertices_to_frames(
        self, 
        vertices: torch.Tensor, 
        origins: torch.Tensor,
        rotations: torch.Tensor
    ) -> torch.Tensor:
        """
        Transform vertices to multiple coordinate frames
        Args:
            vertices: [B, V, 3] vertex positions
            origins: [B, F, 3] frame origins
            rotations: [B, F, 3, 3] frame rotations
        Returns:
            [B, V, F, 3] vertices in each frame
        """
        B, V, _ = vertices.shape
        F = origins.shape[1]
        
        # Expand dimensions for broadcasting
        vertices_exp = vertices.unsqueeze(2)  # [B, V, 1, 3]
        origins_exp = origins.unsqueeze(1)    # [B, 1, F, 3]
        rotations_exp = rotations.unsqueeze(1) # [B, 1, F, 3, 3]
        
        # Translate to frame origins
        vertices_centered = vertices_exp - origins_exp  # [B, V, F, 3]
        
        # Rotate by inverse frame rotation (world to local)
        # R^T * v for each vertex and frame
        vertices_local = torch.matmul(
            rotations_exp.transpose(-1, -2),  # [B, 1, F, 3, 3]
            vertices_centered.unsqueeze(-1)   # [B, V, F, 3, 1]
        ).squeeze(-1)  # [B, V, F, 3]
        
        return vertices_local
    
    def forward(
        self, 
        image_features: Dict[str, torch.Tensor],
        hand_joints: Optional[torch.Tensor] = None,
        mano_vertices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional MANO vertices
        """
        B = image_features['cls_token'].shape[0]
        device = image_features['cls_token'].device
        
        if self.use_mano_vertices and mano_vertices is not None:
            # Subset vertices for efficiency if specified
            if self.vertex_subset is not None and self.vertex_subset < mano_vertices.shape[1]:
                # Randomly sample vertices (or use fixed subset)
                indices = torch.randperm(mano_vertices.shape[1])[:self.vertex_subset]
                mano_vertices_subset = mano_vertices[:, indices]
            else:
                mano_vertices_subset = mano_vertices
            
            V = mano_vertices_subset.shape[1]
            
            # Get coordinate frames from joints
            if hand_joints is None:
                # Estimate joints from vertices (use predefined mapping)
                hand_joints = self._estimate_joints_from_vertices(mano_vertices_subset)
            
            origins, rotations = self.get_coordinate_frames(hand_joints)
            
            # Transform vertices to all frames
            transformed_verts = self.transform_vertices_to_frames(
                mano_vertices_subset, origins, rotations
            )  # [B, V, 22, 3]
            
            # Flatten coordinate dimensions
            transformed_verts = rearrange(transformed_verts, 'b v f c -> b v (f c)')
            
            # Add vertex indices
            vertex_indices = torch.arange(V, device=device)
            vertex_indices = vertex_indices.unsqueeze(0).unsqueeze(-1).expand(B, V, 1)
            
            # Concatenate features
            vertex_features = torch.cat([transformed_verts, vertex_indices], dim=-1)
            
            # Process each vertex through encoder
            # Reshape for batch norm
            vertex_features = rearrange(vertex_features, 'b v f -> (b v) f')
            encoded_verts = self.vertex_encoder(vertex_features)  # [(B*V), hidden]
            encoded_verts = rearrange(encoded_verts, '(b v) h -> b v h', b=B, v=V)
            
            # Attention-based pooling
            attention_scores = self.attention_pool(encoded_verts)  # [B, V, 1]
            attention_weights = F.softmax(attention_scores, dim=1)
            
            # Weighted sum
            global_features = torch.sum(encoded_verts * attention_weights, dim=1)  # [B, hidden]
            
            # Refine global features
            features = self.global_refiner(global_features)
        else:
            # Fallback to joint encoding or image features
            if hand_joints is not None:
                joints_flat = hand_joints.reshape(B, -1)
                features = self.joint_encoder_from_coords(joints_flat)
            else:
                # Use image features as input
                image_feats = image_features['cls_token']
                features = self.joint_encoder_from_image(image_feats)
        
        # Add diversity-promoting features
        diversity_features = self.diversity_proj(features)
        features = features + 0.1 * diversity_features
        
        # Add residual from image features
        features = features + 0.1 * image_features['cls_token']
        
        # Predict outputs
        joints_pred = self.joint_head(features).reshape(B, 21, 3)
        confidence = torch.sigmoid(self.confidence_head(features))
        shape_params = self.shape_head(features)
        
        return {
            'joints_3d': joints_pred,
            'confidence': confidence,
            'shape_params': shape_params,
            'features': features
        }
    
    def _estimate_joints_from_vertices(self, vertices: torch.Tensor) -> torch.Tensor:
        """Estimate joint positions from vertices using predefined mapping"""
        # Simplified version - in practice, use MANO joint regressor
        B = vertices.shape[0]
        device = vertices.device
        
        # Use center of mass as a simple estimate
        joints = torch.zeros(B, 21, 3, device=device, dtype=vertices.dtype)
        
        # Wrist (joint 0) - center of first few vertices
        if vertices.shape[1] > 10:
            joints[:, 0] = vertices[:, :10].mean(dim=1)
        else:
            joints[:, 0] = vertices.mean(dim=1)
        
        # Other joints - distributed along the vertices
        # This is a placeholder - use proper MANO joint regressor
        for i in range(1, 21):
            vertex_idx = int(i * vertices.shape[1] / 21)
            if vertex_idx < vertices.shape[1]:
                joints[:, i] = vertices[:, vertex_idx]
            else:
                joints[:, i] = vertices[:, -1]  # Use last vertex
        
        return joints