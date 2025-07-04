"""
Action Decoder for Video-to-Manipulation Transformer
Converts fused features into high-level manipulation commands
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class ActionDecoder(nn.Module):
    """
    Decoder that generates high-level task commands from fused temporal features
    Output format:
    - ee_target_pose: [x, y, z, rx, ry, rz] - 6D end-effector pose
    - hand_config: grasp type/aperture configuration
    - approach_vector: [x, y, z] - 3D approach direction
    - grasp_force: target grasp force
    """
    
    def __init__(self,
                 input_dim: int = 1024,
                 hidden_dim: int = 512,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 num_grasp_types: int = 10,
                 num_action_steps: int = 10):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_grasp_types = num_grasp_types
        self.num_action_steps = num_action_steps
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Action queries for multi-step prediction
        self.action_queries = nn.Parameter(torch.zeros(num_action_steps, hidden_dim))
        nn.init.normal_(self.action_queries, std=0.02)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output heads
        self.ee_pose_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 6)  # 6D pose (position + rotation)
        )
        
        self.grasp_type_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_grasp_types)  # Grasp type classification
        )
        
        self.grasp_aperture_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)  # Continuous aperture value [0, 1]
        )
        
        self.approach_vector_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # 3D approach direction
        )
        
        self.grasp_force_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)  # Grasp force [0, 1]
        )
        
        self.action_confidence_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)  # Action confidence
        )
        
        # Grasp type embeddings (for known grasp patterns)
        self.grasp_embeddings = nn.Embedding(num_grasp_types, hidden_dim)
        
    def forward(self,
                fused_features: torch.Tensor,
                temporal_features: Optional[torch.Tensor] = None,
                previous_actions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            fused_features: [B, hidden_dim] aggregated features from temporal fusion
            temporal_features: [B, T, hidden_dim] optional temporal features for attention
            previous_actions: [B, num_prev_actions, action_dim] optional previous actions for autoregressive generation
            
        Returns:
            Dictionary containing:
                - ee_target_poses: [B, num_steps, 6] target end-effector poses
                - grasp_types: [B, num_steps, num_grasp_types] grasp type probabilities
                - grasp_apertures: [B, num_steps] grasp aperture values
                - approach_vectors: [B, num_steps, 3] approach directions
                - grasp_forces: [B, num_steps] target grasp forces
                - action_confidence: [B, num_steps] action confidence scores
                - action_features: [B, num_steps, hidden_dim] features for control conversion
        """
        B = fused_features.shape[0]
        device = fused_features.device
        
        # Project input features
        memory = self.input_projection(fused_features).unsqueeze(1)  # [B, 1, hidden_dim]
        
        # If temporal features provided, use them as additional memory
        if temporal_features is not None:
            temporal_memory = self.input_projection(temporal_features)  # [B, T, hidden_dim]
            memory = torch.cat([memory, temporal_memory], dim=1)  # [B, 1+T, hidden_dim]
            
        # Prepare action queries
        action_queries = self.action_queries.unsqueeze(0).expand(B, -1, -1)  # [B, num_steps, hidden_dim]
        
        # If previous actions provided, encode and prepend them
        if previous_actions is not None:
            # Encode previous actions (simplified - would need proper encoding in practice)
            prev_encoded = self.input_projection(previous_actions)  # [B, num_prev, hidden_dim]
            action_queries = torch.cat([prev_encoded, action_queries], dim=1)
            
        # Apply transformer decoder
        action_features = self.transformer_decoder(
            tgt=action_queries,
            memory=memory
        )  # [B, num_steps, hidden_dim]
        
        # Extract features for current steps only
        if previous_actions is not None:
            num_prev = previous_actions.shape[1]
            action_features = action_features[:, num_prev:]
            
        # Generate action predictions
        ee_target_poses = self.ee_pose_head(action_features)  # [B, num_steps, 6]
        
        # Normalize rotation part (last 3 dimensions)
        ee_positions = ee_target_poses[..., :3]
        ee_rotations = F.normalize(ee_target_poses[..., 3:], dim=-1)  # Unit vector for axis-angle
        ee_target_poses = torch.cat([ee_positions, ee_rotations], dim=-1)
        
        # Grasp configuration
        grasp_type_logits = self.grasp_type_head(action_features)  # [B, num_steps, num_grasp_types]
        grasp_types = F.softmax(grasp_type_logits, dim=-1)
        
        grasp_apertures = torch.sigmoid(self.grasp_aperture_head(action_features)).squeeze(-1)  # [B, num_steps]
        
        # Approach and force
        approach_vectors = self.approach_vector_head(action_features)  # [B, num_steps, 3]
        approach_vectors = F.normalize(approach_vectors, dim=-1)  # Normalize to unit vectors
        
        grasp_forces = torch.sigmoid(self.grasp_force_head(action_features)).squeeze(-1)  # [B, num_steps]
        
        # Action confidence
        action_confidence = torch.sigmoid(self.action_confidence_head(action_features)).squeeze(-1)  # [B, num_steps]
        
        # Create hand configuration from grasp type and aperture
        hand_configs = self._create_hand_configs(grasp_types, grasp_apertures, action_features)
        
        return {
            'ee_target_poses': ee_target_poses,
            'grasp_types': grasp_types,
            'grasp_apertures': grasp_apertures,
            'hand_configs': hand_configs,
            'approach_vectors': approach_vectors,
            'grasp_forces': grasp_forces,
            'action_confidence': action_confidence,
            'action_features': action_features
        }
    
    def _create_hand_configs(self,
                           grasp_types: torch.Tensor,
                           apertures: torch.Tensor,
                           features: torch.Tensor) -> torch.Tensor:
        """
        Create hand configuration from grasp type and aperture
        """
        B, num_steps, _ = grasp_types.shape
        
        # Get grasp embeddings weighted by probabilities
        grasp_embeds = self.grasp_embeddings.weight  # [num_grasp_types, hidden_dim]
        weighted_embeds = torch.matmul(grasp_types, grasp_embeds)  # [B, num_steps, hidden_dim]
        
        # Combine with aperture information
        aperture_expanded = apertures.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        hand_features = weighted_embeds * aperture_expanded + features * (1 - aperture_expanded)
        
        # Project to hand configuration space (simplified)
        hand_configs = torch.tanh(hand_features[..., :16])  # Use first 16 dims as joint angles
        
        return hand_configs
    
    def compute_loss(self,
                    predictions: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute action decoder losses
        """
        losses = {}
        
        # End-effector pose loss
        if 'ee_poses_gt' in targets:
            # Position loss (L2)
            position_loss = F.mse_loss(
                predictions['ee_target_poses'][..., :3],
                targets['ee_poses_gt'][..., :3]
            )
            losses['position_loss'] = position_loss
            
            # Rotation loss (geodesic distance)
            pred_rot = predictions['ee_target_poses'][..., 3:]
            target_rot = targets['ee_poses_gt'][..., 3:]
            rotation_loss = 1.0 - F.cosine_similarity(pred_rot, target_rot, dim=-1).mean()
            losses['rotation_loss'] = rotation_loss
            
        # Grasp type loss
        if 'grasp_types_gt' in targets:
            grasp_loss = F.cross_entropy(
                predictions['grasp_types'].reshape(-1, self.num_grasp_types),
                targets['grasp_types_gt'].reshape(-1)
            )
            losses['grasp_type_loss'] = grasp_loss
            
        # Approach vector loss
        if 'approach_vectors_gt' in targets:
            approach_loss = 1.0 - F.cosine_similarity(
                predictions['approach_vectors'].reshape(-1, 3),
                targets['approach_vectors_gt'].reshape(-1, 3),
                dim=-1
            ).mean()
            losses['approach_loss'] = approach_loss
            
        # Force loss
        if 'grasp_forces_gt' in targets:
            force_loss = F.mse_loss(
                predictions['grasp_forces'],
                targets['grasp_forces_gt']
            )
            losses['force_loss'] = force_loss
            
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses


class ActionSequenceGenerator(nn.Module):
    """
    Generates smooth action sequences with temporal consistency
    """
    
    def __init__(self, hidden_dim: int = 512, window_size: int = 5):
        super().__init__()
        
        self.window_size = window_size
        
        # Temporal smoothing
        self.temporal_smooth = nn.Conv1d(
            in_channels=6,  # 6D pose
            out_channels=6,
            kernel_size=window_size,
            padding=window_size // 2,
            groups=6  # Separate convolution per dimension
        )
        
        # Velocity predictor
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 6)  # 6D velocity
        )
        
    def forward(self,
                action_features: torch.Tensor,
                raw_poses: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate smooth action sequences
        """
        B, T, _ = raw_poses.shape
        
        # Apply temporal smoothing
        poses_transposed = raw_poses.transpose(1, 2)  # [B, 6, T]
        smooth_poses = self.temporal_smooth(poses_transposed)
        smooth_poses = smooth_poses.transpose(1, 2)  # [B, T, 6]
        
        # Predict velocities
        velocities = self.velocity_head(action_features)  # [B, T, 6]
        
        # Compute accelerations (finite differences)
        accelerations = torch.zeros_like(velocities)
        if T > 1:
            accelerations[:, 1:] = velocities[:, 1:] - velocities[:, :-1]
            
        return {
            'smooth_poses': smooth_poses,
            'velocities': velocities,
            'accelerations': accelerations
        }