"""
MLP Retargeting for Allegro Hand
Maps human hand configurations to Allegro Hand joint angles (16 DOF)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class AllegroRetargeting(nn.Module):
    """
    MLP-based retargeting from human hand to Allegro Hand
    Architecture: 2 hidden layers, 64 neurons each
    Input: Human hand configuration from decoder
    Output: 16 joint angles for Allegro Hand
    Performance: 30 Hz real-time
    """
    
    def __init__(self,
                 input_dim: int = 16,  # Simplified hand config from decoder
                 hidden_dim: int = 64,
                 output_dim: int = 16,  # Allegro hand DOF
                 use_mano_input: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        
        self.use_mano_input = use_mano_input
        
        # If using MANO parameters as input
        if use_mano_input:
            input_dim = 51  # 48 pose + 3 global rotation
            
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Output scaling layers for joint limits
        self.joint_limit_scale = nn.Parameter(torch.ones(output_dim))
        self.joint_limit_offset = nn.Parameter(torch.zeros(output_dim))
        
        # Allegro hand joint limits (in radians)
        self.register_buffer('joint_lower_limits', torch.tensor([
            -0.47, 0.0, -0.19, -0.162,    # Thumb joints
            -0.196, -0.09, -0.19, -0.162,  # Index finger
            -0.196, -0.09, -0.19, -0.162,  # Middle finger
            -0.196, -0.09, -0.19, -0.162   # Ring finger
        ]))
        
        self.register_buffer('joint_upper_limits', torch.tensor([
            0.47, 1.61, 1.709, 1.618,     # Thumb joints
            0.196, 1.8, 1.709, 1.618,     # Index finger
            0.196, 1.8, 1.709, 1.618,     # Middle finger
            0.196, 1.8, 1.709, 1.618      # Ring finger
        ]))
        
        # Finger mapping for structured control
        self.finger_indices = {
            'thumb': [0, 1, 2, 3],
            'index': [4, 5, 6, 7],
            'middle': [8, 9, 10, 11],
            'ring': [12, 13, 14, 15]
        }
        
        # Grasp-specific bias networks (optional)
        self.grasp_bias_networks = nn.ModuleDict({
            'power': nn.Linear(hidden_dim, output_dim),
            'precision': nn.Linear(hidden_dim, output_dim),
            'lateral': nn.Linear(hidden_dim, output_dim)
        })
        
    def forward(self,
                hand_config: torch.Tensor,
                grasp_type: Optional[torch.Tensor] = None,
                enforce_limits: bool = True) -> Dict[str, torch.Tensor]:
        """
        Args:
            hand_config: [B, input_dim] hand configuration from action decoder
            grasp_type: [B, num_grasp_types] optional grasp type probabilities
            enforce_limits: Whether to enforce joint limits
            
        Returns:
            Dictionary containing:
                - joint_angles: [B, 16] Allegro hand joint angles
                - joint_velocities: [B, 16] estimated joint velocities (if temporal)
                - grasp_synergies: [B, 4] per-finger activation levels
        """
        B = hand_config.shape[0]
        
        # Forward through MLP
        joint_angles_raw = self.mlp(hand_config)  # [B, 16]
        
        # Apply grasp-specific biases if grasp type provided
        if grasp_type is not None:
            # Weighted sum of grasp biases
            grasp_bias = torch.zeros_like(joint_angles_raw)
            
            # Get intermediate features for bias computation
            with torch.no_grad():
                features = self.mlp[0](hand_config)  # First layer features
                
            for grasp_name, bias_net in self.grasp_bias_networks.items():
                if grasp_name == 'power':
                    weight = grasp_type[:, 0:3].sum(dim=1)  # Sum of power grasp types
                elif grasp_name == 'precision':
                    weight = grasp_type[:, 3:6].sum(dim=1)  # Sum of precision types
                else:
                    weight = grasp_type[:, 6:].sum(dim=1)  # Other types
                    
                bias = bias_net(features) * weight.unsqueeze(-1)
                grasp_bias += bias
                
            joint_angles_raw = joint_angles_raw + 0.1 * grasp_bias  # Small bias contribution
            
        # Apply sigmoid and scale to joint limits
        joint_angles = torch.sigmoid(joint_angles_raw)  # [0, 1]
        
        # Scale to joint limits
        joint_range = self.joint_upper_limits - self.joint_lower_limits
        joint_angles = self.joint_lower_limits + joint_angles * joint_range
        
        # Apply learned scaling and offset
        joint_angles = joint_angles * self.joint_limit_scale + self.joint_limit_offset
        
        # Enforce hard limits if requested
        if enforce_limits:
            joint_angles = torch.clamp(
                joint_angles,
                self.joint_lower_limits,
                self.joint_upper_limits
            )
            
        # Compute grasp synergies (per-finger activation)
        grasp_synergies = self._compute_synergies(joint_angles)
        
        # Estimate velocities if we have temporal information
        joint_velocities = torch.zeros_like(joint_angles)  # Placeholder
        
        return {
            'joint_angles': joint_angles,
            'joint_velocities': joint_velocities,
            'grasp_synergies': grasp_synergies,
            'joint_angles_raw': joint_angles_raw  # Before limits
        }
    
    def _compute_synergies(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """
        Compute per-finger synergy values based on joint angles
        """
        B = joint_angles.shape[0]
        synergies = torch.zeros(B, 4, device=joint_angles.device)
        
        for i, (finger, indices) in enumerate(self.finger_indices.items()):
            # Compute average flexion for each finger
            finger_angles = joint_angles[:, indices]
            
            # Normalize by joint ranges
            ranges = self.joint_upper_limits[indices] - self.joint_lower_limits[indices]
            normalized = (finger_angles - self.joint_lower_limits[indices]) / ranges
            
            # Average activation
            synergies[:, i] = normalized.mean(dim=1)
            
        return synergies
    
    def compute_loss(self,
                    predictions: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute retargeting losses
        """
        losses = {}
        
        # Joint angle loss
        if 'allegro_joints_gt' in targets:
            joint_loss = F.mse_loss(
                predictions['joint_angles'],
                targets['allegro_joints_gt']
            )
            losses['joint_loss'] = joint_loss
            
        # Joint limit violation penalty
        lower_violations = F.relu(self.joint_lower_limits - predictions['joint_angles'])
        upper_violations = F.relu(predictions['joint_angles'] - self.joint_upper_limits)
        limit_loss = (lower_violations + upper_violations).mean()
        losses['limit_loss'] = limit_loss * 0.1  # Small weight
        
        # Smoothness loss (if temporal)
        if predictions['joint_angles'].shape[0] > 1:
            diff = predictions['joint_angles'][1:] - predictions['joint_angles'][:-1]
            smoothness_loss = (diff ** 2).mean()
            losses['smoothness_loss'] = smoothness_loss * 0.01
            
        # Synergy consistency loss
        if 'grasp_type' in targets:
            # Ensure synergies match expected patterns for grasp types
            expected_synergies = self._get_expected_synergies(targets['grasp_type'])
            synergy_loss = F.mse_loss(
                predictions['grasp_synergies'],
                expected_synergies
            )
            losses['synergy_loss'] = synergy_loss * 0.1
            
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses
    
    def _get_expected_synergies(self, grasp_types: torch.Tensor) -> torch.Tensor:
        """
        Get expected synergy patterns for different grasp types
        """
        # Simplified synergy patterns
        # [thumb, index, middle, ring]
        synergy_patterns = {
            'power': torch.tensor([0.8, 0.9, 0.9, 0.8]),
            'precision': torch.tensor([0.6, 0.7, 0.3, 0.2]),
            'lateral': torch.tensor([0.7, 0.5, 0.4, 0.3])
        }
        
        # Weight patterns by grasp type probabilities
        B = grasp_types.shape[0]
        expected = torch.zeros(B, 4, device=grasp_types.device)
        
        # Simplified - would need proper grasp type mapping
        expected += synergy_patterns['power'].to(grasp_types.device) * grasp_types[:, 0].unsqueeze(-1)
        
        return expected


class TemporalRetargeting(nn.Module):
    """
    Temporal version of retargeting that considers motion history
    """
    
    def __init__(self,
                 base_retargeting: AllegroRetargeting,
                 history_length: int = 5):
        super().__init__()
        
        self.base_retargeting = base_retargeting
        self.history_length = history_length
        
        # Temporal processing
        self.temporal_encoder = nn.LSTM(
            input_size=16,  # Joint angles
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        
        self.temporal_fusion = nn.Linear(32 + 16, 16)
        
    def forward(self,
                hand_configs: torch.Tensor,
                history: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process with temporal information
        """
        # Get current prediction
        current = self.base_retargeting(hand_configs)
        
        if history is not None and history.shape[1] > 0:
            # Process history
            _, (h_n, _) = self.temporal_encoder(history)
            h_n = h_n.squeeze(0)  # [B, 32]
            
            # Fuse with current
            fused = torch.cat([current['joint_angles'], h_n], dim=-1)
            refined_angles = self.temporal_fusion(fused)
            
            # Update output
            current['joint_angles'] = refined_angles
            
            # Compute velocities
            if history.shape[1] > 0:
                current['joint_velocities'] = refined_angles - history[:, -1]
                
        return current