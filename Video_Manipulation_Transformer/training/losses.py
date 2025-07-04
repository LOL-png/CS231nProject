"""
Loss Functions for Video-to-Manipulation Transformer
Includes all specialized losses for different training stages
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class GraspQualityLoss(nn.Module):
    """
    Grasp quality loss based on analytical grasp metrics
    """
    
    def __init__(self,
                 force_closure_weight: float = 1.0,
                 grasp_wrench_space_weight: float = 0.5,
                 contact_distribution_weight: float = 0.3):
        super().__init__()
        self.force_closure_weight = force_closure_weight
        self.grasp_wrench_space_weight = grasp_wrench_space_weight
        self.contact_distribution_weight = contact_distribution_weight
        
    def forward(self,
                contact_points: torch.Tensor,
                contact_normals: torch.Tensor,
                contact_forces: torch.Tensor) -> torch.Tensor:
        """
        Compute grasp quality loss
        
        Args:
            contact_points: [B, N, 3] contact point positions
            contact_normals: [B, N, 3] contact normal vectors
            contact_forces: [B, N, 3] contact force vectors
            
        Returns:
            Scalar loss value
        """
        B = contact_points.shape[0]
        device = contact_points.device
        
        total_loss = torch.zeros(1, device=device)
        
        for b in range(B):
            # Force closure metric
            fc_score = self._compute_force_closure_score(
                contact_points[b],
                contact_normals[b],
                contact_forces[b]
            )
            fc_loss = 1.0 - fc_score  # Higher score is better
            
            # Grasp wrench space volume (simplified)
            gws_score = self._compute_gws_score(
                contact_points[b],
                contact_normals[b]
            )
            gws_loss = 1.0 - gws_score
            
            # Contact distribution (prefer well-distributed contacts)
            dist_score = self._compute_distribution_score(contact_points[b])
            dist_loss = 1.0 - dist_score
            
            # Combine losses
            batch_loss = (
                self.force_closure_weight * fc_loss +
                self.grasp_wrench_space_weight * gws_loss +
                self.contact_distribution_weight * dist_loss
            )
            
            total_loss += batch_loss
            
        return total_loss / B
    
    def _compute_force_closure_score(self,
                                   points: torch.Tensor,
                                   normals: torch.Tensor,
                                   forces: torch.Tensor) -> torch.Tensor:
        """
        Compute force closure score (0 to 1)
        """
        if points.shape[0] < 3:
            return torch.tensor(0.0, device=points.device)
            
        # Check if forces can balance
        net_force = forces.sum(dim=0)
        force_balance = torch.exp(-net_force.norm())
        
        # Check if normals point inward
        center = points.mean(dim=0)
        to_center = F.normalize(center - points, dim=-1)
        inward_score = torch.clamp((normals * to_center).sum(dim=-1).mean(), 0, 1)
        
        return force_balance * inward_score
    
    def _compute_gws_score(self,
                          points: torch.Tensor,
                          normals: torch.Tensor) -> torch.Tensor:
        """
        Compute grasp wrench space score (simplified)
        """
        if points.shape[0] < 3:
            return torch.tensor(0.0, device=points.device)
            
        # Compute contact wrenches
        wrenches = []
        center = points.mean(dim=0)
        
        for i in range(points.shape[0]):
            # Force component
            force = normals[i]
            
            # Torque component (r × f)
            r = points[i] - center
            torque = torch.cross(r, force)
            
            wrench = torch.cat([force, torque])
            wrenches.append(wrench)
            
        wrenches = torch.stack(wrenches)
        
        # Simplified metric: variance of wrench directions
        wrench_variance = wrenches.std(dim=0).mean()
        score = torch.sigmoid(wrench_variance)
        
        return score
    
    def _compute_distribution_score(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute contact distribution score
        """
        if points.shape[0] < 2:
            return torch.tensor(0.0, device=points.device)
            
        # Compute pairwise distances
        dists = torch.cdist(points, points)
        
        # Exclude diagonal
        mask = ~torch.eye(points.shape[0], dtype=torch.bool, device=points.device)
        dists = dists[mask]
        
        # Prefer uniform distribution
        mean_dist = dists.mean()
        std_dist = dists.std()
        
        # Lower std relative to mean is better
        uniformity = torch.exp(-std_dist / (mean_dist + 1e-6))
        
        return uniformity


class PhysicsConsistencyLoss(nn.Module):
    """
    Loss for ensuring physical consistency in predictions
    """
    
    def __init__(self,
                 penetration_weight: float = 10.0,
                 stability_weight: float = 1.0,
                 energy_weight: float = 0.1):
        super().__init__()
        self.penetration_weight = penetration_weight
        self.stability_weight = stability_weight
        self.energy_weight = energy_weight
        
    def forward(self,
                trajectories: torch.Tensor,
                velocities: Optional[torch.Tensor] = None,
                forces: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute physics consistency loss
        """
        total_loss = 0.0
        
        # Penetration loss (objects shouldn't penetrate each other)
        if trajectories.shape[2] > 1:  # Multiple objects
            penetration_loss = self._compute_penetration_loss(trajectories)
            total_loss += self.penetration_weight * penetration_loss
            
        # Stability loss (objects should come to rest)
        if velocities is not None:
            stability_loss = self._compute_stability_loss(velocities)
            total_loss += self.stability_weight * stability_loss
            
        # Energy conservation loss
        if velocities is not None and forces is not None:
            energy_loss = self._compute_energy_loss(trajectories, velocities, forces)
            total_loss += self.energy_weight * energy_loss
            
        return total_loss
    
    def _compute_penetration_loss(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        Penalize object interpenetration
        """
        B, T, N, _ = trajectories.shape
        
        # Compute pairwise distances between objects
        positions = trajectories[..., :3]  # [B, T, N, 3]
        
        total_penetration = 0.0
        min_distance = 0.05  # Minimum allowed distance
        
        for t in range(T):
            for i in range(N):
                for j in range(i+1, N):
                    dist = (positions[:, t, i] - positions[:, t, j]).norm(dim=-1)
                    penetration = F.relu(min_distance - dist)
                    total_penetration += penetration.mean()
                    
        return total_penetration / (T * N * (N-1) / 2)
    
    def _compute_stability_loss(self, velocities: torch.Tensor) -> torch.Tensor:
        """
        Encourage objects to reach stable states
        """
        # Penalize high velocities at the end of trajectory
        final_velocities = velocities[:, -5:].norm(dim=-1)  # Last 5 timesteps
        return final_velocities.mean()
    
    def _compute_energy_loss(self,
                           positions: torch.Tensor,
                           velocities: torch.Tensor,
                           forces: torch.Tensor) -> torch.Tensor:
        """
        Simplified energy conservation check
        """
        # Compute kinetic energy change
        kinetic_start = 0.5 * (velocities[:, 0] ** 2).sum(dim=-1).mean()
        kinetic_end = 0.5 * (velocities[:, -1] ** 2).sum(dim=-1).mean()
        
        # Compute work done by forces (simplified)
        displacements = positions[:, 1:] - positions[:, :-1]
        work = (forces[:, :-1] * displacements).sum(dim=-1).mean()
        
        # Energy should be conserved (with some dissipation)
        energy_change = abs(kinetic_end - kinetic_start - work)
        
        return energy_change


class TemporalConsistencyLoss(nn.Module):
    """
    Loss for ensuring temporal consistency in predictions
    """
    
    def __init__(self,
                 smoothness_weight: float = 1.0,
                 acceleration_weight: float = 0.5):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.acceleration_weight = acceleration_weight
        
    def forward(self,
                predictions: torch.Tensor,
                timestep: float = 0.02) -> torch.Tensor:
        """
        Compute temporal consistency loss
        
        Args:
            predictions: [B, T, D] temporal predictions
            timestep: Time between frames
        """
        if predictions.shape[1] < 2:
            return torch.tensor(0.0, device=predictions.device)
            
        # Velocity (first derivative)
        velocities = (predictions[:, 1:] - predictions[:, :-1]) / timestep
        
        # Acceleration (second derivative)
        if predictions.shape[1] > 2:
            accelerations = (velocities[:, 1:] - velocities[:, :-1]) / timestep
        else:
            accelerations = None
            
        total_loss = 0.0
        
        # Smoothness loss (penalize high velocities)
        smoothness_loss = velocities.norm(dim=-1).mean()
        total_loss += self.smoothness_weight * smoothness_loss
        
        # Acceleration loss (penalize high accelerations)
        if accelerations is not None:
            accel_loss = accelerations.norm(dim=-1).mean()
            total_loss += self.acceleration_weight * accel_loss
            
        return total_loss


class ActionConsistencyLoss(nn.Module):
    """
    Loss for ensuring action predictions are consistent and achievable
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self,
                actions: Dict[str, torch.Tensor],
                robot_constraints: Optional[Dict] = None) -> torch.Tensor:
        """
        Compute action consistency loss
        """
        total_loss = 0.0
        
        # Grasp type consistency
        if 'grasp_types' in actions:
            # Encourage decisive grasp selection (not uniform distribution)
            entropy = -(actions['grasp_types'] * torch.log(actions['grasp_types'] + 1e-8)).sum(dim=-1)
            total_loss += entropy.mean() * 0.1
            
        # Approach vector consistency with grasp
        if 'approach_vectors' in actions and 'grasp_types' in actions:
            # Different grasp types should have different approach patterns
            # This is a simplified version
            approach_variance = actions['approach_vectors'].std(dim=0).mean()
            total_loss += (1.0 - approach_variance) * 0.1
            
        # Force consistency with grasp type
        if 'grasp_forces' in actions and 'grasp_types' in actions:
            # Power grasps should have higher forces
            power_grasp_prob = actions['grasp_types'][:, :3].sum(dim=-1)  # First 3 are power grasps
            force_power_consistency = F.mse_loss(actions['grasp_forces'], power_grasp_prob)
            total_loss += force_power_consistency * 0.1
            
        return total_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function with all components
    """
    
    def __init__(self, loss_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        
        if loss_weights is None:
            loss_weights = {
                'grasp_quality': 1.0,
                'physics_consistency': 0.5,
                'temporal_consistency': 0.1,
                'action_consistency': 0.2
            }
            
        self.loss_weights = loss_weights
        
        # Initialize individual losses
        self.grasp_quality_loss = GraspQualityLoss()
        self.physics_consistency_loss = PhysicsConsistencyLoss()
        self.temporal_consistency_loss = TemporalConsistencyLoss()
        self.action_consistency_loss = ActionConsistencyLoss()
        
    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                simulation_outputs: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute all losses
        """
        losses = {}
        
        # Grasp quality loss
        if simulation_outputs and 'contact_points' in simulation_outputs:
            grasp_loss = self.grasp_quality_loss(
                simulation_outputs['contact_points'],
                simulation_outputs.get('contact_normals', torch.zeros_like(simulation_outputs['contact_points'])),
                simulation_outputs['contact_forces']
            )
            losses['grasp_quality'] = grasp_loss * self.loss_weights['grasp_quality']
            
        # Physics consistency loss
        if simulation_outputs and 'object_trajectories' in simulation_outputs:
            physics_loss = self.physics_consistency_loss(
                simulation_outputs['object_trajectories'],
                simulation_outputs.get('object_velocities'),
                simulation_outputs.get('contact_forces')
            )
            losses['physics_consistency'] = physics_loss * self.loss_weights['physics_consistency']
            
        # Temporal consistency loss
        if 'ee_target_poses' in predictions:
            temporal_loss = self.temporal_consistency_loss(predictions['ee_target_poses'])
            losses['temporal_consistency'] = temporal_loss * self.loss_weights['temporal_consistency']
            
        # Action consistency loss
        action_loss = self.action_consistency_loss(predictions)
        losses['action_consistency'] = action_loss * self.loss_weights['action_consistency']
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses