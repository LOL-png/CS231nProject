"""
Differentiable Physics Simulation with MuJoCo XLA
Simulates robot manipulation and provides gradients for training
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import os

# Note: In practice, you would use mujoco_mjx for differentiable simulation
# This is a simplified implementation showing the interface
try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("Warning: MuJoCo not available. Using mock simulation.")


class DifferentiableSimulator:
    """
    Differentiable physics simulation for robot manipulation
    Uses MuJoCo XLA (mjx) for GPU-accelerated differentiable simulation
    """
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 device: str = 'cuda',
                 sim_timestep: float = 0.002,
                 control_timestep: float = 0.02):
        """
        Args:
            model_path: Path to MuJoCo XML model file
            device: Device for simulation
            sim_timestep: Physics simulation timestep
            control_timestep: Control loop timestep
        """
        self.device = device
        self.sim_timestep = sim_timestep
        self.control_timestep = control_timestep
        self.substeps = int(control_timestep / sim_timestep)
        
        if MUJOCO_AVAILABLE and model_path and os.path.exists(model_path):
            # Load MuJoCo model
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
        else:
            # Mock model for testing
            self.model = None
            self.data = None
            print("Using mock physics simulation")
            
        # Robot configuration
        self.num_ur_joints = 6
        self.num_allegro_joints = 16
        self.total_joints = self.num_ur_joints + self.num_allegro_joints
        
    def forward(self,
                joint_commands: torch.Tensor,
                initial_state: Optional[Dict[str, torch.Tensor]] = None,
                object_poses: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Run differentiable simulation forward
        
        Args:
            joint_commands: [B, T, 22] joint position commands (6 UR + 16 Allegro)
            initial_state: Optional initial simulation state
            object_poses: [B, num_objects, 7] initial object poses (pos + quat)
            
        Returns:
            Dictionary containing:
                - object_trajectories: [B, T, num_objects, 7] object poses over time
                - contact_forces: [B, T, num_contacts, 3] contact force vectors
                - joint_positions: [B, T, 22] actual joint positions
                - joint_torques: [B, T, 22] applied joint torques
                - grasp_success: [B] binary grasp success indicators
                - simulation_metrics: Dict of additional metrics
        """
        B, T, _ = joint_commands.shape
        device = joint_commands.device
        
        # Initialize outputs
        outputs = {
            'object_trajectories': torch.zeros(B, T, 1, 7, device=device),  # 1 object for now
            'contact_forces': torch.zeros(B, T, 10, 3, device=device),  # Max 10 contacts
            'joint_positions': torch.zeros(B, T, self.total_joints, device=device),
            'joint_torques': torch.zeros(B, T, self.total_joints, device=device),
            'grasp_success': torch.zeros(B, device=device),
            'simulation_metrics': {}
        }
        
        # Simulate each sample in batch
        for b in range(B):
            # Initialize simulation state
            if initial_state is not None:
                self._set_state(initial_state, b)
            else:
                self._reset_simulation()
                
            # Set initial object poses
            if object_poses is not None:
                self._set_object_poses(object_poses[b])
                
            # Run simulation
            for t in range(T):
                # Set joint targets
                target_pos = joint_commands[b, t].cpu().numpy() if joint_commands.is_cuda else joint_commands[b, t].numpy()
                
                if MUJOCO_AVAILABLE and self.model is not None:
                    # Set control targets
                    self.data.ctrl[:self.total_joints] = target_pos
                    
                    # Step simulation
                    for _ in range(self.substeps):
                        mujoco.mj_step(self.model, self.data)
                        
                    # Extract state
                    outputs['joint_positions'][b, t] = torch.tensor(
                        self.data.qpos[:self.total_joints], 
                        device=device
                    )
                    outputs['joint_torques'][b, t] = torch.tensor(
                        self.data.qfrc_actuator[:self.total_joints],
                        device=device
                    )
                    
                    # Get object pose (simplified - assumes single object)
                    obj_pos = self.data.xpos[self.model.nbody - 1]  # Last body
                    obj_quat = self.data.xquat[self.model.nbody - 1]
                    outputs['object_trajectories'][b, t, 0, :3] = torch.tensor(obj_pos, device=device)
                    outputs['object_trajectories'][b, t, 0, 3:] = torch.tensor(obj_quat, device=device)
                    
                    # Extract contact forces
                    contacts = self._extract_contacts()
                    if contacts['forces'].shape[0] > 0:
                        num_contacts = min(contacts['forces'].shape[0], 10)
                        outputs['contact_forces'][b, t, :num_contacts] = contacts['forces'][:num_contacts].to(device)
                else:
                    # Mock simulation
                    outputs['joint_positions'][b, t] = target_pos
                    outputs['joint_torques'][b, t] = torch.randn_like(target_pos) * 0.1
                    
                    # Simple object dynamics
                    if t > 0:
                        outputs['object_trajectories'][b, t] = outputs['object_trajectories'][b, t-1]
                        # Add small motion
                        outputs['object_trajectories'][b, t, 0, :3] += torch.randn(3, device=device) * 0.001
                    
            # Evaluate grasp success
            outputs['grasp_success'][b] = self._evaluate_grasp_success(
                outputs['object_trajectories'][b],
                outputs['contact_forces'][b]
            )
            
        # Compute additional metrics
        outputs['simulation_metrics'] = self._compute_metrics(outputs)
        
        return outputs
    
    def _reset_simulation(self):
        """Reset simulation to initial state"""
        if MUJOCO_AVAILABLE and self.model is not None:
            mujoco.mj_resetData(self.model, self.data)
            
    def _set_state(self, state: Dict[str, torch.Tensor], batch_idx: int):
        """Set simulation state from dictionary"""
        if MUJOCO_AVAILABLE and self.model is not None:
            if 'qpos' in state:
                self.data.qpos[:] = state['qpos'][batch_idx].cpu().numpy()
            if 'qvel' in state:
                self.data.qvel[:] = state['qvel'][batch_idx].cpu().numpy()
                
    def _set_object_poses(self, poses: torch.Tensor):
        """Set object poses in simulation"""
        if MUJOCO_AVAILABLE and self.model is not None:
            # Simplified - would need proper object indexing
            for i, pose in enumerate(poses):
                if i < self.model.nbody - self.total_joints:
                    body_id = self.model.nbody - 1 - i
                    # Set position (would need proper coordinate transform)
                    # self.data.xpos[body_id] = pose[:3].cpu().numpy()
                    # self.data.xquat[body_id] = pose[3:].cpu().numpy()
                    
    def _extract_contacts(self) -> Dict[str, torch.Tensor]:
        """Extract contact information from simulation"""
        contacts = {
            'forces': torch.zeros(0, 3),
            'positions': torch.zeros(0, 3),
            'normals': torch.zeros(0, 3)
        }
        
        if MUJOCO_AVAILABLE and self.model is not None:
            # Extract active contacts
            num_contacts = 0
            force_list = []
            
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                if contact.geom1 >= 0 and contact.geom2 >= 0:
                    # Simple force approximation
                    force = np.zeros(3)
                    mujoco.mj_contactForce(self.model, self.data, i, force)
                    force_list.append(torch.tensor(force))
                    num_contacts += 1
                    
            if num_contacts > 0:
                contacts['forces'] = torch.stack(force_list)
                
        return contacts
    
    def _evaluate_grasp_success(self,
                               object_trajectory: torch.Tensor,
                               contact_forces: torch.Tensor) -> torch.Tensor:
        """
        Evaluate if grasp was successful
        """
        # Check if object was lifted
        initial_height = object_trajectory[0, 0, 2]
        final_height = object_trajectory[-1, 0, 2]
        lifted = (final_height - initial_height) > 0.05  # 5cm threshold
        
        # Check if contact was maintained
        avg_contact_force = contact_forces.sum(dim=-1).mean(dim=-1)  # Average over time and contacts
        contact_maintained = avg_contact_force.mean() > 0.1  # Some force threshold
        
        # Check if object remained stable (low velocity at end)
        if object_trajectory.shape[0] > 1:
            final_velocity = object_trajectory[-1, 0, :3] - object_trajectory[-2, 0, :3]
            stable = final_velocity.norm() < 0.01
        else:
            stable = True
            
        success = lifted and contact_maintained and stable
        return success.float()
    
    def _compute_metrics(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute additional simulation metrics
        """
        metrics = {}
        
        # Object displacement
        object_displacement = (
            outputs['object_trajectories'][:, -1, :, :3] - 
            outputs['object_trajectories'][:, 0, :, :3]
        ).norm(dim=-1).mean()
        metrics['object_displacement'] = object_displacement
        
        # Average contact force
        avg_force = outputs['contact_forces'].norm(dim=-1).mean()
        metrics['avg_contact_force'] = avg_force
        
        # Joint tracking error
        if 'joint_commands' in outputs:
            tracking_error = (
                outputs['joint_positions'] - outputs['joint_commands']
            ).abs().mean()
            metrics['tracking_error'] = tracking_error
            
        # Grasp success rate
        metrics['grasp_success_rate'] = outputs['grasp_success'].mean()
        
        return metrics
    
    def compute_loss(self,
                    outputs: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute physics-based losses
        """
        losses = {}
        
        # Grasp success loss
        if 'grasp_success_gt' in targets:
            success_loss = F.binary_cross_entropy(
                outputs['grasp_success'],
                targets['grasp_success_gt'].float()
            )
            losses['grasp_success_loss'] = success_loss
            
        # Object displacement loss
        if 'target_object_pose' in targets:
            final_poses = outputs['object_trajectories'][:, -1, :, :3]
            displacement_loss = F.mse_loss(
                final_poses,
                targets['target_object_pose'][:, :, :3]
            )
            losses['displacement_loss'] = displacement_loss
            
        # Force regulation loss
        max_force = 10.0  # Maximum allowed force
        force_magnitudes = outputs['contact_forces'].norm(dim=-1)
        force_penalty = F.relu(force_magnitudes - max_force).mean()
        losses['force_penalty'] = force_penalty * 0.01
        
        # Smoothness loss
        if outputs['joint_positions'].shape[1] > 1:
            joint_velocities = outputs['joint_positions'][:, 1:] - outputs['joint_positions'][:, :-1]
            joint_accelerations = joint_velocities[:, 1:] - joint_velocities[:, :-1]
            smoothness_loss = (joint_accelerations ** 2).mean()
            losses['smoothness_loss'] = smoothness_loss * 0.001
            
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses


class GraspQualityMetrics:
    """
    Compute grasp quality metrics for evaluation
    """
    
    @staticmethod
    def compute_grasp_wrench_space(contact_points: torch.Tensor,
                                  contact_normals: torch.Tensor,
                                  friction_coeff: float = 0.5) -> torch.Tensor:
        """
        Compute grasp wrench space (simplified)
        """
        # This would compute the convex hull of contact wrenches
        # For now, return a simple metric
        num_contacts = contact_points.shape[0]
        if num_contacts < 3:
            return torch.tensor(0.0)
            
        # Simplified quality based on number of contacts and spread
        spread = contact_points.std(dim=0).mean()
        quality = min(1.0, num_contacts / 5.0) * spread
        
        return quality
    
    @staticmethod
    def compute_force_closure(contact_points: torch.Tensor,
                            contact_normals: torch.Tensor) -> bool:
        """
        Check if grasp achieves force closure
        """
        # Simplified check - proper implementation would compute
        # whether contact wrenches can resist arbitrary external wrenches
        num_contacts = contact_points.shape[0]
        
        if num_contacts < 3:
            return False
            
        # Check if normals point roughly towards center
        center = contact_points.mean(dim=0)
        to_center = center - contact_points
        to_center = F.normalize(to_center, dim=-1)
        
        dot_products = (contact_normals * to_center).sum(dim=-1)
        pointing_inward = (dot_products > 0.5).sum() >= num_contacts * 0.7
        
        return pointing_inward