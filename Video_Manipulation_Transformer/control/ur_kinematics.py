"""
Analytical Inverse Kinematics for Universal Robot (UR) Arms
Closed-form solution for UR geometry
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class URKinematics:
    """
    Analytical IK solver for UR robot arms
    Method: Closed-form solution for UR geometry
    Input: End-effector pose from decoder
    Output: 6 joint angles for UR arm
    Performance: <1ms computation
    """
    
    def __init__(self, robot_model: str = 'ur5'):
        """
        Initialize UR kinematics solver
        
        Args:
            robot_model: UR robot model ('ur3', 'ur5', 'ur10')
        """
        self.robot_model = robot_model
        
        # DH parameters for different UR models
        self.dh_params = {
            'ur3': {
                'a': [0.0, -0.24365, -0.21325, 0.0, 0.0, 0.0],
                'd': [0.1519, 0.0, 0.0, 0.11235, 0.08535, 0.0819],
                'alpha': [np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0]
            },
            'ur5': {
                'a': [0.0, -0.425, -0.39225, 0.0, 0.0, 0.0],
                'd': [0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823],
                'alpha': [np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0]
            },
            'ur10': {
                'a': [0.0, -0.612, -0.5723, 0.0, 0.0, 0.0],
                'd': [0.1273, 0.0, 0.0, 0.163941, 0.1157, 0.0922],
                'alpha': [np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0]
            }
        }
        
        # Get parameters for this model
        params = self.dh_params[robot_model]
        self.a = torch.tensor(params['a'], dtype=torch.float32)
        self.d = torch.tensor(params['d'], dtype=torch.float32)
        self.alpha = torch.tensor(params['alpha'], dtype=torch.float32)
        
        # Joint limits (radians)
        self.joint_limits = torch.tensor([
            [-2*np.pi, 2*np.pi],  # Joint 1
            [-2*np.pi, 2*np.pi],  # Joint 2
            [-2*np.pi, 2*np.pi],  # Joint 3
            [-2*np.pi, 2*np.pi],  # Joint 4
            [-2*np.pi, 2*np.pi],  # Joint 5
            [-2*np.pi, 2*np.pi]   # Joint 6
        ], dtype=torch.float32)
        
    def forward_kinematics(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """
        Compute forward kinematics
        
        Args:
            joint_angles: [B, 6] joint angles in radians
            
        Returns:
            pose: [B, 4, 4] homogeneous transformation matrix
        """
        B = joint_angles.shape[0]
        device = joint_angles.device
        
        # Move parameters to device
        a = self.a.to(device)
        d = self.d.to(device)
        alpha = self.alpha.to(device)
        
        # Initialize transformation
        T = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1)
        
        # Apply DH transformations
        for i in range(6):
            theta = joint_angles[:, i]
            
            # DH transformation matrix
            T_i = self._dh_transform(theta, d[i], a[i], alpha[i])
            T = torch.matmul(T, T_i)
            
        return T
    
    def inverse_kinematics(self,
                          target_pose: torch.Tensor,
                          elbow_preference: str = 'up',
                          wrist_preference: str = 'no_flip') -> Dict[str, torch.Tensor]:
        """
        Analytical inverse kinematics for UR robots
        
        Args:
            target_pose: [B, 4, 4] or [B, 6] target end-effector pose
                        If [B, 6]: [x, y, z, rx, ry, rz] (position + axis-angle)
            elbow_preference: 'up' or 'down'
            wrist_preference: 'flip' or 'no_flip'
            
        Returns:
            Dictionary containing:
                - joint_angles: [B, 6] joint angles
                - success: [B] boolean success flags
                - num_solutions: [B] number of valid solutions found
        """
        device = target_pose.device
        B = target_pose.shape[0]
        
        # Convert pose to homogeneous matrix if needed
        if target_pose.shape[-1] == 6:
            T_target = self._pose_to_matrix(target_pose)
        else:
            T_target = target_pose
            
        # Move parameters to device
        a = self.a.to(device)
        d = self.d.to(device)
        
        # Extract position and rotation
        p = T_target[:, :3, 3]  # [B, 3]
        R = T_target[:, :3, :3]  # [B, 3, 3]
        
        # Allocate output
        joint_angles = torch.zeros(B, 6, device=device)
        success = torch.ones(B, dtype=torch.bool, device=device)
        
        # Solve for each sample in batch
        for b in range(B):
            try:
                # Get wrist center position
                p_wrist = p[b] - d[5] * R[b, :, 2]
                
                # Solve for first 3 joints (position)
                theta1, theta2, theta3 = self._solve_position(
                    p_wrist, a, d, elbow_preference
                )
                
                # Solve for last 3 joints (orientation)
                R_03 = self._compute_r03(theta1, theta2, theta3, a, d)
                R_36 = torch.matmul(R_03.T, R[b])
                
                theta4, theta5, theta6 = self._solve_orientation(
                    R_36, wrist_preference
                )
                
                # Store solution
                joint_angles[b] = torch.tensor([
                    theta1, theta2, theta3, theta4, theta5, theta6
                ], device=device)
                
            except Exception as e:
                # IK failed for this sample
                success[b] = False
                joint_angles[b] = torch.zeros(6, device=device)
                
        # Wrap angles to [-pi, pi]
        joint_angles = self._wrap_angles(joint_angles)
        
        # Check joint limits
        within_limits = self._check_joint_limits(joint_angles)
        success = success & within_limits.all(dim=1)
        
        return {
            'joint_angles': joint_angles,
            'success': success,
            'num_solutions': success.sum(),
            'within_limits': within_limits
        }
    
    def _dh_transform(self, theta: torch.Tensor, d: float, a: float, alpha: float) -> torch.Tensor:
        """
        Compute DH transformation matrix
        """
        B = theta.shape[0]
        device = theta.device
        
        ct = torch.cos(theta)
        st = torch.sin(theta)
        ca = math.cos(alpha)
        sa = math.sin(alpha)
        
        T = torch.zeros(B, 4, 4, device=device)
        
        T[:, 0, 0] = ct
        T[:, 0, 1] = -st * ca
        T[:, 0, 2] = st * sa
        T[:, 0, 3] = a * ct
        
        T[:, 1, 0] = st
        T[:, 1, 1] = ct * ca
        T[:, 1, 2] = -ct * sa
        T[:, 1, 3] = a * st
        
        T[:, 2, 0] = 0
        T[:, 2, 1] = sa
        T[:, 2, 2] = ca
        T[:, 2, 3] = d
        
        T[:, 3, 3] = 1
        
        return T
    
    def _pose_to_matrix(self, pose: torch.Tensor) -> torch.Tensor:
        """
        Convert 6D pose to homogeneous matrix
        """
        B = pose.shape[0]
        device = pose.device
        
        position = pose[:, :3]
        axis_angle = pose[:, 3:]
        
        # Convert axis-angle to rotation matrix
        angle = torch.norm(axis_angle, dim=1, keepdim=True)
        axis = axis_angle / (angle + 1e-8)
        
        K = torch.zeros(B, 3, 3, device=device)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]
        
        I = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
        R = I + torch.sin(angle).unsqueeze(-1) * K + (1 - torch.cos(angle)).unsqueeze(-1) * torch.matmul(K, K)
        
        # Build homogeneous matrix
        T = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1).clone()
        T[:, :3, :3] = R
        T[:, :3, 3] = position
        
        return T
    
    def _solve_position(self, p_wrist: torch.Tensor, a: torch.Tensor, d: torch.Tensor, 
                       elbow_preference: str) -> Tuple[float, float, float]:
        """
        Solve for first 3 joints using geometric approach
        """
        # Simplified implementation - full version would handle all cases
        x, y, z = p_wrist
        
        # Joint 1
        theta1 = torch.atan2(y, x)
        
        # Distance in xy plane
        r = torch.sqrt(x**2 + y**2)
        
        # Joint 3 (elbow)
        D = (r**2 + (z - d[0])**2 - a[1]**2 - a[2]**2) / (2 * a[1] * a[2])
        
        if abs(D) > 1:
            raise ValueError("Target out of reach")
            
        if elbow_preference == 'up':
            theta3 = torch.atan2(torch.sqrt(1 - D**2), D)
        else:
            theta3 = torch.atan2(-torch.sqrt(1 - D**2), D)
            
        # Joint 2 (shoulder)
        k1 = a[1] + a[2] * torch.cos(theta3)
        k2 = a[2] * torch.sin(theta3)
        
        theta2 = torch.atan2(z - d[0], r) - torch.atan2(k2, k1)
        
        return theta1.item(), theta2.item(), theta3.item()
    
    def _compute_r03(self, theta1: float, theta2: float, theta3: float,
                     a: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """
        Compute rotation matrix from base to joint 3
        """
        # Simplified - would compute full transformation
        device = a.device
        
        c1, s1 = math.cos(theta1), math.sin(theta1)
        c2, s2 = math.cos(theta2), math.sin(theta2)
        c23 = math.cos(theta2 + theta3)
        s23 = math.sin(theta2 + theta3)
        
        R = torch.tensor([
            [c1*c23, -c1*s23, s1],
            [s1*c23, -s1*s23, -c1],
            [s23, c23, 0]
        ], device=device)
        
        return R
    
    def _solve_orientation(self, R_36: torch.Tensor, wrist_preference: str) -> Tuple[float, float, float]:
        """
        Solve for last 3 joints from rotation matrix
        """
        # Extract elements
        r11, r12, r13 = R_36[0, :]
        r21, r22, r23 = R_36[1, :]
        r31, r32, r33 = R_36[2, :]
        
        # Joint 5
        theta5 = torch.atan2(torch.sqrt(r13**2 + r23**2), r33)
        
        # Check for singularity
        if abs(theta5) < 1e-6:
            # Wrist singularity
            theta4 = 0
            theta6 = torch.atan2(-r12, r11)
        else:
            # Joint 4 and 6
            theta4 = torch.atan2(r23, r13)
            theta6 = torch.atan2(-r32, r31)
            
            if wrist_preference == 'flip':
                theta4 += math.pi
                theta5 = -theta5
                theta6 += math.pi
                
        return theta4.item(), theta5.item(), theta6.item()
    
    def _wrap_angles(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Wrap angles to [-pi, pi]
        """
        return torch.atan2(torch.sin(angles), torch.cos(angles))
    
    def _check_joint_limits(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """
        Check if joints are within limits
        """
        lower_ok = joint_angles >= self.joint_limits[:, 0].to(joint_angles.device)
        upper_ok = joint_angles <= self.joint_limits[:, 1].to(joint_angles.device)
        return lower_ok & upper_ok
    
    def jacobian(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """
        Compute manipulator Jacobian
        """
        # Numerical Jacobian computation
        B = joint_angles.shape[0]
        J = torch.zeros(B, 6, 6, device=joint_angles.device)
        
        # Current pose
        T_current = self.forward_kinematics(joint_angles)
        p_current = T_current[:, :3, 3]
        
        # Finite differences
        delta = 1e-6
        for i in range(6):
            # Perturb joint i
            joint_angles_plus = joint_angles.clone()
            joint_angles_plus[:, i] += delta
            
            T_plus = self.forward_kinematics(joint_angles_plus)
            p_plus = T_plus[:, :3, 3]
            
            # Linear velocity part
            J[:, :3, i] = (p_plus - p_current) / delta
            
            # Angular velocity part (simplified)
            R_diff = torch.matmul(T_plus[:, :3, :3], T_current[:, :3, :3].transpose(-1, -2))
            # Extract angular velocity from rotation difference
            J[:, 3:, i] = self._rotation_to_angular_velocity(R_diff, delta)
            
        return J
    
    def _rotation_to_angular_velocity(self, R_diff: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Extract angular velocity from rotation difference
        """
        B = R_diff.shape[0]
        omega = torch.zeros(B, 3, device=R_diff.device)
        
        # Use rotation vector representation
        for b in range(B):
            # Simplified - proper implementation would handle edge cases
            trace = R_diff[b].trace()
            angle = torch.acos((trace - 1) / 2)
            
            if angle > 1e-6:
                omega[b, 0] = (R_diff[b, 2, 1] - R_diff[b, 1, 2]) / (2 * torch.sin(angle))
                omega[b, 1] = (R_diff[b, 0, 2] - R_diff[b, 2, 0]) / (2 * torch.sin(angle))
                omega[b, 2] = (R_diff[b, 1, 0] - R_diff[b, 0, 1]) / (2 * torch.sin(angle))
                omega[b] *= angle / dt
                
        return omega