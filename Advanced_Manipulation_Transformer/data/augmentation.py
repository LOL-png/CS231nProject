import torch
import numpy as np
import random
from typing import Dict, Tuple

class DataAugmentor:
    """Advanced data augmentation for hand pose estimation"""
    
    def __init__(self):
        self.joint_noise_std = 0.005  # 5mm
        self.rotation_range = 10.0    # degrees
        self.scale_range = (0.9, 1.1)
        self.translation_std = 0.02   # 2cm
        
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply augmentations"""
        
        # 1. Joint noise injection (prevents overfitting)
        if 'hand_joints_3d' in sample and random.random() < 0.5:
            noise = torch.randn_like(sample['hand_joints_3d']) * self.joint_noise_std
            sample['hand_joints_3d'] += noise
        
        # 2. 3D rotation augmentation
        if random.random() < 0.5:
            angle = np.radians(random.uniform(-self.rotation_range, self.rotation_range))
            axis = random.choice(['x', 'y', 'z'])
            R = self._get_rotation_matrix(angle, axis)
            
            # Rotate 3D joints
            if 'hand_joints_3d' in sample:
                joints = sample['hand_joints_3d']
                sample['hand_joints_3d'] = torch.matmul(joints, R.T)
            
            # Update object pose
            if 'object_pose' in sample:
                pose = sample['object_pose']
                pose[:3, :3] = torch.matmul(R, pose[:3, :3])
                sample['object_pose'] = pose
        
        # 3. Scale augmentation
        if random.random() < 0.3:
            scale = random.uniform(*self.scale_range)
            if 'hand_joints_3d' in sample:
                sample['hand_joints_3d'] *= scale
            if 'object_pose' in sample:
                sample['object_pose'][:3, 3] *= scale  # Scale translation
        
        # 4. Translation augmentation
        if random.random() < 0.3:
            translation = torch.randn(3) * self.translation_std
            if 'hand_joints_3d' in sample:
                sample['hand_joints_3d'] += translation
            if 'object_pose' in sample:
                sample['object_pose'][:3, 3] += translation
        
        # 5. 2D joint augmentation (consistent with 3D)
        if 'hand_joints_2d' in sample and random.random() < 0.3:
            # Add small 2D noise
            noise_2d = torch.randn_like(sample['hand_joints_2d']) * 2.0  # pixels
            sample['hand_joints_2d'] += noise_2d
        
        # 6. Temporal jitter for sequences
        if 'temporal_offset' in sample:
            sample['temporal_offset'] += torch.randn(1) * 0.1
        
        return sample
    
    def _get_rotation_matrix(self, angle: float, axis: str) -> torch.Tensor:
        """Get 3D rotation matrix"""
        c, s = np.cos(angle), np.sin(angle)
        if axis == 'x':
            R = torch.tensor([
                [1, 0, 0],
                [0, c, -s],
                [0, s, c]
            ], dtype=torch.float32)
        elif axis == 'y':
            R = torch.tensor([
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c]
            ], dtype=torch.float32)
        else:  # z
            R = torch.tensor([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ], dtype=torch.float32)
        return R