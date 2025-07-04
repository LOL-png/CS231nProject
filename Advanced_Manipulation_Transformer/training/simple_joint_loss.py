import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class SimpleJointLoss(nn.Module):
    """
    Simplified loss function that computes:
    1. Per-joint position differences (L2 distance)
    2. Object position differences
    3. Object size error (bounding box dimensions)
    4. Pairwise distances between all hand joints
    
    No fancy weighting, physics constraints, or other complex losses.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Get loss weights from config
        self.hand_weight = config.get('loss_weights', {}).get('hand_coarse', 1.0)
        self.object_weight = config.get('loss_weights', {}).get('object_position', 1.0)
        self.object_size_weight = config.get('loss_weights', {}).get('object_size', 0.5)
        self.pairwise_weight = config.get('loss_weights', {}).get('pairwise_distances', 0.3)
        
        # Optional: weight fingertips more
        self.use_fingertip_weighting = config.get('per_joint_weighting', False)
        self.fingertip_weight = config.get('fingertip_weight', 1.5)
        
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute simple per-joint and object position losses
        
        Args:
            outputs: Model predictions
            targets: Ground truth labels
            
        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}
        
        # 1. Hand joint loss - simple L2 distance per joint
        # Handle both 'hand_joints' and 'hand_joints_3d' keys for ground truth
        hand_gt_key = 'hand_joints_3d' if 'hand_joints_3d' in targets else 'hand_joints'
        
        if 'hand_joints' in outputs and hand_gt_key in targets:
            pred_joints = outputs['hand_joints']  # [B, 21, 3]
            gt_joints = targets[hand_gt_key]      # [B, 21, 3]
            
            # Compute per-joint L2 distance
            per_joint_dist = torch.norm(pred_joints - gt_joints, dim=-1)  # [B, 21]
            
            # Optional: weight fingertips more
            if self.use_fingertip_weighting:
                joint_weights = torch.ones_like(per_joint_dist)
                # Fingertip indices: 4, 8, 12, 16, 20
                fingertip_indices = [4, 8, 12, 16, 20]
                joint_weights[:, fingertip_indices] = self.fingertip_weight
                per_joint_dist = per_joint_dist * joint_weights
            
            # Average over joints and batch
            hand_loss = per_joint_dist.mean()
            losses['hand_joints'] = hand_loss * self.hand_weight
            
            # Also store per-joint errors for analysis (not used in backprop)
            with torch.no_grad():
                for j in range(21):
                    losses[f'joint_{j}_error'] = per_joint_dist[:, j].mean()
        
        # 2. Object position loss - simple L2 distance
        if 'object_positions' in outputs:
            pred_obj_pos = outputs['object_positions']  # [B, num_objects, 3]
            
            # Handle different target formats
            if 'object_pose' in targets:
                # Extract position from 3x4 pose matrix
                target_pose = targets['object_pose']  # [B, 3, 4]
                if target_pose.dim() == 3 and target_pose.shape[1] == 3:
                    # Single object: extract translation
                    gt_obj_pos = target_pose[:, :3, 3]  # [B, 3]
                    
                    # Use only first predicted object
                    obj_distance = torch.norm(pred_obj_pos[:, 0] - gt_obj_pos, dim=-1)
                    obj_loss = obj_distance.mean()
                    losses['object_position'] = obj_loss * self.object_weight
            
            elif 'object_positions' in targets:
                # Direct position targets
                gt_obj_pos = targets['object_positions']
                
                # Handle shape mismatch
                num_objects = min(pred_obj_pos.shape[1], gt_obj_pos.shape[1])
                if num_objects > 0:
                    obj_distances = []
                    for i in range(num_objects):
                        dist = torch.norm(pred_obj_pos[:, i] - gt_obj_pos[:, i], dim=-1)
                        obj_distances.append(dist)
                    
                    obj_loss = torch.stack(obj_distances).mean()
                    losses['object_position'] = obj_loss * self.object_weight
        
        # 3. Object size loss
        if 'object_dimensions' in outputs or 'object_corners' in outputs:
            if 'object_dimensions' in outputs and 'object_dimensions' in targets:
                # Direct dimension prediction
                pred_dims = outputs['object_dimensions']  # [B, num_objects, 3]
                gt_dims = targets['object_dimensions']    # [B, num_objects, 3]
                
                # L2 loss on dimensions (width, height, depth)
                size_loss = torch.norm(pred_dims - gt_dims, dim=-1).mean()
                losses['object_size'] = size_loss * self.object_size_weight
                
            elif 'object_corners' in outputs:
                # Compute size from bounding box corners
                pred_corners = outputs['object_corners']  # [B, num_objects, 8, 3]
                
                # Compute bounding box size from corners
                pred_min = pred_corners.min(dim=2)[0]  # [B, num_objects, 3]
                pred_max = pred_corners.max(dim=2)[0]  # [B, num_objects, 3]
                pred_size = pred_max - pred_min  # [B, num_objects, 3]
                
                if 'object_dimensions' in targets:
                    gt_dims = targets['object_dimensions']
                    size_loss = torch.norm(pred_size - gt_dims, dim=-1).mean()
                    losses['object_size'] = size_loss * self.object_size_weight
                elif 'object_corners' in targets:
                    gt_corners = targets['object_corners']
                    gt_min = gt_corners.min(dim=2)[0]
                    gt_max = gt_corners.max(dim=2)[0]
                    gt_size = gt_max - gt_min
                    
                    size_loss = torch.norm(pred_size - gt_size, dim=-1).mean()
                    losses['object_size'] = size_loss * self.object_size_weight
        
        # 4. Pairwise distance loss between all hand joints
        if 'hand_joints' in outputs and hand_gt_key in targets:
            pred_joints = outputs['hand_joints']  # [B, 21, 3]
            gt_joints = targets[hand_gt_key]      # [B, 21, 3]
            
            # Compute pairwise distances for predicted joints
            # Use cdist to compute all pairwise distances efficiently
            pred_pairwise = torch.cdist(pred_joints, pred_joints)  # [B, 21, 21]
            gt_pairwise = torch.cdist(gt_joints, gt_joints)        # [B, 21, 21]
            
            # Only consider upper triangular part (avoiding diagonal and duplicates)
            batch_size, num_joints = pred_joints.shape[:2]
            triu_indices = torch.triu_indices(num_joints, num_joints, offset=1)
            
            # Extract upper triangular distances
            pred_dists = pred_pairwise[:, triu_indices[0], triu_indices[1]]  # [B, num_pairs]
            gt_dists = gt_pairwise[:, triu_indices[0], triu_indices[1]]      # [B, num_pairs]
            
            # L1 loss on pairwise distances (more robust than L2 for distances)
            pairwise_loss = F.l1_loss(pred_dists, gt_dists)
            losses['pairwise_distances'] = pairwise_loss * self.pairwise_weight
            
            # Optional: Store some specific pairwise distances for analysis
            with torch.no_grad():
                # Example: distance between thumb tip (4) and index tip (8)
                thumb_index_dist = torch.norm(pred_joints[:, 4] - pred_joints[:, 8], dim=-1).mean()
                losses['thumb_index_distance'] = thumb_index_dist
                
                # Distance between wrist (0) and middle fingertip (12)
                wrist_middle_dist = torch.norm(pred_joints[:, 0] - pred_joints[:, 12], dim=-1).mean()
                losses['wrist_middle_distance'] = wrist_middle_dist
        
        # 5. Total loss is sum of all individual losses
        # Filter out analysis metrics (those that shouldn't contribute to backprop)
        trainable_losses = {k: v for k, v in losses.items() 
                           if not k.startswith('joint_') and k not in ['thumb_index_distance', 'wrist_middle_distance']}
        total_loss = sum(trainable_losses.values())
        losses['total'] = total_loss
        
        return losses
    
    def set_epoch(self, epoch: int):
        """For compatibility with the original loss class"""
        pass  # No dynamic weighting in simple loss