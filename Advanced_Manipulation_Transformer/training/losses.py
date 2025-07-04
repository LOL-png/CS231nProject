import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np

class ManipulationLoss(nn.Module):
    """
    Comprehensive loss function addressing all training issues
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Loss weights (carefully tuned for balanced training)
        self.weights = {
            'hand_pose': 1.0,
            'hand_pose_refined': 1.2,  # Higher weight for refined predictions
            'hand_shape': 0.5,
            'hand_2d': 0.3,  # 2D reprojection loss
            'object_pose': 1.0,
            'object_class': 0.5,
            'contact': 0.8,
            'contact_physics': 0.5,  # Physical plausibility
            'diversity': 0.01,  # Prevents mode collapse
            'velocity': 0.05,   # Temporal smoothness
            'penetration': 0.1, # Physical plausibility
            'attention_entropy': 0.001  # Prevents attention collapse
        }
        
        # Override with config
        if 'loss_weights' in config:
            self.weights.update(config['loss_weights'])
        
        # Component losses
        self.hand_pose_loss = AdaptiveMPJPELoss()
        self.shape_loss = nn.MSELoss()
        self.object_pose_loss = SE3Loss()
        self.contact_loss = ContactAwareLoss()
        self.classification_loss = nn.CrossEntropyLoss()
        
        # 2D reprojection loss
        self.reprojection_loss = ReprojectionLoss()
        
        # Physical plausibility loss
        self.physics_loss = PhysicsLoss()
    
    def forward(
        self,
        predictions: Dict[str, Dict[str, torch.Tensor]],
        targets: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
        epoch: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components with dynamic weighting
        """
        losses = {}
        
        # 1. Hand pose loss (adaptive weighting by joint)
        hand_pred = predictions['hand']['joints_3d']
        hand_target = targets['hand_joints_3d']
        losses['hand_pose'] = self.hand_pose_loss(hand_pred, hand_target)
        
        # Refined pose loss if available
        if 'joints_3d_refined' in predictions['hand']:
            hand_refined = predictions['hand']['joints_3d_refined']
            losses['hand_pose_refined'] = self.hand_pose_loss(hand_refined, hand_target)
        
        # 2. Hand shape loss
        if 'shape_params' in predictions['hand'] and 'mano_shape' in targets:
            losses['hand_shape'] = self.shape_loss(
                predictions['hand']['shape_params'],
                targets['mano_shape']
            )
        
        # 3. 2D reprojection loss (helps with 3D accuracy)
        if 'hand_joints_2d' in targets and 'camera_intrinsics' in targets:
            losses['hand_2d'] = self.reprojection_loss(
                hand_pred,
                targets['hand_joints_2d'],
                targets['camera_intrinsics']
            )
        
        # 4. Object pose loss (SE3 geodesic distance)
        if 'object_pose' in targets:
            obj_pred_pos = predictions['objects']['positions'][:, 0]  # First object
            obj_pred_rot = predictions['objects']['rotations'][:, 0]
            losses['object_pose'] = self.object_pose_loss(
                obj_pred_pos, 
                obj_pred_rot, 
                targets['object_pose']
            )
        
        # 5. Object classification loss
        if 'object_id' in targets:
            obj_logits = predictions['objects']['class_logits'][:, 0]  # First object
            losses['object_class'] = self.classification_loss(
                obj_logits,
                targets['object_id']
            )
        
        # 6. Contact loss (encourages realistic interactions)
        losses['contact'] = self.contact_loss(
            predictions['contacts'],
            hand_pred,
            predictions['objects']['positions']
        )
        
        # 7. Physical plausibility losses
        physics_losses = self.physics_loss(
            hand_pred,
            predictions['objects']['positions'],
            predictions['contacts']
        )
        losses.update({f'physics_{k}': v for k, v in physics_losses.items()})
        
        # 8. Diversity loss (prevents mode collapse)
        losses['diversity'] = self.compute_diversity_loss(predictions)
        
        # 9. Velocity loss (temporal smoothness if sequential)
        if 'prev_joints' in targets:
            losses['velocity'] = self.compute_velocity_loss(
                hand_pred,
                targets['prev_joints']
            )
        
        # 10. Attention entropy loss (prevents collapse)
        if model is not None:
            losses['attention_entropy'] = self.compute_attention_entropy_loss(model)
        
        # Dynamic loss weighting based on epoch
        weighted_losses = self.apply_dynamic_weighting(losses, epoch)
        
        # Total loss
        total_loss = sum(weighted_losses.values())
        losses['total'] = total_loss
        
        # Add weighted losses for logging
        losses.update({f'weighted_{k}': v for k, v in weighted_losses.items()})
        
        return losses
    
    def apply_dynamic_weighting(self, losses: Dict[str, torch.Tensor], epoch: int) -> Dict[str, torch.Tensor]:
        """Apply dynamic loss weighting based on training progress"""
        weighted = {}
        
        for key, loss in losses.items():
            if key == 'total':
                continue
                
            base_weight = self.weights.get(key, 1.0)
            
            # Curriculum learning: gradually increase certain losses
            if key in ['hand_pose_refined', 'physics_penetration']:
                # Start low, increase over time
                progress = min(epoch / 50.0, 1.0)  # Full weight by epoch 50
                weight = base_weight * progress
            elif key == 'diversity':
                # Higher weight early to prevent mode collapse
                weight = base_weight * (2.0 - min(epoch / 30.0, 1.0))
            else:
                weight = base_weight
            
            weighted[key] = weight * loss
        
        return weighted
    
    def compute_diversity_loss(self, predictions: Dict) -> torch.Tensor:
        """
        Encourage diverse predictions across batch
        Critical for fixing std=0.0003 issue
        """
        hand_joints = predictions['hand']['joints_3d']
        B = hand_joints.shape[0]
        
        if B > 1:
            # Method 1: Variance-based diversity
            joints_flat = hand_joints.reshape(B, -1)
            variance = torch.var(joints_flat, dim=0)
            var_loss = -torch.log(variance.mean() + 1e-8)
            
            # Method 2: Pairwise distance diversity
            pairwise_dist = torch.cdist(joints_flat, joints_flat)
            # Exclude diagonal
            mask = ~torch.eye(B, dtype=torch.bool, device=pairwise_dist.device)
            valid_dists = pairwise_dist[mask]
            
            # Encourage larger distances
            dist_loss = -torch.log(valid_dists.mean() + 1e-8)
            
            # Combine both
            diversity_loss = 0.5 * var_loss + 0.5 * dist_loss
        else:
            diversity_loss = torch.tensor(0.0, device=hand_joints.device)
        
        return diversity_loss
    
    def compute_velocity_loss(self, current_joints: torch.Tensor, prev_joints: torch.Tensor) -> torch.Tensor:
        """Temporal smoothness loss"""
        velocity = current_joints - prev_joints
        
        # L2 norm of velocity
        velocity_magnitude = torch.norm(velocity, dim=-1)
        
        # Penalize large velocities
        loss = velocity_magnitude.mean()
        
        # Also penalize acceleration if we have it
        if hasattr(self, 'prev_velocity'):
            acceleration = velocity - self.prev_velocity
            loss += 0.5 * torch.norm(acceleration, dim=-1).mean()
        
        self.prev_velocity = velocity.detach()
        
        return loss
    
    def compute_attention_entropy_loss(self, model: nn.Module) -> torch.Tensor:
        """
        Prevent attention entropy collapse
        Based on "Stabilizing Transformer Training" paper
        """
        entropy_losses = []
        
        def get_attention_entropy(module, input, output):
            if hasattr(output, 'attentions') and output.attentions is not None:
                # Attention weights: [B, H, N, N]
                attn_weights = output.attentions
                
                # Compute entropy: -sum(p * log(p))
                entropy = -(attn_weights * torch.log(attn_weights + 1e-9)).sum(dim=-1)
                
                # Average over heads and batch
                avg_entropy = entropy.mean()
                
                # We want high entropy (uniform attention is bad but so is peaked)
                # Target entropy depends on sequence length
                seq_len = attn_weights.shape[-1]
                target_entropy = 0.5 * np.log(seq_len)  # Half of maximum entropy
                
                # Loss encourages entropy near target
                entropy_loss = (avg_entropy - target_entropy) ** 2
                entropy_losses.append(entropy_loss)
        
        # Register hooks
        hooks = []
        for module in model.modules():
            if isinstance(module, nn.MultiheadAttention):
                hook = module.register_forward_hook(get_attention_entropy)
                hooks.append(hook)
        
        # Dummy forward pass to trigger hooks (if needed)
        # In practice, this is called after the main forward pass
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        if entropy_losses:
            return torch.stack(entropy_losses).mean()
        else:
            return torch.tensor(0.0, device=next(model.parameters()).device)

class AdaptiveMPJPELoss(nn.Module):
    """
    MPJPE loss with adaptive per-joint weighting
    Focuses on poorly performing joints
    """
    
    def __init__(self, base_weight: float = 1.0):
        super().__init__()
        self.base_weight = base_weight
        
        # Learnable per-joint weights
        self.joint_weights = nn.Parameter(torch.ones(21))
        
        # Joint importance (fingertips more important)
        importance = torch.ones(21)
        importance[[4, 8, 12, 16, 20]] = 1.5  # Fingertips
        self.register_buffer('joint_importance', importance)
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted MPJPE loss
        Args:
            pred: [B, 21, 3] predicted joints
            target: [B, 21, 3] target joints
        """
        # Per-joint errors
        joint_errors = torch.norm(pred - target, dim=-1)  # [B, 21]
        
        # Adaptive weights (higher weight for worse joints)
        adaptive_weights = F.softplus(self.joint_weights) * self.joint_importance
        
        # Normalize weights
        adaptive_weights = adaptive_weights / adaptive_weights.mean()
        
        # Ensure weights are on the same device as errors
        adaptive_weights = adaptive_weights.to(joint_errors.device)
        
        # Apply weights
        weighted_errors = joint_errors * adaptive_weights
        
        # Mean over joints and batch
        loss = weighted_errors.mean()
        
        return loss * self.base_weight

class SE3Loss(nn.Module):
    """
    Proper SE(3) loss for object poses
    Handles rotation properly unlike L2 loss
    """
    
    def __init__(self, position_weight: float = 1.0):
        super().__init__()
        self.position_weight = position_weight
    
    def forward(
        self,
        pred_pos: torch.Tensor,
        pred_rot: torch.Tensor,  # 6D rotation
        target_pose: torch.Tensor  # 4x4 matrix
    ) -> torch.Tensor:
        # Position loss (Smooth L1 is more robust than L2)
        target_pos = target_pose[:, :3, 3]
        pos_loss = F.smooth_l1_loss(pred_pos, target_pos)
        
        # Rotation loss (convert 6D to matrix first)
        pred_rot_matrix = self.six_d_to_matrix(pred_rot)
        target_rot_matrix = target_pose[:, :3, :3]
        
        # Geodesic distance on SO(3)
        rot_loss = self.geodesic_distance(pred_rot_matrix, target_rot_matrix)
        
        # Combine with balanced weighting, using position_weight
        total_loss = self.position_weight * pos_loss + 0.1 * rot_loss.mean()
        
        return total_loss
    
    def six_d_to_matrix(self, six_d: torch.Tensor) -> torch.Tensor:
        """
        Convert 6D rotation representation to rotation matrix
        Based on "On the Continuity of Rotation Representations in Neural Networks"
        """
        # six_d: [B, 6]
        a1 = six_d[..., :3]
        a2 = six_d[..., 3:]
        
        # Gram-Schmidt process
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (a2 * b1).sum(dim=-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        
        # Stack to form matrix
        matrix = torch.stack([b1, b2, b3], dim=-1)  # [B, 3, 3]
        
        return matrix
    
    def geodesic_distance(self, R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
        """
        Compute geodesic distance between rotation matrices
        Returns distance in radians
        """
        # Compute R1^T @ R2
        R_diff = torch.matmul(R1.transpose(-1, -2), R2)
        
        # Extract trace
        trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        
        # Clamp to avoid numerical issues with arccos
        cos_angle = (trace - 1) / 2
        cos_angle = torch.clamp(cos_angle, -1 + 1e-7, 1 - 1e-7)
        
        # Geodesic distance in radians
        angle = torch.acos(cos_angle)
        
        return angle

class ReprojectionLoss(nn.Module):
    """2D reprojection loss to improve 3D accuracy"""
    
    def forward(
        self,
        joints_3d: torch.Tensor,
        joints_2d_gt: torch.Tensor,
        intrinsics: torch.Tensor
    ) -> torch.Tensor:
        """
        Project 3D joints to 2D and compare with ground truth
        """
        # Project 3D to 2D - ensure same dtype for matmul
        joints_2d_proj = torch.matmul(joints_3d, intrinsics.to(joints_3d.dtype).transpose(-1, -2))
        joints_2d_proj = joints_2d_proj[..., :2] / joints_2d_proj[..., 2:3].clamp(min=0.1)
        
        # L1 loss in 2D (more robust to outliers)
        loss = F.smooth_l1_loss(joints_2d_proj, joints_2d_gt)
        
        return loss

class ContactAwareLoss(nn.Module):
    """Loss for contact prediction with physical constraints"""
    
    def __init__(self):
        super().__init__()
        self.contact_threshold = 0.02  # 2cm threshold for contact
        
    def forward(
        self,
        contact_predictions: Dict[str, torch.Tensor],
        hand_joints: torch.Tensor,
        object_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contact loss with physical constraints
        """
        contact_points = contact_predictions['contact_points']
        # Handle both possible key names
        contact_confidence = contact_predictions.get('contact_confidence', 
                                                   contact_predictions.get('contact_probs'))
        
        # Compute minimum distance from each contact point to hand joints
        # contact_points: [B, N_contacts, 3]
        # hand_joints: [B, 21, 3]
        dists_to_hand = torch.cdist(contact_points, hand_joints)  # [B, N_contacts, 21]
        min_dist_to_hand = dists_to_hand.min(dim=-1)[0]  # [B, N_contacts]
        
        # Similarly for objects
        dists_to_obj = torch.cdist(contact_points, object_positions)  # [B, N_contacts, N_obj]
        min_dist_to_obj = dists_to_obj.min(dim=-1)[0]  # [B, N_contacts]
        
        # Contact points should be close to both hand and object
        proximity_loss = contact_confidence * (min_dist_to_hand + min_dist_to_obj)
        
        # High confidence contacts should be very close
        threshold_loss = contact_confidence * F.relu(
            torch.maximum(min_dist_to_hand, min_dist_to_obj) - self.contact_threshold
        )
        
        # Low confidence for points far from both
        far_penalty = (1 - contact_confidence) * torch.exp(
            -torch.minimum(min_dist_to_hand, min_dist_to_obj) / self.contact_threshold
        )
        
        total_loss = proximity_loss.mean() + threshold_loss.mean() + far_penalty.mean()
        
        return total_loss

class PhysicsLoss(nn.Module):
    """Physical plausibility losses"""
    
    def __init__(self):
        super().__init__()
        self.penetration_threshold = 0.005  # 5mm
        
    def forward(
        self,
        hand_joints: torch.Tensor,
        object_positions: torch.Tensor,
        contact_predictions: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute various physical plausibility losses
        """
        losses = {}
        
        # 1. Joint angle limits (simplified)
        # Compute angles between consecutive joints
        joint_vectors = torch.diff(hand_joints, dim=1)  # [B, 20, 3]
        joint_angles = self.compute_angles(joint_vectors)
        
        # Penalize extreme angles
        angle_loss = F.relu(joint_angles - np.pi * 0.8) + F.relu(-joint_angles)
        losses['joint_angles'] = angle_loss.mean()
        
        # 2. Penetration loss (hand shouldn't penetrate objects)
        # Simplified: use distance between closest points
        hand_obj_dists = torch.cdist(hand_joints, object_positions)  # [B, 21, N_obj]
        min_dists = hand_obj_dists.min(dim=-1)[0]  # [B, 21]
        
        penetration_loss = F.relu(self.penetration_threshold - min_dists)
        losses['penetration'] = penetration_loss.mean()
        
        # 3. Contact consistency
        # If contact confidence is high, forces should be reasonable
        if 'contact_forces' in contact_predictions:
            forces = contact_predictions['contact_forces']
            force_magnitudes = torch.norm(forces, dim=-1)
            
            # Penalize very large forces
            force_loss = F.relu(force_magnitudes - 10.0)  # 10N max
            losses['contact_forces'] = force_loss.mean()
        
        return losses
    
    def compute_angles(self, vectors: torch.Tensor) -> torch.Tensor:
        """Compute angles between consecutive vectors"""
        # Normalize vectors
        vectors_norm = F.normalize(vectors, dim=-1)
        
        # Compute dot products between consecutive vectors
        if vectors_norm.shape[1] > 1:
            dots = (vectors_norm[:, :-1] * vectors_norm[:, 1:]).sum(dim=-1)
        else:
            # Return zero angles if not enough vectors
            return torch.zeros(vectors_norm.shape[0], 0, device=vectors_norm.device)
        
        # Clamp and compute angles
        dots = torch.clamp(dots, -1 + 1e-7, 1 - 1e-7)
        angles = torch.acos(dots)
        
        return angles


class DiversityLoss(nn.Module):
    """Prevent mode collapse by encouraging diverse predictions"""
    
    def __init__(self, margin: float = 0.01):
        super().__init__()
        self.margin = margin
        
    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute diversity loss to prevent identical predictions
        Args:
            predictions: [B, ...] batch of predictions
        """
        B = predictions.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=predictions.device)
        
        # Flatten predictions
        pred_flat = predictions.view(B, -1)
        
        # Compute pairwise distances
        dists = torch.cdist(pred_flat, pred_flat)  # [B, B]
        
        # Mask diagonal (self-distances)
        mask = ~torch.eye(B, dtype=bool, device=predictions.device)
        dists = dists[mask]
        
        # Penalize predictions that are too close
        diversity_loss = F.relu(self.margin - dists).mean()
        
        return diversity_loss


class ComprehensiveLoss(nn.Module):
    """Main loss combining all components with dynamic weighting"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Initialize all loss components
        self.hand_loss = AdaptiveMPJPELoss()
        self.object_loss = SE3Loss(position_weight=config.get('object_position_weight', 1.0))
        self.contact_loss = ContactAwareLoss()
        self.physics_loss = PhysicsLoss()
        self.diversity_loss = DiversityLoss(margin=config.get('diversity_margin', 0.01))
        self.reprojection_loss = ReprojectionLoss()
        
        # Loss weights
        self.weights = config.get('loss_weights', {})
        self.default_weights = {
            'hand_coarse': 1.0,
            'hand_refined': 1.2,
            'object_position': 1.0,
            'object_rotation': 0.5,
            'contact': 0.3,
            'physics': 0.1,
            'diversity': 0.01,
            'reprojection': 0.5,
            'kl': 0.001  # For sigma reparameterization
        }
        
        # Update with provided weights
        for k, v in self.weights.items():
            self.default_weights[k] = v
        
        self.current_epoch = 0
        
    def set_epoch(self, epoch: int):
        """Update epoch for dynamic weighting"""
        self.current_epoch = epoch
        
    def get_weight(self, loss_name: str) -> float:
        """Get dynamic weight for loss component"""
        base_weight = self.default_weights.get(loss_name, 1.0)
        
        # Dynamic scheduling based on epoch
        if loss_name == 'diversity':
            # High early to prevent collapse
            return base_weight * max(1.0, 2.0 - self.current_epoch / 30)
        elif loss_name == 'physics':
            # Increase over time
            return base_weight * min(1.0, self.current_epoch / 50)
        elif loss_name == 'hand_refined':
            # Start low, increase over time
            return base_weight * min(1.0, 0.3 + self.current_epoch / 30)
        else:
            return base_weight
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses
        Args:
            outputs: Model predictions
            targets: Ground truth labels
        Returns:
            Dictionary of individual losses
        """
        losses = {}
        
        # Hand pose losses
        # Handle both 'hand_joints' and 'hand_joints_3d' keys
        hand_gt_key = 'hand_joints_3d' if 'hand_joints_3d' in targets else 'hand_joints'
        
        if 'hand_joints_coarse' in outputs and hand_gt_key in targets:
            losses['hand_coarse'] = self.hand_loss(
                outputs['hand_joints_coarse'],
                targets[hand_gt_key]
            )
        
        if 'hand_joints' in outputs and hand_gt_key in targets:
            losses['hand_refined'] = self.hand_loss(
                outputs['hand_joints'],
                targets[hand_gt_key]
            )
            
            # Add diversity loss
            losses['diversity'] = self.diversity_loss(outputs['hand_joints'])
        
        # Object pose losses
        if 'object_positions' in outputs and 'object_rotations' in outputs and 'object_pose' in targets:
            # Extract positions and rotations separately
            pred_positions = outputs['object_positions']  # [B, max_objects, 3]
            pred_rotations = outputs['object_rotations']   # [B, max_objects, 6]
            
            # Handle target format
            target_poses = targets['object_pose']  # [B, 3, 4] for single object
            
            # For single object target, just use first predicted object
            if target_poses.dim() == 3 and target_poses.shape[1] == 3 and target_poses.shape[2] == 4:
                # Single object case: target is [B, 3, 4]
                obj_loss = self.object_loss(
                    pred_positions[:, 0],  # [B, 3] - first object
                    pred_rotations[:, 0],  # [B, 6] - first object
                    target_poses           # [B, 3, 4]
                )
                losses['object_position'] = obj_loss
                losses['object_rotation'] = torch.tensor(0.0)  # Included in obj_loss
            else:
                # Multiple objects case
                if target_poses.dim() == 3:
                    target_poses = target_poses.unsqueeze(1)  # Add object dimension
                
                total_loss = 0
                num_objects = min(pred_positions.shape[1], target_poses.shape[1])
                
                for i in range(num_objects):
                    obj_loss = self.object_loss(
                        pred_positions[:, i],  # [B, 3]
                        pred_rotations[:, i],  # [B, 6]
                        target_poses[:, i]     # [B, 3, 4]
                    )
                    total_loss = total_loss + obj_loss
                
                if num_objects > 0:
                    losses['object_position'] = total_loss / num_objects
                    losses['object_rotation'] = torch.tensor(0.0)  # Included in total
        
        # Contact loss
        if 'contact_points' in outputs and hand_gt_key in targets:
            # Extract object positions from pose if needed
            object_positions = None
            if 'object_positions' in targets:
                object_positions = targets['object_positions']
            elif 'object_pose' in targets:
                # Extract position from [3, 4] pose matrix
                object_positions = targets['object_pose'][:, :3, 3].unsqueeze(1)  # [B, 1, 3]
            
            if object_positions is not None:
                losses['contact'] = self.contact_loss(
                    outputs,
                    targets[hand_gt_key],
                    object_positions
                )
        
        # Physics losses
        if all(k in outputs for k in ['hand_joints', 'object_positions']):
            physics_dict = self.physics_loss(
                outputs['hand_joints'],
                outputs['object_positions'],
                outputs
            )
            for k, v in physics_dict.items():
                losses[f'physics_{k}'] = v
        
        # Reprojection loss
        if 'hand_joints' in outputs and 'hand_joints_2d' in targets and 'camera_intrinsics' in targets:
            losses['reprojection'] = self.reprojection_loss(
                outputs['hand_joints'],
                targets['hand_joints_2d'],
                targets['camera_intrinsics']
            )
        
        # KL loss from sigma reparameterization
        if 'kl_loss' in outputs:
            losses['kl'] = outputs['kl_loss']
        
        # Apply weights and calculate total
        weighted_losses = {}
        
        # Get device from any tensor in outputs
        device = torch.device('cpu')
        for v in outputs.values():
            if isinstance(v, torch.Tensor):
                device = v.device
                break
        
        total_loss = torch.tensor(0.0, device=device)
        
        for k, v in losses.items():
            if v is not None and isinstance(v, torch.Tensor):
                weight = self.get_weight(k)
                weighted_losses[k] = weight * v
                total_loss = total_loss + weighted_losses[k]
        
        # Add total loss
        weighted_losses['total'] = total_loss
        
        return weighted_losses