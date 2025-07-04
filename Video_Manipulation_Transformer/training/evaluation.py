"""
Evaluation Metrics for Video-to-Manipulation Transformer
Implements COCO, BOP, HPE, and Grasp evaluation protocols
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for the model
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all metrics"""
        self.metrics = defaultdict(list)
        self.predictions = []
        self.ground_truths = []
        
    def update(self,
               predictions: Dict[str, torch.Tensor],
               targets: Dict[str, torch.Tensor],
               simulation_outputs: Optional[Dict[str, torch.Tensor]] = None):
        """
        Update metrics with batch results
        """
        # Store raw predictions and targets
        self.predictions.append(predictions)
        self.ground_truths.append(targets)
        
        # Compute immediate metrics
        batch_metrics = self._compute_batch_metrics(predictions, targets, simulation_outputs)
        
        for key, value in batch_metrics.items():
            self.metrics[key].append(value)
            
    def compute(self) -> Dict[str, float]:
        """
        Compute final aggregated metrics
        """
        results = {}
        
        # Average batch metrics
        for key, values in self.metrics.items():
            if len(values) > 0:
                results[key] = np.mean(values)
                
        # Compute dataset-wide metrics
        if self.predictions and self.ground_truths:
            # COCO metrics
            coco_metrics = self._compute_coco_metrics()
            results.update({f'coco_{k}': v for k, v in coco_metrics.items()})
            
            # BOP metrics
            bop_metrics = self._compute_bop_metrics()
            results.update({f'bop_{k}': v for k, v in bop_metrics.items()})
            
            # HPE metrics
            hpe_metrics = self._compute_hpe_metrics()
            results.update({f'hpe_{k}': v for k, v in hpe_metrics.items()})
            
            # Grasp metrics
            grasp_metrics = self._compute_grasp_metrics()
            results.update({f'grasp_{k}': v for k, v in grasp_metrics.items()})
            
        return results
    
    def _compute_batch_metrics(self,
                             predictions: Dict[str, torch.Tensor],
                             targets: Dict[str, torch.Tensor],
                             simulation_outputs: Optional[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """
        Compute metrics for a single batch
        """
        metrics = {}
        
        # Hand pose accuracy
        if 'joints_3d' in predictions and 'hand_joints_3d' in targets:
            mpjpe = self._compute_mpjpe(
                predictions['joints_3d'],
                targets['hand_joints_3d'].squeeze(1)
            )
            metrics['mpjpe'] = mpjpe.item()
            
        # Object pose accuracy
        if 'positions' in predictions and 'object_poses' in targets:
            position_error = (
                predictions['positions'][:, :targets['object_poses'].shape[1], :] -
                targets['object_poses'][..., :3, 3]
            ).norm(dim=-1).mean()
            metrics['object_position_error'] = position_error.item()
            
        # Action accuracy
        if 'ee_target_poses' in predictions:
            # Trajectory smoothness
            if predictions['ee_target_poses'].shape[1] > 1:
                velocities = predictions['ee_target_poses'][:, 1:] - predictions['ee_target_poses'][:, :-1]
                smoothness = velocities.norm(dim=-1).std(dim=1).mean()
                metrics['trajectory_smoothness'] = smoothness.item()
                
        # Simulation metrics
        if simulation_outputs:
            if 'grasp_success' in simulation_outputs:
                metrics['grasp_success_rate'] = simulation_outputs['grasp_success'].mean().item()
                
            if 'simulation_metrics' in simulation_outputs:
                for k, v in simulation_outputs['simulation_metrics'].items():
                    if isinstance(v, torch.Tensor):
                        metrics[f'sim_{k}'] = v.item()
                        
        return metrics
    
    def _compute_mpjpe(self, pred_joints: torch.Tensor, gt_joints: torch.Tensor) -> torch.Tensor:
        """
        Mean Per Joint Position Error
        """
        return (pred_joints - gt_joints).norm(dim=-1).mean()
    
    def _compute_coco_metrics(self) -> Dict[str, float]:
        """
        COCO-style 2D detection metrics
        """
        metrics = {}
        
        # Simplified implementation - full version would compute:
        # - AP@IoU=0.5:0.95
        # - AP@IoU=0.5
        # - AP@IoU=0.75
        # - AR@1, AR@10, AR@100
        
        # For now, compute basic detection accuracy
        all_preds = []
        all_targets = []
        
        for pred_batch, target_batch in zip(self.predictions, self.ground_truths):
            if 'hand_joints_2d' in pred_batch and 'hand_joints_2d' in target_batch:
                all_preds.append(pred_batch['hand_joints_2d'])
                all_targets.append(target_batch['hand_joints_2d'])
                
        if all_preds:
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            # PCK (Percentage of Correct Keypoints)
            threshold = 10  # pixels
            distances = (all_preds - all_targets).norm(dim=-1)
            pck = (distances < threshold).float().mean()
            metrics['pck@10'] = pck.item()
            
        return metrics
    
    def _compute_bop_metrics(self) -> Dict[str, float]:
        """
        BOP-style 6D pose estimation metrics
        """
        metrics = {}
        
        # Simplified implementation - full version would compute:
        # - VSD (Visible Surface Discrepancy)
        # - MSSD (Maximum Symmetry-aware Surface Distance)
        # - MSPD (Maximum Symmetry-aware Projection Distance)
        
        all_position_errors = []
        all_rotation_errors = []
        
        for pred_batch, target_batch in zip(self.predictions, self.ground_truths):
            if 'positions' in pred_batch and 'object_poses' in target_batch:
                # Position error
                pred_pos = pred_batch['positions']
                gt_pos = target_batch['object_poses'][..., :3, 3]
                
                # Match predictions to ground truth (simplified)
                num_objects = min(pred_pos.shape[1], gt_pos.shape[1])
                pos_error = (pred_pos[:, :num_objects] - gt_pos[:, :num_objects]).norm(dim=-1)
                all_position_errors.append(pos_error)
                
                # Rotation error (if available)
                if 'rotations' in pred_batch:
                    # Convert 6D rotation to matrix
                    # Simplified - would need proper rotation error computation
                    rot_error = torch.rand_like(pos_error) * 0.1  # Placeholder
                    all_rotation_errors.append(rot_error)
                    
        if all_position_errors:
            all_position_errors = torch.cat(all_position_errors, dim=0)
            
            # ADD metric (Average Distance of model points)
            metrics['add'] = all_position_errors.mean().item()
            
            # ADD-S metric (with symmetry) - simplified
            metrics['add_s'] = all_position_errors.mean().item()
            
            # Success rate at different thresholds
            for threshold in [0.05, 0.10]:  # 5cm, 10cm
                success = (all_position_errors < threshold).float().mean()
                metrics[f'add_success@{threshold}'] = success.item()
                
        return metrics
    
    def _compute_hpe_metrics(self) -> Dict[str, float]:
        """
        Hand Pose Estimation metrics
        """
        metrics = {}
        
        all_mpjpe = []
        all_pck = []
        
        for pred_batch, target_batch in zip(self.predictions, self.ground_truths):
            if 'joints_3d' in pred_batch and 'hand_joints_3d' in target_batch:
                pred_joints = pred_batch['joints_3d']
                gt_joints = target_batch['hand_joints_3d'].squeeze(1)
                
                # MPJPE
                mpjpe = self._compute_mpjpe(pred_joints, gt_joints)
                all_mpjpe.append(mpjpe)
                
                # PCK3D
                threshold = 0.03  # 3cm
                distances = (pred_joints - gt_joints).norm(dim=-1)
                pck = (distances < threshold).float().mean(dim=-1)
                all_pck.append(pck)
                
        if all_mpjpe:
            metrics['mpjpe'] = torch.cat(all_mpjpe).mean().item()
            metrics['pck3d@30mm'] = torch.cat(all_pck).mean().item()
            
            # Per-joint breakdown (optional)
            joint_names = ['wrist', 'thumb', 'index', 'middle', 'ring', 'pinky']
            # Would compute per-joint metrics here
            
        return metrics
    
    def _compute_grasp_metrics(self) -> Dict[str, float]:
        """
        Grasp evaluation metrics
        """
        metrics = {}
        
        grasp_successes = []
        force_violations = []
        
        for pred_batch in self.predictions:
            if 'grasp_success' in pred_batch:
                grasp_successes.append(pred_batch['grasp_success'])
                
            if 'contact_forces' in pred_batch:
                # Check force limits
                max_force = 10.0  # N
                violations = (pred_batch['contact_forces'].norm(dim=-1) > max_force).float().mean(dim=-1)
                force_violations.append(violations)
                
        if grasp_successes:
            metrics['grasp_success_rate'] = torch.cat(grasp_successes).mean().item()
            
        if force_violations:
            metrics['force_violation_rate'] = torch.cat(force_violations).mean().item()
            
        # Grasp quality metrics (if available from simulation)
        # - Force closure success rate
        # - Grasp wrench space volume
        # - Contact point distribution
        
        return metrics


class GraspEvaluator:
    """
    Specialized evaluator for grasp quality
    """
    
    @staticmethod
    def evaluate_grasp(contact_points: torch.Tensor,
                      contact_normals: torch.Tensor,
                      contact_forces: torch.Tensor,
                      object_pose: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate grasp quality metrics
        """
        metrics = {}
        
        # Force closure
        force_closure = GraspEvaluator._check_force_closure(
            contact_points, contact_normals, contact_forces
        )
        metrics['force_closure'] = float(force_closure)
        
        # Grasp isotropy (how uniform the grasp is)
        isotropy = GraspEvaluator._compute_grasp_isotropy(
            contact_points, contact_normals
        )
        metrics['grasp_isotropy'] = isotropy
        
        # Grasp stability (resistance to perturbations)
        stability = GraspEvaluator._compute_grasp_stability(
            contact_points, contact_normals, contact_forces
        )
        metrics['grasp_stability'] = stability
        
        return metrics
    
    @staticmethod
    def _check_force_closure(points: torch.Tensor,
                           normals: torch.Tensor,
                           forces: torch.Tensor) -> bool:
        """
        Check if grasp achieves force closure
        """
        if points.shape[0] < 3:
            return False
            
        # Check force balance
        net_force = forces.sum(dim=0)
        force_balanced = net_force.norm() < 0.1
        
        # Check torque balance
        center = points.mean(dim=0)
        torques = torch.stack([
            torch.cross(p - center, f) for p, f in zip(points, forces)
        ])
        net_torque = torques.sum(dim=0)
        torque_balanced = net_torque.norm() < 0.1
        
        return force_balanced and torque_balanced
    
    @staticmethod
    def _compute_grasp_isotropy(points: torch.Tensor,
                              normals: torch.Tensor) -> float:
        """
        Compute how isotropic (uniform) the grasp is
        """
        if points.shape[0] < 3:
            return 0.0
            
        # Compute grasp matrix G
        center = points.mean(dim=0)
        G = []
        
        for i in range(points.shape[0]):
            # Linear part
            G.append(normals[i])
            
            # Angular part
            r = points[i] - center
            G.append(torch.cross(r, normals[i]))
            
        G = torch.stack(G, dim=1)  # [6, N]
        
        # Compute singular values
        U, S, V = torch.svd(G)
        
        # Isotropy is ratio of smallest to largest singular value
        isotropy = S[-1] / (S[0] + 1e-8)
        
        return isotropy.item()
    
    @staticmethod
    def _compute_grasp_stability(points: torch.Tensor,
                               normals: torch.Tensor,
                               forces: torch.Tensor) -> float:
        """
        Compute grasp stability metric
        """
        if points.shape[0] < 3:
            return 0.0
            
        # Simplified stability based on contact configuration
        # Better grasps have:
        # 1. More contacts
        # 2. Well-distributed contacts
        # 3. Appropriate force magnitudes
        
        num_contacts = points.shape[0]
        contact_score = min(1.0, num_contacts / 5.0)
        
        # Distribution score
        pairwise_dists = torch.cdist(points, points)
        avg_dist = pairwise_dists[pairwise_dists > 0].mean()
        dist_score = torch.sigmoid(avg_dist * 10)  # Encourage spread
        
        # Force score
        force_mags = forces.norm(dim=-1)
        force_score = torch.exp(-((force_mags - 2.0) ** 2).mean())  # Prefer ~2N forces
        
        stability = contact_score * dist_score * force_score
        
        return stability.item()