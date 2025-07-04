import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import logging
from scipy.spatial.transform import Rotation as R
import json

logger = logging.getLogger(__name__)

class ComprehensiveEvaluator:
    """
    Comprehensive evaluation for hand and object pose estimation
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.results = {}
        
    def evaluate_model(
        self,
        model: nn.Module,
        dataloader,
        device: torch.device = torch.device('cuda')
    ) -> Dict[str, float]:
        """
        Full evaluation suite
        """
        model.eval()
        
        # Initialize metrics storage
        hand_mpjpe_list = []
        hand_pa_mpjpe_list = []
        object_add_list = []
        object_adds_list = []
        contact_accuracy_list = []
        per_joint_errors = {i: [] for i in range(21)}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = model(batch)
                
                # Evaluate hand pose
                if 'hand_joints' in outputs and 'hand_joints' in batch:
                    hand_metrics = self.evaluate_hand_pose(
                        outputs['hand_joints'],
                        batch['hand_joints'],
                        per_joint_errors
                    )
                    hand_mpjpe_list.extend(hand_metrics['mpjpe'])
                    hand_pa_mpjpe_list.extend(hand_metrics['pa_mpjpe'])
                
                # Evaluate object pose
                if 'object_poses' in outputs and 'object_poses' in batch:
                    object_metrics = self.evaluate_object_pose(
                        outputs['object_poses'],
                        batch['object_poses'],
                        batch.get('object_points')
                    )
                    if object_metrics['add'] is not None:
                        object_add_list.extend(object_metrics['add'])
                    if object_metrics['adds'] is not None:
                        object_adds_list.extend(object_metrics['adds'])
                
                # Evaluate contact predictions
                if 'contact_points' in outputs and 'contact_labels' in batch:
                    contact_acc = self.evaluate_contact_predictions(
                        outputs['contact_points'],
                        outputs.get('contact_probs'),
                        batch['contact_labels']
                    )
                    contact_accuracy_list.append(contact_acc)
        
        # Aggregate metrics
        results = {
            'hand_mpjpe': np.mean(hand_mpjpe_list) if hand_mpjpe_list else 0.0,
            'hand_pa_mpjpe': np.mean(hand_pa_mpjpe_list) if hand_pa_mpjpe_list else 0.0,
            'object_add': np.mean(object_add_list) if object_add_list else 0.0,
            'object_adds': np.mean(object_adds_list) if object_adds_list else 0.0,
            'contact_accuracy': np.mean(contact_accuracy_list) if contact_accuracy_list else 0.0,
        }
        
        # Per-joint analysis
        for joint_idx, errors in per_joint_errors.items():
            if errors:
                results[f'joint_{joint_idx}_mpjpe'] = np.mean(errors)
        
        # Compute AUC metrics
        if hand_mpjpe_list:
            results['hand_auc_20_50'] = self.compute_auc(hand_mpjpe_list, 20, 50)
        
        self.results = results
        return results
    
    def evaluate_hand_pose(
        self,
        pred_joints: torch.Tensor,
        gt_joints: torch.Tensor,
        per_joint_errors: Dict[int, List[float]]
    ) -> Dict[str, List[float]]:
        """
        Evaluate hand pose predictions
        """
        # Compute MPJPE
        mpjpe = torch.norm(pred_joints - gt_joints, dim=-1)  # [B, 21]
        
        # Store per-joint errors
        for joint_idx in range(21):
            per_joint_errors[joint_idx].extend(
                mpjpe[:, joint_idx].cpu().numpy().tolist()
            )
        
        # Mean per sample
        mpjpe_per_sample = mpjpe.mean(dim=1)  # [B]
        
        # PA-MPJPE (Procrustes aligned)
        pa_mpjpe_list = []
        for i in range(pred_joints.shape[0]):
            aligned_pred = self.procrustes_align(
                pred_joints[i].cpu().numpy(),
                gt_joints[i].cpu().numpy()
            )
            pa_mpjpe = np.mean(np.linalg.norm(
                aligned_pred - gt_joints[i].cpu().numpy(), axis=-1
            ))
            pa_mpjpe_list.append(pa_mpjpe)
        
        return {
            'mpjpe': mpjpe_per_sample.cpu().numpy().tolist(),
            'pa_mpjpe': pa_mpjpe_list
        }
    
    def evaluate_object_pose(
        self,
        pred_poses: torch.Tensor,
        gt_poses: torch.Tensor,
        object_points: Optional[torch.Tensor] = None
    ) -> Dict[str, Optional[List[float]]]:
        """
        Evaluate object pose predictions (ADD/ADD-S metrics)
        """
        if object_points is None:
            return {'add': None, 'adds': None}
        
        add_list = []
        adds_list = []
        
        batch_size = pred_poses.shape[0]
        for i in range(batch_size):
            # Extract rotation and translation
            pred_R = pred_poses[i, :3, :3].cpu().numpy()
            pred_t = pred_poses[i, :3, 3].cpu().numpy()
            gt_R = gt_poses[i, :3, :3].cpu().numpy()
            gt_t = gt_poses[i, :3, 3].cpu().numpy()
            
            # Transform object points
            points = object_points[i].cpu().numpy()
            pred_points = points @ pred_R.T + pred_t
            gt_points = points @ gt_R.T + gt_t
            
            # ADD metric
            add = np.mean(np.linalg.norm(pred_points - gt_points, axis=-1))
            add_list.append(add)
            
            # ADD-S metric (symmetric objects)
            adds = self.compute_adds(pred_points, gt_points)
            adds_list.append(adds)
        
        return {
            'add': add_list,
            'adds': adds_list
        }
    
    def evaluate_contact_predictions(
        self,
        pred_contacts: torch.Tensor,
        pred_probs: Optional[torch.Tensor],
        gt_labels: torch.Tensor
    ) -> float:
        """
        Evaluate contact point predictions
        """
        if pred_probs is None:
            # Use distance threshold
            pred_binary = (torch.norm(pred_contacts, dim=-1) < 0.01).float()
        else:
            # Use probability threshold
            pred_binary = (pred_probs > 0.5).float()
        
        # Compute accuracy
        accuracy = (pred_binary == gt_labels).float().mean().item()
        
        return accuracy
    
    def procrustes_align(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """
        Align predicted pose to ground truth using Procrustes analysis
        """
        # Center the points
        pred_centered = pred - pred.mean(axis=0)
        gt_centered = gt - gt.mean(axis=0)
        
        # Compute optimal rotation
        H = pred_centered.T @ gt_centered
        U, _, Vt = np.linalg.svd(H)
        R_opt = Vt.T @ U.T
        
        # Handle reflection
        if np.linalg.det(R_opt) < 0:
            Vt[-1, :] *= -1
            R_opt = Vt.T @ U.T
        
        # Apply transformation
        aligned = pred_centered @ R_opt.T
        
        # Add back centroid
        aligned += gt.mean(axis=0)
        
        return aligned
    
    def compute_adds(self, pred_points: np.ndarray, gt_points: np.ndarray) -> float:
        """
        Compute ADD-S metric for symmetric objects
        """
        # Find closest point for each GT point
        min_distances = []
        for gt_point in gt_points:
            distances = np.linalg.norm(pred_points - gt_point, axis=-1)
            min_distances.append(distances.min())
        
        return np.mean(min_distances)
    
    def compute_auc(
        self,
        errors: List[float],
        min_threshold: float,
        max_threshold: float,
        num_steps: int = 100
    ) -> float:
        """
        Compute Area Under Curve for PCK metric
        """
        errors = np.array(errors)
        thresholds = np.linspace(min_threshold, max_threshold, num_steps)
        pck_values = []
        
        for threshold in thresholds:
            pck = np.mean(errors < threshold)
            pck_values.append(pck)
        
        # Compute AUC using trapezoidal rule
        auc = np.trapz(pck_values, thresholds) / (max_threshold - min_threshold)
        
        return auc
    
    def save_results(self, save_path: str):
        """
        Save evaluation results to file
        """
        # Convert numpy types to Python native types for JSON serialization
        json_results = {}
        for key, value in self.results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, (np.float32, np.float64)):
                json_results[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                json_results[key] = int(value)
            else:
                json_results[key] = value
                
        with open(save_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Saved evaluation results to {save_path}")
    
    def print_summary(self):
        """
        Print evaluation summary
        """
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        if 'hand_mpjpe' in self.results:
            print(f"Hand MPJPE: {self.results['hand_mpjpe']:.2f} mm")
            print(f"Hand PA-MPJPE: {self.results['hand_pa_mpjpe']:.2f} mm")
            print(f"Hand AUC (20-50mm): {self.results.get('hand_auc_20_50', 0):.3f}")
        
        if 'object_add' in self.results:
            print(f"Object ADD: {self.results['object_add']:.2f} mm")
            print(f"Object ADD-S: {self.results['object_adds']:.2f} mm")
        
        if 'contact_accuracy' in self.results:
            print(f"Contact Accuracy: {self.results['contact_accuracy']:.2%}")
        
        # Print worst performing joints
        joint_errors = []
        for i in range(21):
            key = f'joint_{i}_mpjpe'
            if key in self.results:
                joint_errors.append((i, self.results[key]))
        
        if joint_errors:
            joint_errors.sort(key=lambda x: x[1], reverse=True)
            print("\nWorst performing joints:")
            for joint_idx, error in joint_errors[:5]:
                print(f"  Joint {joint_idx}: {error:.2f} mm")
        
        print("="*50 + "\n")