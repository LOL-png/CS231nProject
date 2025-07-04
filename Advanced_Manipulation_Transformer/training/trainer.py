import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import numpy as np
from pathlib import Path
import time
from typing import Dict, Optional, Tuple
import logging

# Import loss function
from .losses import ManipulationLoss

logger = logging.getLogger(__name__)

class ExponentialMovingAverage:
    """EMA for model parameters"""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply(self):
        """Apply shadow parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

class ManipulationTrainer:
    """
    Advanced training loop with all optimizations for H200
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        config: Dict, 
        device: str = 'cuda',
        distributed: bool = False,
        local_rank: int = 0
    ):
        self.model = model
        self.config = config
        self.device = device
        self.distributed = distributed
        self.local_rank = local_rank
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Setup distributed training if needed
        if distributed:
            self.model = DDP(
                self.model, 
                device_ids=[local_rank], 
                output_device=local_rank,
                find_unused_parameters=True
            )
        
        # Optimizer with different LR for different components
        self.optimizer = self._build_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._build_scheduler()
        
        # Mixed precision training (BF16 for H200)
        self.use_amp = config.get('use_amp', True)
        self.amp_dtype = torch.bfloat16 if config.get('use_bf16', True) else torch.float16
        self.scaler = GradScaler(enabled=self.use_amp and self.amp_dtype == torch.float16)
        
        # Loss function
        self.criterion = ManipulationLoss(config)
        
        # Gradient clipping value
        self.grad_clip = config.get('grad_clip', 1.0)
        
        # EMA for stable training
        self.ema = ExponentialMovingAverage(
            self.model, 
            decay=config.get('ema_decay', 0.999)
        )
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_mpjpe = float('inf')
        
        # Logging
        self.log_freq = config.get('log_freq', 100)
        self.val_freq = config.get('val_freq', 1000)
        
        # Gradient accumulation
        self.accumulation_steps = config.get('accumulation_steps', 1)
        
        # Setup logging
        if self.local_rank == 0:
            wandb.init(
                project=config.get('wandb_project', '231nProject'),
                config=config,
                name=config.get('experiment_name', 'default'),
                mode='offline'  # Use offline mode for now
            )
    
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """
        Build optimizer with different learning rates for different components
        Critical for stable training with pretrained components
        """
        # Separate parameters by component
        param_groups = []
        
        # DINOv2 parameters (lower LR for fine-tuning)
        dinov2_params = []
        dinov2_proj_params = []
        
        # Hand encoder parameters
        hand_encoder_params = []
        
        # Other parameters
        other_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'image_encoder.dinov2' in name:
                dinov2_params.append(param)
            elif 'image_encoder' in name and 'dinov2' not in name:
                dinov2_proj_params.append(param)
            elif 'hand_encoder' in name:
                hand_encoder_params.append(param)
            else:
                other_params.append(param)
        
        # Create parameter groups with different learning rates
        # Handle both scalar and list learning rates
        lr_config = self.config.get('learning_rate', 1e-3)
        if isinstance(lr_config, list):
            base_lr = lr_config[0] if lr_config else 1e-3
        else:
            base_lr = lr_config
        
        param_groups = [
            # DINOv2 backbone (very low LR)
            {
                'params': dinov2_params,
                'lr': base_lr * 0.01,  # 1% of base LR
                'weight_decay': 0.01,
                'name': 'dinov2'
            },
            # DINOv2 projection layers
            {
                'params': dinov2_proj_params,
                'lr': base_lr,
                'weight_decay': 0.05,
                'name': 'dinov2_proj'
            },
            # Hand encoder
            {
                'params': hand_encoder_params,
                'lr': base_lr * 0.5,  # 50% of base LR
                'weight_decay': 0.05,
                'name': 'hand_encoder'
            },
            # Other parameters
            {
                'params': other_params,
                'lr': base_lr,
                'weight_decay': 0.05,
                'name': 'other'
            }
        ]
        
        # Filter out empty groups
        param_groups = [g for g in param_groups if len(g['params']) > 0]
        
        # Log parameter counts
        for group in param_groups:
            param_count = sum(p.numel() for p in group['params'])
            logger.info(f"{group['name']}: {param_count/1e6:.2f}M parameters, LR: {group['lr']}")
        
        # Create optimizer (AdamW with weight decay)
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0  # Set per group above
        )
        
        return optimizer
    
    def _build_scheduler(self):
        """Build learning rate scheduler"""
        scheduler_type = self.config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.get('T_0', 10),  # Restart every 10 epochs
                T_mult=self.config.get('T_mult', 2),  # Double period after each restart
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'onecycle':
            # OneCycle for faster convergence
            scheduler = OneCycleLR(
                self.optimizer,
                max_lr=[g['lr'] for g in self.optimizer.param_groups],
                epochs=self.config.get('num_epochs', 100),
                steps_per_epoch=self.config.get('steps_per_epoch', 1000),
                pct_start=0.1,  # 10% warmup
                anneal_strategy='cos'
            )
        else:
            # Simple StepLR
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        
        return scheduler
    
    def train_step(self, batch: Dict) -> Tuple[Dict, Dict[str, float]]:
        """
        Single training step
        
        Returns:
            outputs: Model predictions
            metrics: Dictionary of metrics including loss
        """
        # Forward pass
        with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            # Pass the batch dictionary directly - model will handle extraction
            predictions = self.model(batch)
            
            # Compute losses
            losses = self.criterion(predictions, batch)
            
            # Calculate total loss
            total_loss = torch.tensor(0.0, device=batch['image'].device)
            for loss_name, loss_value in losses.items():
                if loss_name != 'total' and isinstance(loss_value, torch.Tensor):
                    weight = self.criterion.get_weight(loss_name) if hasattr(self.criterion, 'get_weight') else 1.0
                    total_loss = total_loss + weight * loss_value
            
            losses['total'] = total_loss
            total_loss = total_loss / self.accumulation_steps
        
        # Backward pass
        if self.use_amp and self.amp_dtype == torch.float16:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        
        # Update weights if accumulation complete
        if (self.global_step + 1) % self.accumulation_steps == 0:
            # Gradient clipping
            if self.use_amp and self.amp_dtype == torch.float16:
                self.scaler.unscale_(self.optimizer)
            
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.grad_clip
            )
            
            # Optimizer step
            if self.use_amp and self.amp_dtype == torch.float16:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Update EMA
            self.ema.update()
        
        # Update global step
        self.global_step += 1
        
        # Prepare metrics
        metrics = {
            'loss': losses['total'].item(),
            'hand_mpjpe': losses.get('hand_joints_3d', 0.0),
            'object_add': losses.get('object_position', 0.0),
            'grad_norm': grad_norm.item() if 'grad_norm' in locals() else 0.0
        }
        
        return predictions, metrics
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch with proper gradient accumulation and mixed precision
        """
        self.model.train()
        self.epoch = epoch
        
        # Metrics
        total_loss = 0
        loss_components = {}
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(
            train_loader, 
            desc=f'Epoch {epoch}',
            disable=self.local_rank != 0  # Only show on main process
        )
        
        # Training loop
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Adjust learning rate if using OneCycle
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                predictions = self.model(
                    images=batch['image'],
                    mano_vertices=batch.get('mano_vertices'),
                    camera_params={
                        'intrinsics': batch['camera_intrinsics'],
                        'extrinsics': batch.get('camera_extrinsics')
                    } if 'camera_intrinsics' in batch else None
                )
                
                # Compute losses
                losses = self.criterion(
                    predictions, 
                    batch, 
                    self.model,
                    epoch=epoch
                )
                
                # Scale loss for gradient accumulation
                loss = losses['total'] / self.accumulation_steps
            
            # Backward pass
            if self.use_amp and self.amp_dtype == torch.float16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Unscale gradients for clipping
                if self.use_amp and self.amp_dtype == torch.float16:
                    self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip
                )
                
                # Check for NaN gradients
                if torch.isnan(grad_norm):
                    logger.warning(f"NaN gradient detected at step {self.global_step}")
                    self.optimizer.zero_grad()
                    continue
                
                # Optimizer step
                if self.use_amp and self.amp_dtype == torch.float16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Update EMA
                self.ema.update()
                
                # Update global step
                self.global_step += 1
                
                # Logging
                if self.global_step % self.log_freq == 0 and self.local_rank == 0:
                    # Get current learning rates
                    lrs = {f'lr/{g["name"]}': g['lr'] for g in self.optimizer.param_groups}
                    
                    # Log to wandb
                    log_dict = {
                        'train/loss': loss.item() * self.accumulation_steps,
                        'train/grad_norm': grad_norm.item(),
                        'train/epoch': epoch,
                        'train/step': self.global_step,
                        **lrs
                    }
                    
                    # Add loss components
                    for k, v in losses.items():
                        if k != 'total' and isinstance(v, torch.Tensor):
                            log_dict[f'train/{k}'] = v.item()
                    
                    wandb.log(log_dict, step=self.global_step)
            
            # Update metrics
            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            
            # Update loss components
            for k, v in losses.items():
                if k != 'total' and isinstance(v, torch.Tensor):
                    if k not in loss_components:
                        loss_components[k] = 0
                    loss_components[k] += v.item()
            
            # Update progress bar
            if self.local_rank == 0:
                pbar.set_postfix({
                    'loss': f"{loss.item() * self.accumulation_steps:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    'grad': f"{grad_norm.item() if 'grad_norm' in locals() else 0:.2f}"
                })
        
        # Step scheduler (if not OneCycle)
        if not isinstance(self.scheduler, OneCycleLR):
            self.scheduler.step()
        
        # Average metrics
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components.items()}
        
        return {'loss': avg_loss, **avg_components}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validation with EMA model
        """
        self.model.eval()
        self.ema.apply()  # Use EMA weights
        
        # Metrics
        total_loss = 0
        all_mpjpe = []
        all_pa_mpjpe = []
        all_pck = []
        num_batches = 0
        
        # Disable gradient computation
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation', disable=self.local_rank != 0):
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                    predictions = self.model(
                        images=batch['image'],
                        camera_params={
                            'intrinsics': batch['camera_intrinsics'],
                            'extrinsics': batch.get('camera_extrinsics')
                        } if 'camera_intrinsics' in batch else None
                    )
                    
                    # Compute losses
                    losses = self.criterion(predictions, batch)
                
                # Update metrics
                total_loss += losses['total'].item()
                num_batches += 1
                
                # Compute evaluation metrics
                pred_joints = predictions['hand']['joints_3d']
                if 'joints_3d_refined' in predictions['hand']:
                    pred_joints = predictions['hand']['joints_3d_refined']
                
                target_joints = batch['hand_joints_3d']
                
                # MPJPE (Mean Per Joint Position Error)
                mpjpe = torch.norm(pred_joints - target_joints, dim=-1).mean(dim=-1)
                all_mpjpe.extend(mpjpe.cpu().numpy())
                
                # PA-MPJPE (Procrustes Aligned)
                for i in range(pred_joints.shape[0]):
                    pa_mpjpe = compute_pa_mpjpe(
                        pred_joints[i].cpu().numpy(),
                        target_joints[i].cpu().numpy()
                    )
                    all_pa_mpjpe.append(pa_mpjpe)
                
                # PCK (Percentage of Correct Keypoints)
                if 'hand_joints_2d' in batch:
                    pck = compute_pck_batch(
                        pred_joints,
                        target_joints,
                        batch['camera_intrinsics'],
                        threshold=5.0  # 5 pixels
                    )
                    all_pck.extend(pck.cpu().numpy())
        
        # Restore original weights
        self.ema.restore()
        
        # Aggregate metrics
        metrics = {
            'val_loss': total_loss / num_batches,
            'val_mpjpe': np.mean(all_mpjpe) * 1000,  # Convert to mm
            'val_pa_mpjpe': np.mean(all_pa_mpjpe) * 1000,
            'val_mpjpe_std': np.std(all_mpjpe) * 1000
        }
        
        if all_pck:
            metrics['val_pck'] = np.mean(all_pck)
        
        # Log to wandb
        if self.local_rank == 0:
            wandb.log(metrics, step=self.global_step)
        
        return metrics
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        if self.local_rank != 0:
            return
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'ema_state_dict': self.ema.shadow,
            'metrics': metrics,
            'config': self.config
        }
        
        # Save paths
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save latest checkpoint
        latest_path = checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint with MPJPE: {metrics['val_mpjpe']:.2f}mm")
        
        # Save periodic checkpoints
        if self.epoch % self.config.get('save_freq', 10) == 0:
            epoch_path = checkpoint_dir / f'epoch_{self.epoch}.pth'
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load EMA state
        if 'ema_state_dict' in checkpoint:
            self.ema.shadow = checkpoint['ema_state_dict']
        
        # Load training state
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        
        # Load metrics
        metrics = checkpoint.get('metrics', {})
        self.best_val_mpjpe = metrics.get('val_mpjpe', float('inf'))
        
        logger.info(f"Resumed from epoch {self.epoch}, step {self.global_step}")
        
        return metrics
    
    def get_lr(self) -> list:
        """Get current learning rates for all parameter groups"""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def get_gradient_norm(self) -> float:
        """Compute gradient norm for monitoring"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def scheduler_step(self, val_loss: float = None):
        """Step the learning rate scheduler"""
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch data to device"""
        moved_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(self.device, non_blocking=True)
            elif isinstance(value, dict):
                moved_batch[key] = self._move_batch_to_device(value)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                moved_batch[key] = [v.to(self.device, non_blocking=True) for v in value]
            else:
                moved_batch[key] = value
        
        return moved_batch

# Helper functions for evaluation metrics
def compute_pa_mpjpe(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Procrustes-aligned MPJPE
    Aligns prediction to target before computing error
    """
    from scipy.spatial.transform import Rotation
    
    # Center the point clouds
    pred_centered = pred - pred.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    
    # Compute optimal rotation using SVD
    H = pred_centered.T @ target_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute optimal scale
    scale = np.trace(R @ H) / np.sum(pred_centered ** 2)
    
    # Apply transformation
    pred_aligned = scale * (pred_centered @ R.T)
    
    # Compute error
    error = np.linalg.norm(pred_aligned - target_centered, axis=-1).mean()
    
    return error

def compute_pck_batch(
    pred_joints: torch.Tensor,
    target_joints: torch.Tensor,
    intrinsics: torch.Tensor,
    threshold: float = 5.0
) -> torch.Tensor:
    """
    Compute PCK (Percentage of Correct Keypoints) in 2D
    """
    # Project to 2D
    pred_2d = project_3d_to_2d_batch(pred_joints, intrinsics)
    target_2d = project_3d_to_2d_batch(target_joints, intrinsics)
    
    # Compute distances
    dists = torch.norm(pred_2d - target_2d, dim=-1)  # [B, 21]
    
    # Compute PCK
    pck = (dists < threshold).float().mean(dim=-1)  # [B]
    
    return pck

def project_3d_to_2d_batch(joints_3d: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Project 3D joints to 2D"""
    joints_2d_homo = torch.matmul(joints_3d, intrinsics.transpose(-1, -2))
    joints_2d = joints_2d_homo[..., :2] / joints_2d_homo[..., 2:3].clamp(min=0.1)
    return joints_2d