"""
Multi-stage Trainer for Video-to-Manipulation Transformer
Implements the three-stage training strategy described in CLAUDE.md
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import os
from tqdm import tqdm
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging disabled.")

from models.full_model import VideoManipulationTransformer
from data.dexycb_dataset import DexYCBSequenceDataset
from physics.mujoco_sim import DifferentiableSimulator
from control.mlp_retargeting import AllegroRetargeting
from control.ur_kinematics import URKinematics


class MultiStageTrainer:
    """
    Implements the three-stage training strategy:
    1. Encoder pre-training with direct supervision
    2. Frozen encoder training with action decoder
    3. End-to-end fine-tuning with physics gradients
    """
    
    def __init__(self,
                 model: VideoManipulationTransformer,
                 config: Dict,
                 device: str = 'cuda',
                 use_wandb: bool = True):
        """
        Args:
            model: The full transformer model
            config: Training configuration dictionary
            device: Device to train on
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        
        # Initialize components
        self.simulator = DifferentiableSimulator(
            model_path=config.get('mujoco_model_path'),
            device=device
        )
        
        self.allegro_retargeting = AllegroRetargeting().to(device)
        self.ur_kinematics = URKinematics(robot_model=config.get('ur_model', 'ur5'))
        
        # Optimizers for different stages
        self.optimizers = self._create_optimizers()
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        
        # Training state
        self.current_stage = 1
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.init(project="video-manipulation-transformer", config=config)
            
    def _create_optimizers(self) -> Dict[str, optim.Optimizer]:
        """Create optimizers for different training stages"""
        optimizers = {}
        
        # Stage 1: Encoder pre-training
        encoder_params = []
        for encoder in [self.model.hand_encoder, self.model.object_encoder, self.model.contact_encoder]:
            encoder_params.extend(encoder.parameters())
            
        optimizers['stage1'] = optim.AdamW(
            encoder_params,
            lr=self.config.get('stage1_lr', 1e-4),
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        # Stage 2: Frozen encoder training
        decoder_params = []
        decoder_params.extend(self.model.temporal_fusion.parameters())
        decoder_params.extend(self.model.action_decoder.parameters())
        decoder_params.extend(self.allegro_retargeting.parameters())
        
        optimizers['stage2'] = optim.AdamW(
            decoder_params,
            lr=self.config.get('stage2_lr', 5e-5),
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        # Stage 3: End-to-end fine-tuning
        optimizers['stage3'] = optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('stage3_lr', 1e-5),
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        return optimizers
    
    def setup_data_loaders(self, batch_size: int = 32):
        """Setup data loaders for training and validation"""
        train_dataset = DexYCBSequenceDataset(
            split='s0_train',
            sequence_length=self.config.get('sequence_length', 16),
            stride=self.config.get('frame_stride', 1)
        )
        
        val_dataset = DexYCBSequenceDataset(
            split='s0_val',
            sequence_length=self.config.get('sequence_length', 16),
            stride=self.config.get('frame_stride', 1)
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
    def train(self, total_epochs: Dict[str, int]):
        """
        Run the complete three-stage training
        
        Args:
            total_epochs: Dictionary with epochs for each stage
                        e.g., {'stage1': 20, 'stage2': 15, 'stage3': 15}
        """
        print(f"Starting multi-stage training on {self.device}")
        
        # Stage 1: Encoder pre-training
        print("\n=== Stage 1: Encoder Pre-training ===")
        self.current_stage = 1
        self._train_stage(
            num_epochs=total_epochs['stage1'],
            optimizer=self.optimizers['stage1'],
            loss_fn=self._compute_stage1_loss
        )
        
        # Stage 2: Frozen encoder training
        print("\n=== Stage 2: Frozen Encoder Training ===")
        self.current_stage = 2
        self.model.freeze_encoders()
        self._train_stage(
            num_epochs=total_epochs['stage2'],
            optimizer=self.optimizers['stage2'],
            loss_fn=self._compute_stage2_loss
        )
        
        # Stage 3: End-to-end fine-tuning
        print("\n=== Stage 3: End-to-End Fine-tuning ===")
        self.current_stage = 3
        self.model.unfreeze_all()
        self._train_stage(
            num_epochs=total_epochs['stage3'],
            optimizer=self.optimizers['stage3'],
            loss_fn=self._compute_stage3_loss
        )
        
        print("\nTraining completed!")
        
    def _train_stage(self, num_epochs: int, optimizer: optim.Optimizer, loss_fn):
        """Train a single stage"""
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training loop
            train_loss = self._train_epoch(optimizer, loss_fn)
            
            # Validation loop
            val_loss = self._validate_epoch(loss_fn)
            
            # Logging
            print(f"Stage {self.current_stage} - Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if self.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    f'stage{self.current_stage}/train_loss': train_loss,
                    f'stage{self.current_stage}/val_loss': val_loss,
                    'epoch': self.current_epoch
                })
                
            # Save checkpoint if best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(f'best_stage{self.current_stage}.pth')
                
    def _train_epoch(self, optimizer: optim.Optimizer, loss_fn) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Training Stage {self.current_stage}")
        
        for batch in progress_bar:
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(batch, return_intermediates=True)
            
            # Compute loss
            loss_dict = loss_fn(outputs, batch)
            loss = loss_dict['total']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.get('grad_clip', 1.0)
            )
            
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
        return total_loss / num_batches
    
    def _validate_epoch(self, loss_fn) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(batch, return_intermediates=True)
                
                # Compute loss
                loss_dict = loss_fn(outputs, batch)
                loss = loss_dict['total']
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches
    
    def _compute_stage1_loss(self, outputs: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        """Compute loss for Stage 1: Encoder pre-training"""
        losses = {}
        
        # Hand encoder loss
        hand_loss = nn.MSELoss()(
            outputs.get('hand_joints_3d', torch.zeros_like(batch['hand_joints_3d'])),
            batch['hand_joints_3d']
        )
        losses['hand_loss'] = hand_loss
        
        # Object encoder loss
        if 'object_poses' in batch:
            object_loss = nn.MSELoss()(
                outputs.get('object_positions', torch.zeros_like(batch['object_poses'][..., :3, 3])),
                batch['object_poses'][..., :3, 3]
            )
            losses['object_loss'] = object_loss
            
        # Contact loss (if we have contact annotations)
        # This would require contact ground truth in DexYCB
        
        losses['total'] = sum(losses.values())
        return losses
    
    def _compute_stage2_loss(self, outputs: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        """Compute loss for Stage 2: Frozen encoder training"""
        losses = {}
        
        # Action decoder losses
        action_losses = self.model.action_decoder.compute_loss(outputs, batch)
        losses.update(action_losses)
        
        # Robot control losses
        if 'ee_target_poses' in outputs:
            # Convert to joint commands
            joint_commands = self._actions_to_joints(outputs)
            
            # Simulate
            sim_outputs = self.simulator.forward(
                joint_commands,
                object_poses=batch.get('object_poses')
            )
            
            # Physics losses
            physics_losses = self.simulator.compute_loss(sim_outputs, batch)
            for k, v in physics_losses.items():
                losses[f'physics_{k}'] = v * self.config.get('physics_weight', 0.1)
                
        losses['total'] = sum(v for k, v in losses.items() if 'total' not in k)
        return losses
    
    def _compute_stage3_loss(self, outputs: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        """Compute loss for Stage 3: End-to-end fine-tuning"""
        losses = {}
        
        # All losses from stage 1 and 2
        stage1_losses = self._compute_stage1_loss(outputs, batch)
        stage2_losses = self._compute_stage2_loss(outputs, batch)
        
        # Combine with appropriate weights
        for k, v in stage1_losses.items():
            if k != 'total':
                losses[f'encoder_{k}'] = v * self.config.get('encoder_weight', 0.1)
                
        for k, v in stage2_losses.items():
            if k != 'total':
                losses[k] = v
                
        losses['total'] = sum(v for k, v in losses.items() if 'total' not in k)
        return losses
    
    def _actions_to_joints(self, outputs: Dict) -> torch.Tensor:
        """Convert action decoder outputs to joint commands"""
        B = outputs['ee_target_poses'].shape[0]
        T = outputs['ee_target_poses'].shape[1]
        
        joint_commands = torch.zeros(B, T, 22, device=self.device)
        
        for b in range(B):
            for t in range(T):
                # UR arm IK
                ur_result = self.ur_kinematics.inverse_kinematics(
                    outputs['ee_target_poses'][b, t].unsqueeze(0)
                )
                
                if ur_result['success'][0]:
                    joint_commands[b, t, :6] = ur_result['joint_angles'][0]
                    
                # Allegro hand retargeting
                allegro_result = self.allegro_retargeting(
                    outputs['hand_configs'][b, t].unsqueeze(0),
                    outputs['grasp_types'][b, t].unsqueeze(0)
                )
                joint_commands[b, t, 6:] = allegro_result['joint_angles'][0]
                
        return joint_commands
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch data to device"""
        moved_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                moved_batch[k] = v.to(self.device)
            else:
                moved_batch[k] = v
        return moved_batch
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'allegro_retargeting_state_dict': self.allegro_retargeting.state_dict(),
            'optimizer_state_dict': self.optimizers[f'stage{self.current_stage}'].state_dict(),
            'stage': self.current_stage,
            'epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        save_path = os.path.join(self.config.get('checkpoint_dir', 'checkpoints'), filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")
        
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.allegro_retargeting.load_state_dict(checkpoint['allegro_retargeting_state_dict'])
        
        self.current_stage = checkpoint['stage']
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Checkpoint loaded from {filepath}")
        
        
class H200Optimizer:
    """
    Optimizations specific to H200 GPU
    """
    
    @staticmethod
    def setup_mixed_precision():
        """Setup mixed precision training for H200"""
        # Use BF16 for better numerical stability on H200
        from torch.cuda.amp import GradScaler
        scaler = GradScaler('cuda')
        return scaler
    
    @staticmethod
    def setup_gradient_checkpointing(model: nn.Module):
        """Enable gradient checkpointing for memory efficiency"""
        # Enable for transformer layers
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True
                
    @staticmethod
    def optimize_data_loading(loader: DataLoader):
        """Optimize data loading for H200"""
        # Prefetch to GPU
        loader.pin_memory = True
        loader.num_workers = min(16, os.cpu_count())
        loader.prefetch_factor = 2