#!/usr/bin/env python3
"""
Minimal W&B Sweep with torch.compile fix
"""

import os
import sys
import wandb
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path

# Setup paths
os.environ['DEX_YCB_DIR'] = '/home/n231/231nProjectV2/dex-ycb-toolkit/data'
project_root = Path('.').absolute()
sys.path.insert(0, str(project_root))

# Import the compile fix
from fixes.torch_compile_fix import optimize_for_h200_with_compile_fix, create_compile_config

def train():
    # Initialize wandb
    run = wandb.init()
    
    # Load default config
    config = OmegaConf.load('configs/default_config.yaml')
    
    # Override with sweep parameters (matching train_full_featured.ipynb parameters)
    config.training.learning_rate = wandb.config.learning_rate
    config.training.batch_size = wandb.config.batch_size
    config.model.hidden_dim = wandb.config.hidden_dim
    config.model.dropout = wandb.config.dropout
    config.model.num_refinement_steps = wandb.config.num_refinement_steps
    config.model.freeze_layers = wandb.config.freeze_layers
    
    # Loss weights
    config.loss.loss_weights.hand_coarse = wandb.config.loss_weight_hand_coarse
    config.loss.loss_weights.hand_refined = wandb.config.loss_weight_hand_refined
    config.loss.loss_weights.object_position = wandb.config.loss_weight_object_position
    config.loss.loss_weights.contact = wandb.config.loss_weight_contact
    config.loss.loss_weights.diversity = wandb.config.loss_weight_diversity
    
    # Additional sweep parameters if they exist
    if hasattr(wandb.config, 'loss_weight_object_rotation'):
        config.loss.loss_weights.object_rotation = wandb.config.loss_weight_object_rotation
    if hasattr(wandb.config, 'loss_weight_physics'):
        config.loss.loss_weights.physics = wandb.config.loss_weight_physics
    if hasattr(wandb.config, 'loss_weight_reprojection'):
        config.loss.loss_weights.reprojection = wandb.config.loss_weight_reprojection
    
    # Augmentation parameters if they exist
    if hasattr(wandb.config, 'aug_rotation_range'):
        config.data.augmentation.rotation_range = wandb.config.aug_rotation_range
    if hasattr(wandb.config, 'aug_translation_std'):
        config.data.augmentation.translation_std = wandb.config.aug_translation_std
    if hasattr(wandb.config, 'aug_scale_min'):
        config.data.augmentation.scale_range = [wandb.config.aug_scale_min, wandb.config.aug_scale_max]
    if hasattr(wandb.config, 'aug_color_jitter'):
        config.data.augmentation.color_jitter = wandb.config.aug_color_jitter
    if hasattr(wandb.config, 'aug_joint_noise_std'):
        config.data.augmentation.joint_noise_std = wandb.config.aug_joint_noise_std
    
    # Other parameters
    if hasattr(wandb.config, 'scheduler_type'):
        config.training.scheduler = wandb.config.scheduler_type
    if hasattr(wandb.config, 'fingertip_weight'):
        config.loss.fingertip_weight = wandb.config.fingertip_weight
    if hasattr(wandb.config, 'per_joint_weighting'):
        config.loss.per_joint_weighting = wandb.config.per_joint_weighting
    if hasattr(wandb.config, 'diversity_margin'):
        config.loss.diversity_margin = wandb.config.diversity_margin
    
    # Training settings
    config.training.weight_decay = wandb.config.weight_decay
    config.training.grad_clip = wandb.config.grad_clip
    config.training.num_epochs = 20  # Shorter for sweep
    config.training.use_wandb = True
    config.training.use_amp = True
    config.training.use_bf16 = True
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Import components
    from models.unified_model import UnifiedManipulationTransformer
    from training.losses import ComprehensiveLoss
    from data.gpu_cached_dataset import create_gpu_cached_dataloaders
    from solutions.mode_collapse import ModeCollapsePreventionModule
    
    try:
        # Create GPU-cached dataloaders (smaller for sweep)
        gpu_config = {
            'gpu_max_samples': wandb.config.get('max_samples', 50000),
            'gpu_max_samples_val': 5000,
            'gpu_cache_path': './gpu_cache_sweep',
            'batch_size': config.training.batch_size,
            'use_bfloat16': config.training.use_bf16,
            'preload_dinov2': False
        }
        
        train_loader, val_loader = create_gpu_cached_dataloaders(gpu_config)
        
        # Create model
        model = UnifiedManipulationTransformer(config.model)
        
        # Apply mode collapse prevention
        if config.optimizations.use_mode_collapse_prevention:
            mode_collapse_config = {
                'noise_std': 0.01,
                'drop_path_rate': 0.1,
                'mixup_alpha': 0.2
            }
            model = ModeCollapsePreventionModule.wrap_model(model, mode_collapse_config)
        
        # Initialize weights (from train_full_featured.ipynb)
        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.xavier_uniform_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
        
        for name, module in model.named_modules():
            if 'dinov2' not in name or 'encoder.layer.' not in name:
                init_weights(module)
        
        # Create compile configuration based on batch size
        compile_config = create_compile_config(config.training.batch_size, config.model)
        
        # Apply optimizations with compile fix
        model = optimize_for_h200_with_compile_fix(model, compile_config)
        model = model.to(device)
        
        # Log that we're using the compile fix
        wandb.log({
            'compile_fix_applied': True,
            'compile_mode': compile_config['compile_mode']
        })
        
        # Create optimizer with parameter groups (from train_full_featured.ipynb)
        dinov2_params = []
        encoder_params = []
        decoder_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'dinov2' in name:
                dinov2_params.append(param)
            elif 'decoder' in name:
                decoder_params.append(param)
            elif 'encoder' in name:
                encoder_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = []
        if dinov2_params:
            param_groups.append({
                'params': dinov2_params,
                'lr': config.training.learning_rate * 0.01,
                'name': 'dinov2'
            })
        if encoder_params:
            param_groups.append({
                'params': encoder_params,
                'lr': config.training.learning_rate * 0.5,
                'name': 'encoders'
            })
        if decoder_params:
            param_groups.append({
                'params': decoder_params,
                'lr': config.training.learning_rate,
                'name': 'decoders'
            })
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': config.training.learning_rate,
                'name': 'other'
            })
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=config.training.weight_decay, fused=True)
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Loss function
        criterion = ComprehensiveLoss(config.loss)
        
        # Training loop (simplified from train_full_featured.ipynb)
        best_val_mpjpe = float('inf')
        
        for epoch in range(config.training.num_epochs):
            # Update loss epoch
            criterion.set_epoch(epoch)
            
            # Train
            model.train()
            train_loss = 0
            train_samples = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Convert BFloat16 images to Float32 for DINOv2
                if batch['image'].dtype == torch.bfloat16:
                    batch['image'] = batch['image'].float()
                
                optimizer.zero_grad()
                
                # Forward pass
                try:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        outputs = model(batch)
                        losses = criterion(outputs, batch)
                        loss = losses['total'] if isinstance(losses, dict) else losses
                except RuntimeError as e:
                    # If we get a compile error, log it and skip this batch
                    if "Tensors must have same number of dimensions" in str(e):
                        wandb.log({
                            'compile_error': True,
                            'error_message': str(e),
                            'batch_idx': batch_idx,
                            'epoch': epoch
                        })
                        continue
                    else:
                        raise e
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
                optimizer.step()
                
                train_loss += loss.item() * batch['image'].shape[0]
                train_samples += batch['image'].shape[0]
                
                # Log batch metrics
                if batch_idx % 50 == 0:
                    wandb.log({
                        'train/batch_loss': loss.item(),
                        'train/lr': optimizer.param_groups[0]['lr']
                    })
            
            train_loss /= train_samples
            
            # Validation
            model.eval()
            val_loss = 0
            val_mpjpe = 0
            val_samples = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    if batch['image'].dtype == torch.bfloat16:
                        batch['image'] = batch['image'].float()
                    
                    try:
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                            outputs = model(batch)
                            losses = criterion(outputs, batch)
                            loss = losses['total'] if isinstance(losses, dict) else losses
                    except RuntimeError as e:
                        if "Tensors must have same number of dimensions" in str(e):
                            continue
                        else:
                            raise e
                    
                    val_loss += loss.item() * batch['image'].shape[0]
                    
                    # Compute MPJPE
                    if 'hand_joints' in outputs:
                        pred = outputs['hand_joints']
                        gt = batch['joints_3d']
                        mpjpe = ((pred - gt).norm(dim=-1)).mean()
                        val_mpjpe += mpjpe.item() * batch['image'].shape[0]
                    
                    val_samples += batch['image'].shape[0]
            
            val_loss /= val_samples
            val_mpjpe /= val_samples
            
            # Update best
            if val_mpjpe < best_val_mpjpe:
                best_val_mpjpe = val_mpjpe
            
            # Log epoch metrics
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'val/loss': val_loss,
                'val/mpjpe': val_mpjpe,
                'val/best_mpjpe': best_val_mpjpe
            })
            
            # Update scheduler
            scheduler.step()
        
        # Final logging
        wandb.log({
            'final/best_val_mpjpe': best_val_mpjpe,
            'final/train_loss': train_loss,
            'final/val_loss': val_loss
        })
        
    except Exception as e:
        # Log any errors
        wandb.log({
            'error': str(e),
            'error_type': type(e).__name__
        })
        raise e
    
    finally:
        wandb.finish()


if __name__ == '__main__':
    train()