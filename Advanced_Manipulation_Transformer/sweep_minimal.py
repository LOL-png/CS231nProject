#!/usr/bin/env python3
"""
Minimal W&B Sweep - Based on train_full_featured.ipynb
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
    from optimizations.pytorch_native_optimization import optimize_for_h200
    
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
        
        # Apply optimizations
        model = optimize_for_h200(model, compile_mode='reduce-overhead')
        model = model.to(device)
        
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
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(batch)
                    losses = criterion(outputs, batch)
                    loss = losses['total'] if isinstance(losses, dict) else losses
                
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
                    
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        outputs = model(batch)
                        losses = criterion(outputs, batch)
                        loss = losses['total'] if isinstance(losses, dict) else losses
                    
                    batch_size = batch['image'].shape[0]
                    val_samples += batch_size
                    val_loss += loss.item() * batch_size
                    
                    if 'hand_joints' in outputs and 'hand_joints' in batch:
                        mpjpe = torch.norm(outputs['hand_joints'] - batch['hand_joints'], dim=-1).mean()
                        val_mpjpe += mpjpe.item() * 1000 * batch_size
            
            val_loss /= val_samples
            val_mpjpe /= val_samples
            
            # Update best
            if val_mpjpe < best_val_mpjpe:
                best_val_mpjpe = val_mpjpe
            
            # Update scheduler
            scheduler.step()
            
            # Log epoch metrics
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'val/loss': val_loss,
                'val/hand_mpjpe': val_mpjpe,
                'val/best_mpjpe': best_val_mpjpe
            })
            
            print(f"Epoch {epoch}: val_mpjpe={val_mpjpe:.2f}mm (best={best_val_mpjpe:.2f}mm)")
        
        # Log final metrics
        wandb.log({
            'final/best_val_mpjpe': best_val_mpjpe,
            'final/epochs_trained': config.training.num_epochs
        })
        
    except Exception as e:
        print(f"Training failed: {e}")
        wandb.log({'error': str(e), 'val/hand_mpjpe': 1000.0})
        raise

# Sweep configuration matching train_full_featured.ipynb
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val/hand_mpjpe',
        'goal': 'minimize'
    },
    'parameters': {
        # Main hyperparameters from notebook
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,
            'max': 5e-3
        },
        'batch_size': {
            'values': [16, 32, 64]
        },
        'hidden_dim': {
            'values': [512, 768, 1024]
        },
        'dropout': {
            'values': [0.1, 0.2, 0.3]
        },
        'num_refinement_steps': {
            'values': [1, 2, 3]
        },
        'freeze_layers': {
            'values': [8, 10, 12]
        },
        
        # Loss weights (key ones from notebook)
        'loss_weight_hand_coarse': {
            'distribution': 'uniform',
            'min': 0.8,
            'max': 1.2
        },
        'loss_weight_hand_refined': {
            'distribution': 'uniform',
            'min': 1.0,
            'max': 1.5
        },
        'loss_weight_object_position': {
            'distribution': 'uniform',
            'min': 0.8,
            'max': 1.2
        },
        'loss_weight_contact': {
            'distribution': 'uniform',
            'min': 0.2,
            'max': 0.5
        },
        'loss_weight_diversity': {
            'distribution': 'log_uniform_values',
            'min': 0.005,
            'max': 0.05
        },
        
        # Optimizer settings
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 0.001,
            'max': 0.1
        },
        'grad_clip': {
            'values': [0.5, 1.0, 2.0]
        },
        
        # Dataset size for sweep
        'max_samples': {
            'values': [50000, 100000]
        }
    }
}

if __name__ == '__main__':
    # Run sweep
    project_name = 'amt-minimal'
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    print(f"Starting sweep {sweep_id}")
    print(f"View at: https://wandb.ai/{wandb.api.default_entity}/{project_name}/sweeps/{sweep_id}")
    wandb.agent(sweep_id, train, count=10)