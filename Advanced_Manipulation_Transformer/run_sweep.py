#!/usr/bin/env python3
"""
W&B Sweep Runner for Advanced Manipulation Transformer

This script loads the sweep configuration, initializes a W&B sweep,
modifies the training configuration based on sweep parameters,
and runs the training with proper logging.
"""

import os
import sys
import torch
import wandb
import yaml
import argparse
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import training components
from models.unified_model import UnifiedManipulationTransformer
from training.trainer import ManipulationTrainer
from training.losses import ComprehensiveLoss
from evaluation.evaluator import ComprehensiveEvaluator
from data.gpu_cached_dataset import create_gpu_cached_dataloaders
from solutions.mode_collapse import ModeCollapsePreventionModule
from optimizations.pytorch_native_optimization import optimize_for_h200


def load_base_config():
    """Load the base configuration file"""
    config_path = project_root / "configs" / "default_config.yaml"
    config = OmegaConf.load(config_path)
    return config


def update_config_from_sweep(config, sweep_config):
    """Update configuration with sweep parameters"""
    # Learning and optimization
    config.training.learning_rate = sweep_config.get('learning_rate', config.training.learning_rate)
    config.training.batch_size = sweep_config.get('batch_size', config.training.batch_size)
    config.training.weight_decay = sweep_config.get('weight_decay', config.training.weight_decay)
    config.training.grad_clip = sweep_config.get('grad_clip', config.training.grad_clip)
    config.training.ema_decay = sweep_config.get('ema_decay', config.training.ema_decay)
    
    # Model architecture
    config.model.hidden_dim = sweep_config.get('hidden_dim', config.model.hidden_dim)
    config.model.contact_hidden_dim = sweep_config.get('contact_hidden_dim', config.model.contact_hidden_dim)
    config.model.dropout = sweep_config.get('dropout', config.model.dropout)
    config.model.num_refinement_steps = sweep_config.get('num_refinement_steps', config.model.num_refinement_steps)
    config.model.freeze_layers = sweep_config.get('freeze_layers', config.model.freeze_layers)
    config.model.use_attention_fusion = sweep_config.get('use_attention_fusion', config.model.use_attention_fusion)
    config.model.use_sigma_reparam = sweep_config.get('use_sigma_reparam', config.model.use_sigma_reparam)
    
    # Loss weights
    config.loss.loss_weights.hand_coarse = sweep_config.get('loss_weight_hand_coarse', config.loss.loss_weights.hand_coarse)
    config.loss.loss_weights.hand_refined = sweep_config.get('loss_weight_hand_refined', config.loss.loss_weights.hand_refined)
    config.loss.loss_weights.object_position = sweep_config.get('loss_weight_object_position', config.loss.loss_weights.object_position)
    config.loss.loss_weights.object_rotation = sweep_config.get('loss_weight_object_rotation', config.loss.loss_weights.object_rotation)
    config.loss.loss_weights.contact = sweep_config.get('loss_weight_contact', config.loss.loss_weights.contact)
    config.loss.loss_weights.physics = sweep_config.get('loss_weight_physics', config.loss.loss_weights.physics)
    config.loss.loss_weights.diversity = sweep_config.get('loss_weight_diversity', config.loss.loss_weights.diversity)
    config.loss.loss_weights.reprojection = sweep_config.get('loss_weight_reprojection', config.loss.loss_weights.reprojection)
    config.loss.loss_weights.kl = sweep_config.get('loss_weight_kl', config.loss.loss_weights.kl)
    
    # Multi-rate learning
    config.training.multi_rate.pretrained = sweep_config.get('lr_mult_pretrained', config.training.multi_rate.pretrained)
    config.training.multi_rate.new_encoders = sweep_config.get('lr_mult_new_encoders', config.training.multi_rate.new_encoders)
    
    # Scheduler settings
    config.training.T_0 = sweep_config.get('scheduler_T_0', config.training.T_0)
    config.training.min_lr = sweep_config.get('scheduler_min_lr', config.training.min_lr)
    
    # Augmentation settings
    config.data.augmentation.rotation_range = sweep_config.get('aug_rotation_range', config.data.augmentation.rotation_range)
    config.data.augmentation.scale_range = [
        sweep_config.get('aug_scale_min', config.data.augmentation.scale_range[0]),
        sweep_config.get('aug_scale_max', config.data.augmentation.scale_range[1])
    ]
    config.data.augmentation.translation_std = sweep_config.get('aug_translation_std', config.data.augmentation.translation_std)
    config.data.augmentation.color_jitter = sweep_config.get('aug_color_jitter', config.data.augmentation.color_jitter)
    config.data.augmentation.joint_noise_std = sweep_config.get('aug_joint_noise_std', config.data.augmentation.joint_noise_std)
    
    # Memory optimization
    config.optimizations.memory.gradient_checkpointing = sweep_config.get('gradient_checkpointing', config.optimizations.memory.gradient_checkpointing)
    config.optimizations.memory.checkpoint_ratio = sweep_config.get('checkpoint_ratio', config.optimizations.memory.checkpoint_ratio)
    
    # Mixed precision
    config.training.use_amp = sweep_config.get('use_amp', config.training.use_amp)
    config.training.use_bf16 = sweep_config.get('use_bf16', config.training.use_bf16)
    
    return config


def train_sweep():
    """Main training function for sweep"""
    # Initialize wandb run (this gets the sweep parameters)
    run = wandb.init()
    sweep_config = wandb.config
    
    # Load base configuration
    config = load_base_config()
    
    # Update configuration with sweep parameters
    config = update_config_from_sweep(config, sweep_config)
    
    # Update experiment name and output directory for this sweep run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.experiment_name = f"sweep_{run.id}_{timestamp}"
    config.output_dir = f"outputs/sweeps/{run.id}"
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(f"{config.output_dir}/checkpoints", exist_ok=True)
    
    # Log the full configuration
    wandb.config.update(OmegaConf.to_container(config))
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataloaders with GPU caching for speed
    print(f"Creating dataloaders with batch size {config.training.batch_size}...")
    gpu_config = {
        'gpu_max_samples': min(10000, config.training.batch_size * 1000),  # Limit samples for sweep
        'gpu_max_samples_val': min(2000, config.training.batch_size * 100),
        'gpu_cache_path': './gpu_cache_sweep',
        'batch_size': config.training.batch_size,
        'use_bfloat16': config.training.use_bf16,
        'preload_dinov2': False
    }
    
    try:
        train_loader, val_loader = create_gpu_cached_dataloaders(gpu_config)
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        wandb.finish(exit_code=1)
        return
    
    # Create model
    print("Creating model...")
    model = UnifiedManipulationTransformer(config.model)
    
    # Apply mode collapse prevention if enabled
    if config.optimizations.use_mode_collapse_prevention:
        mode_collapse_config = {
            'noise_std': 0.01,
            'drop_path_rate': 0.1,
            'mixup_alpha': 0.2
        }
        model = ModeCollapsePreventionModule.wrap_model(model, mode_collapse_config)
    
    # Initialize weights
    def init_weights(module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            if hasattr(module, 'weight') and module.weight is not None:
                torch.nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.01)
    
    # Apply initialization to non-pretrained layers
    for name, module in model.named_modules():
        if 'dinov2' not in name or 'encoder.layer.' not in name:
            init_weights(module)
    
    # Apply optimizations and move to device
    model = optimize_for_h200(model, compile_mode='reduce-overhead')
    model = model.to(device)
    
    # Create optimizer with parameter groups
    param_groups = []
    
    # Group parameters
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
    
    # Create parameter groups with different learning rates
    if dinov2_params:
        param_groups.append({
            'params': dinov2_params,
            'lr': config.training.learning_rate * config.training.multi_rate.pretrained,
            'name': 'dinov2'
        })
    
    if encoder_params:
        param_groups.append({
            'params': encoder_params,
            'lr': config.training.learning_rate * config.training.multi_rate.new_encoders,
            'name': 'encoders'
        })
    
    if decoder_params:
        param_groups.append({
            'params': decoder_params,
            'lr': config.training.learning_rate * config.training.multi_rate.decoders,
            'name': 'decoders'
        })
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': config.training.learning_rate,
            'name': 'other'
        })
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.training.weight_decay, fused=True)
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.training.T_0,
        T_mult=2,
        eta_min=config.training.min_lr
    )
    
    # Create loss function
    criterion = ComprehensiveLoss(config.loss)
    
    # Training loop
    best_val_mpjpe = float('inf')
    early_stop_patience = 10
    early_stop_counter = 0
    
    print(f"\nStarting training for {config.training.num_epochs} epochs...")
    
    for epoch in range(config.training.num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_mpjpe = 0
        train_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Convert BFloat16 images to Float32 for DINOv2
            if config.training.use_bf16 and 'image' in batch and batch['image'].dtype == torch.bfloat16:
                batch['image'] = batch['image'].float()
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', dtype=torch.bfloat16 if config.training.use_bf16 else torch.float16):
                outputs = model(batch)
                losses = criterion(outputs, batch)
                loss = losses['total'] if isinstance(losses, dict) else losses
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            optimizer.step()
            
            # Calculate metrics
            batch_size = batch['image'].shape[0]
            train_samples += batch_size
            train_loss += loss.item() * batch_size
            
            if 'hand_joints' in outputs and 'hand_joints' in batch:
                with torch.no_grad():
                    mpjpe = torch.norm(outputs['hand_joints'] - batch['hand_joints'], dim=-1).mean()
                    train_mpjpe += mpjpe.item() * 1000 * batch_size  # Convert to mm
            
            # Log batch metrics
            if batch_idx % 50 == 0:
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/batch_mpjpe': mpjpe.item() * 1000 if 'hand_joints' in outputs else 0,
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'system/gpu_memory_gb': torch.cuda.memory_allocated() / 1e9
                })
        
        # Average training metrics
        train_loss /= train_samples
        train_mpjpe /= train_samples
        
        # Validation
        model.eval()
        val_loss = 0
        val_mpjpe = 0
        val_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                if config.training.use_bf16 and 'image' in batch and batch['image'].dtype == torch.bfloat16:
                    batch['image'] = batch['image'].float()
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16 if config.training.use_bf16 else torch.float16):
                    outputs = model(batch)
                    losses = criterion(outputs, batch)
                    loss = losses['total'] if isinstance(losses, dict) else losses
                
                batch_size = batch['image'].shape[0]
                val_samples += batch_size
                val_loss += loss.item() * batch_size
                
                if 'hand_joints' in outputs and 'hand_joints' in batch:
                    mpjpe = torch.norm(outputs['hand_joints'] - batch['hand_joints'], dim=-1).mean()
                    val_mpjpe += mpjpe.item() * 1000 * batch_size
        
        # Average validation metrics
        val_loss /= val_samples
        val_mpjpe /= val_samples
        
        # Update learning rate
        scheduler.step()
        
        # Log epoch metrics
        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss,
            'train/hand_mpjpe': train_mpjpe,
            'val/loss': val_loss,
            'val/hand_mpjpe': val_mpjpe,
            'val/best_mpjpe': best_val_mpjpe,
        })
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{config.training.num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train MPJPE: {train_mpjpe:.2f}mm, "
              f"Val Loss: {val_loss:.4f}, Val MPJPE: {val_mpjpe:.2f}mm")
        
        # Save best model
        if val_mpjpe < best_val_mpjpe:
            best_val_mpjpe = val_mpjpe
            early_stop_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mpjpe': best_val_mpjpe,
                'config': config
            }, f"{config.output_dir}/checkpoints/best_model.pth")
        else:
            early_stop_counter += 1
        
        # Early stopping
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Log final metrics
    wandb.log({
        'final/best_val_mpjpe': best_val_mpjpe,
        'final/epochs_trained': epoch + 1,
    })
    
    # Clean up
    wandb.finish()
    torch.cuda.empty_cache()


def main():
    """Main entry point for sweep runner"""
    parser = argparse.ArgumentParser(description='Run W&B sweep for Advanced Manipulation Transformer')
    parser.add_argument('--sweep_id', type=str, default=None, help='W&B sweep ID to join')
    parser.add_argument('--project', type=str, default='advanced-manipulation-transformer-sweep', 
                       help='W&B project name')
    parser.add_argument('--count', type=int, default=1, help='Number of sweep runs to execute')
    args = parser.parse_args()
    
    if args.sweep_id:
        # Join existing sweep
        wandb.agent(args.sweep_id, function=train_sweep, count=args.count, project=args.project)
    else:
        # Create new sweep
        with open('configs/sweep_config.yaml', 'r') as f:
            sweep_config = yaml.safe_load(f)
        
        # Initialize sweep
        sweep_id = wandb.sweep(sweep_config, project=args.project)
        print(f"Created sweep with ID: {sweep_id}")
        print(f"View sweep at: https://wandb.ai/{wandb.api.default_entity}/{args.project}/sweeps/{sweep_id}")
        
        # Run agent
        wandb.agent(sweep_id, function=train_sweep, count=args.count)


if __name__ == "__main__":
    # Set DEX_YCB_DIR environment variable if not set
    import os
    if 'DEX_YCB_DIR' not in os.environ:
        os.environ['DEX_YCB_DIR'] = '/home/n231/231nProjectV2/dex-ycb-toolkit/data'
    
    main()