#!/usr/bin/env python3
"""
Advanced Manipulation Transformer Training Script
Supports all optimizations: FP8, FlashAttention, FSDP, etc.
"""

import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Optional

# Set DEX_YCB_DIR environment variable
os.environ['DEX_YCB_DIR'] = '/home/n231/231nProjectV2/dex-ycb-toolkit/data'

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.enhanced_dexycb import EnhancedDexYCBDataset
from models.unified_model import UnifiedManipulationTransformer
from training.trainer import ManipulationTrainer
from training.losses import ComprehensiveLoss
from evaluation.evaluator import ComprehensiveEvaluator
from debugging.model_debugger import ModelDebugger
from solutions.mode_collapse import ModeCollapsePreventionModule
from optimizations.memory_management import MemoryOptimizer
from optimizations.distributed_training import DistributedTrainingSetup
from optimizations.flash_attention import replace_with_flash_attention
from optimizations.fp8_mixed_precision import enable_fp8_training
from optimizations.data_loading import OptimizedDataLoader

logger = logging.getLogger(__name__)

class TrainingPipeline:
    """Complete training pipeline with all optimizations"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup distributed training if available
        self.distributed = DistributedTrainingSetup.init_distributed()
        if self.distributed:
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        
        # Initialize components
        self._setup_logging()
        self._setup_datasets()
        self._setup_model()
        self._setup_training()
        self._setup_debugging()
    
    def _setup_logging(self):
        """Setup logging and experiment tracking"""
        if self.rank == 0:
            # Setup wandb
            if self.config.training.use_wandb:
                wandb.init(
                    project="advanced-manipulation-transformer",
                    config=OmegaConf.to_container(self.config),
                    name=self.config.experiment_name
                )
            
            # Create output directories
            self.output_dir = Path(self.config.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            self.checkpoint_dir = self.output_dir / "checkpoints"
            self.checkpoint_dir.mkdir(exist_ok=True)
    
    def _setup_datasets(self):
        """Setup datasets and dataloaders"""
        logger.info("Setting up datasets...")
        
        # Create datasets
        self.train_dataset = EnhancedDexYCBDataset(
            dexycb_root=self.config.data.root_dir,
            split=self.config.data.train_split,
            sequence_length=self.config.data.sequence_length,
            augment=True
        )
        
        self.val_dataset = EnhancedDexYCBDataset(
            dexycb_root=self.config.data.root_dir,
            split=self.config.data.val_split,
            sequence_length=1,  # No sequences for validation
            augment=False
        )
        
        # Create optimized dataloaders
        self.train_loader = OptimizedDataLoader.create_dataloader(
            self.train_dataset,
            batch_size=self.config.training.batch_size // self.world_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            distributed=self.distributed
        )
        
        self.val_loader = OptimizedDataLoader.create_dataloader(
            self.val_dataset,
            batch_size=self.config.training.batch_size // self.world_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            distributed=self.distributed
        )
        
        logger.info(f"Train samples: {len(self.train_dataset)}, "
                   f"Val samples: {len(self.val_dataset)}")
    
    def _setup_model(self):
        """Setup model with all optimizations"""
        logger.info("Setting up model...")
        
        # Create base model
        self.model = UnifiedManipulationTransformer(self.config.model)
        
        # Apply optimizations based on config
        if self.config.optimizations.use_flash_attention:
            self.model = replace_with_flash_attention(self.model)
            logger.info("Enabled FlashAttention")
        
        if self.config.optimizations.use_mode_collapse_prevention:
            # Wrap with mode collapse prevention
            mode_collapse_config = {
                'noise_std': 0.01,
                'drop_path_rate': 0.1,
                'mixup_alpha': 0.2
            }
            self.model = ModeCollapsePreventionModule.wrap_model(
                self.model, mode_collapse_config
            )
            logger.info("Enabled mode collapse prevention")
        
        # Apply memory optimizations for H200
        if self.config.optimizations.use_memory_optimization:
            self.model = MemoryOptimizer.optimize_model_for_h200(
                self.model, self.config
            )
            logger.info("Applied H200 memory optimizations")
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Setup distributed training
        if self.distributed:
            if self.config.optimizations.use_fsdp:
                self.model = DistributedTrainingSetup.wrap_model_fsdp(
                    self.model, self.config, self.config.training.mixed_precision
                )
            else:
                self.model = DistributedTrainingSetup.wrap_model_ddp(
                    self.model, mixed_precision=self.config.training.mixed_precision
                )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    def _setup_training(self):
        """Setup training components"""
        logger.info("Setting up training components...")
        
        # Enable FP8 if configured
        if self.config.optimizations.use_fp8 and torch.cuda.get_device_capability()[0] >= 9:
            self.model = enable_fp8_training(self.model)
            logger.info("Enabled FP8 mixed precision")
        
        # Create trainer
        self.trainer = ManipulationTrainer(
            model=self.model,
            config=self.config.training,
            device=self.device,
            distributed=self.distributed,
            local_rank=self.rank
        )
        
        # Setup evaluator
        self.evaluator = ComprehensiveEvaluator(self.config.evaluation)
    
    def _setup_debugging(self):
        """Setup debugging tools"""
        if self.config.debug.enabled and self.rank == 0:
            debug_dir = self.output_dir / "debug"
            debug_dir.mkdir(exist_ok=True)
            self.debugger = ModelDebugger(self.model, str(debug_dir))
        else:
            self.debugger = None
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Debug initial model if enabled
        if self.debugger and self.config.debug.debug_initial_model:
            sample_batch = next(iter(self.train_loader))
            self.debugger.analyze_model(sample_batch)
            self.debugger.debug_prediction_diversity(self.val_loader, num_batches=5)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(self.config.training.num_epochs):
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Save checkpoint
            if self.rank == 0:
                is_best = val_metrics['loss'] < best_val_loss
                if is_best:
                    best_val_loss = val_metrics['loss']
                
                self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Log metrics
            if self.rank == 0:
                self.log_metrics(epoch, train_metrics, val_metrics)
        
        # Final evaluation
        if self.rank == 0:
            self.final_evaluation()
        
        # Cleanup
        if self.distributed:
            DistributedTrainingSetup.cleanup()
    
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch"""
        # Delegate to trainer
        train_metrics = self.trainer.train_epoch(self.train_loader, epoch)
        
        # Reduce metrics across processes if distributed
        if self.distributed:
            train_metrics = DistributedTrainingSetup.all_reduce_metrics(train_metrics)
        
        return train_metrics
    
    def validate(self, epoch: int) -> dict:
        """Validate model"""
        # Delegate to trainer
        val_metrics = self.trainer.validate(self.val_loader)
        
        # Reduce metrics across processes if distributed
        if self.distributed:
            val_metrics = DistributedTrainingSetup.all_reduce_metrics(val_metrics)
        
        return val_metrics
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'metrics': metrics,
            'config': OmegaConf.to_container(self.config)
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with val_loss: {metrics['loss']:.4f}")
    
    def log_metrics(self, epoch: int, train_metrics: dict, val_metrics: dict):
        """Log metrics to wandb and console"""
        # Console logging
        logger.info(f"Epoch {epoch+1}/{self.config.training.num_epochs}")
        logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                   f"MPJPE: {train_metrics['hand_mpjpe']:.2f}")
        logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                   f"MPJPE: {val_metrics['hand_mpjpe']:.2f}")
        
        # Wandb logging
        if self.config.training.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'train/hand_mpjpe': train_metrics['hand_mpjpe'],
                'train/object_add': train_metrics['object_add'],
                'val/loss': val_metrics['loss'],
                'val/hand_mpjpe': val_metrics['hand_mpjpe'],
                'val/object_add': val_metrics['object_add'],
                'lr': self.trainer.optimizer.param_groups[0]['lr']
            })
    
    def final_evaluation(self):
        """Perform final comprehensive evaluation"""
        logger.info("Running final evaluation...")
        
        # Load best model
        best_checkpoint = torch.load(self.checkpoint_dir / "best_model.pth")
        self.model.load_state_dict(best_checkpoint['model_state_dict'])
        
        # Run evaluation
        results = self.evaluator.evaluate_model(self.model, self.val_loader, self.device)
        
        # Save results
        self.evaluator.save_results(str(self.output_dir / "final_results.json"))
        self.evaluator.print_summary()
        
        # Debug final model if enabled
        if self.debugger and self.config.debug.debug_final_model:
            self.debugger.debug_prediction_diversity(self.val_loader, num_batches=10)


@hydra.main(version_base=None, config_path="configs", config_name="default_config")
def main(config: DictConfig):
    """Main entry point"""
    # Print config
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(config))
    
    # Create and run training pipeline
    pipeline = TrainingPipeline(config)
    pipeline.train()


if __name__ == "__main__":
    main()