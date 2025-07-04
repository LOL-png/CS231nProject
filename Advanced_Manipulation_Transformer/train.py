import os
import torch
import torch.nn as nn
from pathlib import Path
import yaml
import argparse
import logging
from typing import Optional

# Set DEX_YCB_DIR environment variable
os.environ['DEX_YCB_DIR'] = '/home/n231/231nProjectV2/dex-ycb-toolkit/data'

# Import all components
from models.unified_model import UnifiedManipulationTransformer
from data.enhanced_dexycb import EnhancedDexYCBDataset
from training.trainer import ManipulationTrainer
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def main(config_path: str = "configs/default_config.yaml"):
    """
    Main training script
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Configuration loaded")
    logger.info(f"Experiment: {config['experiment_name']}")
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create datasets
    logger.info("Loading datasets...")
    
    train_dataset = EnhancedDexYCBDataset(
        dexycb_root=config['data']['root_dir'],  # Use root_dir instead of dexycb_root
        split='train',
        sequence_length=config['data']['sequence_length'],
        augment=True,  # Default value
        use_cache=True  # Default value
    )
    
    val_dataset = EnhancedDexYCBDataset(
        dexycb_root=config['data']['root_dir'],  # Use root_dir instead of dexycb_root
        split='val',
        sequence_length=1,  # No sequences for validation
        augment=False,
        use_cache=True  # Default value
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'] * 2,
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # Create model
    logger.info("Creating model...")
    model = UnifiedManipulationTransformer(config['model'])
    
    logger.info(f"Model created with {model.count_parameters():.2f}M parameters")
    
    # Create trainer
    trainer = ManipulationTrainer(
        model=model,
        config=config['training'],
        device=device
    )
    
    # Load checkpoint if specified
    if config['checkpoint']['resume_from']:
        logger.info(f"Resuming from checkpoint: {config['checkpoint']['resume_from']}")
        trainer.load_checkpoint(config['checkpoint']['resume_from'])
    
    # Training loop
    logger.info("Starting training...")
    
    best_val_mpjpe = float('inf')
    
    for epoch in range(trainer.epoch, config['training']['num_epochs']):
        # Train epoch
        train_metrics = trainer.train_epoch(train_loader, epoch)
        
        logger.info(f"Epoch {epoch} - Train loss: {train_metrics['loss']:.4f}")
        
        # Validation
        if epoch % config['training'].get('val_epochs', 1) == 0:
            val_metrics = trainer.validate(val_loader)
            
            logger.info(f"Epoch {epoch} - Val MPJPE: {val_metrics['val_mpjpe']:.2f}mm")
            
            # Save checkpoint
            is_best = val_metrics['val_mpjpe'] < best_val_mpjpe
            if is_best:
                best_val_mpjpe = val_metrics['val_mpjpe']
            
            trainer.save_checkpoint(val_metrics, is_best)
    
    logger.info("Training completed!")
    logger.info(f"Best validation MPJPE: {best_val_mpjpe:.2f}mm")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Advanced Manipulation Transformer")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    main(args.config)