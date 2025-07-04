"""
Main entry point for Video-to-Manipulation Transformer training
"""

import argparse
import torch
import yaml
import os
from datetime import datetime

from models.full_model import VideoManipulationTransformer
from training.trainer import MultiStageTrainer
from training.evaluation import EvaluationMetrics


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict) -> VideoManipulationTransformer:
    """Create model from configuration"""
    model = VideoManipulationTransformer(
        hand_encoder_config=config.get('hand_encoder'),
        object_encoder_config=config.get('object_encoder'),
        contact_encoder_config=config.get('contact_encoder'),
        temporal_fusion_config=config.get('temporal_fusion'),
        action_decoder_config=config.get('action_decoder'),
        patch_size=config.get('patch_size', 16),
        image_size=tuple(config.get('image_size', [224, 224])),
        max_seq_length=config.get('max_seq_length', 16),
        dropout=config.get('dropout', 0.1)
    )
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train Video-to-Manipulation Transformer')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--eval_only', action='store_true',
                       help='Run evaluation only')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for this experiment')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update config with command line arguments
    config['device'] = args.device
    config['batch_size'] = args.batch_size
    config['num_workers'] = args.num_workers
    
    # Set experiment name
    if args.experiment_name:
        config['experiment_name'] = args.experiment_name
    else:
        config['experiment_name'] = f"video_manip_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    # Create model
    print("Creating model...")
    model = create_model(config)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = MultiStageTrainer(
        model=model,
        config=config,
        device=args.device,
        use_wandb=config.get('use_wandb', True)
    )
    
    # Setup data loaders
    print("Setting up data loaders...")
    trainer.setup_data_loaders(batch_size=args.batch_size)
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
        
    if args.eval_only:
        # Evaluation mode
        print("Running evaluation...")
        evaluator = EvaluationMetrics()
        
        model.eval()
        with torch.no_grad():
            for batch in trainer.val_loader:
                batch = trainer._move_batch_to_device(batch)
                outputs = model(batch, return_intermediates=True)
                evaluator.update(outputs, batch)
                
        metrics = evaluator.compute()
        
        print("\nEvaluation Results:")
        for key, value in sorted(metrics.items()):
            print(f"{key}: {value:.4f}")
            
    else:
        # Training mode
        print("\nStarting training...")
        
        # Define epochs for each stage
        stage_epochs = {
            'stage1': config.get('stage1_epochs', 20),
            'stage2': config.get('stage2_epochs', 15),
            'stage3': config.get('stage3_epochs', 15)
        }
        
        # Run training
        trainer.train(stage_epochs)
        
        print("\nTraining completed!")
        
        # Final evaluation
        print("\nRunning final evaluation...")
        evaluator = EvaluationMetrics()
        
        model.eval()
        with torch.no_grad():
            for batch in trainer.val_loader:
                batch = trainer._move_batch_to_device(batch)
                outputs = model(batch, return_intermediates=True)
                evaluator.update(outputs, batch)
                
        metrics = evaluator.compute()
        
        print("\nFinal Evaluation Results:")
        for key, value in sorted(metrics.items()):
            print(f"{key}: {value:.4f}")


def create_default_config():
    """Create a default configuration file"""
    config = {
        # Model architecture
        'patch_size': 16,
        'image_size': [224, 224],
        'max_seq_length': 16,
        'dropout': 0.1,
        
        # Encoder configurations
        'hand_encoder': {
            'hidden_dim': 512,
            'num_layers': 6,
            'num_heads': 8,
            'mlp_dim': 2048,
            'dropout': 0.1
        },
        
        'object_encoder': {
            'hidden_dim': 512,
            'num_layers': 6,
            'num_heads': 8,
            'mlp_dim': 2048,
            'dropout': 0.1,
            'max_objects': 10
        },
        
        'contact_encoder': {
            'hidden_dim': 256,
            'num_layers': 4,
            'num_heads': 8,
            'mlp_dim': 1024,
            'dropout': 0.1,
            'num_contact_points': 10
        },
        
        # Temporal fusion
        'temporal_fusion': {
            'hidden_dim': 1024,
            'num_layers': 8,
            'num_heads': 16,
            'mlp_dim': 4096,
            'dropout': 0.1,
            'window_size': 8
        },
        
        # Action decoder
        'action_decoder': {
            'hidden_dim': 512,
            'num_layers': 4,
            'num_heads': 8,
            'dropout': 0.1,
            'num_grasp_types': 10,
            'num_action_steps': 10
        },
        
        # Training settings
        'batch_size': 32,
        'num_workers': 8,
        'sequence_length': 16,
        'frame_stride': 1,
        
        # Learning rates
        'stage1_lr': 1e-4,
        'stage2_lr': 5e-5,
        'stage3_lr': 1e-5,
        'weight_decay': 0.01,
        'grad_clip': 1.0,
        
        # Stage epochs
        'stage1_epochs': 20,
        'stage2_epochs': 15,
        'stage3_epochs': 15,
        
        # Loss weights
        'encoder_weight': 0.1,
        'physics_weight': 0.1,
        
        # Paths
        'checkpoint_dir': 'checkpoints',
        'mujoco_model_path': None,
        'ur_model': 'ur5',
        
        # Logging
        'use_wandb': True
    }
    
    return config


if __name__ == '__main__':
    # Check if config file exists, create default if not
    if not os.path.exists('config.yaml'):
        print("Creating default config.yaml...")
        config = create_default_config()
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print("Default config.yaml created. Please review and modify as needed.")
        
    main()