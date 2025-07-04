#!/usr/bin/env python3
"""Debug script to test the errors"""

import os
import sys
import torch

# Set environment variable
os.environ['DEX_YCB_DIR'] = '/home/n231/231nProjectV2/dex-ycb-toolkit/data'

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_mode_collapse_prevention():
    """Test ModeCollapsePreventionModule.wrap_model"""
    print("Testing ModeCollapsePreventionModule...")
    try:
        from solutions.mode_collapse import ModeCollapsePreventionModule
        
        # Create a dummy model
        model = torch.nn.Linear(10, 10)
        
        # Test wrap_model
        config = {
            'noise_std': 0.01,
            'drop_path_rate': 0.1,
            'mixup_alpha': 0.2
        }
        wrapped_model = ModeCollapsePreventionModule.wrap_model(model, config)
        print("✓ ModeCollapsePreventionModule.wrap_model works correctly")
        
    except Exception as e:
        print(f"✗ Error in ModeCollapsePreventionModule: {e}")
        import traceback
        traceback.print_exc()

def test_trainer_methods():
    """Test ManipulationTrainer methods"""
    print("\nTesting ManipulationTrainer methods...")
    try:
        from training.trainer import ManipulationTrainer
        from models.unified_model import UnifiedManipulationTransformer
        
        # Create dummy config
        config = {
            'learning_rate': 1e-3,
            'num_epochs': 100,
            'use_amp': True,
            'use_bf16': True,
            'grad_clip': 1.0,
            'ema_decay': 0.999,
            'scheduler': 'cosine',
            'T_0': 10,
            'min_lr': 1e-6
        }
        
        # Create dummy model config
        model_config = {
            'freeze_layers': 12,
            'hidden_dim': 256,
            'contact_hidden_dim': 128,
            'max_objects': 10,
            'num_object_classes': 21,
            'num_contact_points': 10,
            'dropout': 0.1
        }
        
        # Create model and trainer
        model = UnifiedManipulationTransformer(model_config)
        trainer = ManipulationTrainer(model=model, config=config)
        
        # Test get_lr method
        lr_list = trainer.get_lr()
        print(f"✓ get_lr() returns: {lr_list}")
        
        # Test get_gradient_norm
        grad_norm = trainer.get_gradient_norm()
        print(f"✓ get_gradient_norm() returns: {grad_norm}")
        
        # Test scheduler_step
        trainer.scheduler_step(0.5)
        print("✓ scheduler_step() works correctly")
        
    except Exception as e:
        print(f"✗ Error in ManipulationTrainer: {e}")
        import traceback
        traceback.print_exc()

def test_dataset_initialization():
    """Test EnhancedDexYCBDataset initialization"""
    print("\nTesting EnhancedDexYCBDataset initialization...")
    try:
        from data.enhanced_dexycb import EnhancedDexYCBDataset
        
        # Test with dexycb_root parameter
        dataset = EnhancedDexYCBDataset(
            dexycb_root='../../dex-ycb-toolkit',
            split='train',
            sequence_length=1,
            augment=True
        )
        print(f"✓ EnhancedDexYCBDataset initialized successfully with {len(dataset)} samples")
        
    except Exception as e:
        print(f"✗ Error in EnhancedDexYCBDataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Debugging Advanced Manipulation Transformer errors...\n")
    
    test_mode_collapse_prevention()
    test_trainer_methods()
    test_dataset_initialization()
    
    print("\nDebug script completed!")