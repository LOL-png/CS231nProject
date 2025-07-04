#!/usr/bin/env python3
"""
Test fixes for train_full_featured.ipynb issues
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Dict

# Set environment
os.environ['DEX_YCB_DIR'] = '/home/n231/231nProjectV2/dex-ycb-toolkit/data'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_forward_with_dict_input():
    """Test that model can handle dictionary input"""
    print("Testing model forward with dictionary input...")
    
    try:
        from models.unified_model import UnifiedManipulationTransformer
        from debugging.model_debugger import ModelDebugger
        
        # Create model
        config = {
            'hidden_dim': 256, 
            'freeze_layers': 6,
            'use_sigma_reparam': False  # Disable for testing
        }
        model = UnifiedManipulationTransformer(config)
        
        # Create sample batch (as from dataloader)
        batch_size = 2
        sample_batch = {
            'image': torch.randn(batch_size, 3, 224, 224),
            'hand_joints_3d': torch.randn(batch_size, 21, 3),
            'hand_joints_2d': torch.randn(batch_size, 21, 2),
            'mano_pose': torch.randn(batch_size, 51),
            'mano_shape': torch.randn(batch_size, 10),
            'object_pose': torch.randn(batch_size, 3, 4),
            'object_id': torch.randint(0, 10, (batch_size,)),
            'object_id': torch.randint(0, 10, (batch_size,)),
            'camera_intrinsics': torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1),
            'camera_extrinsics': torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1),
            'has_hand': torch.ones(batch_size, dtype=torch.bool),
        }
        
        # Test 1: Model forward with dict input
        print("  Testing model(dict)...")
        outputs = model(sample_batch)
        print(f"  ✅ Model accepts dictionary input")
        print(f"  Output keys: {list(outputs.keys())}")
        
        # Test 2: Model forward with separate args
        print("\n  Testing model(images=...)...")
        outputs2 = model(images=sample_batch['image'])
        print(f"  ✅ Model accepts separate arguments")
        
        # Test 3: Debugger compatibility
        print("\n  Testing debugger.analyze_model(batch)...")
        debugger = ModelDebugger(model, save_dir="./test_debug")
        # Just test forward pass analysis
        debugger.analyze_forward_pass(sample_batch)
        print(f"  ✅ Debugger works with batch input")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_function_with_dataset_keys():
    """Test that loss function handles dataset key names"""
    print("\nTesting loss function with dataset keys...")
    
    try:
        from training.losses import ComprehensiveLoss
        from models.unified_model import UnifiedManipulationTransformer
        
        # Create model and loss
        config = {
            'hidden_dim': 256, 
            'freeze_layers': 6,
            'use_sigma_reparam': False  # Disable for testing
        }
        model = UnifiedManipulationTransformer(config)
        
        loss_config = {
            'loss_weights': {
                'hand_coarse': 1.0,
                'hand_refined': 1.0,
                'object_position': 1.0,
                'diversity': 0.01
            }
        }
        criterion = ComprehensiveLoss(loss_config)
        
        # Create batch with dataset keys
        batch_size = 2
        batch = {
            'image': torch.randn(batch_size, 3, 224, 224),
            'hand_joints_3d': torch.randn(batch_size, 21, 3),  # Dataset uses hand_joints_3d
            'hand_joints_2d': torch.randn(batch_size, 21, 2),
            'object_pose': torch.randn(batch_size, 3, 4),
            'object_id': torch.randint(0, 10, (batch_size,)),  # Dataset uses object_pose (singular)
            'camera_intrinsics': torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1),
        }
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(batch)
        
        # Test loss computation
        print("  Testing loss computation...")
        losses = criterion(outputs, batch)
        
        print(f"  ✅ Loss accepts dataset keys")
        print(f"  Loss components: {list(losses.keys())}")
        print(f"  Total loss: {losses['total'].item():.4f}")
        
        # Check that we have some non-zero losses
        non_zero_losses = [k for k, v in losses.items() if k != 'total' and v.item() > 0]
        if non_zero_losses:
            print(f"  ✅ Non-zero losses: {non_zero_losses}")
        else:
            print(f"  ⚠️  All losses are zero - might need to check computation")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step():
    """Test the training step with all components"""
    print("\nTesting complete training step...")
    
    try:
        from models.unified_model import UnifiedManipulationTransformer
        from training.trainer import ManipulationTrainer
        from training.losses import ComprehensiveLoss
        from omegaconf import OmegaConf
        
        # Create config
        config = OmegaConf.create({
            'model': {
                'hidden_dim': 256, 
                'freeze_layers': 6,
                'use_sigma_reparam': False  # Disable for testing
            },
            'training': {
                'batch_size': 2,
                'learning_rate': 1e-4,
                'weight_decay': 0.01,
                'use_amp': True,
                'accumulation_steps': 1,
                'grad_clip': 1.0
            },
            'loss': {
                'loss_weights': {
                    'hand_coarse': 1.0,
                    'hand_refined': 1.0,
                    'object_position': 1.0,
                    'diversity': 0.01
                }
            }
        })
        
        # Create model
        model = UnifiedManipulationTransformer(config.model)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create trainer
        trainer = ManipulationTrainer(
            model=model,
            config=config.training,
            device=device
        )
        
        # Replace with comprehensive loss
        trainer.criterion = ComprehensiveLoss(config.loss)
        
        # Create sample batch
        batch = {
            'image': torch.randn(2, 3, 224, 224).to(device),
            'hand_joints_3d': torch.randn(2, 21, 3).to(device),
            'hand_joints_2d': torch.randn(2, 21, 2).to(device),
            'object_pose': torch.randn(2, 3, 4).to(device),
            'camera_intrinsics': torch.eye(3).unsqueeze(0).repeat(2, 1, 1).to(device),
        }
        
        # Run training step
        print("  Running train_step...")
        outputs, metrics = trainer.train_step(batch)
        
        print(f"  ✅ Training step completed")
        print(f"  Metrics: {metrics}")
        
        # Check outputs
        if 'hand_joints' in outputs:
            print(f"  Hand predictions shape: {outputs['hand_joints'].shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing notebook fixes")
    print("=" * 60)
    
    tests = [
        test_model_forward_with_dict_input,
        test_loss_function_with_dataset_keys,
        test_training_step
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test.__name__}: {status}")
    
    if all(results):
        print("\n✅ All tests passed! The notebook should work now.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
    print("=" * 60)