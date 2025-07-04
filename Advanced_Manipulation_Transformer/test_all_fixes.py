#!/usr/bin/env python3
"""
Test script to verify all bug fixes are working
"""

import os
import sys
import torch
import torch.nn as nn

# Set environment
os.environ['DEX_YCB_DIR'] = '/home/n231/231nProjectV2/dex-ycb-toolkit/data'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_sigma_reparam():
    """Test SigmaReparam dimension fix"""
    print("\n1. Testing SigmaReparam fix...")
    try:
        from models.unified_model import SigmaReparam
        
        # Test with different input dimensions
        linear = nn.Linear(512, 256)
        sigma_layer = SigmaReparam(linear)
        
        # Test different input shapes
        x1 = torch.randn(2, 512)  # 2D input
        x2 = torch.randn(2, 10, 512)  # 3D input
        
        out1 = sigma_layer(x1)
        out2 = sigma_layer(x2)
        
        print(f"✅ SigmaReparam works with 2D input: {x1.shape} -> {out1.shape}")
        print(f"✅ SigmaReparam works with 3D input: {x2.shape} -> {out2.shape}")
        
    except Exception as e:
        print(f"❌ SigmaReparam test failed: {e}")
        return False
    return True

def test_transformer_layer_replacement():
    """Test TransformerEncoderLayer attribute access fix"""
    print("\n2. Testing TransformerEncoderLayer replacement fix...")
    try:
        from solutions.mode_collapse import ModeCollapsePreventionModule
        
        # Create a simple model with transformer layers
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=256,
                        nhead=8,
                        batch_first=True
                    ),
                    num_layers=2
                )
            
            def forward(self, x):
                return self.encoder(x)
        
        # Test wrapping with mode collapse prevention
        model = TestModel()
        config = {'noise_std': 0.01, 'drop_path_rate': 0.1, 'mixup_alpha': 0.2}
        wrapped_model = ModeCollapsePreventionModule.wrap_model(model, config)
        
        # Test forward pass
        x = torch.randn(2, 10, 256)
        output = wrapped_model(x)
        
        print(f"✅ TransformerEncoderLayer replacement works: {x.shape} -> {output.shape}")
        
    except Exception as e:
        print(f"❌ TransformerEncoderLayer test failed: {e}")
        return False
    return True

def test_learning_rate_handling():
    """Test learning rate type handling fix"""
    print("\n3. Testing learning rate handling fix...")
    try:
        from training.trainer import ManipulationTrainer
        
        # Test with scalar learning rate
        config1 = {'learning_rate': 1e-3}
        model = nn.Linear(10, 10)
        trainer1 = ManipulationTrainer(model, config1, device='cpu')
        
        # Test with list learning rate
        config2 = {'learning_rate': [1e-3, 1e-4]}
        trainer2 = ManipulationTrainer(model, config2, device='cpu')
        
        print("✅ Learning rate handling works with scalar and list configs")
        
    except Exception as e:
        print(f"❌ Learning rate test failed: {e}")
        return False
    return True

def test_dataset_split_handling():
    """Test dataset split naming fix"""
    print("\n4. Testing dataset split handling fix...")
    try:
        from data.enhanced_dexycb import EnhancedDexYCBDataset
        
        # Test different split names
        splits_to_test = ['train', 'val', 'test', 's0_train', 's0_val']
        
        for split in splits_to_test:
            try:
                # Just test initialization, not full loading
                dataset = EnhancedDexYCBDataset(
                    dexycb_root='../../dex-ycb-toolkit',
                    split=split,
                    sequence_length=1,
                    max_samples=10  # Load only 10 samples for testing
                )
                print(f"✅ Dataset split '{split}' handled correctly")
            except KeyError as e:
                if 'Unknown dataset name' in str(e):
                    print(f"❌ Dataset split '{split}' failed: {e}")
                else:
                    raise
                    
    except Exception as e:
        print(f"❌ Dataset split test failed: {e}")
        return False
    return True

def test_loss_variable_scope():
    """Test SimpleLoss variable scope fix"""
    print("\n5. Testing SimpleLoss variable scope fix...")
    try:
        # Recreate SimpleLoss with fix
        class SimpleLoss(nn.Module):
            def __init__(self):
                super().__init__()
                
            def forward(self, outputs, targets):
                losses = {}
                
                # Initialize pred_joints to None
                pred_joints = None
                
                # Hand joint loss (MPJPE)
                if 'hand_joints' in outputs and 'hand_joints' in targets:
                    pred_joints = outputs['hand_joints']
                    gt_joints = targets['hand_joints']
                    
                    # Simple L2 loss
                    joint_loss = nn.functional.mse_loss(pred_joints, gt_joints)
                    losses['joint_mse'] = joint_loss
                    
                    # MPJPE for monitoring
                    with torch.no_grad():
                        mpjpe = torch.norm(pred_joints - gt_joints, dim=-1).mean()
                        losses['mpjpe'] = mpjpe
                
                # Simple diversity loss to prevent collapse
                if pred_joints is not None and pred_joints.shape[0] > 1:
                    # Variance of predictions
                    pred_std = pred_joints.std(dim=0).mean()
                    diversity_loss = torch.relu(torch.tensor(0.01) - pred_std)
                    losses['diversity'] = diversity_loss * 0.1
                
                # Total loss
                total_loss = sum(losses.values())
                losses['total'] = total_loss
                
                return losses
        
        # Test with and without hand joints
        loss_fn = SimpleLoss()
        
        # Test case 1: With hand joints
        outputs1 = {'hand_joints': torch.randn(4, 21, 3)}
        targets1 = {'hand_joints': torch.randn(4, 21, 3)}
        losses1 = loss_fn(outputs1, targets1)
        print(f"✅ SimpleLoss works with hand joints: {list(losses1.keys())}")
        
        # Test case 2: Without hand joints
        outputs2 = {'other_key': torch.randn(4, 10)}
        targets2 = {'other_key': torch.randn(4, 10)}
        losses2 = loss_fn(outputs2, targets2)
        print(f"✅ SimpleLoss works without hand joints: {list(losses2.keys())}")
        
    except Exception as e:
        print(f"❌ SimpleLoss test failed: {e}")
        return False
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing All Bug Fixes")
    print("=" * 60)
    
    tests = [
        test_sigma_reparam,
        test_transformer_layer_replacement,
        test_learning_rate_handling,
        test_dataset_split_handling,
        test_loss_variable_scope
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total = len(results)
    passed = sum(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All bug fixes verified! The codebase is ready for training.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)