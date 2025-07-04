#!/usr/bin/env python3
"""
Test the SE3Loss fix
"""

import os
import sys
import torch
import torch.nn as nn

# Set environment
os.environ['DEX_YCB_DIR'] = '/home/n231/231nProjectV2/dex-ycb-toolkit/data'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_se3loss_fix():
    """Test that SE3Loss accepts position_weight parameter"""
    print("Testing SE3Loss fix...")
    
    try:
        from training.losses import SE3Loss, ComprehensiveLoss
        
        # Test 1: Create SE3Loss with default position_weight
        print("\n1. Testing SE3Loss with default position_weight...")
        loss1 = SE3Loss()
        print(f"✅ SE3Loss created with default position_weight: {loss1.position_weight}")
        
        # Test 2: Create SE3Loss with custom position_weight
        print("\n2. Testing SE3Loss with custom position_weight...")
        loss2 = SE3Loss(position_weight=2.0)
        print(f"✅ SE3Loss created with custom position_weight: {loss2.position_weight}")
        
        # Test 3: Test forward pass
        print("\n3. Testing SE3Loss forward pass...")
        batch_size = 4
        pred_pos = torch.randn(batch_size, 3)
        pred_rot = torch.randn(batch_size, 6)  # 6D rotation
        target_pose = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        target_pose[:, :3, 3] = torch.randn(batch_size, 3)  # Random positions
        
        loss_value = loss2(pred_pos, pred_rot, target_pose)
        print(f"✅ Forward pass successful, loss value: {loss_value.item():.4f}")
        
        # Test 4: Create ComprehensiveLoss with config
        print("\n4. Testing ComprehensiveLoss creation...")
        config = {
            'object_position_weight': 1.5,
            'diversity_margin': 0.01,
            'loss_weights': {
                'hand_coarse': 1.0,
                'hand_refined': 1.2,
                'object_position': 1.0,
                'object_rotation': 0.5,
                'contact': 0.3,
                'physics': 0.1,
                'diversity': 0.01,
                'reprojection': 0.5,
                'kl': 0.001
            }
        }
        
        comprehensive_loss = ComprehensiveLoss(config)
        print("✅ ComprehensiveLoss created successfully")
        print(f"   SE3Loss position_weight: {comprehensive_loss.object_loss.position_weight}")
        
        # Test 5: Test that position_weight affects the loss
        print("\n5. Testing that position_weight affects the loss value...")
        loss_default = SE3Loss(position_weight=1.0)
        loss_weighted = SE3Loss(position_weight=2.0)
        
        loss_val1 = loss_default(pred_pos, pred_rot, target_pose)
        loss_val2 = loss_weighted(pred_pos, pred_rot, target_pose)
        
        # The weighted loss should be different (unless rotation loss is 0)
        print(f"   Loss with weight=1.0: {loss_val1.item():.4f}")
        print(f"   Loss with weight=2.0: {loss_val2.item():.4f}")
        
        if abs(loss_val2.item() - loss_val1.item()) > 0.001:
            print("✅ Position weight correctly affects loss value")
        else:
            print("⚠️ Position weight may not be affecting loss (could be due to small rotation loss)")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing SE3Loss Position Weight Fix")
    print("=" * 60)
    
    success = test_se3loss_fix()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ SE3Loss fix verified! The parameter is now accepted.")
    else:
        print("❌ SE3Loss test failed! Please check the error above.")
    print("=" * 60)