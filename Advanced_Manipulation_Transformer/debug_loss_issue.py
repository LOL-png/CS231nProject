#!/usr/bin/env python3
"""Debug the loss computation issue"""

import os
import sys
import torch

os.environ['DEX_YCB_DIR'] = '/home/n231/231nProjectV2/dex-ycb-toolkit/data'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.unified_model import UnifiedManipulationTransformer
from training.losses import ComprehensiveLoss

# Create simple test
config = {
    'hidden_dim': 256, 
    'freeze_layers': 6,
    'use_sigma_reparam': False
}
model = UnifiedManipulationTransformer(config)

# Create batch
batch = {
    'image': torch.randn(2, 3, 224, 224),
    'hand_joints_3d': torch.randn(2, 21, 3),
    'hand_joints_2d': torch.randn(2, 21, 2),
    'object_pose': torch.randn(2, 3, 4),
    'camera_intrinsics': torch.eye(3).unsqueeze(0).repeat(2, 1, 1),
}

# Forward pass
with torch.no_grad():
    outputs = model(batch)

print("Output shapes:")
for k, v in outputs.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape}")
    elif isinstance(v, dict):
        print(f"  {k}: dict with keys {list(v.keys())}")

# Check specific outputs
if 'object_positions' in outputs:
    print(f"\nobject_positions shape: {outputs['object_positions'].shape}")
if 'object_rotations' in outputs:
    print(f"object_rotations shape: {outputs['object_rotations'].shape}")
if 'contact_probs' in outputs:
    print(f"contact_probs type: {type(outputs['contact_probs'])}")
    if isinstance(outputs['contact_probs'], torch.Tensor):
        print(f"contact_probs shape: {outputs['contact_probs'].shape}")
if 'contacts' in outputs:
    print(f"\ncontacts dict keys: {list(outputs['contacts'].keys())}")
    for k, v in outputs['contacts'].items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape {v.shape}")
        else:
            print(f"  {k}: type {type(v)}")

# Create loss
loss_config = {
    'loss_weights': {
        'hand_coarse': 1.0,
        'hand_refined': 1.0,
        'object_position': 1.0,
        'diversity': 0.01
    }
}
criterion = ComprehensiveLoss(loss_config)

try:
    losses = criterion(outputs, batch)
    print("\nLoss computation successful!")
    print("Loss components:", list(losses.keys()))
except Exception as e:
    print(f"\nError in loss computation: {e}")
    import traceback
    traceback.print_exc()