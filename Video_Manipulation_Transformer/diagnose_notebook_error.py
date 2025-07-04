#!/usr/bin/env python
"""
Diagnose the exact mask shape error from the notebook
"""

import os
import sys
import torch
from torch.utils.data import DataLoader

# Set environment
os.environ['DEX_YCB_DIR'] = '/home/n231/231nProjectV2/dex-ycb-toolkit/data'

# Add project root to path
project_root = os.path.abspath('.')
sys.path.insert(0, project_root)

# Import modules
from data.dexycb_dataset import DexYCBDataset

# Create dataset and loader
train_dataset = DexYCBDataset(split='s0_train', max_objects=10)
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

# Get a batch
batch = next(iter(train_loader))

print("Batch data shapes:")
for key, value in batch.items():
    if isinstance(value, torch.Tensor):
        print(f"  {key}: {value.shape}")

# Check hand_joints_2d specifically
hand_joints_2d = batch['hand_joints_2d']
print(f"\nhand_joints_2d shape: {hand_joints_2d.shape}")
print(f"hand_joints_2d.dim(): {hand_joints_2d.dim()}")

# Test the process_batch logic
print("\n--- Testing process_batch logic ---")

# The notebook code checks dim == 4 and shape[1] == 1
if hand_joints_2d.dim() == 4 and hand_joints_2d.shape[1] == 1:
    print("Condition met: squeezing dimension 1")
    hand_joints_2d = hand_joints_2d.squeeze(1)
    print(f"After squeeze: {hand_joints_2d.shape}")
else:
    print("Condition NOT met - no squeeze")

# Now test mask computation
print("\n--- Testing mask computation ---")

# Test what happens if we compute mask without proper view/flatten
try:
    # This is what would cause [8, 2] shape
    wrong_mask = ~torch.all(hand_joints_2d == -1, dim=(1, 2))
    print(f"Wrong mask (dim=(1,2)) shape: {wrong_mask.shape}")
except Exception as e:
    print(f"Error with dim=(1,2): {e}")

# Correct mask computation
correct_mask = ~torch.all(hand_joints_2d.view(hand_joints_2d.shape[0], -1) == -1, dim=1)
print(f"Correct mask shape: {correct_mask.shape}")

# Try indexing
print("\n--- Testing indexing ---")
try:
    # This would fail with wrong mask
    if wrong_mask.dim() == 2:
        print(f"Wrong mask would fail - shape {wrong_mask.shape} can't index [8, 21, 2]")
    
    # This works with correct mask
    result = hand_joints_2d[correct_mask]
    print(f"Correct indexing works: result shape {result.shape}")
except Exception as e:
    print(f"Indexing error: {e}")