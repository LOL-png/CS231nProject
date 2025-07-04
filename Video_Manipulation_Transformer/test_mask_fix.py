"""
Test the mask fix for the training error
"""
import torch

# Simulate the data shapes from the error
batch_size = 8

# Hand joints 2D with shape [B, 1, 21, 2]
hand_joints_2d = torch.randn(batch_size, 1, 21, 2)
# Some samples have invalid hands (all -1)
hand_joints_2d[2] = -1
hand_joints_2d[5] = -1

print(f"hand_joints_2d shape: {hand_joints_2d.shape}")

# Old method that caused the error
if hand_joints_2d.dim() == 4 and hand_joints_2d.shape[1] == 1:
    hand_joints_2d_squeezed = hand_joints_2d.squeeze(1)
    print(f"After squeeze(1): {hand_joints_2d_squeezed.shape}")

# This was producing wrong shape
old_valid_hands = ~torch.all(hand_joints_2d_squeezed == -1, dim=(1, 2))
print(f"\nOld mask shape (WRONG): {old_valid_hands.shape}")
print(f"Old mask: {old_valid_hands}")

# New method - correct
new_valid_hands = ~torch.all(hand_joints_2d_squeezed.view(hand_joints_2d_squeezed.shape[0], -1) == -1, dim=1)
print(f"\nNew mask shape (CORRECT): {new_valid_hands.shape}")
print(f"New mask: {new_valid_hands}")

# Test indexing
try:
    result = hand_joints_2d[new_valid_hands]
    print(f"\nIndexing works! Result shape: {result.shape}")
except Exception as e:
    print(f"\nIndexing failed: {e}")

# Also test with 3D hand joints
hand_joints_3d = torch.randn(batch_size, 1, 21, 3)
hand_joints_3d[2] = -1
hand_joints_3d[5] = -1

if hand_joints_3d.dim() == 4 and hand_joints_3d.shape[1] == 1:
    hand_joints_3d = hand_joints_3d.squeeze(1)

valid_hands_3d = ~torch.all(hand_joints_3d.view(hand_joints_3d.shape[0], -1) == -1, dim=1)
print(f"\n3D joints mask shape: {valid_hands_3d.shape}")
print(f"3D joints mask: {valid_hands_3d}")