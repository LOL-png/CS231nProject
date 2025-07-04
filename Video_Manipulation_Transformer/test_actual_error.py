"""
Test to reproduce the actual error from the notebook
"""
import torch

# Simulate the exact scenario from the error
batch_size = 8

# Hand joints 2D with shape [B, 1, 21, 2]
hand_joints_2d = torch.randn(batch_size, 1, 21, 2)

print(f"Original hand_joints_2d shape: {hand_joints_2d.shape}")

# What happens in process_batch function
# Line: hand_joints_2d = batch['hand_joints_2d']
# Line: if hand_joints_2d.dim() == 3 and hand_joints_2d.shape[1] == 1:
#     hand_joints_2d = hand_joints_2d.squeeze(1)

# The notebook code checks for dim == 3, but the data has dim == 4!
print(f"hand_joints_2d.dim() = {hand_joints_2d.dim()}")

# So the squeeze doesn't happen, and we compute mask on [8, 1, 21, 2]
# Line: valid_hands = ~torch.all(hand_joints_2d == -1, dim=(1, 2))
print("\nComputing mask on unsqueezed tensor...")
try:
    # This is what was happening - computing on [8, 1, 21, 2] with dim=(1,2)
    # This reduces dims 1 and 2, leaving dims 0 and 3, giving shape [8, 2]
    wrong_mask = ~torch.all(hand_joints_2d == -1, dim=(1, 2))
    print(f"Wrong mask shape: {wrong_mask.shape}")
    print(f"This explains the error: mask shape [8, 2] doesn't match tensor shape!")
except Exception as e:
    print(f"Error: {e}")

# The fix is to check for dim == 4 instead of dim == 3
if hand_joints_2d.dim() == 4 and hand_joints_2d.shape[1] == 1:
    hand_joints_2d = hand_joints_2d.squeeze(1)
    print(f"\nAfter fixing dim check and squeezing: {hand_joints_2d.shape}")
    
# Now compute mask correctly
correct_mask = ~torch.all(hand_joints_2d.view(hand_joints_2d.shape[0], -1) == -1, dim=1)
print(f"Correct mask shape: {correct_mask.shape}")