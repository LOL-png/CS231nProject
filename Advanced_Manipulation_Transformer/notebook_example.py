#!/usr/bin/env python3
"""
Example showing how to use the fixed model in a notebook

This demonstrates:
1. Model accepts dictionary input from dataloader
2. Loss function handles dataset key names correctly
3. Training step works with all components
"""

import torch
from models.unified_model import UnifiedManipulationTransformer
from training.losses import ComprehensiveLoss
from training.trainer import ManipulationTrainer
from omegaconf import OmegaConf

# Configuration
config = OmegaConf.create({
    'model': {
        'hidden_dim': 1024,
        'freeze_layers': 6,
        'use_sigma_reparam': True,  # Enable to prevent mode collapse
        'dropout': 0.1,
        'max_objects': 10,
        'contact_hidden_dim': 512,
        'num_contact_points': 10,
        'use_attention_fusion': True,
        'num_refinement_steps': 2
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'use_amp': True,
        'accumulation_steps': 1,
        'grad_clip': 1.0
    },
    'loss': {
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
})

# Create model
model = UnifiedManipulationTransformer(config.model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Create loss function
criterion = ComprehensiveLoss(config.loss)

# Create trainer
trainer = ManipulationTrainer(
    model=model,
    config=config.training,
    device=device
)
# Replace trainer's criterion with our comprehensive loss
trainer.criterion = criterion

# Example batch from dataloader (as it would come from DexYCB)
batch = {
    'image': torch.randn(32, 3, 224, 224).to(device),
    'hand_joints_3d': torch.randn(32, 21, 3).to(device),
    'hand_joints_2d': torch.randn(32, 21, 2).to(device),
    'mano_pose': torch.randn(32, 51).to(device),
    'mano_shape': torch.randn(32, 10).to(device),
    'object_pose': torch.randn(32, 3, 4).to(device),  # Single object per scene
    'object_id': torch.randint(0, 10, (32,)).to(device),
    'camera_intrinsics': torch.eye(3).unsqueeze(0).repeat(32, 1, 1).to(device),
    'camera_extrinsics': torch.eye(4).unsqueeze(0).repeat(32, 1, 1).to(device),
    'has_hand': torch.ones(32, dtype=torch.bool).to(device),
}

# Forward pass - model now accepts dictionary input directly
outputs = model(batch)

# Check outputs
print("Model outputs:")
for k, v in outputs.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: shape {v.shape}")
    elif isinstance(v, dict):
        print(f"  {k}: dict with {len(v)} keys")

# Compute losses - loss function now handles dataset key names
losses = criterion(outputs, batch)
print("\nLoss components:")
for k, v in losses.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.item():.4f}")

# Run a training step
outputs, metrics = trainer.train_step(batch)
print(f"\nTraining metrics: {metrics}")

print("\n✅ Everything is working correctly!")
print("\nKey fixes applied:")
print("1. Model forward() accepts dictionary input from dataloader")
print("2. Loss function maps 'hand_joints_3d' -> hand ground truth") 
print("3. Loss function handles 'object_pose' (singular) from dataset")
print("4. Pixel alignment module dimension mismatch fixed")
print("5. Contact predictions properly mapped (contact_confidence -> contact_probs)")
print("6. SE3Loss accepts correct arguments")
print("7. Device mismatches resolved")
print("8. Autocast API updated for PyTorch 2.5")