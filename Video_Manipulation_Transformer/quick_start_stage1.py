#!/usr/bin/env python
"""
Quick start script for Stage 1 training of Video-to-Manipulation Transformer
This script demonstrates a complete working pipeline without Jupyter notebook complications
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Set environment
os.environ['DEX_YCB_DIR'] = '/home/n231/231nProjectV2/dex-ycb-toolkit/data'

# Add project root to path
project_root = os.path.abspath('.')
sys.path.insert(0, project_root)

# Import our modules
from models.encoders.hand_encoder import HandPoseEncoder
from models.encoders.object_encoder import ObjectPoseEncoder
from models.encoders.contact_encoder import ContactDetectionEncoder
from data.dexycb_dataset import DexYCBDataset
from data.preprocessing import VideoPreprocessor

# Config
config = {
    'batch_size': 8,
    'patch_size': 16,
    'image_size': [224, 224],
    'learning_rate': 1e-4,
    'grad_clip': 1.0,
    'num_epochs': 1,  # Just 1 epoch for quick test
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create dataset and loader
print("\n1. Creating dataset...")
train_dataset = DexYCBDataset(split='s0_train', max_objects=10)
val_dataset = DexYCBDataset(split='s0_val', max_objects=10)

train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print(f"Training samples: {len(train_dataset):,}")
print(f"Validation samples: {len(val_dataset):,}")

# Create models
print("\n2. Creating models...")
patch_dim = 3 * config['patch_size'] * config['patch_size']

hand_encoder = HandPoseEncoder(
    input_dim=patch_dim,
    hidden_dim=512,
    num_layers=6,
    num_heads=8,
    dropout=0.1
).to(device)

object_encoder = ObjectPoseEncoder(
    input_dim=patch_dim,
    hidden_dim=512,
    num_layers=6,
    num_heads=8,
    dropout=0.1
).to(device)

contact_encoder = ContactDetectionEncoder(
    input_dim=patch_dim,
    hidden_dim=256,
    num_layers=4,
    num_heads=8,
    dropout=0.1
).to(device)

preprocessor = VideoPreprocessor(
    image_size=tuple(config['image_size']),
    patch_size=config['patch_size']
)

# Setup optimizers
optimizer_hand = optim.AdamW(hand_encoder.parameters(), lr=config['learning_rate'])
optimizer_object = optim.AdamW(object_encoder.parameters(), lr=config['learning_rate'])
optimizer_contact = optim.AdamW(contact_encoder.parameters(), lr=config['learning_rate'])

mse_loss = nn.MSELoss()

# Simple training loop
print("\n3. Starting training...")
print("Note: Hand region extraction disabled to avoid edge case errors")
hand_encoder.train()
object_encoder.train()
contact_encoder.train()

for epoch in range(config['num_epochs']):
    total_hand_loss = 0
    total_object_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}')
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # Preprocess
        B = batch['color'].shape[0]
        processed_images = []
        
        for i in range(B):
            img = preprocessor.preprocess_frame(batch['color'][i])
            processed_images.append(img)
        
        images = torch.stack(processed_images)
        patches = preprocessor.create_patches(images)
        
        # Hand encoder
        hand_output = hand_encoder(patches)
        
        # Hand loss
        hand_gt = batch['hand_joints_3d'].to(device)
        if hand_gt.dim() == 4 and hand_gt.shape[1] == 1:
            hand_gt = hand_gt.squeeze(1)
        
        # Create mask for valid hands - ensure shape [B]
        valid_hands = ~torch.all(hand_gt.view(hand_gt.shape[0], -1) == -1, dim=1)
        
        if valid_hands.any():
            hand_loss = mse_loss(
                hand_output['joints_3d'][valid_hands],
                hand_gt[valid_hands]
            )
            
            optimizer_hand.zero_grad()
            hand_loss.backward()
            torch.nn.utils.clip_grad_norm_(hand_encoder.parameters(), config['grad_clip'])
            optimizer_hand.step()
            
            total_hand_loss += hand_loss.item()
        
        # Object encoder
        object_output = object_encoder(patches, object_ids=batch['ycb_ids'])
        
        # Object loss
        if batch['object_poses'].shape[1] > 0:
            valid_objects = ~torch.all(batch['object_poses'] == 0, dim=(2, 3))
            if valid_objects.any():
                object_positions_gt = batch['object_poses'][:, :, :3, 3].to(device)
                num_pred_objects = min(object_output['positions'].shape[1], batch['object_poses'].shape[1])
                valid_mask = valid_objects[:, :num_pred_objects]
                
                if valid_mask.any():
                    pred_positions = object_output['positions'][:, :num_pred_objects]
                    gt_positions = object_positions_gt[:, :num_pred_objects]
                    
                    pred_flat = pred_positions[valid_mask]
                    gt_flat = gt_positions[valid_mask]
                    
                    object_loss = mse_loss(pred_flat, gt_flat)
                    
                    optimizer_object.zero_grad()
                    object_loss.backward()
                    torch.nn.utils.clip_grad_norm_(object_encoder.parameters(), config['grad_clip'])
                    optimizer_object.step()
                    
                    total_object_loss += object_loss.item()
        
        # Contact encoder (forward only, no loss)
        contact_output = contact_encoder(
            hand_output['features'],
            object_output['features']
        )
        
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'hand_loss': f'{total_hand_loss/num_batches:.4f}',
            'obj_loss': f'{total_object_loss/num_batches:.4f}'
        })
        
        # Stop after 100 batches for quick test
        if batch_idx >= 100:
            break
    
    print(f"\nEpoch {epoch+1} completed!")
    print(f"  Average hand loss: {total_hand_loss/num_batches:.4f}")
    print(f"  Average object loss: {total_object_loss/num_batches:.4f}")

# Quick validation
print("\n4. Running validation...")
hand_encoder.eval()
object_encoder.eval()

total_mpjpe = 0
num_valid_hands = 0

with torch.no_grad():
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= 10:  # Just 10 batches for quick validation
            break
            
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # Process
        B = batch['color'].shape[0]
        processed_images = []
        
        for i in range(B):
            img = preprocessor.preprocess_frame(batch['color'][i])
            processed_images.append(img)
        
        images = torch.stack(processed_images)
        patches = preprocessor.create_patches(images)
        
        # Hand predictions
        hand_output = hand_encoder(patches)
        hand_gt = batch['hand_joints_3d'].to(device)
        if hand_gt.dim() == 4 and hand_gt.shape[1] == 1:
            hand_gt = hand_gt.squeeze(1)
        
        valid_hands = ~torch.all(hand_gt.view(hand_gt.shape[0], -1) == -1, dim=1)
        
        if valid_hands.any():
            # MPJPE metric
            mpjpe = (hand_output['joints_3d'][valid_hands] - hand_gt[valid_hands]).norm(dim=-1).mean()
            total_mpjpe += mpjpe.item() * valid_hands.sum().item()
            num_valid_hands += valid_hands.sum().item()

if num_valid_hands > 0:
    avg_mpjpe = total_mpjpe / num_valid_hands
    print(f"\nValidation MPJPE: {avg_mpjpe*1000:.2f}mm")

# Save checkpoints
print("\n5. Saving checkpoints...")
checkpoint_dir = 'checkpoints/stage1'
os.makedirs(checkpoint_dir, exist_ok=True)

torch.save({
    'model_state_dict': hand_encoder.state_dict(),
    'optimizer_state_dict': optimizer_hand.state_dict(),
}, os.path.join(checkpoint_dir, 'hand_encoder_quick.pth'))

torch.save({
    'model_state_dict': object_encoder.state_dict(),
    'optimizer_state_dict': optimizer_object.state_dict(),
}, os.path.join(checkpoint_dir, 'object_encoder_quick.pth'))

torch.save({
    'model_state_dict': contact_encoder.state_dict(),
    'optimizer_state_dict': optimizer_contact.state_dict(),
}, os.path.join(checkpoint_dir, 'contact_encoder_quick.pth'))

print(f"Checkpoints saved to {checkpoint_dir}")
print("\n✓ Stage 1 quick start completed successfully!")