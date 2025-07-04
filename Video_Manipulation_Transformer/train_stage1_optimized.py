#!/usr/bin/env python
"""
Optimized Stage 1 training for H200 GPU (140GB memory, 700W power)
Maximizes GPU utilization with mixed precision, large batches, and parallel training
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from tqdm import tqdm
import time

# Set environment
os.environ['DEX_YCB_DIR'] = '/home/n231/231nProjectV2/dex-ycb-toolkit/data'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Disable for better performance
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'  # H200 architecture

# Enable TF32 for better performance on H100/H200
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Add project root to path
project_root = os.path.abspath('.')
sys.path.insert(0, project_root)

from models.encoders.hand_encoder import HandPoseEncoder
from models.encoders.object_encoder import ObjectPoseEncoder
from models.encoders.contact_encoder import ContactDetectionEncoder
from data.dexycb_dataset import DexYCBDataset
from data.preprocessing import VideoPreprocessor

# Optimized config for H200
config = {
    # Data settings - maximize batch size
    'batch_size': 256,  # Start with 256, can go higher
    'accumulation_steps': 4,  # Effective batch size = 1024
    'num_workers': 16,  # Use multiple CPU cores for data loading
    'prefetch_factor': 4,  # Prefetch batches
    'pin_memory': True,
    'persistent_workers': True,
    
    # Model settings - scale up
    'patch_size': 16,
    'image_size': [224, 224],
    'hand_hidden_dim': 1024,  # Double the hidden dim
    'object_hidden_dim': 1024,
    'contact_hidden_dim': 512,
    'num_layers': 12,  # More layers
    
    # Training settings
    'learning_rate': 1e-3,  # Higher LR for larger batch
    'num_epochs': 5,
    'grad_clip': 1.0,
    'warmup_steps': 500,
    
    # Mixed precision
    'use_amp': True,
    'amp_dtype': torch.bfloat16,  # Better for H200
    
    # Logging
    'log_interval': 10,
    'val_interval': 100,
    
    # Memory settings
    'gradient_checkpointing': True,  # Trade compute for memory
    'compile_model': True,  # torch.compile for better performance
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Memory and performance monitoring
def print_gpu_stats():
    if torch.cuda.is_available():
        print(f"\nGPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        # Note: Power usage needs nvidia-ml-py to read

# Create datasets with optimized settings
print("\n1. Creating datasets...")
train_dataset = DexYCBDataset(split='s0_train', max_objects=10)
val_dataset = DexYCBDataset(split='s0_val', max_objects=10)

# Create data loaders with optimization
train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=config['num_workers'],
    pin_memory=config['pin_memory'],
    prefetch_factor=config['prefetch_factor'],
    persistent_workers=config['persistent_workers'],
    drop_last=True  # For consistent batch sizes
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print(f"Training samples: {len(train_dataset):,}")
print(f"Validation samples: {len(val_dataset):,}")
print(f"Batches per epoch: {len(train_loader)}")

# Create scaled-up models
print("\n2. Creating scaled-up models...")
patch_dim = 3 * config['patch_size'] * config['patch_size']

# Larger models for better GPU utilization
hand_encoder = HandPoseEncoder(
    input_dim=patch_dim,
    hidden_dim=config['hand_hidden_dim'],
    num_layers=config['num_layers'],
    num_heads=16,  # More attention heads
    mlp_dim=4096,  # Larger MLP
    dropout=0.1
).to(device)

object_encoder = ObjectPoseEncoder(
    input_dim=patch_dim,
    hidden_dim=config['object_hidden_dim'],
    num_layers=config['num_layers'],
    num_heads=16,
    mlp_dim=4096,
    dropout=0.1,
    max_objects=10
).to(device)

contact_encoder = ContactDetectionEncoder(
    input_dim=patch_dim,
    hidden_dim=config['contact_hidden_dim'],
    num_layers=8,  # More layers
    num_heads=16,
    mlp_dim=2048,
    dropout=0.1
).to(device)

# Enable gradient checkpointing if requested
if config['gradient_checkpointing']:
    # Note: Would need to implement checkpointing in model forward passes
    print("Gradient checkpointing enabled (requires model modification)")

# Compile models for better performance (PyTorch 2.0+)
if config['compile_model'] and hasattr(torch, 'compile'):
    print("Compiling models with torch.compile...")
    hand_encoder = torch.compile(hand_encoder, mode='max-autotune')
    object_encoder = torch.compile(object_encoder, mode='max-autotune')
    contact_encoder = torch.compile(contact_encoder, mode='max-autotune')

preprocessor = VideoPreprocessor(
    image_size=tuple(config['image_size']),
    patch_size=config['patch_size']
)

# Count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nScaled model parameters:")
print(f"Hand encoder: {count_parameters(hand_encoder)/1e6:.1f}M")
print(f"Object encoder: {count_parameters(object_encoder)/1e6:.1f}M")
print(f"Contact encoder: {count_parameters(contact_encoder)/1e6:.1f}M")
print(f"Total: {(count_parameters(hand_encoder) + count_parameters(object_encoder) + count_parameters(contact_encoder))/1e6:.1f}M")

print_gpu_stats()

# Setup optimizers with larger learning rate for large batch
optimizer_hand = optim.AdamW(hand_encoder.parameters(), lr=config['learning_rate'], weight_decay=0.01)
optimizer_object = optim.AdamW(object_encoder.parameters(), lr=config['learning_rate'], weight_decay=0.01)
optimizer_contact = optim.AdamW(contact_encoder.parameters(), lr=config['learning_rate'], weight_decay=0.01)

# Learning rate schedulers
scheduler_hand = optim.lr_scheduler.OneCycleLR(
    optimizer_hand,
    max_lr=config['learning_rate'],
    epochs=config['num_epochs'],
    steps_per_epoch=len(train_loader) // config['accumulation_steps']
)
scheduler_object = optim.lr_scheduler.OneCycleLR(
    optimizer_object,
    max_lr=config['learning_rate'],
    epochs=config['num_epochs'],
    steps_per_epoch=len(train_loader) // config['accumulation_steps']
)
scheduler_contact = optim.lr_scheduler.OneCycleLR(
    optimizer_contact,
    max_lr=config['learning_rate'],
    epochs=config['num_epochs'],
    steps_per_epoch=len(train_loader) // config['accumulation_steps']
)

# Mixed precision training
scaler_hand = GradScaler(enabled=config['use_amp'])
scaler_object = GradScaler(enabled=config['use_amp'])
scaler_contact = GradScaler(enabled=config['use_amp'])

# Loss functions
mse_loss = nn.MSELoss()

# Parallel training function
def train_epoch_parallel(epoch):
    """Train all three encoders in parallel for one epoch"""
    hand_encoder.train()
    object_encoder.train()
    contact_encoder.train()
    
    total_loss = 0
    num_batches = 0
    
    # Timing
    data_time = 0
    compute_time = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}')
    
    for batch_idx, batch in enumerate(progress_bar):
        start_time = time.time()
        
        # Move to device (async transfer)
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)
        
        data_time += time.time() - start_time
        compute_start = time.time()
        
        # Preprocess
        B = batch['color'].shape[0]
        processed_images = []
        
        # Batch preprocessing
        for i in range(B):
            img = preprocessor.preprocess_frame(batch['color'][i])
            processed_images.append(img)
        
        images = torch.stack(processed_images)
        patches = preprocessor.create_patches(images)
        
        # Prepare ground truth
        hand_gt = batch['hand_joints_3d']
        if hand_gt.dim() == 4 and hand_gt.shape[1] == 1:
            hand_gt = hand_gt.squeeze(1)
        
        valid_hands = ~torch.all(hand_gt.view(hand_gt.shape[0], -1) == -1, dim=1)
        
        # === PARALLEL FORWARD PASSES ===
        
        # Hand encoder with mixed precision
        with autocast(device_type='cuda', dtype=config['amp_dtype'], enabled=config['use_amp']):
            hand_output = hand_encoder(patches)
            
            if valid_hands.any():
                hand_loss = mse_loss(
                    hand_output['joints_3d'][valid_hands],
                    hand_gt[valid_hands]
                ) / config['accumulation_steps']
            else:
                hand_loss = torch.tensor(0.0, device=device)
        
        # Object encoder with mixed precision
        with autocast(device_type='cuda', dtype=config['amp_dtype'], enabled=config['use_amp']):
            object_output = object_encoder(patches, object_ids=batch.get('ycb_ids', None))
            
            # Object loss computation
            if batch['object_poses'].shape[1] > 0:
                valid_objects = ~torch.all(batch['object_poses'] == 0, dim=(2, 3))
                if valid_objects.any():
                    object_positions_gt = batch['object_poses'][:, :, :3, 3]
                    num_pred_objects = min(object_output['positions'].shape[1], batch['object_poses'].shape[1])
                    valid_mask = valid_objects[:, :num_pred_objects]
                    
                    if valid_mask.any():
                        pred_positions = object_output['positions'][:, :num_pred_objects]
                        gt_positions = object_positions_gt[:, :num_pred_objects]
                        pred_flat = pred_positions[valid_mask]
                        gt_flat = gt_positions[valid_mask]
                        object_loss = mse_loss(pred_flat, gt_flat) / config['accumulation_steps']
                    else:
                        object_loss = torch.tensor(0.0, device=device)
                else:
                    object_loss = torch.tensor(0.0, device=device)
            else:
                object_loss = torch.tensor(0.0, device=device)
        
        # Contact encoder with mixed precision
        with autocast(device_type='cuda', dtype=config['amp_dtype'], enabled=config['use_amp']):
            # Detach features to train independently
            contact_output = contact_encoder(
                hand_output['features'].detach(),
                object_output['features'].detach()
            )
            # No ground truth for contacts in DexYCB
            contact_loss = torch.tensor(0.0, device=device)
        
        # === PARALLEL BACKWARD PASSES ===
        
        # Scale losses and backward
        if hand_loss.requires_grad:
            scaler_hand.scale(hand_loss).backward()
        if object_loss.requires_grad:
            scaler_object.scale(object_loss).backward()
        if contact_loss.requires_grad:
            scaler_contact.scale(contact_loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % config['accumulation_steps'] == 0:
            # Unscale and clip gradients
            scaler_hand.unscale_(optimizer_hand)
            scaler_object.unscale_(optimizer_object)
            scaler_contact.unscale_(optimizer_contact)
            
            torch.nn.utils.clip_grad_norm_(hand_encoder.parameters(), config['grad_clip'])
            torch.nn.utils.clip_grad_norm_(object_encoder.parameters(), config['grad_clip'])
            torch.nn.utils.clip_grad_norm_(contact_encoder.parameters(), config['grad_clip'])
            
            # Optimizer steps
            scaler_hand.step(optimizer_hand)
            scaler_object.step(optimizer_object)
            scaler_contact.step(optimizer_contact)
            
            # Update scalers
            scaler_hand.update()
            scaler_object.update()
            scaler_contact.update()
            
            # Zero gradients
            optimizer_hand.zero_grad(set_to_none=True)
            optimizer_object.zero_grad(set_to_none=True)
            optimizer_contact.zero_grad(set_to_none=True)
            
            # Update schedulers
            scheduler_hand.step()
            scheduler_object.step()
            scheduler_contact.step()
        
        compute_time += time.time() - compute_start
        
        # Update metrics
        total_loss += hand_loss.item() * config['accumulation_steps'] + object_loss.item() * config['accumulation_steps']
        num_batches += 1
        
        # Update progress bar
        gpu_mem = torch.cuda.memory_allocated() / 1e9
        progress_bar.set_postfix({
            'hand': f'{hand_loss.item()*config["accumulation_steps"]:.4f}',
            'obj': f'{object_loss.item()*config["accumulation_steps"]:.4f}',
            'gpu_mem': f'{gpu_mem:.1f}GB',
            'data_ms': f'{1000*data_time/num_batches:.1f}',
            'compute_ms': f'{1000*compute_time/num_batches:.1f}'
        })
        
        # Print GPU stats periodically
        if (batch_idx + 1) % config['log_interval'] == 0:
            print_gpu_stats()
    
    return total_loss / max(num_batches, 1)

# Training loop
print("\n3. Starting optimized training...")
print(f"Effective batch size: {config['batch_size'] * config['accumulation_steps']}")
print_gpu_stats()

for epoch in range(config['num_epochs']):
    start_epoch = time.time()
    
    # Train
    train_loss = train_epoch_parallel(epoch)
    
    epoch_time = time.time() - start_epoch
    samples_per_sec = len(train_dataset) / epoch_time
    
    print(f"\nEpoch {epoch+1} completed in {epoch_time:.1f}s")
    print(f"Training throughput: {samples_per_sec:.1f} samples/sec")
    print(f"Average loss: {train_loss:.4f}")
    print_gpu_stats()

# Save optimized checkpoints
print("\n4. Saving checkpoints...")
checkpoint_dir = 'checkpoints/stage1_optimized'
os.makedirs(checkpoint_dir, exist_ok=True)

torch.save({
    'model_state_dict': hand_encoder.state_dict(),
    'optimizer_state_dict': optimizer_hand.state_dict(),
    'config': config,
}, os.path.join(checkpoint_dir, 'hand_encoder_optimized.pth'))

torch.save({
    'model_state_dict': object_encoder.state_dict(),
    'optimizer_state_dict': optimizer_object.state_dict(),
    'config': config,
}, os.path.join(checkpoint_dir, 'object_encoder_optimized.pth'))

torch.save({
    'model_state_dict': contact_encoder.state_dict(),
    'optimizer_state_dict': optimizer_contact.state_dict(),
    'config': config,
}, os.path.join(checkpoint_dir, 'contact_encoder_optimized.pth'))

print(f"Optimized checkpoints saved to {checkpoint_dir}")
print("\n✓ Optimized Stage 1 training completed!")