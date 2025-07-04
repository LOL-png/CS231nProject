"""
Test script to diagnose training errors
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Set environment
os.environ['DEX_YCB_DIR'] = '/home/n231/231nProjectV2/dex-ycb-toolkit/data'

# Add project root to path
project_root = os.path.abspath('.')
sys.path.insert(0, project_root)

# Import modules
from models.encoders.hand_encoder import HandPoseEncoder
from models.encoders.object_encoder import ObjectPoseEncoder
from models.encoders.contact_encoder import ContactDetectionEncoder
from data.dexycb_dataset import DexYCBDataset
from data.preprocessing import VideoPreprocessor

# Config
config = {
    'batch_size': 2,  # Very small batch
    'patch_size': 16,
    'image_size': [224, 224],
    'learning_rate': 1e-4,
    'grad_clip': 1.0,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create dataset and loader
print("\n1. Creating dataset...")
try:
    train_dataset = DexYCBDataset(split='s0_train', max_objects=10)
    print(f"Dataset created with {len(train_dataset)} samples")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    print("DataLoader created")
except Exception as e:
    print(f"Error creating dataset: {e}")
    raise

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

print("Models created")

# Setup optimizers
optimizer_hand = optim.AdamW(hand_encoder.parameters(), lr=config['learning_rate'])
optimizer_object = optim.AdamW(object_encoder.parameters(), lr=config['learning_rate'])
optimizer_contact = optim.AdamW(contact_encoder.parameters(), lr=config['learning_rate'])

mse_loss = nn.MSELoss()

# Test one batch
print("\n3. Testing one training batch...")
try:
    batch = next(iter(train_loader))
    print(f"\nBatch loaded successfully:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Move to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    
    # Preprocess
    print("\n4. Testing preprocessing...")
    B = batch['color'].shape[0]
    processed_images = []
    
    for i in range(B):
        img = preprocessor.preprocess_frame(batch['color'][i])
        processed_images.append(img)
    
    images = torch.stack(processed_images)
    print(f"Preprocessed images: {images.shape}")
    
    # Create patches
    patches = preprocessor.create_patches(images)
    print(f"Patches: {patches.shape}")
    
    # Test hand encoder
    print("\n5. Testing hand encoder...")
    try:
        hand_output = hand_encoder(patches)
        print(f"Hand encoder outputs:")
        for key, value in hand_output.items():
            print(f"  {key}: {value.shape}")
        
        # Test loss computation
        hand_gt = batch['hand_joints_3d'].to(device)
        if hand_gt.dim() == 3 and hand_gt.shape[1] == 1:
            hand_gt = hand_gt.squeeze(1)
        
        # Create proper mask shape [B]
        # hand_gt shape is [B, 21, 3], we need mask shape [B]
        # Check if all values in a hand are -1 (invalid)
        valid_hands = ~torch.all(hand_gt.view(hand_gt.shape[0], -1) == -1, dim=1)
        print(f"Valid hands mask shape: {valid_hands.shape}, values: {valid_hands}")
        
        if valid_hands.any():
            hand_loss = mse_loss(
                hand_output['joints_3d'][valid_hands],
                hand_gt[valid_hands]
            )
            print(f"Hand loss: {hand_loss.item():.4f}")
        else:
            print("No valid hands in batch")
            
    except Exception as e:
        print(f"Error in hand encoder: {e}")
        import traceback
        traceback.print_exc()
    
    # Test object encoder
    print("\n6. Testing object encoder...")
    try:
        # Handle ycb_ids
        ycb_ids = batch.get('ycb_ids', None)
        if isinstance(ycb_ids, torch.Tensor):
            # Already a tensor (from our updated dataset)
            pass
        elif isinstance(ycb_ids, list):
            # Convert list to tensor
            max_obj = max(len(ids) for ids in ycb_ids)
            ycb_ids_tensor = torch.full((len(ycb_ids), max_obj), -1, dtype=torch.long, device=device)
            for i, ids in enumerate(ycb_ids):
                ycb_ids_tensor[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            ycb_ids = ycb_ids_tensor
        
        print(f"YCB IDs shape: {ycb_ids.shape if torch.is_tensor(ycb_ids) else 'list'}")
        
        object_output = object_encoder(patches, object_ids=ycb_ids)
        print(f"Object encoder outputs:")
        for key, value in object_output.items():
            print(f"  {key}: {value.shape}")
            
    except Exception as e:
        print(f"Error in object encoder: {e}")
        import traceback
        traceback.print_exc()
    
    # Test contact encoder
    print("\n7. Testing contact encoder...")
    try:
        contact_output = contact_encoder(
            hand_output['features'],
            object_output['features']
        )
        print(f"Contact encoder outputs:")
        for key, value in contact_output.items():
            print(f"  {key}: {value.shape}")
            
    except Exception as e:
        print(f"Error in contact encoder: {e}")
        import traceback
        traceback.print_exc()
    
except Exception as e:
    print(f"Error in training test: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test Complete ===")