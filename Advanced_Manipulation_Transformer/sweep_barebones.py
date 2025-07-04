#!/usr/bin/env python3
"""
Barebones W&B Sweep for Advanced Manipulation Transformer
Modeled after typical W&B sweep notebooks
"""

import os
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Set environment variable
os.environ['DEX_YCB_DIR'] = '/home/n231/231nProjectV2/dex-ycb-toolkit/data'

# Simple dummy dataset for testing
class DummyDataset(Dataset):
    def __init__(self, split='train', size=1000):
        self.size = size
        self.split = split
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Random data
        image = torch.randn(3, 224, 224)
        hand_joints = torch.randn(21, 3)
        return {
            'image': image,
            'hand_joints': hand_joints
        }

# Simple model
class SimpleModel(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, hidden_dim)
        self.head = nn.Linear(hidden_dim, 21 * 3)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x).flatten(1)
        x = torch.relu(self.fc(x))
        x = self.head(x)
        return x.reshape(-1, 21, 3)

def train():
    # Initialize wandb
    run = wandb.init()
    config = wandb.config
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    train_dataset = DummyDataset('train', size=1000)
    val_dataset = DummyDataset('val', size=100)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # Create model
    model = SimpleModel(hidden_dim=config.hidden_dim).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(config.epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            images = batch['image'].to(device)
            targets = batch['hand_joints'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        val_mpjpe = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                targets = batch['hand_joints'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                # Calculate MPJPE
                mpjpe = torch.norm(outputs - targets, dim=-1).mean().item() * 1000  # mm
                
                val_loss += loss.item()
                val_mpjpe += mpjpe
        
        val_loss /= len(val_loader)
        val_mpjpe /= len(val_loader)
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_mpjpe': val_mpjpe
        })
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_mpjpe={val_mpjpe:.2f}mm")

# Define sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_mpjpe',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'min': 0.0001,
            'max': 0.01,
            'distribution': 'log_uniform_values'
        },
        'batch_size': {
            'values': [16, 32, 64]
        },
        'hidden_dim': {
            'values': [256, 512, 1024]
        },
        'epochs': {
            'value': 10
        }
    }
}

if __name__ == '__main__':
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, project='amt-barebones-test')
    
    # Run sweep agent
    wandb.agent(sweep_id, function=train, count=5)