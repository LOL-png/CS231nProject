#!/usr/bin/env python3
"""
Simple W&B Sweep Example - Exactly like notebook style
"""

import wandb
import random

# 1. Define the training function
def train():
    # Initialize a new wandb run
    wandb.init()
    
    # Get hyperparameters from wandb
    lr = wandb.config.learning_rate
    batch_size = wandb.config.batch_size
    epochs = wandb.config.epochs
    
    # Simulate training - replace this with your actual training code
    for epoch in range(epochs):
        # Simulate metrics
        train_loss = 1.0 / (epoch + 1) * lr * 100  # Fake decreasing loss
        val_loss = train_loss + random.uniform(0, 0.1)
        val_mpjpe = 300 / (epoch + 1) + random.uniform(-10, 10)  # Fake MPJPE in mm
        
        # Log metrics to wandb
        wandb.log({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_mpjpe': val_mpjpe,
            'epoch': epoch
        })
    
    # Log final metric
    wandb.log({'final_mpjpe': val_mpjpe})

# 2. Define sweep configuration
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
            'values': [16, 32, 64, 128]
        },
        'epochs': {
            'value': 10
        }
    }
}

# 3. Initialize sweep
sweep_id = wandb.sweep(sweep_config, project='amt-test')

# 4. Run the sweep agent
wandb.agent(sweep_id, train, count=10)