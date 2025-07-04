# Complete Usage Guide for Transition Merger Model

## Current Status
- ✅ Model architecture implemented
- ✅ Training framework ready
- ❌ No pre-trained weights available
- ❌ Need HOISDF outputs to train

## Step-by-Step Usage

### Step 1: Get HOISDF Outputs

First, you need HOISDF outputs from your videos. You have two options:

#### Option A: Use Pre-trained HOISDF Model
```python
# Load your trained HOISDF model
import sys
sys.path.append('../HOISDF')
from main.model import get_model
import torch

# Load HOISDF model
hoisdf_model = get_model('test')
checkpoint = torch.load('../HOISDF/outputs/your_checkpoint.pth')
hoisdf_model.load_state_dict(checkpoint['model_state_dict'])
hoisdf_model.eval()

# Process videos to get HOISDF outputs
# This extracts MANO parameters, SDFs, contact points, etc.
```

#### Option B: Use Synthetic Data (for testing)
```python
# Run the notebook to create synthetic data
# This is in train_transitions.ipynb
```

### Step 2: Prepare Your Data

```python
import pickle
from pathlib import Path

# Create data directory
data_dir = Path('data/hoisdf_outputs')
data_dir.mkdir(parents=True, exist_ok=True)

# For each video, extract and save HOISDF outputs
for video in your_videos:
    # Process through HOISDF
    outputs = hoisdf_model(video)
    
    # Save the outputs
    save_data = {
        'mano_params': outputs['mano_params'].cpu(),  # [T, 51]
        'hand_sdf': outputs['hand_sdf'].cpu(),        # [T, 64, 64, 64]
        'object_sdf': outputs['object_sdf'].cpu(),    # [T, 64, 64, 64]
        'contact_points': outputs['contact_points'].cpu(),
        'contact_frames': outputs['contact_frames'].cpu(),
        'hand_vertices': outputs['hand_vertices'].cpu(),
        'object_center': outputs['object_center'].cpu()
    }
    
    with open(data_dir / f'{video_name}.pkl', 'wb') as f:
        pickle.dump(save_data, f)
```

### Step 3: Train the Transition Model

```python
from transition_merger_model import TransitionMergerModel
from transition_dataset import HOISDFTransitionDataset, TransitionTrainer
import torch

# Configuration
config = {
    'tokenizer': {
        'mano_dim': 51,
        'sdf_resolution': 64,
        'hidden_dim': 256,
        'num_tokens': 256
    },
    'transformer': {
        'input_dim': 256,
        'hidden_dim': 512,
        'num_heads': 8,
        'num_layers': 6,
        'mano_dim': 51,
        'chunk_size': 50,
        'dropout': 0.1
    },
    'diffuser': {
        'mano_dim': 51,
        'hidden_dim': 256,
        'condition_dim': 512,
        'num_timesteps': 100
    },
    'batch_size': 4,
    'learning_rate': 1e-4,
    'num_epochs': 100,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'loss_weights': {
        'mano_recon': 1.0,
        'contact': 0.5,
        'smooth': 0.1,
        'boundary': 0.5,
        'contrastive': 0.2,
        'diffusion': 0.5
    }
}

# Create datasets
train_dataset = HOISDFTransitionDataset(
    data_dir='data/hoisdf_outputs',
    mode='train'
)

val_dataset = HOISDFTransitionDataset(
    data_dir='data/hoisdf_outputs',
    mode='val'
)

# Initialize model
model = TransitionMergerModel(config).to(config['device'])

# Create trainer and train
trainer = TransitionTrainer(model, train_dataset, val_dataset, config)
trainer.train(config['num_epochs'])

# Trained weights will be saved as 'checkpoint_best.pth'
```

### Step 4: Use the Trained Model

```python
# Load trained model
checkpoint = torch.load('checkpoint_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load two sequences you want to merge
with open('data/sequence1.pkl', 'rb') as f:
    seq1_data = pickle.load(f)
with open('data/sequence2.pkl', 'rb') as f:
    seq2_data = pickle.load(f)

# Create HOISDFOutputs
from transition_merger_model import HOISDFOutputs

outputs1 = HOISDFOutputs(**seq1_data)
outputs2 = HOISDFOutputs(**seq2_data)

# Generate transition
with torch.no_grad():
    result = model(outputs1, outputs2, transition_length=30, mode='inference')
    
# Extract the transition
transition_mano = result['transformer']['refined_mano']  # [1, 30, 51]

# The transition contains MANO parameters that smoothly connect seq1 to seq2
```

## Where to Find Trained Weights

Currently, there are **NO pre-trained weights** available because:

1. This is a new model we just created
2. It needs to be trained on HOISDF outputs
3. Training requires your specific data

After training, weights will be saved as:
- `checkpoint_best.pth` - Best validation performance
- Contains model weights, optimizer state, and training config

## Quick Start (Without Real Data)

To test the system immediately:

```bash
# 1. Open Jupyter notebook
jupyter notebook train_transitions.ipynb

# 2. Run all cells to:
#    - Create synthetic HOISDF data
#    - Train a small model
#    - Generate sample transitions

# 3. This will create a checkpoint_best.pth file
```

## File Structure

```
Test_Model/
├── transition_merger_model.py    # Model implementation
├── transition_dataset.py         # Dataset and training
├── train_transitions.ipynb       # Training notebook
├── data/
│   └── hoisdf_outputs/          # Your HOISDF outputs go here
└── checkpoint_best.pth          # Trained weights (after training)
```

## Next Steps

1. **Get HOISDF Outputs**: Either from your trained HOISDF model or synthetic data
2. **Train the Model**: Run the training notebook or script
3. **Use for Inference**: Load trained weights and generate transitions

## Common Issues

**Q: Where do I get HOISDF model?**
A: You need to train it using the HOISDF code in the parent directory, or use synthetic data for testing.

**Q: How long does training take?**
A: Depends on dataset size. With 100 sequences, expect 2-4 hours on GPU.

**Q: Can I use it without training?**
A: No, the model needs to learn transition patterns from your data.

**Q: What if I don't have paired sequences?**
A: The model automatically finds compatible pairs based on hand pose similarity.