"""
Quick start script to test the transition merger model
This creates synthetic data and trains a small model for demonstration
"""

import torch
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

from transition_merger_model import TransitionMergerModel, HOISDFOutputs
from transition_dataset import HOISDFTransitionDataset, TransitionTrainer

def create_synthetic_data(output_dir='data/synthetic_hoisdf'):
    """Create synthetic HOISDF outputs for testing"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating synthetic HOISDF data...")
    
    # Define different motion types
    motion_types = {
        'reach_forward': lambda t: torch.stack([
            0.2 * t,                    # X: forward
            -0.3 + 0.1 * torch.sin(t * np.pi),  # Y: slight up-down
            0.1 * torch.cos(t * 2 * np.pi)      # Z: side motion
        ], dim=1),
        
        'reach_up': lambda t: torch.stack([
            0.1 * torch.sin(t * np.pi),    # X: slight forward-back
            -0.3 + 0.4 * t,                # Y: upward
            0.05 * torch.cos(t * np.pi)    # Z: minimal
        ], dim=1),
        
        'reach_side': lambda t: torch.stack([
            0.1 * torch.cos(t * np.pi),    # X: minimal
            -0.2 + 0.05 * torch.sin(t * 2 * np.pi),  # Y: slight
            0.3 * t                         # Z: sideways
        ], dim=1),
        
        'grasp_close': lambda t: torch.stack([
            0.15 + 0.05 * torch.cos(t * 4 * np.pi),  # X: vibration
            -0.25,                          # Y: fixed
            0.1                             # Z: fixed
        ], dim=1)
    }
    
    # Generate sequences
    num_frames = 100
    sequences_created = 0
    
    for motion_name, motion_fn in motion_types.items():
        for variant in range(5):  # 5 variants of each motion
            t = torch.linspace(0, 1, num_frames)
            
            # Create MANO parameters
            mano_params = torch.zeros(num_frames, 51)
            
            # Translation (first 3 params)
            mano_params[:, :3] = motion_fn(t)
            
            # Add some variation
            mano_params[:, :3] += torch.randn(num_frames, 3) * 0.02
            
            # Pose parameters (next 45 params) - add some finger motion
            for finger in range(5):
                finger_motion = 0.3 * torch.sin(t * (2 + finger) * np.pi + variant)
                mano_params[:, 3 + finger*3:6 + finger*3] = finger_motion.unsqueeze(1)
            
            # Shape parameters (last 3 params) - keep constant
            mano_params[:, 48:51] = torch.randn(1, 3) * 0.1
            
            # Create synthetic SDFs and other data
            data = {
                'mano_params': mano_params,
                'hand_sdf': torch.randn(num_frames, 64, 64, 64) * 0.1,
                'object_sdf': torch.randn(num_frames, 64, 64, 64) * 0.1,
                'contact_points': torch.zeros(num_frames, 10, 3),
                'contact_frames': torch.zeros(num_frames, 10),
                'hand_vertices': torch.zeros(num_frames, 778, 3),
                'object_center': torch.tensor([0.2, -0.2, 0.0]).unsqueeze(0).repeat(num_frames, 1)
            }
            
            # Add some contact frames for grasping motions
            if 'grasp' in motion_name:
                data['contact_frames'][50:80, :5] = 1.0
                data['contact_points'][50:80, :5] = torch.randn(30, 5, 3) * 0.05
            
            # Save
            filename = output_dir / f"{motion_name}_{variant:03d}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
            
            sequences_created += 1
    
    print(f"Created {sequences_created} synthetic sequences in {output_dir}")
    return output_dir

def train_small_model(data_dir, num_epochs=10):
    """Train a small model for demonstration"""
    
    print("\nSetting up training...")
    
    # Configuration for small model
    config = {
        'tokenizer': {
            'mano_dim': 51,
            'sdf_resolution': 64,
            'hidden_dim': 128,  # Smaller for faster training
            'num_tokens': 128
        },
        'transformer': {
            'input_dim': 128,
            'hidden_dim': 256,  # Smaller
            'num_heads': 4,     # Fewer heads
            'num_layers': 3,    # Fewer layers
            'mano_dim': 51,
            'chunk_size': 30,   # Smaller chunks
            'dropout': 0.1
        },
        'diffuser': {
            'mano_dim': 51,
            'hidden_dim': 128,
            'condition_dim': 256,
            'num_timesteps': 50  # Fewer diffusion steps
        },
        'batch_size': 2,
        'num_workers': 0,
        'learning_rate': 1e-3,  # Higher LR for faster convergence
        'weight_decay': 1e-5,
        'num_epochs': num_epochs,
        'transition_length': 30,
        'log_interval': 5,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'loss_weights': {
            'mano_recon': 1.0,
            'contact': 0.1,
            'smooth': 0.5,
            'boundary': 0.1,
            'contrastive': 0.1,
            'diffusion': 0.1
        }
    }
    
    # Create datasets
    train_dataset = HOISDFTransitionDataset(
        data_dir=str(data_dir),
        mode='train',
        similarity_threshold=0.2  # Lower threshold for more pairs
    )
    
    val_dataset = HOISDFTransitionDataset(
        data_dir=str(data_dir),
        mode='val',
        similarity_threshold=0.2
    )
    
    print(f"Training pairs: {len(train_dataset)}")
    print(f"Validation pairs: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        print("No training pairs found! Check your data.")
        return None
    
    # Initialize model
    model = TransitionMergerModel(config).to(config['device'])
    print(f"Model on device: {config['device']}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = TransitionTrainer(model, train_dataset, val_dataset, config)
    trainer.train(num_epochs)
    
    return model, config

def test_model(model, data_dir, config):
    """Test the trained model"""
    
    print("\nTesting trained model...")
    
    # Load two different sequences
    files = list(Path(data_dir).glob("*.pkl"))
    if len(files) < 2:
        print("Not enough sequences for testing")
        return
    
    # Pick two sequences with different motion types
    seq1_file = [f for f in files if 'reach_forward' in f.name][0]
    seq2_file = [f for f in files if 'reach_up' in f.name][0]
    
    print(f"Testing transition: {seq1_file.name} → {seq2_file.name}")
    
    # Load sequences
    with open(seq1_file, 'rb') as f:
        seq1_data = pickle.load(f)
    with open(seq2_file, 'rb') as f:
        seq2_data = pickle.load(f)
    
    # Convert to HOISDFOutputs and add batch dimension
    outputs1 = HOISDFOutputs(
        **{k: torch.tensor(v).unsqueeze(0).to(config['device']) 
           for k, v in seq1_data.items()}
    )
    outputs2 = HOISDFOutputs(
        **{k: torch.tensor(v).unsqueeze(0).to(config['device']) 
           for k, v in seq2_data.items()}
    )
    
    # Generate transition
    model.eval()
    with torch.no_grad():
        result = model(outputs1, outputs2, transition_length=30, mode='inference')
    
    # Extract results
    transition_mano = result['transformer']['refined_mano'][0].cpu().numpy()
    
    # Visualize
    plt.figure(figsize=(15, 5))
    
    # Plot X translation
    plt.subplot(1, 3, 1)
    seq1_end = seq1_data['mano_params'][-20:, 0].numpy()
    seq2_start = seq2_data['mano_params'][:20, 0].numpy()
    
    plt.plot(np.arange(0, 20), seq1_end, 'b-', linewidth=2, label='Seq1 end')
    plt.plot(np.arange(20, 50), transition_mano[:, 0], 'r-', linewidth=3, label='Generated transition')
    plt.plot(np.arange(50, 70), seq2_start, 'g-', linewidth=2, label='Seq2 start')
    plt.xlabel('Frame')
    plt.ylabel('X Translation')
    plt.title('X-axis Movement')
    plt.legend()
    plt.grid(True)
    
    # Plot Y translation
    plt.subplot(1, 3, 2)
    seq1_end = seq1_data['mano_params'][-20:, 1].numpy()
    seq2_start = seq2_data['mano_params'][:20, 1].numpy()
    
    plt.plot(np.arange(0, 20), seq1_end, 'b-', linewidth=2)
    plt.plot(np.arange(20, 50), transition_mano[:, 1], 'r-', linewidth=3)
    plt.plot(np.arange(50, 70), seq2_start, 'g-', linewidth=2)
    plt.xlabel('Frame')
    plt.ylabel('Y Translation')
    plt.title('Y-axis Movement')
    plt.grid(True)
    
    # Plot 3D trajectory
    ax = plt.subplot(1, 3, 3, projection='3d')
    
    # Full trajectory
    full_x = np.concatenate([seq1_data['mano_params'][-20:, 0], 
                             transition_mano[:, 0], 
                             seq2_data['mano_params'][:20, 0]])
    full_y = np.concatenate([seq1_data['mano_params'][-20:, 1], 
                             transition_mano[:, 1], 
                             seq2_data['mano_params'][:20, 1]])
    full_z = np.concatenate([seq1_data['mano_params'][-20:, 2], 
                             transition_mano[:, 2], 
                             seq2_data['mano_params'][:20, 2]])
    
    ax.plot(full_x[:20], full_y[:20], full_z[:20], 'b-', linewidth=2, label='Seq1')
    ax.plot(full_x[20:50], full_y[20:50], full_z[20:50], 'r-', linewidth=3, label='Transition')
    ax.plot(full_x[50:], full_y[50:], full_z[50:], 'g-', linewidth=2, label='Seq2')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Hand Trajectory')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('transition_result.png', dpi=150)
    plt.show()
    
    print("\nTransition generation complete!")
    print("Results saved to 'transition_result.png'")

def main():
    """Run the complete quick start demo"""
    
    print("=== Transition Merger Quick Start ===\n")
    
    # Step 1: Create synthetic data
    data_dir = create_synthetic_data()
    
    # Step 2: Train small model
    model, config = train_small_model(data_dir, num_epochs=10)
    
    if model is not None:
        # Step 3: Test the model
        test_model(model, data_dir, config)
        
        print("\n=== Quick Start Complete! ===")
        print("\nNext steps:")
        print("1. Replace synthetic data with real HOISDF outputs")
        print("2. Train for more epochs (100+)")
        print("3. Use larger model configuration")
        print("4. Load checkpoint_best.pth for best results")
    else:
        print("\nTraining failed. Check your setup.")

if __name__ == "__main__":
    main()