"""
Dataset and training framework for transition learning using pairwise HOISDF outputs
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import random
import pickle
import json
from pathlib import Path
from dataclasses import dataclass
from transition_merger_model import HOISDFOutputs


class HOISDFTransitionDataset(Dataset):
    """Dataset for learning transitions between HOISDF outputs using pairwise comparisons."""
    
    def __init__(self, 
                 data_dir: str,
                 sequence_length: int = 100,
                 transition_length: int = 30,
                 context_frames: int = 20,
                 similarity_threshold: float = 0.3,
                 mode: str = 'train'):
        """
        Args:
            data_dir: Directory containing HOISDF output files
            sequence_length: Expected length of each sequence
            transition_length: Number of frames for transition
            context_frames: Number of frames to use as context before/after transition
            similarity_threshold: Threshold for determining similar hand poses
            mode: 'train' or 'val'
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.transition_length = transition_length
        self.context_frames = context_frames
        self.similarity_threshold = similarity_threshold
        self.mode = mode
        
        # Load all sequences
        self.sequences = self._load_sequences()
        
        # Create pairwise combinations based on similarity
        self.pairs = self._create_pairs()
        
        print(f"Loaded {len(self.sequences)} sequences")
        print(f"Created {len(self.pairs)} valid pairs for {mode}")
        
    def _load_sequences(self) -> List[Dict]:
        """Load all HOISDF output sequences from directory."""
        sequences = []
        
        # Support multiple file formats
        for file_path in self.data_dir.glob("*.pkl"):
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                sequences.append({
                    'name': file_path.stem,
                    'data': data,
                    'path': file_path
                })
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        # Also support .pt files
        for file_path in self.data_dir.glob("*.pt"):
            try:
                data = torch.load(file_path)
                sequences.append({
                    'name': file_path.stem,
                    'data': data,
                    'path': file_path
                })
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        return sequences
    
    def _compute_hand_similarity(self, seq1: Dict, seq2: Dict) -> float:
        """Compute similarity between end pose of seq1 and start pose of seq2."""
        # Get MANO parameters from end of seq1 and start of seq2
        mano1_end = seq1['data']['mano_params'][-self.context_frames:].mean(dim=0)
        mano2_start = seq2['data']['mano_params'][:self.context_frames].mean(dim=0)
        
        # Compute similarity based on pose parameters (ignore translation)
        pose1 = mano1_end[3:48]  # 45-dim pose parameters
        pose2 = mano2_start[3:48]
        
        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            pose1.unsqueeze(0), 
            pose2.unsqueeze(0)
        ).item()
        
        return similarity
    
    def _compute_object_compatibility(self, seq1: Dict, seq2: Dict) -> float:
        """Check if objects are compatible for transition."""
        # Get object information
        obj1_center = seq1['data']['object_center'][-1]
        obj2_center = seq2['data']['object_center'][0]
        
        # Check if objects are reasonably close
        distance = torch.norm(obj1_center - obj2_center).item()
        
        # Compatible if objects are within reasonable distance
        return 1.0 if distance < 0.5 else 0.0  # 50cm threshold
    
    def _create_pairs(self) -> List[Tuple[int, int, float]]:
        """Create pairs of sequences based on similarity."""
        pairs = []
        
        # Split sequences for train/val
        n_sequences = len(self.sequences)
        if self.mode == 'train':
            indices = list(range(int(0.8 * n_sequences)))
        else:
            indices = list(range(int(0.8 * n_sequences), n_sequences))
            
        # Create all possible pairs
        for i in indices:
            for j in indices:
                if i == j:
                    continue
                    
                # Compute similarity metrics
                hand_sim = self._compute_hand_similarity(
                    self.sequences[i], 
                    self.sequences[j]
                )
                obj_compat = self._compute_object_compatibility(
                    self.sequences[i], 
                    self.sequences[j]
                )
                
                # Add pair if similarity is above threshold
                if hand_sim > self.similarity_threshold:
                    pairs.append((i, j, hand_sim * obj_compat))
                    
        # Sort pairs by compatibility score
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        return pairs
    
    def _extract_hoisdf_outputs(self, data: Dict) -> HOISDFOutputs:
        """Convert loaded data to HOISDFOutputs format."""
        return HOISDFOutputs(
            mano_params=data['mano_params'],
            hand_sdf=data.get('hand_sdf', torch.zeros(100, 64, 64, 64)),
            object_sdf=data.get('object_sdf', torch.zeros(100, 64, 64, 64)),
            contact_points=data.get('contact_points', torch.zeros(100, 10, 3)),
            contact_frames=data.get('contact_frames', torch.zeros(100, 10)),
            hand_vertices=data.get('hand_vertices', torch.zeros(100, 778, 3)),
            object_center=data.get('object_center', torch.zeros(100, 3))
        )
    
    def _create_synthetic_transition(self, outputs1: HOISDFOutputs, 
                                   outputs2: HOISDFOutputs) -> HOISDFOutputs:
        """Create synthetic ground truth transition between two sequences."""
        T = self.transition_length
        
        # Extract end of seq1 and start of seq2
        mano1_end = outputs1.mano_params[-self.context_frames:]
        mano2_start = outputs2.mano_params[:self.context_frames]
        
        # Create smooth transition using cubic interpolation
        t = torch.linspace(0, 1, T).unsqueeze(1)
        
        # Separate translation and pose for better interpolation
        trans1 = mano1_end[-1, :3]
        trans2 = mano2_start[0, :3]
        pose1 = mano1_end[-1, 3:48]
        pose2 = mano2_start[0, 3:48]
        shape = mano1_end[-1, 48:51]  # Use shape from first sequence
        
        # Smooth translation with cubic easing
        ease_t = t * t * (3.0 - 2.0 * t)  # Smooth step function
        trans_interp = trans1 * (1 - ease_t) + trans2 * ease_t
        
        # Interpolate pose parameters with SLERP-like behavior
        pose_interp = pose1 * (1 - t) + pose2 * t
        
        # Combine into MANO parameters
        mano_transition = torch.cat([
            trans_interp,
            pose_interp,
            shape.unsqueeze(0).repeat(T, 1)
        ], dim=1)
        
        # Create transition outputs (simplified - you'd compute actual SDFs)
        transition_outputs = HOISDFOutputs(
            mano_params=mano_transition,
            hand_sdf=torch.zeros(T, 64, 64, 64),
            object_sdf=torch.zeros(T, 64, 64, 64),
            contact_points=torch.zeros(T, 10, 3),
            contact_frames=torch.zeros(T, 10),
            hand_vertices=torch.zeros(T, 778, 3),
            object_center=outputs2.object_center[:T]  # Use target object position
        )
        
        return transition_outputs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """Get a pair of sequences and their transition."""
        seq1_idx, seq2_idx, compatibility = self.pairs[idx]
        
        # Load sequences
        seq1_data = self.sequences[seq1_idx]['data']
        seq2_data = self.sequences[seq2_idx]['data']
        
        # Extract HOISDF outputs
        outputs1 = self._extract_hoisdf_outputs(seq1_data)
        outputs2 = self._extract_hoisdf_outputs(seq2_data)
        
        # Create synthetic ground truth transition
        gt_transition = self._create_synthetic_transition(outputs1, outputs2)
        
        return {
            'outputs1': outputs1,
            'outputs2': outputs2,
            'gt_transition': gt_transition,
            'compatibility': compatibility,
            'names': (self.sequences[seq1_idx]['name'], self.sequences[seq2_idx]['name'])
        }


class TransitionTrainer:
    """Trainer for transition merger model using pairwise comparisons."""
    
    def __init__(self, 
                 model: nn.Module,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 config: Dict):
        self.model = model
        self.config = config
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            collate_fn=self.collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            collate_fn=self.collate_fn
        )
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs']
        )
        
        # Loss function
        from transition_merger_model import TransitionLoss
        self.criterion = TransitionLoss(config['loss_weights'])
        
        # Tracking
        self.best_val_loss = float('inf')
        self.device = config['device']
        
    def collate_fn(self, batch):
        """Custom collate function to handle HOISDFOutputs."""
        # Stack outputs for batch processing
        outputs1_list = [item['outputs1'] for item in batch]
        outputs2_list = [item['outputs2'] for item in batch]
        gt_transition_list = [item['gt_transition'] for item in batch]
        
        # Create batched HOISDFOutputs
        def stack_outputs(outputs_list):
            return HOISDFOutputs(
                mano_params=torch.stack([o.mano_params for o in outputs_list]),
                hand_sdf=torch.stack([o.hand_sdf for o in outputs_list]),
                object_sdf=torch.stack([o.object_sdf for o in outputs_list]),
                contact_points=torch.stack([o.contact_points for o in outputs_list]),
                contact_frames=torch.stack([o.contact_frames for o in outputs_list]),
                hand_vertices=torch.stack([o.hand_vertices for o in outputs_list]),
                object_center=torch.stack([o.object_center for o in outputs_list])
            )
        
        return {
            'outputs1': stack_outputs(outputs1_list),
            'outputs2': stack_outputs(outputs2_list),
            'gt_transition': stack_outputs(gt_transition_list),
            'compatibility': torch.tensor([item['compatibility'] for item in batch]),
            'names': [item['names'] for item in batch]
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            outputs1 = self._to_device(batch['outputs1'])
            outputs2 = self._to_device(batch['outputs2'])
            gt_transition = self._to_device(batch['gt_transition'])
            
            # Forward pass
            model_outputs = self.model(
                outputs1, outputs2,
                transition_length=self.config['transition_length'],
                mode='train'
            )
            
            # Compute loss
            losses = self.criterion(model_outputs, gt_transition, self.model)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += losses['total'].item()
            
            if batch_idx % self.config['log_interval'] == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}")
                for k, v in losses.items():
                    if k != 'total' and not k.startswith('diffusion_'):
                        print(f"  {k}: {v.item():.4f}")
                        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                outputs1 = self._to_device(batch['outputs1'])
                outputs2 = self._to_device(batch['outputs2'])
                gt_transition = self._to_device(batch['gt_transition'])
                
                # Forward pass
                model_outputs = self.model(
                    outputs1, outputs2,
                    transition_length=self.config['transition_length'],
                    mode='inference'
                )
                
                # Compute loss
                losses = self.criterion(model_outputs, gt_transition, self.model)
                total_loss += losses['total'].item()
                
        return total_loss / len(self.val_loader)
    
    def _to_device(self, outputs: HOISDFOutputs) -> HOISDFOutputs:
        """Move HOISDFOutputs to device."""
        return HOISDFOutputs(
            mano_params=outputs.mano_params.to(self.device),
            hand_sdf=outputs.hand_sdf.to(self.device),
            object_sdf=outputs.object_sdf.to(self.device),
            contact_points=outputs.contact_points.to(self.device),
            contact_frames=outputs.contact_frames.to(self.device),
            hand_vertices=outputs.hand_vertices.to(self.device),
            object_center=outputs.object_center.to(self.device)
        )
    
    def train(self, num_epochs):
        """Full training loop."""
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step()
            
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
                
    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        torch.save(checkpoint, f'checkpoint_best.pth')
        print(f"Saved checkpoint at epoch {epoch} with val_loss {val_loss:.4f}")


# Example usage
if __name__ == "__main__":
    from transition_merger_model import TransitionMergerModel
    
    # Configuration
    config = {
        # Model config
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
        
        # Training config
        'batch_size': 4,
        'num_workers': 4,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_epochs': 100,
        'transition_length': 30,
        'log_interval': 10,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        
        # Loss weights
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
        data_dir='path/to/hoisdf/outputs',
        mode='train'
    )
    
    val_dataset = HOISDFTransitionDataset(
        data_dir='path/to/hoisdf/outputs',
        mode='val'
    )
    
    # Initialize model
    model = TransitionMergerModel(config).to(config['device'])
    
    # Create trainer
    trainer = TransitionTrainer(model, train_dataset, val_dataset, config)
    
    # Train
    trainer.train(config['num_epochs'])