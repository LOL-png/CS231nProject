"""
GPU-Cached Dataset for Advanced Manipulation Transformer
Adapted from the original Video Manipulation Transformer's gpu_only_dataset.py
Keeps entire dataset in GPU memory for maximum performance
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import cv2
from pathlib import Path
import os
import sys
from tqdm import tqdm
import logging

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'dex-ycb-toolkit'))

logger = logging.getLogger(__name__)

class GPUCachedDataset:
    """
    Dataset that lives entirely on GPU memory
    Pre-loads and pre-processes all data to achieve 100GB+ GPU memory usage
    """
    
    def __init__(self, 
                 split: str = 'train',
                 max_samples: int = 100000,  # Load 100k samples to use 100GB+ memory
                 image_size: Tuple[int, int] = (224, 224),
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float32,
                 cache_path: Optional[str] = None,
                 normalize: bool = True,
                 load_dinov2_features: bool = False):
        """
        Args:
            split: Dataset split ('train', 'val', 'test')
            max_samples: Maximum samples to load (adjust based on GPU memory)
            image_size: Target image size
            device: GPU device
            dtype: Data type (float32 or bfloat16)
            cache_path: Path to save/load preprocessed data
            normalize: Apply ImageNet normalization
            load_dinov2_features: Pre-extract DINOv2 features (saves compute)
        """
        self.split = split
        self.max_samples = max_samples
        self.image_size = image_size
        self.device = device
        self.dtype = dtype
        self.cache_path = cache_path
        self.normalize = normalize
        self.load_dinov2_features = load_dinov2_features
        
        # Memory estimation
        samples_per_gb = 1000  # Rough estimate
        estimated_memory = max_samples / samples_per_gb
        print(f"Estimated GPU memory usage: {estimated_memory:.1f} GB")
        
        # Check for cached data first
        cache_file = None
        if cache_path:
            os.makedirs(cache_path, exist_ok=True)
            cache_file = f"{cache_path}/{split}_gpu_cache_{max_samples}.pt"
            
            if os.path.exists(cache_file):
                print(f"Loading cached GPU dataset from {cache_file}...")
                self._load_cache(cache_file)
                return
        
        print(f"Building GPU dataset for {split} split...")
        print(f"Target: {max_samples} samples on GPU")
        self._build_dataset()
        
        if cache_file:
            self._save_cache(cache_file)
        
        self.num_samples = len(self.data['images'])
        print(f"✓ GPU dataset ready with {self.num_samples} samples")
        print(f"  Memory usage: {self._get_memory_usage():.1f} GB")
    
    def _build_dataset(self):
        """Build dataset directly on GPU"""
        try:
            # Import dex-ycb-toolkit
            from dex_ycb_toolkit.factory import get_dataset
            
            # Map split names
            if self.split == 'train':
                dex_split = 's0_train'
            elif self.split == 'val':
                dex_split = 's0_val'
            elif self.split == 'test':
                dex_split = 's0_val'  # Use val for test
            else:
                dex_split = f's0_{self.split}'
            
            dex_dataset = get_dataset(dex_split)
            
        except ImportError:
            logger.error("Could not import dex-ycb-toolkit. Make sure it's installed.")
            raise
        
        # Pre-allocate GPU tensors
        num_samples = min(len(dex_dataset), self.max_samples)
        
        # Allocate tensors
        print(f"Allocating GPU memory for {num_samples} samples...")
        self.data = {
            'images': torch.zeros((num_samples, 3, *self.image_size), 
                                device=self.device, dtype=self.dtype),
            'hand_joints': torch.full((num_samples, 21, 3), -1.0,
                                    device=self.device, dtype=self.dtype),
            'hand_joints_valid': torch.zeros((num_samples, 21),
                                           device=self.device, dtype=torch.bool),
            'mano_pose': torch.zeros((num_samples, 51),
                                   device=self.device, dtype=self.dtype),
            'mano_shape': torch.zeros((num_samples, 10),
                                    device=self.device, dtype=self.dtype),
            'object_poses': torch.zeros((num_samples, 10, 3, 4), 
                                      device=self.device, dtype=self.dtype),
            'object_ids': torch.zeros((num_samples, 10), 
                                    device=self.device, dtype=torch.long),
            'num_objects': torch.zeros((num_samples,), 
                                     device=self.device, dtype=torch.long),
            'has_hand': torch.zeros((num_samples,), 
                                  device=self.device, dtype=torch.bool),
            'camera_intrinsics': torch.zeros((num_samples, 3, 3),
                                           device=self.device, dtype=self.dtype),
        }
        
        # Additional features if requested
        if self.load_dinov2_features:
            # DINOv2-large outputs 1024-dim features
            self.data['dinov2_features'] = torch.zeros((num_samples, 1024),
                                                      device=self.device, dtype=self.dtype)
        
        # Load data in batches to avoid CPU memory issues
        batch_size = 100
        print("Loading and preprocessing data...")
        
        # Progress bar
        pbar = tqdm(total=num_samples, desc="Loading to GPU")
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            
            # Process batch
            for i in range(start_idx, end_idx):
                try:
                    sample = dex_dataset[i]
                    
                    # Load and preprocess image directly to GPU
                    color_path = sample['color_file']
                    img = cv2.imread(color_path)
                    if img is None:
                        logger.warning(f"Could not load image: {color_path}")
                        continue
                    
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.image_size)
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                    
                    # Normalize if requested
                    if self.normalize:
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        img_tensor = (img_tensor - mean) / std
                    
                    self.data['images'][i] = img_tensor.to(self.device, dtype=self.dtype)
                    
                    # Load labels
                    label_path = sample['label_file']
                    labels = np.load(label_path)
                    
                    # Hand data
                    if 'joint_3d' in labels and labels['joint_3d'].shape[0] > 0:
                        joints_3d = torch.from_numpy(labels['joint_3d'][0])
                        self.data['hand_joints'][i] = joints_3d.to(self.device, dtype=self.dtype)
                        self.data['hand_joints_valid'][i] = True
                        self.data['has_hand'][i] = True
                    
                    if 'pose_m' in labels and labels['pose_m'].shape[0] > 0:
                        pose = torch.from_numpy(labels['pose_m'][0])
                        # Handle different pose dimensions (48 or 51)
                        if pose.shape[0] == 48:
                            # Pad to 51 if needed
                            pose = torch.cat([pose, torch.zeros(3)])
                        elif pose.shape[0] > 51:
                            pose = pose[:51]
                        self.data['mano_pose'][i, :pose.shape[0]] = pose.to(self.device, dtype=self.dtype)
                    
                    # Object data
                    if 'pose_y' in labels:
                        obj_poses = labels['pose_y']
                        num_objs = min(len(obj_poses), 10)
                        if num_objs > 0:
                            obj_tensor = torch.from_numpy(obj_poses[:num_objs])
                            self.data['object_poses'][i, :num_objs] = obj_tensor.to(self.device, dtype=self.dtype)
                        self.data['num_objects'][i] = num_objs
                    
                    # Object IDs
                    ycb_ids = sample.get('ycb_ids', [])
                    if ycb_ids:
                        num_ids = min(len(ycb_ids), 10)
                        id_tensor = torch.tensor(ycb_ids[:num_ids], dtype=torch.long)
                        self.data['object_ids'][i, :num_ids] = id_tensor.to(self.device)
                    
                    # Camera intrinsics
                    if 'intrinsics' in sample:
                        K = sample['intrinsics']
                        if isinstance(K, dict):
                            # Convert dict to matrix
                            K_matrix = np.array([[K['fx'], 0, K['ppx']],
                                               [0, K['fy'], K['ppy']],
                                               [0, 0, 1]], dtype=np.float32)
                        else:
                            K_matrix = K
                        self.data['camera_intrinsics'][i] = torch.from_numpy(K_matrix).to(self.device, dtype=self.dtype)
                    
                    # MANO shape parameters
                    if 'mano_betas' in sample:
                        betas = torch.tensor(sample['mano_betas'], dtype=self.dtype)
                        self.data['mano_shape'][i] = betas.to(self.device)
                    
                except Exception as e:
                    logger.warning(f"Error processing sample {i}: {e}")
                    continue
                
                pbar.update(1)
        
        pbar.close()
        
        # Extract DINOv2 features if requested
        if self.load_dinov2_features:
            print("Extracting DINOv2 features...")
            self._extract_dinov2_features()
    
    def _extract_dinov2_features(self):
        """Pre-extract DINOv2 features for all images"""
        try:
            from transformers import AutoModel
            
            # Load DINOv2
            model = AutoModel.from_pretrained('facebook/dinov2-large')
            model = model.to(self.device)
            model.eval()
            
            batch_size = 32
            with torch.no_grad():
                for i in tqdm(range(0, self.num_samples, batch_size), desc="Extracting DINOv2"):
                    end_idx = min(i + batch_size, self.num_samples)
                    batch_images = self.data['images'][i:end_idx]
                    
                    # DINOv2 expects 224x224 images
                    if batch_images.shape[-1] != 224:
                        batch_images = torch.nn.functional.interpolate(
                            batch_images, size=(224, 224), mode='bilinear'
                        )
                    
                    # Extract features
                    outputs = model(batch_images)
                    features = outputs.last_hidden_state.mean(dim=1)  # Global pool
                    self.data['dinov2_features'][i:end_idx] = features
                    
            del model  # Free memory
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.warning(f"Could not extract DINOv2 features: {e}")
            self.data.pop('dinov2_features', None)
    
    def _save_cache(self, cache_file: str):
        """Save preprocessed data to disk"""
        print(f"Saving cache to {cache_file}...")
        
        # Move to CPU for saving (to save GPU memory temporarily)
        cpu_data = {}
        for key, value in self.data.items():
            if isinstance(value, torch.Tensor):
                cpu_data[key] = value.cpu()
            else:
                cpu_data[key] = value
        
        # Save with compression
        torch.save(cpu_data, cache_file, pickle_protocol=4)
        print(f"✓ Cache saved ({os.path.getsize(cache_file) / 1e9:.1f} GB)")
    
    def _load_cache(self, cache_file: str):
        """Load preprocessed data from disk"""
        print(f"Loading cache from {cache_file}...")
        cpu_data = torch.load(cache_file, map_location='cpu')
        
        # Move to GPU
        self.data = {}
        for key, value in cpu_data.items():
            if isinstance(value, torch.Tensor):
                self.data[key] = value.to(self.device, dtype=self.dtype)
            else:
                self.data[key] = value
        
        self.num_samples = len(self.data['images'])
        print(f"✓ Loaded {self.num_samples} samples")
    
    def _get_memory_usage(self):
        """Get GPU memory usage in GB"""
        return torch.cuda.memory_allocated(self.device) / 1e9
    
    def get_batch_generator(self, batch_size: int, shuffle: bool = True, drop_last: bool = False):
        """
        Generate batches directly from GPU memory
        Zero CPU overhead - everything stays on GPU
        """
        indices = torch.arange(self.num_samples, device=self.device)
        
        if shuffle:
            indices = indices[torch.randperm(self.num_samples, device=self.device)]
        
        # Handle drop_last
        num_batches = self.num_samples // batch_size
        if not drop_last and self.num_samples % batch_size != 0:
            num_batches += 1
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, self.num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Create batch - everything stays on GPU
            batch = {}
            for key, value in self.data.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value[batch_indices]
            
            # Rename keys to match expected format
            output_batch = {
                'image': batch['images'],
                'hand_joints': batch['hand_joints'],
                'hand_joints_valid': batch['hand_joints_valid'],
                'mano_pose': batch['mano_pose'],
                'mano_shape': batch['mano_shape'],
                'object_poses': batch['object_poses'],
                'object_ids': batch['object_ids'],
                'num_objects': batch['num_objects'],
                'has_hand': batch['has_hand'],
                'camera_intrinsics': batch['camera_intrinsics'],
            }
            
            # Add DINOv2 features if available
            if 'dinov2_features' in batch:
                output_batch['dinov2_features'] = batch['dinov2_features']
            
            yield output_batch
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Return a sample - already on GPU!"""
        return {
            'image': self.data['images'][idx],
            'hand_joints': self.data['hand_joints'][idx],
            'hand_joints_valid': self.data['hand_joints_valid'][idx],
            'mano_pose': self.data['mano_pose'][idx],
            'mano_shape': self.data['mano_shape'][idx],
            'object_poses': self.data['object_poses'][idx],
            'object_ids': self.data['object_ids'][idx],
            'num_objects': self.data['num_objects'][idx],
            'has_hand': self.data['has_hand'][idx],
            'camera_intrinsics': self.data['camera_intrinsics'][idx],
            'dinov2_features': self.data.get('dinov2_features', [None] * self.num_samples)[idx],
        }


class GPUDataLoader:
    """
    DataLoader replacement for GPU-cached dataset
    Provides familiar interface while keeping everything on GPU
    """
    def __init__(self, 
                 gpu_dataset: GPUCachedDataset, 
                 batch_size: int, 
                 shuffle: bool = True,
                 drop_last: bool = False):
        self.dataset = gpu_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
    
    def __iter__(self):
        return self.dataset.get_batch_generator(self.batch_size, self.shuffle, self.drop_last)
    
    def __len__(self):
        if self.drop_last:
            return self.dataset.num_samples // self.batch_size
        else:
            return (self.dataset.num_samples + self.batch_size - 1) // self.batch_size


def test_gpu_dataset():
    """Test the GPU-cached dataset"""
    print("Testing GPU-Cached Dataset...")
    
    # Create small test dataset
    dataset = GPUCachedDataset(
        split='train',
        max_samples=1000,  # Start small for testing
        cache_path='./gpu_cache_test'
    )
    
    # Create dataloader
    loader = GPUDataLoader(dataset, batch_size=32, shuffle=True)
    
    # Test iteration
    print(f"\nTesting dataloader with {len(loader)} batches...")
    for i, batch in enumerate(loader):
        if i == 0:
            print("\nFirst batch contents:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape} on {value.device}")
        if i >= 2:
            break
    
    print(f"\nGPU Memory Usage: {dataset._get_memory_usage():.2f} GB")
    print("✓ Test passed!")


# Notebook cell helper to replace standard dataloader
def create_gpu_cached_dataloaders(config: dict):
    """
    Create GPU-cached dataloaders for training
    
    Usage in notebook:
    ```python
    from data.gpu_cached_dataset import create_gpu_cached_dataloaders
    
    # Replace standard dataloaders with GPU-cached versions
    train_loader, val_loader = create_gpu_cached_dataloaders(config)
    ```
    """
    print("Creating GPU-cached datasets...")
    print(f"Target memory usage: {config.get('gpu_max_samples', 100000) / 1000:.1f} GB")
    
    # Create train dataset
    train_dataset = GPUCachedDataset(
        split='train',
        max_samples=config.get('gpu_max_samples', 100000),
        cache_path=config.get('gpu_cache_path', './gpu_cache'),
        dtype=torch.bfloat16 if config.get('use_bfloat16', True) else torch.float32,
        load_dinov2_features=config.get('preload_dinov2', False)
    )
    
    # Create val dataset (smaller)
    val_dataset = GPUCachedDataset(
        split='val',
        max_samples=config.get('gpu_max_samples_val', 10000),
        cache_path=config.get('gpu_cache_path', './gpu_cache'),
        dtype=torch.bfloat16 if config.get('use_bfloat16', True) else torch.float32,
        load_dinov2_features=config.get('preload_dinov2', False)
    )
    
    # Create loaders
    train_loader = GPUDataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=True,
        drop_last=True
    )
    
    val_loader = GPUDataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        drop_last=False
    )
    
    print(f"\n✓ GPU Datasets created:")
    print(f"  Train: {len(train_dataset)} samples ({train_dataset._get_memory_usage():.1f} GB)")
    print(f"  Val: {len(val_dataset)} samples ({val_dataset._get_memory_usage():.1f} GB)")
    print(f"  Total GPU memory: {train_dataset._get_memory_usage() + val_dataset._get_memory_usage():.1f} GB")
    
    return train_loader, val_loader


if __name__ == "__main__":
    test_gpu_dataset()