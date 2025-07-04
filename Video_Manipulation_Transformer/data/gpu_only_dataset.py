"""
Pure GPU Dataset Implementation
Eliminates ALL CPU-GPU transfers by keeping everything on GPU
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import cv2
from pathlib import Path
import os


class GPUOnlyDataset:
    """
    Dataset that lives entirely on GPU memory
    Pre-loads and pre-processes all data to eliminate CPU bottlenecks
    """
    
    def __init__(self, 
                 split: str = 's0_train',
                 max_samples: int = 50000,
                 image_size: Tuple[int, int] = (224, 224),
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float32,
                 cache_path: Optional[str] = None):
        """
        Args:
            split: Dataset split
            max_samples: Maximum samples to load (memory limited)
            image_size: Target image size
            device: GPU device
            dtype: Data type (float32 or bfloat16)
            cache_path: Path to save/load preprocessed data
        """
        self.split = split
        self.max_samples = max_samples
        self.image_size = image_size
        self.device = device
        self.dtype = dtype
        self.cache_path = cache_path
        
        # Check for cached data first
        if cache_path and os.path.exists(f"{cache_path}/{split}_gpu_cache.pt"):
            print(f"Loading cached GPU dataset from {cache_path}...")
            self._load_cache()
        else:
            print(f"Building GPU dataset for {split}...")
            self._build_dataset()
            if cache_path:
                self._save_cache()
        
        self.num_samples = len(self.data['color'])
        print(f"✓ GPU dataset ready with {self.num_samples} samples")
        print(f"  Memory usage: {self._get_memory_usage():.1f} GB")
    
    def _build_dataset(self):
        """Build dataset directly on GPU"""
        # Import here to avoid circular dependency
        from dex_ycb_toolkit.factory import get_dataset
        dex_dataset = get_dataset(self.split)
        
        # Pre-allocate GPU tensors
        num_samples = min(len(dex_dataset), self.max_samples)
        
        # Allocate tensors
        print(f"Allocating GPU memory for {num_samples} samples...")
        self.data = {
            'color': torch.zeros((num_samples, 3, *self.image_size), 
                               device=self.device, dtype=self.dtype),
            'hand_joints_3d': torch.full((num_samples, 21, 3), -1.0,
                                        device=self.device, dtype=self.dtype),
            'hand_joints_2d': torch.full((num_samples, 21, 2), -1.0,
                                        device=self.device, dtype=self.dtype),
            'hand_pose': torch.zeros((num_samples, 51),  # MANO pose is 51D
                                   device=self.device, dtype=self.dtype),
            'object_poses': torch.zeros((num_samples, 10, 3, 4), 
                                      device=self.device, dtype=self.dtype),
            'ycb_ids': torch.zeros((num_samples, 10), 
                                 device=self.device, dtype=torch.long),
            'num_objects': torch.zeros((num_samples,), 
                                     device=self.device, dtype=torch.long),
            'mano_side': [],  # Keep as list
            'mano_betas': torch.zeros((num_samples, 10), 
                                     device=self.device, dtype=self.dtype),
            'has_hand': torch.zeros((num_samples,), 
                                  device=self.device, dtype=torch.bool),
        }
        
        # Load data in batches to avoid CPU memory issues
        batch_size = 100
        print("Loading and preprocessing data...")
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            
            # Process batch
            for i in range(start_idx, end_idx):
                sample = dex_dataset[i]
                
                # Load and preprocess image directly to GPU
                color_path = sample['color_file']
                img = cv2.imread(color_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.image_size)
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                self.data['color'][i] = img_tensor.to(self.device, dtype=self.dtype)
                
                # Load labels
                label_path = sample['label_file']
                labels = np.load(label_path)
                
                # Hand data
                if 'joint_3d' in labels and labels['joint_3d'].shape[0] > 0:
                    joints_3d = torch.from_numpy(labels['joint_3d'][0])
                    self.data['hand_joints_3d'][i] = joints_3d.to(self.device, dtype=self.dtype)
                    self.data['has_hand'][i] = True
                
                if 'joint_2d' in labels:
                    joints_2d = torch.from_numpy(labels['joint_2d'][0])
                    self.data['hand_joints_2d'][i] = joints_2d.to(self.device, dtype=self.dtype)
                
                if 'pose_m' in labels:
                    pose = torch.from_numpy(labels['pose_m'][0])
                    # Handle different pose dimensions (48 or 51)
                    if pose.shape[0] == 48:
                        # Pad to 51 if needed
                        pose = torch.cat([pose, torch.zeros(3)])
                    elif pose.shape[0] > 51:
                        pose = pose[:51]
                    self.data['hand_pose'][i, :pose.shape[0]] = pose.to(self.device, dtype=self.dtype)
                
                # Object data
                if 'pose_y' in labels:
                    obj_poses = labels['pose_y']
                    num_objs = min(len(obj_poses), 10)
                    if num_objs > 0:
                        obj_tensor = torch.from_numpy(obj_poses[:num_objs])
                        self.data['object_poses'][i, :num_objs] = obj_tensor.to(self.device, dtype=self.dtype)
                    self.data['num_objects'][i] = num_objs
                
                # YCB IDs
                ycb_ids = sample.get('ycb_ids', [])
                if ycb_ids:
                    num_ids = min(len(ycb_ids), 10)
                    id_tensor = torch.tensor(ycb_ids[:num_ids], dtype=torch.long)
                    self.data['ycb_ids'][i, :num_ids] = id_tensor.to(self.device)
                
                # Metadata
                self.data['mano_side'].append(sample.get('mano_side', 'right'))
                if 'mano_betas' in sample:
                    beta_tensor = torch.tensor(sample['mano_betas'], dtype=self.dtype)
                    self.data['mano_betas'][i] = beta_tensor.to(self.device)
            
            if (end_idx % 1000) == 0:
                print(f"  Loaded {end_idx}/{num_samples} samples to GPU "
                      f"({self._get_memory_usage():.1f} GB used)")
        
        # Normalize images (ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        self.data['color'] = (self.data['color'] - mean) / std
        
    def _save_cache(self):
        """Save preprocessed data to disk"""
        if not self.cache_path:
            return
        
        os.makedirs(self.cache_path, exist_ok=True)
        cache_file = f"{self.cache_path}/{self.split}_gpu_cache.pt"
        
        print(f"Saving cache to {cache_file}...")
        # Move to CPU for saving
        cpu_data = {}
        for key, value in self.data.items():
            if isinstance(value, torch.Tensor):
                cpu_data[key] = value.cpu()
            else:
                cpu_data[key] = value
        
        torch.save(cpu_data, cache_file)
        print(f"✓ Cache saved")
    
    def _load_cache(self):
        """Load preprocessed data from disk"""
        cache_file = f"{self.cache_path}/{self.split}_gpu_cache.pt"
        
        print(f"Loading cache from {cache_file}...")
        cpu_data = torch.load(cache_file)
        
        # Move to GPU
        self.data = {}
        for key, value in cpu_data.items():
            if isinstance(value, torch.Tensor):
                self.data[key] = value.to(self.device, dtype=self.dtype)
            else:
                self.data[key] = value
    
    def _get_memory_usage(self):
        """Get GPU memory usage in GB"""
        return torch.cuda.memory_allocated(self.device) / 1e9
    
    def get_batch_generator(self, batch_size: int, shuffle: bool = True):
        """
        Generate batches directly from GPU memory
        Zero CPU overhead - everything stays on GPU
        """
        indices = torch.arange(self.num_samples, device=self.device)
        
        if shuffle:
            indices = indices[torch.randperm(self.num_samples, device=self.device)]
        
        for start_idx in range(0, self.num_samples, batch_size):
            end_idx = min(start_idx + batch_size, self.num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Create batch - everything stays on GPU
            batch = {}
            for key, value in self.data.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value[batch_indices]
                elif isinstance(value, list):
                    # Handle list data (mano_side)
                    batch[key] = [value[idx.item()] for idx in batch_indices]
            
            yield batch
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Return a sample - already on GPU!"""
        return {
            'color': self.data['color'][idx],
            'hand_joints_3d': self.data['hand_joints_3d'][idx],
            'hand_joints_2d': self.data['hand_joints_2d'][idx],
            'hand_pose': self.data['hand_pose'][idx],
            'object_poses': self.data['object_poses'][idx],
            'ycb_ids': self.data['ycb_ids'][idx],
            'num_objects': self.data['num_objects'][idx],
            'has_hand': self.data['has_hand'][idx],
        }


class GPUBatchGenerator:
    """
    Wrapper to make GPU dataset compatible with training loops
    """
    def __init__(self, gpu_dataset: GPUOnlyDataset, batch_size: int, shuffle: bool = True):
        self.dataset = gpu_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self):
        return self.dataset.get_batch_generator(self.batch_size, self.shuffle)
    
    def __len__(self):
        return (self.dataset.num_samples + self.batch_size - 1) // self.batch_size


def benchmark_gpu_dataset():
    """Benchmark GPU dataset performance"""
    import time
    
    print("Benchmarking GPU-only dataset...")
    
    # Create dataset
    dataset = GPUOnlyDataset(
        split='s0_train',
        max_samples=10000,
        image_size=(224, 224),
        dtype=torch.float32
    )
    
    # Create batch generator
    batch_gen = GPUBatchGenerator(dataset, batch_size=512, shuffle=True)
    
    # Warmup
    for _ in range(5):
        batch = next(iter(batch_gen))
    
    torch.cuda.synchronize()
    
    # Benchmark
    num_epochs = 3
    total_batches = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        for batch in batch_gen:
            # Simulate some computation
            _ = batch['color'].mean()
            torch.cuda.synchronize()
            total_batches += 1
    
    total_time = time.time() - start_time
    
    print(f"\nBenchmark Results:")
    print(f"  Total batches: {total_batches}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Batches/sec: {total_batches/total_time:.1f}")
    print(f"  Samples/sec: {total_batches * 512 / total_time:.1f}")
    print(f"  GPU Memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")


if __name__ == "__main__":
    benchmark_gpu_dataset()