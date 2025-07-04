"""
DexYCB Dataset Loader for Video-to-Manipulation Transformer
Loads and preprocesses data from the DexYCB dataset
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dex-ycb-toolkit'))
from dex_ycb_toolkit.factory import get_dataset as get_dexycb_dataset


def dexycb_collate_fn(batch):
    """
    Custom collate function to handle variable-sized data in DexYCB
    """
    # Find max number of objects across the batch
    max_objects = max(sample['object_poses'].shape[0] for sample in batch)
    
    collated = {}
    
    for key in batch[0].keys():
        if key == 'object_poses':
            # Pad object poses to max_objects
            padded_poses = []
            for sample in batch:
                poses = sample[key]
                if poses.shape[0] < max_objects:
                    # Pad with zeros
                    pad_size = max_objects - poses.shape[0]
                    padding = torch.zeros(pad_size, *poses.shape[1:])
                    poses = torch.cat([poses, padding], dim=0)
                padded_poses.append(poses)
            collated[key] = torch.stack(padded_poses)
            
        elif key == 'ycb_ids':
            # Pad ycb_ids lists to same length
            padded_ids = []
            for sample in batch:
                ids = sample[key]
                if len(ids) < max_objects:
                    # Pad with -1 (invalid ID)
                    ids = ids + [-1] * (max_objects - len(ids))
                padded_ids.append(ids)
            collated[key] = padded_ids
            
        elif isinstance(batch[0][key], torch.Tensor):
            # Stack tensors
            try:
                collated[key] = torch.stack([sample[key] for sample in batch])
            except:
                # If stacking fails, just return as list
                collated[key] = [sample[key] for sample in batch]
                
        elif isinstance(batch[0][key], (int, float, str)):
            # Lists of primitives
            collated[key] = [sample[key] for sample in batch]
            
        elif isinstance(batch[0][key], dict):
            # Dictionaries (like intrinsics)
            collated[key] = [sample[key] for sample in batch]
            
        else:
            # Default: just create a list
            collated[key] = [sample[key] for sample in batch]
    
    return collated


class DexYCBDataset(Dataset):
    """
    Dataset class for loading DexYCB data with proper preprocessing
    for the Video-to-Manipulation Transformer
    """
    
    def __init__(self, split='s0_train', sequence_length=16, stride=1, max_objects=10):
        """
        Args:
            split: Dataset split (s0_train, s0_val, etc.)
            sequence_length: Number of frames per sequence
            stride: Frame sampling stride
            max_objects: Maximum number of objects to pad to
        """
        self.split = split
        self.sequence_length = sequence_length
        self.stride = stride
        self.max_objects = max_objects
        
        # Load DexYCB dataset
        self.dexycb = get_dexycb_dataset(split)
        self.length = len(self.dexycb)
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        """
        Returns a sample with annotations
        """
        # Get sample from DexYCB
        sample = self.dexycb[idx]
        
        # Load color image
        color_img = cv2.imread(sample['color_file'])
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        
        # Load depth image if available
        depth_img = None
        if os.path.exists(sample['depth_file']):
            depth_img = cv2.imread(sample['depth_file'], cv2.IMREAD_ANYDEPTH)
        
        # Load labels
        label = np.load(sample['label_file'])
        
        # Pad object poses to max_objects
        object_poses = label['pose_y']  # [num_obj, 3, 4]
        num_objects = object_poses.shape[0]
        if num_objects < self.max_objects:
            # Pad with zeros
            padding = np.zeros((self.max_objects - num_objects, 3, 4), dtype=object_poses.dtype)
            object_poses = np.concatenate([object_poses, padding], axis=0)
        elif num_objects > self.max_objects:
            # Truncate if too many objects
            object_poses = object_poses[:self.max_objects]
            
        # Pad YCB IDs
        ycb_ids = sample['ycb_ids']
        if len(ycb_ids) < self.max_objects:
            ycb_ids = ycb_ids + [-1] * (self.max_objects - len(ycb_ids))
        elif len(ycb_ids) > self.max_objects:
            ycb_ids = ycb_ids[:self.max_objects]
        
        # Convert to tensors and extract relevant information
        data = {
            'color': torch.from_numpy(color_img).permute(2, 0, 1).float() / 255.0,  # [3, H, W] normalized
            'depth': torch.from_numpy(depth_img).float() if depth_img is not None else None,
            'segmentation': torch.from_numpy(label['seg']).long(),
            'object_poses': torch.from_numpy(object_poses).float(),  # [max_objects, 3, 4]
            'hand_pose': torch.from_numpy(label['pose_m']).float(),  # [1, 51] MANO params
            'hand_joints_3d': torch.from_numpy(label['joint_3d']).float(),  # [1, 21, 3]
            'hand_joints_2d': torch.from_numpy(label['joint_2d']).float(),  # [1, 21, 2]
            'intrinsics': sample['intrinsics'],
            'ycb_ids': torch.tensor(ycb_ids).long(),  # Now a tensor of fixed size
            'mano_side': sample['mano_side'],
            'mano_betas': torch.tensor(sample['mano_betas']).float(),
            'num_objects': torch.tensor(num_objects).long()  # Store original number of objects
        }
        
        return data
    
    def get_sequence(self, start_idx, end_idx=None):
        """
        Get a sequence of frames for temporal processing
        """
        if end_idx is None:
            end_idx = min(start_idx + self.sequence_length * self.stride, len(self))
            
        sequence = []
        for i in range(start_idx, end_idx, self.stride):
            if i < len(self):
                sequence.append(self[i])
                
        return sequence


class DexYCBSequenceDataset(Dataset):
    """
    Dataset that returns sequences of frames for temporal training
    """
    
    def __init__(self, split='s0_train', sequence_length=16, stride=1, overlap=8):
        self.base_dataset = DexYCBDataset(split, sequence_length, stride)
        self.sequence_length = sequence_length
        self.stride = stride
        self.overlap = overlap
        
        # Calculate valid sequence starts
        self.valid_starts = []
        step = sequence_length - overlap
        for i in range(0, len(self.base_dataset) - sequence_length * stride, step):
            self.valid_starts.append(i)
            
    def __len__(self):
        return len(self.valid_starts)
        
    def __getitem__(self, idx):
        start_idx = self.valid_starts[idx]
        sequence = self.base_dataset.get_sequence(start_idx)
        
        # Stack sequence data
        stacked_data = {}
        for key in sequence[0].keys():
            if sequence[0][key] is not None and isinstance(sequence[0][key], torch.Tensor):
                stacked_data[key] = torch.stack([frame[key] for frame in sequence])
            else:
                # For non-tensor data, just take from first frame
                stacked_data[key] = sequence[0][key]
                
        stacked_data['sequence_length'] = len(sequence)
        
        return stacked_data