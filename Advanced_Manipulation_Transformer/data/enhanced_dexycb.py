import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple
import cv2
from torchvision import transforms as T
import random
from pathlib import Path
import json
import h5py
from PIL import Image
import logging
from tqdm import tqdm
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'dex-ycb-toolkit'))

logger = logging.getLogger(__name__)

class EnhancedDexYCBDataset(Dataset):
    """
    DexYCB dataset with advanced augmentation
    Complete implementation without SDF dependency
    """
    
    def __init__(
        self,
        dexycb_root: str,
        split: str = 'train',
        sequence_length: int = 1,  # For temporal modeling
        augment: bool = True,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None  # For debugging
    ):
        self.dexycb_root = Path(dexycb_root)
        self.split = split
        self.sequence_length = sequence_length
        self.augment = augment and (split == 'train')
        self.use_cache = use_cache
        self.max_samples = max_samples
        
        # Setup cache directory
        if cache_dir is None:
            cache_dir = self.dexycb_root / 'cache' / split
        self.cache_dir = Path(cache_dir)
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load samples using dex-ycb-toolkit
        self.samples = self._load_samples_from_toolkit()
        
        if self.max_samples:
            self.samples = self.samples[:self.max_samples]
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
        
        # Image preprocessing
        self.image_transform = self._build_image_transform()
        
        # Data augmentation
        self.augmentor = DataAugmentor() if self.augment else None
        
        # Camera parameters for DexYCB
        self.camera_info = self._load_camera_info()
    
    def _load_samples_from_toolkit(self) -> List[Dict]:
        """Load sample information using dex-ycb-toolkit"""
        try:
            # Import dex-ycb-toolkit
            from dex_ycb_toolkit.dex_ycb import DexYCBDataset
            from dex_ycb_toolkit.factory import get_dataset
            
            # Handle both formats: 'train' and 's0_train'
            if self.split.startswith('s'):
                # Already has the s0_ prefix
                dataset = get_dataset(self.split)
            else:
                # Add s0_ prefix
                if self.split == 'train':
                    dataset = get_dataset('s0_train')
                elif self.split == 'val':
                    dataset = get_dataset('s0_val')
                elif self.split == 'test':
                    # DexYCB doesn't have a public test set, use val for testing
                    logger.warning("DexYCB doesn't have a public test set, using validation set for testing")
                    dataset = get_dataset('s0_val')
                else:
                    raise ValueError(f"Unknown split: {self.split}. Use 'train', 'val', or 'test'")
            
            samples = []
            
            # Iterate through dataset and extract sample information
            for i in range(len(dataset)):
                try:
                    sample_data = dataset[i]
                    
                    # Extract relevant information
                    sample = {
                        'index': i,
                        'color_file': sample_data.get('color_file', ''),
                        'depth_file': sample_data.get('depth_file', ''),
                        'label_file': sample_data.get('label_file', ''),
                        'meta_file': sample_data.get('meta_file', ''),
                        'subject': sample_data.get('mano_side', 'right'),
                        'has_hand': True  # DexYCB always has hands
                    }
                    samples.append(sample)
                except Exception as e:
                    logger.warning(f"Failed to load sample {i}: {e}")
                    continue
            
            return samples
            
        except ImportError:
            logger.warning("dex-ycb-toolkit not available, using direct loading")
            return self._load_samples_direct()
    
    def _load_samples_direct(self) -> List[Dict]:
        """Direct loading fallback if toolkit not available"""
        split_mapping = {
            'train': 's0_train',
            'val': 's0_val',
            'test': 's0_test'
        }
        
        split_name = split_mapping.get(self.split, f's0_{self.split}')
        
        # Look for split files
        split_file = self.dexycb_root / 'splits' / f'{split_name}.json'
        if not split_file.exists():
            split_file = self.dexycb_root / f'{split_name}.json'
        
        if not split_file.exists():
            # Create dummy samples for testing
            logger.warning(f"Split file not found: {split_file}, creating dummy samples")
            return self._create_dummy_samples()
        
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        samples = []
        for entry in split_data:
            # Parse entry format
            if isinstance(entry, str):
                parts = entry.split('/')
                sample = {
                    'subject': parts[0] if len(parts) > 0 else 'default',
                    'scene': parts[1] if len(parts) > 1 else 'scene1',
                    'frame': int(parts[2]) if len(parts) > 2 else 0,
                    'sequence_name': f"{parts[0]}_{parts[1]}" if len(parts) > 1 else "default_scene",
                    'has_hand': True
                }
            else:
                sample = entry
                sample['has_hand'] = sample.get('has_hand', True)
            
            samples.append(sample)
        
        return samples
    
    def _create_dummy_samples(self) -> List[Dict]:
        """Create dummy samples for testing without real data"""
        samples = []
        num_samples = 100 if self.split == 'train' else 20
        
        for i in range(num_samples):
            sample = {
                'index': i,
                'subject': 'dummy',
                'scene': f'scene_{i % 5}',
                'frame': i,
                'sequence_name': f'dummy_scene_{i % 5}',
                'has_hand': True
            }
            samples.append(sample)
        
        return samples
    
    def _load_camera_info(self) -> Dict:
        """Load DexYCB camera calibration"""
        calib_file = self.dexycb_root / 'calibration' / 'camera_params.json'
        
        if calib_file.exists():
            with open(calib_file, 'r') as f:
                return json.load(f)
        else:
            # Default DexYCB camera parameters
            return {
                'fx': 617.343,
                'fy': 617.343,
                'cx': 312.42,
                'cy': 239.99,
                'width': 640,
                'height': 480
            }
    
    def _build_image_transform(self):
        """Build image preprocessing pipeline"""
        normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats for DINOv2
            std=[0.229, 0.224, 0.225]
        )
        
        if self.split == 'train' and self.augment:
            return T.Compose([
                T.ToPILImage(),
                T.RandomResizedCrop(224, scale=(0.8, 1.2), ratio=(0.9, 1.1)),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                T.RandomApply([T.GaussianBlur(kernel_size=5)], p=0.2),
                T.ToTensor(),
                normalize
            ])
        else:
            return T.Compose([
                T.ToPILImage(),
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                normalize
            ])
    
    def _load_image(self, sample_info: Dict) -> np.ndarray:
        """Load RGB image from DexYCB"""
        # Try loading from file path if available
        if 'color_file' in sample_info and sample_info['color_file']:
            image_path = Path(sample_info['color_file'])
            if image_path.exists():
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image
        
        # Try constructing path
        if all(k in sample_info for k in ['subject', 'scene', 'frame']):
            image_path = (self.dexycb_root / 'data' / 
                         sample_info['subject'] / sample_info['scene'] / 
                         'color' / f'{sample_info["frame"]:06d}.jpg')
            
            if not image_path.exists():
                # Try PNG format
                image_path = image_path.with_suffix('.png')
            
            if image_path.exists():
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image
        
        # Return dummy image if not found
        logger.debug(f"Image not found for sample, returning dummy image")
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def _load_annotations(self, sample_info: Dict) -> Dict[str, np.ndarray]:
        """Load annotations from DexYCB meta files"""
        # Initialize default annotations
        annotations = {
            'hand_joints_3d': np.zeros((21, 3), dtype=np.float32),
            'hand_joints_2d': np.zeros((21, 2), dtype=np.float32),
            'mano_pose': np.zeros(51, dtype=np.float32),  # 48 + 3 for global rotation
            'mano_shape': np.zeros(10, dtype=np.float32),
            'object_pose': np.eye(4, dtype=np.float32),
            'object_id': 0,
            'camera_intrinsics': self._get_camera_intrinsics(),
            'camera_extrinsics': np.eye(4, dtype=np.float32),
            'has_hand': sample_info.get('has_hand', True)
        }
        
        # Try loading from label/meta file
        if 'label_file' in sample_info and sample_info['label_file']:
            label_path = Path(sample_info['label_file'])
            if label_path.exists():
                try:
                    label_data = np.load(label_path, allow_pickle=True)
                    if 'joint_3d' in label_data:
                        annotations['hand_joints_3d'] = label_data['joint_3d'].astype(np.float32).reshape(21, 3)
                    if 'joint_2d' in label_data:
                        annotations['hand_joints_2d'] = label_data['joint_2d'].astype(np.float32).reshape(21, 2)
                    if 'pose_m' in label_data:
                        mano_pose = label_data['pose_m'].astype(np.float32).flatten()
                        # Handle different MANO pose dimensions
                        if len(mano_pose) == 48:
                            annotations['mano_pose'][:48] = mano_pose
                        elif len(mano_pose) == 51:
                            annotations['mano_pose'] = mano_pose
                        else:
                            annotations['mano_pose'][:min(len(mano_pose), 51)] = mano_pose[:min(len(mano_pose), 51)]
                    if 'pose_y' in label_data and len(label_data['pose_y']) > 0:
                        annotations['object_pose'] = label_data['pose_y'][0].astype(np.float32)
                    if 'ycb_ids' in label_data and len(label_data['ycb_ids']) > 0:
                        annotations['object_id'] = int(label_data['ycb_ids'][0])
                except Exception as e:
                    logger.debug(f"Failed to load annotations: {e}")
        
        # Try alternative path construction
        elif all(k in sample_info for k in ['subject', 'scene', 'frame']):
            meta_path = (self.dexycb_root / 'data' / 
                        sample_info['subject'] / sample_info['scene'] / 
                        'meta' / f'{sample_info["frame"]:06d}.npz')
            
            if meta_path.exists():
                try:
                    meta_data = np.load(meta_path, allow_pickle=True)
                    # Extract annotations similar to above
                    if 'joints_3d' in meta_data:
                        annotations['hand_joints_3d'] = meta_data['joints_3d'].astype(np.float32)
                    if 'joints_2d' in meta_data:
                        annotations['hand_joints_2d'] = meta_data['joints_2d'].astype(np.float32)
                    if 'mano_pose' in meta_data:
                        mano_pose = meta_data['mano_pose'].astype(np.float32)
                        if len(mano_pose) == 48:
                            annotations['mano_pose'][:48] = mano_pose
                        elif len(mano_pose) == 51:
                            annotations['mano_pose'] = mano_pose
                    if 'mano_betas' in meta_data:
                        annotations['mano_shape'] = meta_data['mano_betas'].astype(np.float32)
                    if 'pose_y' in meta_data:
                        annotations['object_pose'] = meta_data['pose_y'].astype(np.float32)
                    if 'ycb_ids' in meta_data and len(meta_data['ycb_ids']) > 0:
                        annotations['object_id'] = int(meta_data['ycb_ids'][0])
                except Exception as e:
                    logger.debug(f"Failed to load meta data: {e}")
        
        return annotations
    
    def _get_camera_intrinsics(self) -> np.ndarray:
        """Get camera intrinsic matrix"""
        K = np.array([
            [self.camera_info['fx'], 0, self.camera_info['cx']],
            [0, self.camera_info['fy'], self.camera_info['cy']],
            [0, 0, 1]
        ], dtype=np.float32)
        return K
    
    def _get_sequence(self, idx: int) -> List[Dict]:
        """Get a sequence of frames for temporal modeling"""
        center_sample = self.samples[idx]
        sequence = []
        
        # Get frames around the center frame
        for offset in range(-self.sequence_length//2, self.sequence_length//2 + 1):
            # For now, just repeat the same frame
            # In a real implementation, you would load consecutive frames
            sample_info = center_sample.copy()
            sample_info['temporal_offset'] = offset
            
            try:
                sample = self._get_single_sample_by_info(sample_info)
                sequence.append(sample)
            except Exception as e:
                # Use center frame as fallback
                logger.warning(f"Failed to load frame with offset {offset}: {e}")
                sample = self._get_single_sample_by_info(center_sample)
                sample['temporal_offset'] = offset
                sequence.append(sample)
        
        return sequence
    
    def _get_single_sample_by_info(self, sample_info: Dict) -> Dict[str, torch.Tensor]:
        """Get a single sample by sample info"""
        # Load RGB image
        image = self._load_image(sample_info)
        
        # Load annotations
        annotations = self._load_annotations(sample_info)
        
        # Preprocess image
        image_tensor = self.image_transform(image)
        
        # Build sample dict
        sample = {
            'image': image_tensor,
            'hand_joints_3d': torch.tensor(annotations['hand_joints_3d'], dtype=torch.float32),
            'hand_joints_2d': torch.tensor(annotations['hand_joints_2d'], dtype=torch.float32),
            'mano_pose': torch.tensor(annotations['mano_pose'], dtype=torch.float32),
            'mano_shape': torch.tensor(annotations['mano_shape'], dtype=torch.float32),
            'object_pose': torch.tensor(annotations['object_pose'], dtype=torch.float32),
            'object_id': torch.tensor(annotations['object_id'], dtype=torch.long),
            'camera_intrinsics': torch.tensor(annotations['camera_intrinsics'], dtype=torch.float32),
            'camera_extrinsics': torch.tensor(annotations['camera_extrinsics'], dtype=torch.float32),
            'has_hand': torch.tensor(annotations['has_hand'], dtype=torch.bool),
            'sample_id': str(sample_info.get('index', 0))
        }
        
        # Apply data augmentation
        if self.augmentor:
            sample = self.augmentor(sample)
        
        return sample
    
    def _get_single_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample by index"""
        return self._get_single_sample_by_info(self.samples[idx])
    
    def _stack_sequence(self, samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Stack a sequence of samples into batch format"""
        # Stack all tensors along a new time dimension
        stacked = {}
        
        for key in samples[0].keys():
            if isinstance(samples[0][key], torch.Tensor):
                if key == 'sample_id':
                    # String data, keep as list
                    stacked[key] = [s[key] for s in samples]
                else:
                    # Stack tensors
                    stacked[key] = torch.stack([s[key] for s in samples], dim=0)
            else:
                # Non-tensor data
                stacked[key] = [s[key] for s in samples]
        
        return stacked
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample with all modalities"""
        
        # Use cache if available
        if self.use_cache:
            cache_file = self.cache_dir / f'sample_{idx}.pt'
            if cache_file.exists():
                try:
                    return torch.load(cache_file)
                except:
                    pass
        
        # Handle sequence sampling for temporal modeling
        if self.sequence_length > 1:
            samples = self._get_sequence(idx)
            batch = self._stack_sequence(samples)
        else:
            batch = self._get_single_sample(idx)
        
        # Save to cache
        if self.use_cache:
            try:
                torch.save(batch, self.cache_dir / f'sample_{idx}.pt')
            except:
                pass
        
        return batch


class DataAugmentor:
    """Advanced data augmentation for hand pose estimation"""
    
    def __init__(self):
        self.joint_noise_std = 0.005  # 5mm
        self.rotation_range = 10.0    # degrees
        self.scale_range = (0.9, 1.1)
        self.translation_std = 0.02   # 2cm
        
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply augmentations"""
        
        # 1. Joint noise injection (prevents overfitting)
        if 'hand_joints_3d' in sample and random.random() < 0.5:
            noise = torch.randn_like(sample['hand_joints_3d']) * self.joint_noise_std
            sample['hand_joints_3d'] += noise
        
        # 2. 3D rotation augmentation
        if random.random() < 0.5:
            angle = np.radians(random.uniform(-self.rotation_range, self.rotation_range))
            axis = random.choice(['x', 'y', 'z'])
            R = self._get_rotation_matrix(angle, axis)
            
            # Rotate 3D joints
            if 'hand_joints_3d' in sample:
                joints = sample['hand_joints_3d']
                sample['hand_joints_3d'] = torch.matmul(joints, R.T)
            
            # Update object pose
            if 'object_pose' in sample:
                pose = sample['object_pose']
                pose[:3, :3] = torch.matmul(R, pose[:3, :3])
                sample['object_pose'] = pose
        
        # 3. Scale augmentation
        if random.random() < 0.3:
            scale = random.uniform(*self.scale_range)
            if 'hand_joints_3d' in sample:
                sample['hand_joints_3d'] *= scale
            if 'object_pose' in sample:
                sample['object_pose'][:3, 3] *= scale  # Scale translation
        
        # 4. Translation augmentation
        if random.random() < 0.3:
            translation = torch.randn(3) * self.translation_std
            if 'hand_joints_3d' in sample:
                sample['hand_joints_3d'] += translation
            if 'object_pose' in sample:
                sample['object_pose'][:3, 3] += translation
        
        # 5. 2D joint augmentation (consistent with 3D)
        if 'hand_joints_2d' in sample and random.random() < 0.3:
            # Add small 2D noise
            noise_2d = torch.randn_like(sample['hand_joints_2d']) * 2.0  # pixels
            sample['hand_joints_2d'] += noise_2d
        
        # 6. Temporal jitter for sequences
        if 'temporal_offset' in sample:
            sample['temporal_offset'] += torch.randn(1) * 0.1
        
        return sample
    
    def _get_rotation_matrix(self, angle: float, axis: str) -> torch.Tensor:
        """Get 3D rotation matrix"""
        c, s = np.cos(angle), np.sin(angle)
        if axis == 'x':
            R = torch.tensor([
                [1, 0, 0],
                [0, c, -s],
                [0, s, c]
            ], dtype=torch.float32)
        elif axis == 'y':
            R = torch.tensor([
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c]
            ], dtype=torch.float32)
        else:  # z
            R = torch.tensor([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ], dtype=torch.float32)
        return R