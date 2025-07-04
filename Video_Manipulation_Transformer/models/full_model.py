"""
Full Video-to-Manipulation Transformer Model
Combines all encoders, temporal fusion, and action decoder
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from .encoders.hand_encoder import HandPoseEncoder
from .encoders.object_encoder import ObjectPoseEncoder
from .encoders.contact_encoder import ContactDetectionEncoder
from .temporal_fusion import TemporalFusionEncoder
from .action_decoder import ActionDecoder
from data.preprocessing import VideoPreprocessor, PositionalEncoding


class VideoManipulationTransformer(nn.Module):
    """
    Complete transformer model for video-to-manipulation
    Processes video sequences and outputs robot manipulation commands
    """
    
    def __init__(self,
                 # Encoder configs
                 hand_encoder_config: Optional[Dict] = None,
                 object_encoder_config: Optional[Dict] = None,
                 contact_encoder_config: Optional[Dict] = None,
                 # Fusion config
                 temporal_fusion_config: Optional[Dict] = None,
                 # Decoder config
                 action_decoder_config: Optional[Dict] = None,
                 # General configs
                 patch_size: int = 16,
                 image_size: Tuple[int, int] = (224, 224),
                 max_seq_length: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        
        self.patch_size = patch_size
        self.image_size = image_size
        self.max_seq_length = max_seq_length
        
        # Calculate patch dimensions
        self.num_patches_h = image_size[0] // patch_size
        self.num_patches_w = image_size[1] // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.patch_dim = 3 * patch_size * patch_size
        
        # Initialize preprocessor
        self.preprocessor = VideoPreprocessor(
            image_size=image_size,
            patch_size=patch_size,
            normalize=True
        )
        
        # Initialize positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=512,  # Will be projected to match encoder dims
            max_len=max_seq_length
        )
        
        # Default configs
        if hand_encoder_config is None:
            hand_encoder_config = {
                'input_dim': self.patch_dim,
                'hidden_dim': 512,
                'num_layers': 6,
                'num_heads': 8,
                'dropout': dropout
            }
            
        if object_encoder_config is None:
            object_encoder_config = {
                'input_dim': self.patch_dim,
                'hidden_dim': 512,
                'num_layers': 6,
                'num_heads': 8,
                'dropout': dropout
            }
            
        if contact_encoder_config is None:
            contact_encoder_config = {
                'input_dim': self.patch_dim,
                'hidden_dim': 256,
                'num_layers': 4,
                'num_heads': 8,
                'dropout': dropout
            }
            
        if temporal_fusion_config is None:
            temporal_fusion_config = {
                'hand_dim': hand_encoder_config['hidden_dim'],
                'object_dim': object_encoder_config['hidden_dim'],
                'contact_dim': contact_encoder_config['hidden_dim'],
                'hidden_dim': 1024,
                'num_layers': 8,
                'num_heads': 16,
                'dropout': dropout,
                'max_seq_length': max_seq_length
            }
            
        if action_decoder_config is None:
            action_decoder_config = {
                'input_dim': temporal_fusion_config['hidden_dim'],
                'hidden_dim': 512,
                'num_layers': 4,
                'num_heads': 8,
                'dropout': dropout
            }
            
        # Initialize encoders
        self.hand_encoder = HandPoseEncoder(**hand_encoder_config)
        self.object_encoder = ObjectPoseEncoder(**object_encoder_config)
        self.contact_encoder = ContactDetectionEncoder(**contact_encoder_config)
        
        # Initialize temporal fusion
        self.temporal_fusion = TemporalFusionEncoder(**temporal_fusion_config)
        
        # Initialize action decoder
        self.action_decoder = ActionDecoder(**action_decoder_config)
        
        # Spatial positional encoding projections
        self.spatial_pos_proj_hand = nn.Linear(512, hand_encoder_config['hidden_dim'])
        self.spatial_pos_proj_obj = nn.Linear(512, object_encoder_config['hidden_dim'])
        
    def forward(self,
                video_sequence: Dict[str, torch.Tensor],
                return_intermediates: bool = False) -> Dict[str, torch.Tensor]:
        """
        Process video sequence and generate manipulation commands
        
        Args:
            video_sequence: Dictionary containing:
                - color: [B, T, H, W, 3] or [B, T, 3, H, W] RGB frames
                - hand_joints_2d: [B, T, 21, 2] 2D hand joint positions
                - segmentation: [B, T, H, W] segmentation masks
                - ycb_ids: List of object IDs in the scene
            return_intermediates: Whether to return intermediate encoder outputs
            
        Returns:
            Dictionary containing action decoder outputs and optionally intermediate features
        """
        # Extract dimensions
        if video_sequence['color'].dim() == 5:
            B, T, C, H, W = video_sequence['color'].shape
            if C != 3:  # Channel last format
                video_sequence['color'] = video_sequence['color'].permute(0, 1, 4, 2, 3)
                B, T, C, H, W = video_sequence['color'].shape
        else:
            raise ValueError("Expected 5D tensor for video sequence")
            
        device = video_sequence['color'].device
        
        # Process each frame
        hand_features_list = []
        object_features_list = []
        contact_features_list = []
        
        # Get spatial positional encoding
        spatial_encoding = self.positional_encoding.encode_spatial(
            self.num_patches_h, 
            self.num_patches_w
        ).to(device)
        
        spatial_encoding_hand = self.spatial_pos_proj_hand(spatial_encoding)
        spatial_encoding_obj = self.spatial_pos_proj_obj(spatial_encoding)
        
        for t in range(T):
            # Extract frame
            frame = video_sequence['color'][:, t]  # [B, 3, H, W]
            
            # Preprocess frame
            frame = self.preprocessor.preprocess_frame(frame)
            
            # Create patches
            patches = self.preprocessor.create_patches(frame.unsqueeze(0) if frame.dim() == 3 else frame)
            
            # Extract hand region
            if 'hand_joints_2d' in video_sequence:
                hand_joints = video_sequence['hand_joints_2d'][:, t]  # [B, 21, 2]
                hand_region = self.preprocessor.get_hand_region(frame.unsqueeze(0), hand_joints.unsqueeze(1))
                hand_patches = self.preprocessor.create_patches(hand_region)
            else:
                hand_patches = patches  # Use full image if no hand joints
                
            # Extract object regions
            if 'segmentation' in video_sequence:
                seg = video_sequence['segmentation'][:, t]  # [B, H, W]
                object_regions = self.preprocessor.get_object_regions(
                    frame.unsqueeze(0), 
                    seg.unsqueeze(0),
                    video_sequence.get('ycb_ids', [])
                )
                # Use first object region or full patches
                if object_regions:
                    obj_patches = self.preprocessor.create_patches(list(object_regions.values())[0])
                else:
                    obj_patches = patches
            else:
                obj_patches = patches
                
            # Encode features
            hand_output = self.hand_encoder(hand_patches, spatial_encoding_hand)
            object_output = self.object_encoder(obj_patches, spatial_encoding=spatial_encoding_obj)
            
            # Contact detection using encoded features
            contact_output = self.contact_encoder(
                hand_features=hand_output['features'],
                object_features=object_output['features']
            )
            
            # Collect features
            hand_features_list.append(hand_output['features'])
            object_features_list.append(object_output['features'])
            contact_features_list.append(contact_output['features'])
            
        # Stack temporal features
        hand_features_temporal = torch.stack(hand_features_list, dim=1)  # [B, T, hand_dim]
        object_features_temporal = torch.stack(object_features_list, dim=1)  # [B, T, obj_dim]
        contact_features_temporal = torch.stack(contact_features_list, dim=1)  # [B, T, contact_dim]
        
        # Temporal fusion
        fusion_output = self.temporal_fusion(
            hand_features=hand_features_temporal,
            object_features=object_features_temporal,
            contact_features=contact_features_temporal
        )
        
        # Action decoding
        action_output = self.action_decoder(
            fused_features=fusion_output['aggregated_features'],
            temporal_features=fusion_output['fused_features']
        )
        
        # Prepare output
        output = {
            **action_output,  # All action decoder outputs
            'fusion_features': fusion_output['aggregated_features']
        }
        
        if return_intermediates:
            output.update({
                'hand_features': hand_features_temporal,
                'object_features': object_features_temporal,
                'contact_features': contact_features_temporal,
                'temporal_features': fusion_output['fused_features'],
                'attention_weights': fusion_output.get('attention_weights')
            })
            
        return output
    
    def compute_loss(self,
                    predictions: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor],
                    loss_weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute all losses for the model
        """
        if loss_weights is None:
            loss_weights = {
                'hand': 1.0,
                'object': 1.0,
                'contact': 0.5,
                'action': 2.0
            }
            
        losses = {}
        
        # Encoder losses (if training with auxiliary supervision)
        if 'hand_features' in predictions and 'hand_joints_3d' in targets:
            hand_losses = self.hand_encoder.compute_loss(
                {'joints_3d': predictions.get('hand_joints_3d', torch.zeros_like(targets['hand_joints_3d']))},
                targets
            )
            for k, v in hand_losses.items():
                losses[f'hand_{k}'] = v * loss_weights['hand']
                
        if 'object_features' in predictions and 'object_poses' in targets:
            object_losses = self.object_encoder.compute_loss(
                {'positions': predictions.get('object_positions', torch.zeros(1, 10, 3))},
                targets
            )
            for k, v in object_losses.items():
                losses[f'object_{k}'] = v * loss_weights['object']
                
        if 'contact_features' in predictions:
            # Contact losses would need ground truth contact annotations
            pass
            
        # Action decoder losses
        action_losses = self.action_decoder.compute_loss(predictions, targets)
        for k, v in action_losses.items():
            losses[f'action_{k}'] = v * loss_weights['action']
            
        # Total loss
        losses['total'] = sum(v for k, v in losses.items() if 'total' not in k)
        
        return losses
    
    def freeze_encoders(self):
        """Freeze encoder parameters for stage 2 training"""
        for encoder in [self.hand_encoder, self.object_encoder, self.contact_encoder]:
            for param in encoder.parameters():
                param.requires_grad = False
                
    def unfreeze_all(self):
        """Unfreeze all parameters for end-to-end training"""
        for param in self.parameters():
            param.requires_grad = True