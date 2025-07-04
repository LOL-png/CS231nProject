import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import logging
import math

# Import required components
from .encoders.dinov2_encoder import DINOv2ImageEncoder
from .encoders.hand_encoder import MultiCoordinateHandEncoder
from .encoders.object_encoder import ObjectPoseEncoder, ContactEncoder
from .pixel_aligned import PixelAlignedRefinement
from .decoders.object_pose_decoder import ObjectPoseDecoder
from .decoders.contact_decoder import ContactDecoder

logger = logging.getLogger(__name__)

# Sigma reparameterization module
class SigmaReparam(nn.Module):
    """Apply σ-reparameterization to a linear layer"""
    
    def __init__(self, linear_layer: nn.Linear):
        super().__init__()
        self.linear = linear_layer
        self.sigma = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # Spectral normalization
        weight = self.linear.weight
        # Handle both 1D and 2D weights properly
        if weight.dim() == 1:
            weight_norm = weight / (weight.norm() + 1e-8)
        else:
            weight_norm = weight / (weight.norm(dim=1, keepdim=True) + 1e-8)
        
        # Apply linear with normalized weight and learned scale
        out = F.linear(x, weight_norm * self.sigma, self.linear.bias)
        return out

# Main unified model
class UnifiedManipulationTransformer(nn.Module):
    """
    Complete model integrating all components with σ-reparameterization
    to prevent mode collapse and attention entropy collapse
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        
        # Core components
        self.image_encoder = DINOv2ImageEncoder(
            freeze_layers=config.get('freeze_layers', 12),
            output_dim=config.get('hidden_dim', 1024),
            dropout=config.get('dropout', 0.1)
        )
        
        self.hand_encoder = MultiCoordinateHandEncoder(
            hidden_dim=config.get('hidden_dim', 1024),
            use_mano_vertices=config.get('use_mano_vertices', True),
            dropout=config.get('dropout', 0.1)
        )
        
        # Object encoder using the imported one
        self.object_encoder = ObjectPoseEncoder(
            input_dim=config.get('hidden_dim', 1024),
            hidden_dim=config.get('hidden_dim', 1024),
            dropout=config.get('dropout', 0.1),
            max_objects=config.get('max_objects', 10)
        )
        
        # Contact encoder
        self.contact_encoder = ContactEncoder(
            hand_feat_dim=config.get('hidden_dim', 1024),
            object_feat_dim=config.get('hidden_dim', 1024),
            hidden_dim=config.get('contact_hidden_dim', 512),
            dropout=config.get('dropout', 0.1),
            num_contact_points=config.get('num_contact_points', 10)
        )
        
        self.pixel_aligner = PixelAlignedRefinement(
            image_feat_dim=config.get('hidden_dim', 1024),
            point_feat_dim=256,
            num_refinement_steps=config.get('num_refinement_steps', 2)
        )
        
        # Feature fusion (without SDF)
        fusion_input_dim = config.get('hidden_dim', 1024) * 3  # Image + hand + object
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, config.get('hidden_dim', 1024)),
            nn.LayerNorm(config.get('hidden_dim', 1024)),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(config.get('hidden_dim', 1024), config.get('hidden_dim', 1024))
        )
        
        # Attention-based fusion as alternative
        self.use_attention_fusion = config.get('use_attention_fusion', True)
        if self.use_attention_fusion:
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=config.get('hidden_dim', 1024),
                num_heads=8,
                dropout=config.get('dropout', 0.1),
                batch_first=True
            )
        
        # Task-specific decoders
        self.object_decoder = ObjectPoseDecoder(config)
        self.contact_decoder = ContactDecoder(config)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply σ-reparameterization
        self.use_sigma_reparam = config.get('use_sigma_reparam', True)
        if self.use_sigma_reparam:
            self.apply_sigma_reparam()
        
        logger.info(f"Initialized UnifiedManipulationTransformer with {self.count_parameters()}M parameters")
    
    def _init_weights(self, module):
        """Initialize weights with better strategies"""
        if isinstance(module, nn.Linear):
            # Xavier initialization with gain based on activation
            nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def apply_sigma_reparam(self):
        """
        Apply σ-reparameterization to prevent attention collapse
        Critical for solving mode collapse issue
        """
        # Get parent modules to check context
        parent_modules = {}
        for name, module in self.named_modules():
            if '.' in name:
                parent_name = name.rsplit('.', 1)[0]
                if parent_name in dict(self.named_modules()):
                    parent_modules[name] = dict(self.named_modules())[parent_name]
        
        for name, module in self.named_modules():
            # Skip certain layers
            if any(skip in name for skip in ['norm', 'embedding', 'head']):
                continue
            
            # Skip layers inside MultiheadAttention modules
            if name in parent_modules:
                parent = parent_modules[name]
                if isinstance(parent, nn.MultiheadAttention):
                    continue
            
            # Skip attention-related modules
            if any(attn in name for attn in ['self_attn', 'multihead', 'attention']):
                continue
                
            if isinstance(module, nn.Linear):
                # Replace with sigma-reparameterized version
                parent_name = '.'.join(name.split('.')[:-1])
                module_name = name.split('.')[-1]
                
                if parent_name:
                    parent = self.get_submodule(parent_name)
                    setattr(parent, module_name, SigmaReparam(module))
                
        logger.info("Applied σ-reparameterization to linear layers")
    
    def get_submodule(self, name: str) -> nn.Module:
        """Get submodule by name"""
        module = self
        for part in name.split('.'):
            module = getattr(module, part)
        return module
    
    def count_parameters(self) -> float:
        """Count trainable parameters in millions"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
    
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        mano_vertices: Optional[torch.Tensor] = None,
        camera_params: Optional[Dict[str, torch.Tensor]] = None,
        return_features: bool = False,
        **kwargs
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Forward pass through entire model
        
        Args:
            images: [B, 3, 224, 224] RGB images (or dict with 'image' key)
            mano_vertices: Optional [B, 778, 3] MANO vertices
            camera_params: Optional dict with camera parameters
            return_features: Whether to return intermediate features
            **kwargs: Additional arguments (for dict input compatibility)
            
        Returns:
            Dictionary with 'hand', 'objects', 'contacts' predictions
        """
        # Handle dictionary input (from data loader)
        if images is None and 'image' in kwargs:
            images = kwargs['image']
        elif isinstance(images, dict):
            # Extract from dictionary
            batch_dict = images
            images = batch_dict.get('image')
            if mano_vertices is None and 'mano_vertices' in batch_dict:
                mano_vertices = batch_dict.get('mano_vertices')
            if camera_params is None and 'camera_intrinsics' in batch_dict:
                camera_params = {
                    'intrinsics': batch_dict.get('camera_intrinsics'),
                    'extrinsics': batch_dict.get('camera_extrinsics')
                }
        
        if images is None:
            raise ValueError("No images provided to forward pass")
            
        B = images.shape[0]
        device = images.device
        
        # Extract image features
        image_features = self.image_encoder(images)
        
        # Initial hand prediction from image
        hand_outputs = self.hand_encoder(
            image_features,
            hand_joints=None,  # Will be predicted
            mano_vertices=mano_vertices
        )
        
        # Object detection from image features
        object_outputs = self.object_encoder(image_features)
        
        # Contact prediction using hand and object features
        contact_outputs = self.contact_encoder(
            hand_outputs['features'],
            object_outputs['features']
        )
        
        # Feature fusion
        if self.use_attention_fusion:
            # Stack features for attention
            feature_list = [
                image_features['cls_token'].unsqueeze(1),
                hand_outputs['features'].unsqueeze(1),
                object_outputs['features'].mean(dim=1).unsqueeze(1)  # Pool object features
            ]
            
            combined_features = torch.cat(feature_list, dim=1)  # [B, 3, hidden_dim]
            
            # Self-attention fusion
            fused_features, attention_weights = self.fusion_attention(
                combined_features,
                combined_features,
                combined_features
            )
            
            # Global features from attended tokens
            global_features = fused_features.mean(dim=1)  # [B, hidden_dim]
        else:
            # Simple concatenation and MLP fusion
            feature_list = [
                image_features['cls_token'], 
                hand_outputs['features'],
                object_outputs['features'].mean(dim=1)
            ]
            
            concatenated = torch.cat(feature_list, dim=-1)
            global_features = self.feature_fusion(concatenated)
        
        # Use decoders for refined predictions
        object_refined = self.object_decoder(
            global_features,
            image_features,
            hand_outputs['joints_3d']
        )
        
        contact_refined = self.contact_decoder(
            global_features,
            hand_outputs['joints_3d'],
            object_refined['positions']
        )
        
        # Merge outputs
        object_outputs.update(object_refined)
        contact_outputs.update(contact_refined)
        
        # Pixel-aligned refinement for hand joints
        if camera_params is not None:
            refinement_output = self.pixel_aligner(
                hand_outputs['joints_3d'],
                image_features,
                camera_params
            )
            hand_outputs['joints_3d_refined'] = refinement_output['refined_points']
            hand_outputs['refinement_confidence'] = refinement_output['confidence']
            
            # Also refine object positions
            obj_refinement = self.pixel_aligner(
                object_outputs['positions'].reshape(B, -1, 3),
                image_features,
                camera_params
            )
            object_outputs['positions_refined'] = obj_refinement['refined_points'].reshape(
                B, -1, 3
            )
        
        # Prepare output in format expected by loss function
        outputs = {
            # Hand predictions
            'hand_joints_coarse': hand_outputs['joints_3d'],
            'hand_joints': hand_outputs.get('joints_3d_refined', hand_outputs['joints_3d']),
            'hand_confidence': hand_outputs.get('confidence'),
            'hand_shape': hand_outputs.get('shape_params'),
            
            # Object predictions
            'object_positions': object_outputs.get('positions'),
            'object_rotations': object_outputs.get('rotations'),
            'object_poses': torch.cat([
                object_outputs.get('positions', torch.zeros(B, 10, 3, device=device)),
                object_outputs.get('rotations', torch.zeros(B, 10, 4, device=device))
            ], dim=-1) if 'positions' in object_outputs else None,
            
            # Contact predictions
            'contact_points': contact_outputs.get('contact_points'),
            'contact_probs': contact_outputs.get('contact_confidence'),  # Map contact_confidence to contact_probs
            'contact_forces': contact_outputs.get('contact_forces'),
            
            # Original nested structure (for backward compatibility)
            'hand': hand_outputs,
            'objects': object_outputs,
            'contacts': contact_outputs
        }
        
        # Add features if requested
        if return_features:
            outputs['features'] = {
                'global': global_features,
                'image': image_features,
                'attention_weights': attention_weights if self.use_attention_fusion else None
            }
        
        # Add KL loss if using sigma reparameterization
        if self.use_sigma_reparam:
            kl_losses = []
            for module in self.modules():
                if hasattr(module, 'kl_loss') and module.kl_loss is not None:
                    kl_losses.append(module.kl_loss)
            if kl_losses:
                outputs['kl_loss'] = sum(kl_losses) / len(kl_losses)
        
        return outputs