import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np

class ModeCollapsePreventionModule(nn.Module):
    """
    Collection of techniques to prevent mode collapse
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # 1. Noise injection layers
        self.noise_std = config.get('noise_std', 0.01)
        
        # 2. Stochastic depth (drop path)
        self.drop_path_rate = config.get('drop_path_rate', 0.1)
        
        # 3. Mixup augmentation
        self.mixup_alpha = config.get('mixup_alpha', 0.2)
        
        # 4. Feature perturbation
        self.feature_noise = FeatureNoise(std=0.05)
    
    def add_noise_to_features(self, features: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Add noise to features during training"""
        if training and self.noise_std > 0:
            noise = torch.randn_like(features) * self.noise_std
            features = features + noise
        return features
    
    def mixup_batch(self, x: torch.Tensor, y: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], float]:
        """Apply mixup augmentation"""
        if self.training and self.mixup_alpha > 0:
            batch_size = x.shape[0]
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            
            # Random permutation
            index = torch.randperm(batch_size).to(x.device)
            
            # Mix inputs
            mixed_x = lam * x + (1 - lam) * x[index]
            
            # Mix targets
            mixed_y = {}
            for key, value in y.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    mixed_y[key] = lam * value + (1 - lam) * value[index]
                else:
                    mixed_y[key] = value
            
            return mixed_x, mixed_y, lam
        
        return x, y, 1.0
    
    @staticmethod
    def wrap_model(model: nn.Module, config: Dict) -> nn.Module:
        """
        Wrap an existing model with mode collapse prevention features
        
        Args:
            model: The model to wrap
            config: Configuration for mode collapse prevention
        
        Returns:
            Wrapped model with mode collapse prevention
        """
        class WrappedModel(nn.Module):
            def __init__(self, base_model, prevention_config):
                super().__init__()
                self.base_model = base_model
                self.prevention = ModeCollapsePreventionModule(prevention_config)
                
                # Replace transformer layers if they exist
                self._replace_transformer_layers()
            
            def _replace_transformer_layers(self):
                """Replace standard transformer layers with improved ones"""
                for name, module in self.base_model.named_children():
                    if isinstance(module, nn.TransformerEncoderLayer):
                        # Extract d_model from the self_attn layer
                        d_model = module.self_attn.embed_dim
                        nhead = module.self_attn.num_heads
                        
                        # Extract dropout from the layers
                        dropout = 0.1  # default
                        if hasattr(module, 'dropout') and hasattr(module.dropout, 'p'):
                            dropout = module.dropout.p
                        elif hasattr(module, 'dropout1') and hasattr(module.dropout1, 'p'):
                            dropout = module.dropout1.p
                        
                        # Replace with improved layer
                        improved_layer = ImprovedTransformerLayer(
                            d_model=d_model,
                            nhead=nhead,
                            dropout=dropout,
                            drop_path=self.prevention.drop_path_rate
                        )
                        setattr(self.base_model, name, improved_layer)
                    elif hasattr(module, 'children'):
                        # Recursively replace in submodules
                        self._replace_in_module(module)
            
            def _replace_in_module(self, module):
                """Recursively replace transformer layers in a module"""
                for name, child in module.named_children():
                    if isinstance(child, nn.TransformerEncoderLayer):
                        # Extract d_model from the self_attn layer
                        d_model = child.self_attn.embed_dim
                        nhead = child.self_attn.num_heads
                        
                        # Extract dropout from the layers
                        dropout = 0.1  # default
                        if hasattr(child, 'dropout') and hasattr(child.dropout, 'p'):
                            dropout = child.dropout.p
                        elif hasattr(child, 'dropout1') and hasattr(child.dropout1, 'p'):
                            dropout = child.dropout1.p
                        
                        improved_layer = ImprovedTransformerLayer(
                            d_model=d_model,
                            nhead=nhead,
                            dropout=dropout,
                            drop_path=self.prevention.drop_path_rate
                        )
                        setattr(module, name, improved_layer)
                    elif hasattr(child, 'children'):
                        self._replace_in_module(child)
            
            def forward(self, *args, **kwargs):
                # Apply mixup to images if present
                if 'images' in kwargs and self.training:
                    images = kwargs['images']
                    batch_size = images.shape[0]
                    
                    # Apply feature noise
                    images = self.prevention.add_noise_to_features(images, self.training)
                    kwargs['images'] = images
                
                # Forward through base model
                outputs = self.base_model(*args, **kwargs)
                
                # Add noise to output features if they exist
                if isinstance(outputs, dict):
                    for key in ['hand', 'object', 'contact']:
                        if key in outputs and 'features' in outputs[key]:
                            outputs[key]['features'] = self.prevention.add_noise_to_features(
                                outputs[key]['features'], self.training
                            )
                
                return outputs
        
        return WrappedModel(model, config)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        
        return output

class FeatureNoise(nn.Module):
    """Add structured noise to features"""
    
    def __init__(self, std: float = 0.05):
        super().__init__()
        self.std = std
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Compute feature statistics
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            
            # Add scaled noise
            noise = torch.randn_like(x) * self.std * std
            x = x + noise
        
        return x

# Integration example
class ImprovedTransformerLayer(nn.Module):
    """Transformer layer with mode collapse prevention"""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, drop_path: float = 0.1):
        super().__init__()
        
        # Standard transformer components
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # FFN with modifications
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            FeatureNoise(std=0.02),  # Add noise in FFN
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        # Drop path
        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)
        
        # Temperature for attention
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                is_causal: bool = False) -> torch.Tensor:
        """Forward pass matching TransformerEncoderLayer signature"""
        # For compatibility, we use 'src' instead of 'x'
        x = src
        
        # Self-attention with temperature scaling
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            is_causal=is_causal
        )
        attn_out = attn_out * self.temperature
        
        # Residual with drop path
        x = x + self.drop_path1(attn_out)
        x = self.norm1(x)
        
        # FFN with drop path
        ffn_out = self.ffn(x)
        x = x + self.drop_path2(ffn_out)
        x = self.norm2(x)
        
        return x