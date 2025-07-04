"""
Temporal Fusion Encoder for Video-to-Manipulation Transformer
Integrates outputs from all three encoders across temporal dimension
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class TemporalFusionEncoder(nn.Module):
    """
    Temporal fusion encoder that integrates hand, object, and contact features
    Architecture: 8 layers, 1024 dim, 16 attention heads
    Key features:
    - Temporal multi-head attention
    - Cross-modal fusion between hand/object/contact features
    - Sliding window attention (8-16 frames)
    """
    
    def __init__(self,
                 hand_dim: int = 512,
                 object_dim: int = 512,
                 contact_dim: int = 256,
                 hidden_dim: int = 1024,
                 num_layers: int = 8,
                 num_heads: int = 16,
                 mlp_dim: int = 4096,
                 dropout: float = 0.1,
                 max_seq_length: int = 16,
                 window_size: int = 8):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.window_size = window_size
        
        # Feature projections to common dimension
        self.hand_projection = nn.Linear(hand_dim, hidden_dim)
        self.object_projection = nn.Linear(object_dim, hidden_dim)
        self.contact_projection = nn.Linear(contact_dim, hidden_dim)
        
        # Modality embeddings
        self.modality_embeddings = nn.Parameter(torch.zeros(3, hidden_dim))
        nn.init.normal_(self.modality_embeddings, std=0.02)
        
        # Temporal positional encoding
        self.temporal_pos_encoding = self._create_temporal_encoding(max_seq_length, hidden_dim)
        
        # Transformer encoder with sliding window attention
        self.layers = nn.ModuleList([
            TemporalFusionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                window_size=window_size if i < num_layers // 2 else None  # Use window attention for first half
            )
            for i in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Temporal aggregation
        self.temporal_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def _create_temporal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal temporal positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # [1, max_len, d_model]
        
    def forward(self,
                hand_features: torch.Tensor,
                object_features: torch.Tensor,
                contact_features: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            hand_features: [B, T, hand_dim] temporal hand features
            object_features: [B, T, object_dim] temporal object features
            contact_features: [B, T, contact_dim] temporal contact features
            attention_mask: [B, T] optional attention mask for padded sequences
            
        Returns:
            Dictionary containing:
                - fused_features: [B, T, hidden_dim] temporally fused features
                - aggregated_features: [B, hidden_dim] aggregated features for action decoder
                - attention_weights: Cross-modal attention weights for visualization
        """
        B, T = hand_features.shape[:2]
        device = hand_features.device
        
        # Project features to common dimension
        hand_proj = self.hand_projection(hand_features)  # [B, T, hidden_dim]
        object_proj = self.object_projection(object_features)  # [B, T, hidden_dim]
        contact_proj = self.contact_projection(contact_features)  # [B, T, hidden_dim]
        
        # Add modality embeddings
        hand_proj = hand_proj + self.modality_embeddings[0]
        object_proj = object_proj + self.modality_embeddings[1]
        contact_proj = contact_proj + self.modality_embeddings[2]
        
        # Stack modalities
        x = torch.stack([hand_proj, object_proj, contact_proj], dim=2)  # [B, T, 3, hidden_dim]
        x = x.view(B, T * 3, self.hidden_dim)  # [B, T*3, hidden_dim]
        
        # Add temporal positional encoding
        temporal_pe = self.temporal_pos_encoding[:, :T].to(device)
        temporal_pe = temporal_pe.repeat(1, 3, 1)  # Repeat for each modality
        x = x + temporal_pe
        
        # Create attention mask for modalities and time
        if attention_mask is not None:
            # Expand mask for 3 modalities
            attention_mask = attention_mask.unsqueeze(1).repeat(1, 3, 1)  # [B, 3, T]
            attention_mask = attention_mask.view(B, T * 3)  # [B, T*3]
            
        # Apply transformer layers
        attention_weights_list = []
        for layer in self.layers:
            x, attn_weights = layer(x, attention_mask)
            if attn_weights is not None:
                attention_weights_list.append(attn_weights)
                
        # Final normalization
        x = self.final_norm(x)
        
        # Reshape back to temporal structure
        x = x.view(B, T, 3, self.hidden_dim)
        
        # Average across modalities for fused features
        fused_features = x.mean(dim=2)  # [B, T, hidden_dim]
        
        # Temporal aggregation
        if attention_mask is not None:
            # Mask out padded timesteps
            mask = attention_mask.view(B, T, 3)[:, :, 0].unsqueeze(-1)  # [B, T, 1]
            fused_features = fused_features * mask
            
            # Weighted average
            lengths = mask.sum(dim=1)  # [B, 1]
            aggregated = fused_features.sum(dim=1) / lengths.clamp(min=1)
        else:
            # Simple average
            aggregated = fused_features.mean(dim=1)  # [B, hidden_dim]
            
        # Apply temporal pooling
        aggregated_features = self.temporal_pool(aggregated)
        
        return {
            'fused_features': fused_features,
            'aggregated_features': aggregated_features,
            'attention_weights': attention_weights_list[-1] if attention_weights_list else None,
            'modality_features': x  # [B, T, 3, hidden_dim] for detailed analysis
        }


class TemporalFusionLayer(nn.Module):
    """
    Single layer of temporal fusion transformer with optional sliding window attention
    """
    
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 mlp_dim: int,
                 dropout: float = 0.1,
                 window_size: Optional[int] = None):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.window_size = window_size
        
        # Multi-head attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
    def forward(self,
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [B, T*3, hidden_dim] input features
            attention_mask: [B, T*3] attention mask
            
        Returns:
            x: [B, T*3, hidden_dim] output features
            attn_weights: Optional attention weights for visualization
        """
        B, L, D = x.shape
        
        # Create sliding window mask if specified
        if self.window_size is not None and L > self.window_size * 3:
            window_mask = self._create_sliding_window_mask(L, self.window_size * 3, x.device)
            if attention_mask is not None:
                # Combine with existing mask
                attention_mask = attention_mask.unsqueeze(1) & window_mask.unsqueeze(0)
            else:
                attention_mask = window_mask.unsqueeze(0).expand(B, -1, -1)
                
        # Self-attention with residual
        attn_output, attn_weights = self.self_attention(
            query=x,
            key=x,
            value=x,
            attn_mask=attention_mask if attention_mask is not None and attention_mask.dim() == 3 else None,
            key_padding_mask=attention_mask if attention_mask is not None and attention_mask.dim() == 2 else None,
            need_weights=True
        )
        x = self.norm1(x + attn_output)
        
        # Cross-modal attention (between different timesteps and modalities)
        # Reshape to separate time and modality
        T = L // 3
        x_reshaped = x.view(B, T, 3, D)
        
        # Attend across modalities for each timestep
        cross_attn_output = torch.zeros_like(x_reshaped)
        for t in range(T):
            for m in range(3):
                # Current modality attends to other modalities at same timestep
                query = x_reshaped[:, t, m:m+1]  # [B, 1, D]
                key_value = x_reshaped[:, t]  # [B, 3, D]
                
                attn_out, _ = self.cross_attention(query, key_value, key_value)
                cross_attn_output[:, t, m] = attn_out.squeeze(1)
                
        cross_attn_output = cross_attn_output.view(B, L, D)
        x = self.norm2(x + cross_attn_output)
        
        # Feed-forward network with residual
        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)
        
        return x, attn_weights
    
    def _create_sliding_window_mask(self, seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
        """Create sliding window attention mask"""
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = True
            
        return mask