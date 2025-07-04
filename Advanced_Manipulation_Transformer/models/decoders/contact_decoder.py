import torch
import torch.nn as nn
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class ContactDecoder(nn.Module):
    """Decoder for hand-object contacts"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        hidden_dim = config.get('contact_hidden_dim', 512)
        num_contact_points = config.get('num_contact_points', 10)
        dropout = config.get('dropout', 0.1)
        global_feature_dim = config.get('hidden_dim', 1024)  # Main model hidden dim
        
        # Contact point queries
        self.contact_queries = nn.Parameter(
            torch.randn(num_contact_points, hidden_dim) * 0.02
        )
        
        # Feature projection
        self.hand_proj = nn.Linear(21 * 3, hidden_dim)
        self.obj_proj = nn.Linear(10 * 3, hidden_dim)  # Max 10 objects
        self.global_proj = nn.Linear(global_feature_dim, hidden_dim)  # From global features
        
        # Point-wise projection for attention
        self.hand_point_proj = nn.Linear(3, hidden_dim)  # Per hand joint
        self.obj_point_proj = nn.Linear(3, hidden_dim)   # Per object position
        
        # Cross-attention between hand and objects
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(2)
        ])
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        
        # Output heads
        self.contact_point_head = nn.Linear(hidden_dim, 3)
        self.contact_confidence_head = nn.Linear(hidden_dim, 1)
        self.contact_type_head = nn.Linear(hidden_dim, 4)  # none/light/firm/manipulation
        self.contact_force_head = nn.Linear(hidden_dim, 3)
        self.interaction_type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 6)  # Global interaction type
        )
    
    def forward(
        self,
        global_features: torch.Tensor,
        hand_joints: torch.Tensor,
        object_positions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        B = global_features.shape[0]
        
        # Project features
        hand_feat = self.hand_proj(hand_joints.reshape(B, -1))
        obj_feat = self.obj_proj(object_positions.reshape(B, -1))
        global_feat = self.global_proj(global_features)
        
        # Combine features
        combined_feat = hand_feat + obj_feat + global_feat
        combined_feat = combined_feat.unsqueeze(1)  # [B, 1, hidden]
        
        # Expand contact queries
        queries = self.contact_queries.unsqueeze(0).expand(B, -1, -1)
        
        # Cross-attention to find contact regions
        # Project hand joints and objects to feature space
        hand_feat_seq = self.hand_point_proj(hand_joints)  # [B, 21, hidden]
        obj_feat_seq = self.obj_point_proj(object_positions)  # [B, N_obj, hidden]
        
        for i, attn in enumerate(self.cross_attention):
            if i == 0:
                # Attend to hand joints
                queries, _ = attn(
                    query=queries,
                    key=hand_feat_seq,
                    value=hand_feat_seq
                )
            else:
                # Attend to objects
                queries, _ = attn(
                    query=queries,
                    key=obj_feat_seq,
                    value=obj_feat_seq
                )
        
        # Decode contacts
        memory = torch.cat([combined_feat, queries], dim=1)
        decoded = self.decoder(
            tgt=queries,
            memory=memory
        )
        
        # Predict outputs
        contact_points = self.contact_point_head(decoded)
        contact_confidence = torch.sigmoid(self.contact_confidence_head(decoded)).squeeze(-1)
        contact_types = self.contact_type_head(decoded)
        contact_forces = self.contact_force_head(decoded)
        
        # Global interaction type from pooled features
        pooled = decoded.mean(dim=1)
        interaction_type = self.interaction_type_head(pooled)
        
        return {
            'contact_points': contact_points,
            'contact_confidence': contact_confidence,
            'contact_types': contact_types,
            'contact_forces': contact_forces,
            'interaction_type': interaction_type,
            'features': decoded
        }