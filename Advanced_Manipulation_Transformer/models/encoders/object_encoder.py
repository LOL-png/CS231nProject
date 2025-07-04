import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ObjectPoseEncoder(nn.Module):
    """
    Object pose encoder for multi-object detection and pose estimation
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 1024,
        num_layers: int = 6,
        num_heads: int = 16,
        dropout: float = 0.1,
        max_objects: int = 10
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_objects = max_objects
        
        # Input projection from image features
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Object queries (learnable embeddings)
        self.object_queries = nn.Parameter(
            torch.randn(max_objects, hidden_dim) * 0.02
        )
        
        # Transformer encoder for feature extraction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers // 2)
        
        # Transformer decoder for object detection
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers // 2)
        
        # Output heads
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)
        )
        
        self.rotation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 6)  # 6D rotation representation
        )
        
        self.confidence_head = nn.Linear(hidden_dim, 1)
        self.class_head = nn.Linear(hidden_dim, 100)  # Assuming 100 object classes
        
    def forward(
        self,
        image_features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        Args:
            image_features: Dictionary containing patch_tokens from image encoder
        Returns:
            Dictionary with object predictions
        """
        B = image_features['patch_tokens'].shape[0]
        
        # Project image features
        features = self.input_proj(image_features['patch_tokens'])  # [B, N, hidden_dim]
        
        # Encode features
        encoded_features = self.transformer_encoder(features)
        
        # Expand object queries
        queries = self.object_queries.unsqueeze(0).expand(B, -1, -1)  # [B, max_objects, hidden_dim]
        
        # Decode objects
        decoded = self.transformer_decoder(
            tgt=queries,
            memory=encoded_features
        )  # [B, max_objects, hidden_dim]
        
        # Predict outputs
        positions = self.position_head(decoded)  # [B, max_objects, 3]
        rotations = self.rotation_head(decoded)  # [B, max_objects, 6]
        confidence = torch.sigmoid(self.confidence_head(decoded))  # [B, max_objects, 1]
        class_logits = self.class_head(decoded)  # [B, max_objects, num_classes]
        
        return {
            'positions': positions,
            'rotations': rotations,
            'confidence': confidence.squeeze(-1),
            'class_logits': class_logits,
            'features': decoded
        }


class ContactEncoder(nn.Module):
    """
    Contact detection encoder for hand-object interaction
    """
    
    def __init__(
        self,
        hand_feat_dim: int = 1024,
        object_feat_dim: int = 1024,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_contact_points: int = 10
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_contact_points = num_contact_points
        
        # Feature projection
        self.hand_proj = nn.Linear(hand_feat_dim, hidden_dim)
        self.object_proj = nn.Linear(object_feat_dim, hidden_dim)
        
        # Contact point queries
        self.contact_queries = nn.Parameter(
            torch.randn(num_contact_points, hidden_dim) * 0.02
        )
        
        # Cross-attention between hand and objects
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(2)
        ])
        
        # Self-attention transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.contact_point_head = nn.Linear(hidden_dim, 3)
        self.contact_confidence_head = nn.Linear(hidden_dim, 1)
        self.contact_type_head = nn.Linear(hidden_dim, 4)  # none/light/firm/manipulation
        self.contact_force_head = nn.Linear(hidden_dim, 3)
        
    def forward(
        self,
        hand_features: torch.Tensor,
        object_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        Args:
            hand_features: [B, hidden_dim] hand features
            object_features: [B, max_objects, hidden_dim] object features
        Returns:
            Dictionary with contact predictions
        """
        B = hand_features.shape[0]
        
        # Project features
        hand_feat = self.hand_proj(hand_features).unsqueeze(1)  # [B, 1, hidden_dim]
        obj_feat = self.object_proj(object_features)  # [B, max_objects, hidden_dim]
        
        # Expand contact queries
        queries = self.contact_queries.unsqueeze(0).expand(B, -1, -1)  # [B, num_contacts, hidden_dim]
        
        # Cross-attention: queries attend to hand features
        queries, _ = self.cross_attention[0](
            query=queries,
            key=hand_feat,
            value=hand_feat
        )
        
        # Cross-attention: queries attend to object features
        queries, _ = self.cross_attention[1](
            query=queries,
            key=obj_feat,
            value=obj_feat
        )
        
        # Self-attention refinement
        refined_features = self.transformer(queries)  # [B, num_contacts, hidden_dim]
        
        # Predict outputs
        contact_points = self.contact_point_head(refined_features)  # [B, num_contacts, 3]
        contact_confidence = torch.sigmoid(self.contact_confidence_head(refined_features)).squeeze(-1)  # [B, num_contacts]
        contact_types = self.contact_type_head(refined_features)  # [B, num_contacts, 4]
        contact_forces = self.contact_force_head(refined_features)  # [B, num_contacts, 3]
        
        return {
            'contact_points': contact_points,
            'contact_confidence': contact_confidence,
            'contact_types': contact_types,
            'contact_forces': contact_forces,
            'features': refined_features
        }