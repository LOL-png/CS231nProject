import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class ObjectPoseDecoder(nn.Module):
    """Decoder for object pose and classification"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        hidden_dim = config.get('hidden_dim', 1024)
        num_objects = config.get('max_objects', 10)
        num_classes = config.get('num_object_classes', 100)
        dropout = config.get('dropout', 0.1)
        
        # Object queries (learnable embeddings)
        self.object_queries = nn.Parameter(
            torch.randn(num_objects, hidden_dim) * 0.02
        )
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=4)
        
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
        self.class_head = nn.Linear(hidden_dim, num_classes)
    
    def forward(
        self, 
        global_features: torch.Tensor,
        image_features: Dict[str, torch.Tensor],
        hand_joints: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        B = global_features.shape[0]
        
        # Expand object queries
        queries = self.object_queries.unsqueeze(0).expand(B, -1, -1)
        
        # Use image patch tokens as memory
        memory = image_features['patch_tokens']
        
        # Decode objects
        decoded = self.transformer(
            tgt=queries,
            memory=memory
        )
        
        # Predict outputs
        positions = self.position_head(decoded)
        rotations = self.rotation_head(decoded)
        confidence = torch.sigmoid(self.confidence_head(decoded))
        class_logits = self.class_head(decoded)
        
        return {
            'positions': positions,
            'rotations': rotations,
            'confidence': confidence.squeeze(-1),
            'class_logits': class_logits,
            'features': decoded
        }