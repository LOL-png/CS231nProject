"""
Transition Merger Model using HOISDF outputs
Merges two video sequences by learning smooth transitions between them
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class HOISDFOutputs:
    """Structure for HOISDF outputs over time"""
    mano_params: torch.Tensor  # [T, 51] (3 trans + 45 pose + 3 shape)
    hand_sdf: torch.Tensor     # [T, D, H, W] 
    object_sdf: torch.Tensor   # [T, D, H, W]
    contact_points: torch.Tensor  # [T, N, 3] contact locations
    contact_frames: torch.Tensor  # [T, N] which frames have contact
    hand_vertices: torch.Tensor   # [T, 778, 3] MANO mesh vertices
    object_center: torch.Tensor   # [T, 3] object center position
    

class HOISDFTokenizer(nn.Module):
    """Tokenizes HOISDF outputs for transformer input"""
    
    def __init__(self, 
                 mano_dim: int = 51,
                 sdf_resolution: int = 64,
                 hidden_dim: int = 256,
                 num_tokens: int = 256):
        super().__init__()
        
        # MANO parameter encoder
        self.mano_encoder = nn.Sequential(
            nn.Linear(mano_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # SDF encoder (processes hand and object SDFs)
        self.sdf_encoder = nn.Sequential(
            nn.Conv3d(2, 32, 3, padding=1),  # 2 channels: hand + object
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(128, hidden_dim)
        )
        
        # Contact encoder
        self.contact_encoder = nn.Sequential(
            nn.Linear(4, 64),  # 3D position + contact strength
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )
        
        # Combine all features into tokens
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, num_tokens)
        )
        
    def forward(self, hoisdf_outputs: HOISDFOutputs) -> torch.Tensor:
        """
        Convert HOISDF outputs to transformer tokens
        
        Returns:
            tokens: [B, T, num_tokens]
        """
        T = hoisdf_outputs.mano_params.shape[0]
        
        # Add batch dimension if not present
        if hoisdf_outputs.mano_params.dim() == 2:
            mano_params = hoisdf_outputs.mano_params.unsqueeze(0)  # [1, T, 51]
            hand_sdf = hoisdf_outputs.hand_sdf.unsqueeze(0)
            object_sdf = hoisdf_outputs.object_sdf.unsqueeze(0)
            contact_points = hoisdf_outputs.contact_points.unsqueeze(0)
            batch_size = 1
        else:
            mano_params = hoisdf_outputs.mano_params
            hand_sdf = hoisdf_outputs.hand_sdf
            object_sdf = hoisdf_outputs.object_sdf
            contact_points = hoisdf_outputs.contact_points
            batch_size = mano_params.shape[0]
            
        tokens = []
        
        for t in range(T):
            # Encode MANO parameters
            mano_feat = self.mano_encoder(mano_params[:, t])
            
            # Encode SDFs (combine hand and object)
            combined_sdf = torch.stack([
                hand_sdf[:, t],
                object_sdf[:, t]
            ], dim=1)
            sdf_feat = self.sdf_encoder(combined_sdf)
            
            # Encode contact information
            # Average pool contact points and add contact strength
            if contact_points.shape[2] > 0:
                contact_strength = (torch.abs(hand_sdf[:, t]) < 0.01).float().mean()
                contact_info = torch.cat([
                    contact_points[:, t].mean(dim=1),  # Average contact position
                    contact_strength.unsqueeze(-1)
                ], dim=-1)
            else:
                contact_info = torch.zeros(batch_size, 4).to(mano_params.device)
                
            contact_feat = self.contact_encoder(contact_info)
            
            # Fuse all features
            combined_feat = torch.cat([mano_feat, sdf_feat, contact_feat], dim=-1)
            token = self.feature_fusion(combined_feat)
            tokens.append(token)
            
        return torch.stack(tokens, dim=1)


class TransitionTransformer(nn.Module):
    """Transformer for learning transitions between video sequences"""
    
    def __init__(self,
                 input_dim: int = 256,
                 hidden_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 mano_dim: int = 51,
                 chunk_size: int = 50,
                 dropout: float = 0.1):
        super().__init__()
        
        self.chunk_size = chunk_size
        self.mano_dim = mano_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding for sequence position
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim))
        
        # Video encoding (which video: 1 or 2)
        self.video_embedding = nn.Embedding(3, hidden_dim)  # 0: video1, 1: video2, 2: transition
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output heads
        self.mano_head = nn.Linear(hidden_dim, chunk_size * mano_dim)
        self.boundary_head = nn.Linear(hidden_dim, 1)
        self.transition_quality_head = nn.Linear(hidden_dim, 1)
        self.task_embedding_head = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, 
                tokens_video1: torch.Tensor,
                tokens_video2: torch.Tensor,
                transition_length: int = 30) -> Dict[str, torch.Tensor]:
        """
        Process two video sequences and generate transition
        
        Args:
            tokens_video1: [B, T1, input_dim] tokens from first video
            tokens_video2: [B, T2, input_dim] tokens from second video
            transition_length: Number of frames for transition
            
        Returns:
            Dictionary with transition predictions
        """
        B = tokens_video1.shape[0]
        T1 = tokens_video1.shape[1]
        T2 = tokens_video2.shape[1]
        device = tokens_video1.device
        
        # Create combined sequence with video indicators
        # Take last 20 frames of video1, transition, first 20 frames of video2
        context_frames = 20
        
        # Prepare tokens
        v1_context = tokens_video1[:, -context_frames:]
        v2_context = tokens_video2[:, :context_frames]
        
        # Project inputs
        v1_proj = self.input_proj(v1_context)
        v2_proj = self.input_proj(v2_context)
        
        # Add video embeddings
        v1_embed = self.video_embedding(torch.zeros(B, context_frames, dtype=torch.long, device=device))
        v2_embed = self.video_embedding(torch.ones(B, context_frames, dtype=torch.long, device=device))
        
        v1_proj = v1_proj + v1_embed
        v2_proj = v2_proj + v2_embed
        
        # Concatenate sequences
        sequence = torch.cat([v1_proj, v2_proj], dim=1)
        
        # Add positional encoding
        T_total = sequence.shape[1]
        sequence = sequence + self.pos_encoding[:, :T_total, :]
        
        # Transformer encoding
        hidden_states = self.transformer(sequence)
        
        # Generate transition predictions
        # Use the middle hidden states to predict transition
        mid_start = context_frames - transition_length // 2
        mid_end = context_frames + transition_length // 2
        transition_states = hidden_states[:, mid_start:mid_end]
        
        # Predict MANO parameters for transition
        mano_chunks = self.mano_head(transition_states)
        mano_chunks = mano_chunks.reshape(B, transition_length, self.chunk_size, self.mano_dim)
        
        # Predict boundaries
        boundaries = torch.sigmoid(self.boundary_head(hidden_states))
        
        # Predict transition quality score
        quality_scores = torch.sigmoid(self.transition_quality_head(transition_states))
        
        # Get task embeddings
        task_embeddings = self.task_embedding_head(hidden_states)
        
        return {
            'mano_chunks': mano_chunks,
            'boundaries': boundaries,
            'transition_quality': quality_scores,
            'task_embeddings': task_embeddings,
            'hidden_states': hidden_states,
            'transition_states': transition_states
        }


class TransitionDiffuser(nn.Module):
    """Diffusion model for refining transitions with contact awareness"""
    
    def __init__(self,
                 mano_dim: int = 51,
                 hidden_dim: int = 256,
                 condition_dim: int = 512,
                 num_timesteps: int = 100):
        super().__init__()
        
        self.mano_dim = mano_dim
        self.num_timesteps = num_timesteps
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Condition embedding (transformer states + contact info)
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim + 4, hidden_dim),  # +4 for contact info
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Denoising network with residual connections
        self.denoiser = nn.ModuleList([
            nn.Sequential(
                nn.Linear(mano_dim + hidden_dim * 2, hidden_dim * 4),
                nn.SiLU(),
                nn.Linear(hidden_dim * 4, hidden_dim * 4)
            ),
            nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 2)
            ),
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, mano_dim)
            )
        ])
        
        # Noise schedule
        self.register_buffer('betas', torch.linspace(0.0001, 0.02, num_timesteps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
    def forward(self, 
                x: torch.Tensor, 
                t: torch.Tensor, 
                condition: torch.Tensor,
                contact_info: torch.Tensor) -> torch.Tensor:
        """
        Predict noise for denoising step
        
        Args:
            x: [B, mano_dim] noisy MANO parameters
            t: [B] diffusion timestep
            condition: [B, condition_dim] conditioning from transformer
            contact_info: [B, 4] contact strength and position
            
        Returns:
            [B, mano_dim] predicted noise
        """
        # Embed time
        t_embed = self.time_embed(t.unsqueeze(1).float() / self.num_timesteps)
        
        # Embed condition with contact info
        condition_with_contact = torch.cat([condition, contact_info], dim=1)
        c_embed = self.condition_embed(condition_with_contact)
        
        # Initial features
        h = torch.cat([x, t_embed, c_embed], dim=1)
        
        # Pass through denoising layers
        for layer in self.denoiser:
            h = layer(h)
            
        return h
    
    def sample(self, 
               condition: torch.Tensor, 
               contact_info: torch.Tensor,
               num_samples: int = 1) -> torch.Tensor:
        """Generate MANO parameters through reverse diffusion"""
        B = condition.shape[0]
        device = condition.device
        
        # Start from noise
        x = torch.randn(B, num_samples, self.mano_dim).to(device)
        
        # Reverse diffusion process
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((B,), t, device=device)
            
            # Reshape for batch processing
            x_flat = x.reshape(B * num_samples, self.mano_dim)
            condition_exp = condition.unsqueeze(1).expand(-1, num_samples, -1).reshape(B * num_samples, -1)
            contact_exp = contact_info.unsqueeze(1).expand(-1, num_samples, -1).reshape(B * num_samples, -1)
            t_batch_exp = t_batch.unsqueeze(1).expand(-1, num_samples).reshape(-1)
            
            # Predict noise
            noise = self.forward(x_flat, t_batch_exp, condition_exp, contact_exp)
            noise = noise.reshape(B, num_samples, self.mano_dim)
            
            # DDPM update
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            
            if t > 0:
                noise_scale = torch.sqrt(self.betas[t])
                x = (x - (1 - alpha) / torch.sqrt(1 - alpha_cumprod) * noise) / torch.sqrt(alpha)
                x = x + noise_scale * torch.randn_like(x)
            else:
                x = (x - (1 - alpha) / torch.sqrt(1 - alpha_cumprod) * noise) / torch.sqrt(alpha)
                
        return x


class TransitionMergerModel(nn.Module):
    """Complete model for merging video transitions using HOISDF outputs"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Initialize components
        self.tokenizer = HOISDFTokenizer(**config['tokenizer'])
        self.transformer = TransitionTransformer(**config['transformer'])
        self.diffuser = TransitionDiffuser(**config['diffuser'])
        
        # Fusion layer for diffuser conditioning
        self.condition_fusion = nn.Linear(
            config['transformer']['hidden_dim'],
            config['diffuser']['condition_dim']
        )
        
    def forward(self, 
                hoisdf_outputs1: HOISDFOutputs,
                hoisdf_outputs2: HOISDFOutputs,
                transition_length: int = 30,
                mode: str = 'train') -> Dict[str, torch.Tensor]:
        """
        Generate smooth transition between two video sequences
        
        Args:
            hoisdf_outputs1: HOISDF outputs from first video
            hoisdf_outputs2: HOISDF outputs from second video
            transition_length: Number of frames for transition
            mode: 'train' or 'inference'
            
        Returns:
            Dictionary with transition predictions
        """
        # Tokenize HOISDF outputs
        tokens1 = self.tokenizer(hoisdf_outputs1)
        tokens2 = self.tokenizer(hoisdf_outputs2)
        
        # Generate transition with transformer
        transformer_outputs = self.transformer(
            tokens1, tokens2, transition_length
        )
        
        # Prepare conditioning for diffuser
        transition_states = transformer_outputs['transition_states']
        B, T_trans = transition_states.shape[:2]
        
        # Apply diffusion refinement if in inference mode
        if mode == 'inference':
            refined_mano = []
            
            for t in range(T_trans):
                # Get condition from transformer
                condition = self.condition_fusion(transition_states[:, t])
                
                # Estimate contact info at this timestep
                # Linear interpolation of contact from end of video1 to start of video2
                alpha = t / T_trans
                contact_info = torch.zeros(B, 4).to(condition.device)
                # You would compute actual contact info from HOISDF outputs
                
                # Refine with diffusion
                refined_params = self.diffuser.sample(
                    condition, 
                    contact_info,
                    num_samples=1
                ).squeeze(1)
                
                refined_mano.append(refined_params)
                
            transformer_outputs['refined_mano'] = torch.stack(refined_mano, dim=1)
        else:
            # During training, just use transformer predictions
            transformer_outputs['refined_mano'] = transformer_outputs['mano_chunks'][:, :, 0, :]
            
        return {
            'transformer': transformer_outputs,
            'tokens': {'video1': tokens1, 'video2': tokens2}
        }


class TransitionLoss(nn.Module):
    """Loss functions for transition learning"""
    
    def __init__(self, weights: Dict[str, float]):
        super().__init__()
        self.weights = weights
        
    def mano_reconstruction_loss(self, pred_mano: torch.Tensor, 
                                gt_mano: torch.Tensor) -> torch.Tensor:
        """L2 loss for MANO parameter reconstruction"""
        return F.mse_loss(pred_mano, gt_mano)
    
    def contact_consistency_loss(self, pred_mano: torch.Tensor,
                                hoisdf_outputs: HOISDFOutputs) -> torch.Tensor:
        """Ensure predicted MANO maintains contact when needed"""
        # This would compute whether predicted MANO parameters
        # maintain contact at frames where original had contact
        # Simplified version:
        contact_frames = (hoisdf_outputs.contact_frames.sum(dim=-1) > 0).float()
        
        # Penalize large movements during contact
        if pred_mano.shape[1] > 1:
            mano_velocity = torch.diff(pred_mano, dim=1).norm(dim=-1)
            contact_velocity_penalty = (mano_velocity * contact_frames[:, 1:]).mean()
            return contact_velocity_penalty
        return torch.tensor(0.0).to(pred_mano.device)
    
    def smoothness_loss(self, pred_mano: torch.Tensor) -> torch.Tensor:
        """Penalize jerky movements"""
        if pred_mano.shape[1] > 2:
            # Velocity
            velocity = torch.diff(pred_mano, dim=1)
            # Acceleration
            acceleration = torch.diff(velocity, dim=1)
            
            return velocity.norm(dim=-1).mean() + acceleration.norm(dim=-1).mean()
        return torch.tensor(0.0).to(pred_mano.device)
    
    def boundary_loss(self, pred_boundaries: torch.Tensor,
                     gt_boundaries: torch.Tensor) -> torch.Tensor:
        """BCE loss for boundary detection"""
        return F.binary_cross_entropy(pred_boundaries, gt_boundaries)
    
    def contrastive_loss(self, embeddings: torch.Tensor,
                        video_labels: torch.Tensor) -> torch.Tensor:
        """Contrastive loss for task embeddings"""
        B, T, D = embeddings.shape
        
        # Simple contrastive: embeddings from same video should be similar
        loss = 0
        for i in range(B):
            for t in range(T-1):
                if video_labels[i, t] == video_labels[i, t+1]:
                    # Same video - should be similar
                    loss += (1 - F.cosine_similarity(
                        embeddings[i, t].unsqueeze(0),
                        embeddings[i, t+1].unsqueeze(0)
                    ))
                else:
                    # Different videos - should be different
                    loss += F.relu(0.5 + F.cosine_similarity(
                        embeddings[i, t].unsqueeze(0),
                        embeddings[i, t+1].unsqueeze(0)
                    ))
        
        return loss / (B * (T-1))
    
    def diffusion_loss(self, diffuser: TransitionDiffuser,
                      gt_mano: torch.Tensor,
                      conditions: torch.Tensor,
                      contact_info: torch.Tensor) -> torch.Tensor:
        """Diffusion denoising loss"""
        B = gt_mano.shape[0]
        device = gt_mano.device
        
        # Sample random timesteps
        t = torch.randint(0, diffuser.num_timesteps, (B,), device=device)
        
        # Add noise to ground truth MANO parameters
        noise = torch.randn_like(gt_mano)
        alpha_cumprod = diffuser.alphas_cumprod[t]
        noisy_mano = torch.sqrt(alpha_cumprod).unsqueeze(1) * gt_mano + \
                     torch.sqrt(1 - alpha_cumprod).unsqueeze(1) * noise
        
        # Predict noise
        pred_noise = diffuser(noisy_mano, t, conditions, contact_info)
        
        return F.mse_loss(pred_noise, noise)
    
    def forward(self, 
                model_outputs: Dict,
                gt_transition: HOISDFOutputs,
                model: TransitionMergerModel) -> Dict[str, torch.Tensor]:
        """Compute all losses"""
        losses = {}
        
        # MANO reconstruction loss
        pred_mano = model_outputs['transformer']['refined_mano']
        gt_mano = gt_transition.mano_params
        losses['mano_recon'] = self.mano_reconstruction_loss(pred_mano, gt_mano) * \
                               self.weights.get('mano_recon', 1.0)
        
        # Contact consistency
        losses['contact'] = self.contact_consistency_loss(pred_mano, gt_transition) * \
                           self.weights.get('contact', 0.5)
        
        # Smoothness
        losses['smooth'] = self.smoothness_loss(pred_mano) * \
                          self.weights.get('smooth', 0.1)
        
        # Boundary detection (if available)
        if 'boundaries' in model_outputs['transformer']:
            # Create pseudo ground truth boundaries
            T = model_outputs['transformer']['boundaries'].shape[1]
            gt_boundaries = torch.zeros_like(model_outputs['transformer']['boundaries'])
            # Mark transition region
            transition_start = T // 3
            transition_end = 2 * T // 3
            gt_boundaries[:, transition_start:transition_end] = 1.0
            
            losses['boundary'] = self.boundary_loss(
                model_outputs['transformer']['boundaries'],
                gt_boundaries
            ) * self.weights.get('boundary', 0.5)
        
        # Contrastive loss
        embeddings = model_outputs['transformer']['task_embeddings']
        # Create video labels (0 for video1, 1 for video2)
        T = embeddings.shape[1]
        video_labels = torch.cat([
            torch.zeros(embeddings.shape[0], T//2),
            torch.ones(embeddings.shape[0], T//2)
        ], dim=1).to(embeddings.device)
        
        losses['contrastive'] = self.contrastive_loss(embeddings, video_labels) * \
                               self.weights.get('contrastive', 0.2)
        
        # Diffusion loss (sample some timesteps)
        if hasattr(model, 'diffuser'):
            transition_states = model_outputs['transformer']['transition_states']
            for t in range(0, transition_states.shape[1], 5):  # Every 5th frame
                condition = model.condition_fusion(transition_states[:, t])
                contact_info = torch.zeros(condition.shape[0], 4).to(condition.device)
                
                losses[f'diffusion_{t}'] = self.diffusion_loss(
                    model.diffuser,
                    gt_mano[:, t],
                    condition,
                    contact_info
                ) * self.weights.get('diffusion', 0.5)
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'tokenizer': {
            'mano_dim': 51,
            'sdf_resolution': 64,
            'hidden_dim': 256,
            'num_tokens': 256
        },
        'transformer': {
            'input_dim': 256,
            'hidden_dim': 512,
            'num_heads': 8,
            'num_layers': 6,
            'mano_dim': 51,
            'chunk_size': 50,
            'dropout': 0.1
        },
        'diffuser': {
            'mano_dim': 51,
            'hidden_dim': 256,
            'condition_dim': 512,
            'num_timesteps': 100
        }
    }
    
    # Loss weights
    loss_weights = {
        'mano_recon': 1.0,
        'contact': 0.5,
        'smooth': 0.1,
        'boundary': 0.5,
        'contrastive': 0.2,
        'diffusion': 0.5
    }
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransitionMergerModel(config).to(device)
    criterion = TransitionLoss(loss_weights)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test with dummy data
    T1, T2 = 100, 100
    
    # Create dummy HOISDF outputs
    dummy_outputs1 = HOISDFOutputs(
        mano_params=torch.randn(T1, 51).to(device),
        hand_sdf=torch.randn(T1, 64, 64, 64).to(device),
        object_sdf=torch.randn(T1, 64, 64, 64).to(device),
        contact_points=torch.randn(T1, 10, 3).to(device),
        contact_frames=torch.randint(0, 2, (T1, 10)).float().to(device),
        hand_vertices=torch.randn(T1, 778, 3).to(device),
        object_center=torch.randn(T1, 3).to(device)
    )
    
    dummy_outputs2 = HOISDFOutputs(
        mano_params=torch.randn(T2, 51).to(device),
        hand_sdf=torch.randn(T2, 64, 64, 64).to(device),
        object_sdf=torch.randn(T2, 64, 64, 64).to(device),
        contact_points=torch.randn(T2, 10, 3).to(device),
        contact_frames=torch.randint(0, 2, (T2, 10)).float().to(device),
        hand_vertices=torch.randn(T2, 778, 3).to(device),
        object_center=torch.randn(T2, 3).to(device)
    )
    
    # Forward pass
    outputs = model(dummy_outputs1, dummy_outputs2, transition_length=30, mode='train')
    print("Forward pass successful!")
    print(f"Transition MANO shape: {outputs['transformer']['refined_mano'].shape}")