import torch
import torch.nn as nn
from typing import Dict, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TrainingStabilizer:
    """
    Solutions for training instability issues
    """
    
    @staticmethod
    def create_robust_optimizer(model: nn.Module, config: Dict) -> Tuple[torch.optim.Optimizer, Dict]:
        """
        Create optimizer with stability features
        """
        # Separate parameters by stability requirements
        stable_params = []
        sensitive_params = []
        
        for name, param in model.named_parameters():
            if any(s in name for s in ['norm', 'bias', 'positional']):
                sensitive_params.append(param)
            else:
                stable_params.append(param)
        
        # Different settings for different parameters
        optimizer = torch.optim.AdamW([
            {
                'params': stable_params,
                'lr': config['learning_rate'],
                'betas': (0.9, 0.999),
                'weight_decay': 0.01
            },
            {
                'params': sensitive_params,
                'lr': config['learning_rate'] * 0.1,
                'betas': (0.9, 0.999),
                'weight_decay': 0.0  # No weight decay for these
            }
        ])
        
        # Add gradient centralization
        optimizer = GradientCentralization(optimizer)
        
        return optimizer
    
    @staticmethod
    def detect_and_handle_anomalies(loss: torch.Tensor, model: nn.Module, step: int) -> torch.Tensor:
        """
        Detect and handle training anomalies
        """
        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"NaN/Inf loss detected at step {step}")
            
            # Reset to last good checkpoint
            return None
        
        # Check for loss explosion
        if hasattr(detect_and_handle_anomalies, 'loss_history'):
            loss_history = detect_and_handle_anomalies.loss_history
            
            if len(loss_history) > 10:
                recent_mean = np.mean(loss_history[-10:])
                if loss.item() > recent_mean * 10:
                    logger.warning(f"Loss explosion detected at step {step}: {loss.item()} vs {recent_mean}")
                    
                    # Skip this batch
                    return None
        else:
            detect_and_handle_anomalies.loss_history = []
        
        detect_and_handle_anomalies.loss_history.append(loss.item())
        
        # Check gradient norms
        total_grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_grad_norm += param_norm ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        if total_grad_norm > 100:
            logger.warning(f"Large gradient norm at step {step}: {total_grad_norm}")
        
        return loss

class GradientCentralization:
    """
    Gradient Centralization for more stable training
    """
    
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
    def step(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None and len(p.shape) > 1:
                    # Centralize gradients
                    p.grad.data -= p.grad.data.mean(dim=0, keepdim=True)
        
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)