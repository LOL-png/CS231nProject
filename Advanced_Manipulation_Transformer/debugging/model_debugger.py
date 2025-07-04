import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.hooks import RemovableHandle
import logging
import os

logger = logging.getLogger(__name__)

class ModelDebugger:
    """
    Comprehensive debugging tools for the Advanced Manipulation Transformer
    """
    
    def __init__(self, model: nn.Module, save_dir: str = "debug_outputs"):
        self.model = model
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Storage for debugging data
        self.activations = {}
        self.gradients = {}
        self.gradient_norms = {}
        self.hooks = []
        
    def analyze_model(self, sample_batch: Dict[str, torch.Tensor]):
        """
        Perform comprehensive model analysis
        """
        logger.info("Starting model analysis...")
        
        # 1. Check model architecture
        self.print_model_summary()
        
        # 2. Analyze forward pass
        self.analyze_forward_pass(sample_batch)
        
        # 3. Analyze backward pass
        self.analyze_backward_pass(sample_batch)
        
        # 4. Check for common issues
        self.check_common_issues()
        
        # 5. Visualize results
        self.visualize_analysis()
        
        logger.info("Model analysis complete")
    
    def print_model_summary(self):
        """
        Print detailed model architecture summary
        """
        print("\n" + "="*50)
        print("MODEL ARCHITECTURE SUMMARY")
        print("="*50)
        
        total_params = 0
        trainable_params = 0
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                params = sum(p.numel() for p in module.parameters())
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                
                if params > 0:
                    print(f"{name}: {module.__class__.__name__} - "
                          f"Params: {params:,} (Trainable: {trainable:,})")
                
                total_params += params
                trainable_params += trainable
        
        print(f"\nTotal Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
        print("="*50 + "\n")
    
    def analyze_forward_pass(self, sample_batch: Dict[str, torch.Tensor]):
        """
        Analyze activations during forward pass
        """
        logger.info("Analyzing forward pass...")
        
        # Register hooks
        self._register_forward_hooks()
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(sample_batch)
        
        # Analyze activations
        for name, activation in self.activations.items():
            stats = self._compute_tensor_stats(activation)
            logger.info(f"Activation {name}: shape={activation.shape}, "
                       f"mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
                       f"has_nan={stats['has_nan']}, has_inf={stats['has_inf']}")
            
            # Check for dead neurons
            if stats['zero_fraction'] > 0.5:
                logger.warning(f"Layer {name} has {stats['zero_fraction']:.1%} dead neurons!")
        
        # Clean up hooks
        self._remove_hooks()
        
        return outputs
    
    def analyze_backward_pass(self, sample_batch: Dict[str, torch.Tensor]):
        """
        Analyze gradients during backward pass
        """
        logger.info("Analyzing backward pass...")
        
        # Register hooks
        self._register_backward_hooks()
        
        # Forward and backward pass
        self.model.train()
        outputs = self.model(sample_batch)
        
        # Create dummy loss
        if isinstance(outputs, dict) and 'hand_joints' in outputs:
            loss = outputs['hand_joints'].mean()
        else:
            loss = outputs.mean() if isinstance(outputs, torch.Tensor) else outputs[0].mean()
        
        loss.backward()
        
        # Analyze gradients
        for name, grad in self.gradients.items():
            stats = self._compute_tensor_stats(grad)
            logger.info(f"Gradient {name}: shape={grad.shape}, "
                       f"mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
                       f"norm={stats['norm']:.6f}")
            
            # Check for vanishing/exploding gradients
            if stats['norm'] < 1e-7:
                logger.warning(f"Vanishing gradient in {name}!")
            elif stats['norm'] > 100:
                logger.warning(f"Exploding gradient in {name}!")
        
        # Clean up
        self._remove_hooks()
        self.model.zero_grad()
    
    def check_common_issues(self):
        """
        Check for common model issues
        """
        logger.info("Checking for common issues...")
        
        issues_found = []
        
        # 1. Check for uninitialized parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if torch.all(param == 0):
                    issues_found.append(f"Parameter {name} is all zeros")
                elif torch.isnan(param).any():
                    issues_found.append(f"Parameter {name} contains NaN")
                elif torch.isinf(param).any():
                    issues_found.append(f"Parameter {name} contains Inf")
        
        # 2. Check for duplicate parameters
        param_ids = set()
        for name, param in self.model.named_parameters():
            param_id = id(param)
            if param_id in param_ids:
                issues_found.append(f"Duplicate parameter found: {name}")
            param_ids.add(param_id)
        
        # 3. Check batch norm statistics
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                if module.running_mean is None:
                    issues_found.append(f"BatchNorm {name} has no running statistics")
                elif torch.all(module.running_var == 1) and torch.all(module.running_mean == 0):
                    issues_found.append(f"BatchNorm {name} statistics not updated")
        
        # Report issues
        if issues_found:
            logger.warning("Issues found:")
            for issue in issues_found:
                logger.warning(f"  - {issue}")
        else:
            logger.info("No common issues found")
    
    def visualize_analysis(self):
        """
        Create visualizations of the analysis
        """
        # 1. Activation distributions
        self._plot_activation_distributions()
        
        # 2. Gradient flow
        self._plot_gradient_flow()
        
        # 3. Parameter distributions
        self._plot_parameter_distributions()
    
    def debug_prediction_diversity(self, dataloader, num_batches: int = 10):
        """
        Check if model produces diverse predictions
        """
        logger.info("Checking prediction diversity...")
        
        predictions = []
        self.model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                outputs = self.model(batch)
                
                if isinstance(outputs, dict) and 'hand_joints' in outputs:
                    pred = outputs['hand_joints']
                else:
                    pred = outputs
                
                predictions.append(pred.cpu())
        
        # Analyze diversity
        predictions = torch.cat(predictions, dim=0)
        
        # Compute statistics
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        # Check for mode collapse
        avg_std = std_pred.mean().item()
        logger.info(f"Average prediction std: {avg_std:.6f}")
        
        if avg_std < 0.001:
            logger.error("SEVERE MODE COLLAPSE DETECTED! Model produces nearly constant predictions")
        elif avg_std < 0.01:
            logger.warning("Low prediction diversity - potential mode collapse")
        else:
            logger.info("Prediction diversity appears normal")
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot prediction distribution
        axes[0].hist(predictions.flatten().numpy(), bins=50, alpha=0.7)
        axes[0].set_title("Prediction Distribution")
        axes[0].set_xlabel("Value")
        axes[0].set_ylabel("Count")
        
        # Plot std heatmap
        if std_pred.dim() == 2:
            im = axes[1].imshow(std_pred.numpy(), cmap='hot', aspect='auto')
            axes[1].set_title("Prediction Std Dev Heatmap")
            axes[1].set_xlabel("Feature")
            axes[1].set_ylabel("Joint")
            plt.colorbar(im, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "prediction_diversity.png"))
        plt.close()
    
    def _register_forward_hooks(self):
        """Register hooks to capture activations"""
        def hook_fn(name):
            def hook(module, input, output):
                self.activations[name] = output.detach().cpu()
            return hook
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0 and len(list(module.parameters())) > 0:
                handle = module.register_forward_hook(hook_fn(name))
                self.hooks.append(handle)
    
    def _register_backward_hooks(self):
        """Register hooks to capture gradients"""
        def hook_fn(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    self.gradients[name] = grad_output[0].detach().cpu()
                    self.gradient_norms[name] = grad_output[0].norm().item()
            return hook
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0 and len(list(module.parameters())) > 0:
                handle = module.register_backward_hook(hook_fn(name))
                self.hooks.append(handle)
    
    def _remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def _compute_tensor_stats(self, tensor: torch.Tensor) -> Dict[str, float]:
        """Compute statistics for a tensor"""
        return {
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'min': tensor.min().item(),
            'max': tensor.max().item(),
            'norm': tensor.norm().item(),
            'has_nan': torch.isnan(tensor).any().item(),
            'has_inf': torch.isinf(tensor).any().item(),
            'zero_fraction': (tensor == 0).float().mean().item()
        }
    
    def _plot_activation_distributions(self):
        """Plot activation distributions"""
        if not self.activations:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (name, activation) in enumerate(list(self.activations.items())[:4]):
            ax = axes[idx]
            
            # Flatten activation
            act_flat = activation.flatten().numpy()
            
            # Plot histogram
            ax.hist(act_flat, bins=50, alpha=0.7, density=True)
            ax.axvline(act_flat.mean(), color='red', linestyle='--', label='Mean')
            ax.set_title(f"{name}")
            ax.set_xlabel("Activation Value")
            ax.set_ylabel("Density")
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "activation_distributions.png"))
        plt.close()
    
    def _plot_gradient_flow(self):
        """Plot gradient flow through layers"""
        if not self.gradient_norms:
            return
        
        layers = list(self.gradient_norms.keys())
        grad_norms = list(self.gradient_norms.values())
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(layers)), grad_norms)
        plt.xlabel("Layers")
        plt.ylabel("Gradient Norm")
        plt.title("Gradient Flow")
        plt.xticks(range(len(layers)), layers, rotation=90)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "gradient_flow.png"))
        plt.close()
    
    def _plot_parameter_distributions(self):
        """Plot parameter distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        param_groups = {
            'weights': [],
            'biases': [],
            'norms': [],
            'embeddings': []
        }
        
        for name, param in self.model.named_parameters():
            param_flat = param.detach().cpu().flatten().numpy()
            
            if 'bias' in name:
                param_groups['biases'].extend(param_flat)
            elif 'norm' in name:
                param_groups['norms'].extend(param_flat)
            elif 'embed' in name or 'positional' in name:
                param_groups['embeddings'].extend(param_flat)
            else:
                param_groups['weights'].extend(param_flat)
        
        for idx, (group_name, values) in enumerate(param_groups.items()):
            if idx >= 4 or not values:
                continue
                
            ax = axes[idx]
            ax.hist(values, bins=50, alpha=0.7, density=True)
            ax.set_title(f"{group_name.capitalize()} Distribution")
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            
            # Add statistics
            if values:
                mean = np.mean(values)
                std = np.std(values)
                ax.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.3f}')
                ax.text(0.05, 0.95, f'Std: {std:.3f}', transform=ax.transAxes,
                       verticalalignment='top')
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "parameter_distributions.png"))
        plt.close()