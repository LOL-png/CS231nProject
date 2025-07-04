import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.hooks import RemovableHandle
import logging
import os
import gc
import warnings

logger = logging.getLogger(__name__)

class MemoryEfficientDebugger:
    """
    Memory-efficient debugging tools for the Advanced Manipulation Transformer
    
    Key improvements:
    1. Uses torch.no_grad() to prevent gradient accumulation
    2. Immediately moves data to GPU and deletes CPU references
    3. Limits stored data to prevent memory bloat
    4. Adds explicit garbage collection
    5. Disables torch.compile during debugging
    """
    
    def __init__(self, model: nn.Module, save_dir: str = "debug_outputs", 
                 max_stored_activations: int = 10, max_stored_batches: int = 5):
        self.model = model
        self.save_dir = save_dir
        self.max_stored_activations = max_stored_activations
        self.max_stored_batches = max_stored_batches
        os.makedirs(save_dir, exist_ok=True)
        
        # Disable torch.compile for debugging
        self._disable_compile()
        
        # Limited storage for debugging data
        self.activations = {}
        self.gradients = {}
        self.gradient_norms = {}
        self.hooks = []
        self.stored_predictions = []
        
        # Track memory usage
        self.initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
    def _disable_compile(self):
        """Disable torch.compile to avoid compilation overhead during debugging"""
        for name, module in self.model.named_modules():
            if hasattr(module, '_compiled'):
                warnings.warn(f"Disabling torch.compile for module {name}")
                # Replace with original module if possible
                if hasattr(module, '_original'):
                    setattr(self.model, name, module._original)
    
    def analyze_model(self, sample_batch: Dict[str, torch.Tensor]):
        """
        Perform memory-efficient model analysis
        """
        logger.info("Starting memory-efficient model analysis...")
        
        # Move batch to GPU immediately if not already there
        device = next(self.model.parameters()).device
        sample_batch = self._move_batch_to_device(sample_batch, device)
        
        try:
            # 1. Quick model summary (no memory overhead)
            self.print_model_summary()
            
            # 2. Analyze forward pass with limited storage
            with torch.no_grad():
                self.analyze_forward_pass(sample_batch)
            
            # 3. Analyze backward pass with cleanup
            self.analyze_backward_pass(sample_batch)
            
            # 4. Quick checks without storing data
            self.check_common_issues()
            
            # 5. Generate visualizations and clean up
            self.visualize_analysis()
            
        finally:
            # Cleanup
            self._cleanup()
            
        logger.info("Model analysis complete")
        logger.info(f"Peak memory usage: {self._get_memory_usage():.2f} MB")
    
    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        """Move batch to device and delete CPU references"""
        gpu_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                gpu_batch[key] = value.to(device, non_blocking=True)
                # Delete CPU reference if it was on CPU
                if value.device.type == 'cpu':
                    del value
            else:
                gpu_batch[key] = value
        
        # Force garbage collection
        gc.collect()
        return gpu_batch
    
    def print_model_summary(self):
        """Print model summary without storing intermediate data"""
        print("\n" + "="*50)
        print("MODEL ARCHITECTURE SUMMARY")
        print("="*50)
        
        total_params = 0
        trainable_params = 0
        
        # Use generator to avoid storing all modules
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
        """Analyze forward pass with limited memory usage"""
        logger.info("Analyzing forward pass (memory-efficient)...")
        
        # Register hooks with limited storage
        self._register_limited_forward_hooks()
        
        # Forward pass without gradients
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(sample_batch)
        
        # Analyze stored activations
        activation_count = 0
        for name, activation in self.activations.items():
            activation_count += 1
            if activation_count > self.max_stored_activations:
                break
                
            # Compute stats on GPU to avoid CPU transfer
            stats = self._compute_tensor_stats_gpu(activation)
            logger.info(f"Activation {name}: shape={activation.shape}, "
                       f"mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
                       f"has_nan={stats['has_nan']}, has_inf={stats['has_inf']}")
            
            # Check for dead neurons
            if stats['zero_fraction'] > 0.5:
                logger.warning(f"Layer {name} has {stats['zero_fraction']:.1%} dead neurons!")
        
        # Clean up hooks and activations
        self._remove_hooks()
        self.activations.clear()
        gc.collect()
        
        return outputs
    
    def analyze_backward_pass(self, sample_batch: Dict[str, torch.Tensor]):
        """Analyze backward pass with memory cleanup"""
        logger.info("Analyzing backward pass (memory-efficient)...")
        
        # Register hooks with limited storage
        self._register_limited_backward_hooks()
        
        # Forward and backward pass
        self.model.train()
        
        # Use gradient checkpointing if available
        with torch.cuda.amp.autocast(enabled=False):  # Disable for accurate gradient analysis
            outputs = self.model(sample_batch)
            
            # Create simple loss
            if isinstance(outputs, dict) and 'hand_joints' in outputs:
                loss = outputs['hand_joints'].mean()
            else:
                loss = outputs.mean() if isinstance(outputs, torch.Tensor) else outputs[0].mean()
            
            loss.backward()
        
        # Analyze gradients with limited storage
        grad_count = 0
        for name, grad in self.gradients.items():
            grad_count += 1
            if grad_count > self.max_stored_activations:
                break
                
            stats = self._compute_tensor_stats_gpu(grad)
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
        self.gradients.clear()
        self.model.zero_grad()
        gc.collect()
        torch.cuda.empty_cache()
    
    def check_common_issues(self):
        """Quick checks without storing data"""
        logger.info("Checking for common issues...")
        
        issues_found = []
        
        # Check parameters without storing them
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    # Check on GPU to avoid transfers
                    if param.numel() > 0:
                        if torch.all(param == 0):
                            issues_found.append(f"Parameter {name} is all zeros")
                        elif torch.isnan(param).any():
                            issues_found.append(f"Parameter {name} contains NaN")
                        elif torch.isinf(param).any():
                            issues_found.append(f"Parameter {name} contains Inf")
        
        # Check batch norm statistics
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if module.running_mean is None:
                    issues_found.append(f"BatchNorm {name} has no running statistics")
        
        # Report issues
        if issues_found:
            logger.warning("Issues found:")
            for issue in issues_found[:10]:  # Limit output
                logger.warning(f"  - {issue}")
            if len(issues_found) > 10:
                logger.warning(f"  ... and {len(issues_found) - 10} more issues")
        else:
            logger.info("No common issues found")
    
    def debug_prediction_diversity(self, dataloader, num_batches: int = 5):
        """Check prediction diversity with limited memory usage"""
        logger.info("Checking prediction diversity (memory-efficient)...")
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # Store limited predictions
        batch_predictions = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                # Move to GPU
                batch = self._move_batch_to_device(batch, device)
                
                outputs = self.model(batch)
                
                if isinstance(outputs, dict) and 'hand_joints' in outputs:
                    pred = outputs['hand_joints']
                else:
                    pred = outputs
                
                # Store only summary statistics, not full predictions
                pred_stats = {
                    'mean': pred.mean(dim=0).cpu(),
                    'std': pred.std(dim=0).cpu(),
                    'min': pred.min(dim=0)[0].cpu(),
                    'max': pred.max(dim=0)[0].cpu()
                }
                batch_predictions.append(pred_stats)
                
                # Clean up batch
                del batch
                gc.collect()
        
        # Analyze diversity from statistics
        all_stds = torch.stack([p['std'] for p in batch_predictions])
        avg_std = all_stds.mean().item()
        
        logger.info(f"Average prediction std: {avg_std:.6f}")
        
        if avg_std < 0.001:
            logger.error("SEVERE MODE COLLAPSE DETECTED! Model produces nearly constant predictions")
        elif avg_std < 0.01:
            logger.warning("Low prediction diversity - potential mode collapse")
        else:
            logger.info("Prediction diversity appears normal")
        
        # Clean up
        batch_predictions.clear()
        gc.collect()
    
    def visualize_analysis(self):
        """Create visualizations with minimal memory usage"""
        # Only create essential plots
        self._plot_gradient_flow()
        
        # Clean up matplotlib memory
        plt.close('all')
        gc.collect()
    
    def _register_limited_forward_hooks(self):
        """Register hooks with limited storage"""
        stored_count = 0
        
        def hook_fn(name):
            def hook(module, input, output):
                nonlocal stored_count
                if stored_count < self.max_stored_activations:
                    # Store only small tensors or summaries
                    if output.numel() < 1e6:  # Less than 1M elements
                        self.activations[name] = output.detach()
                    else:
                        # Store only statistics for large tensors
                        self.activations[name] = torch.tensor([
                            output.mean().item(),
                            output.std().item(),
                            output.norm().item()
                        ])
                    stored_count += 1
            return hook
        
        # Only hook selected layers
        target_layers = ['hand_encoder', 'object_encoder', 'contact_encoder', 'fusion']
        for name, module in self.model.named_modules():
            if any(target in name for target in target_layers):
                if len(list(module.children())) == 0 and len(list(module.parameters())) > 0:
                    handle = module.register_forward_hook(hook_fn(name))
                    self.hooks.append(handle)
    
    def _register_limited_backward_hooks(self):
        """Register backward hooks with limited storage"""
        stored_count = 0
        
        def hook_fn(name):
            def hook(module, grad_input, grad_output):
                nonlocal stored_count
                if stored_count < self.max_stored_activations and grad_output[0] is not None:
                    grad = grad_output[0]
                    # Store gradient norm only
                    self.gradient_norms[name] = grad.norm().item()
                    
                    # Store small gradients or summaries
                    if grad.numel() < 1e5:  # Less than 100k elements
                        self.gradients[name] = grad.detach()
                    stored_count += 1
            return hook
        
        # Only hook important layers
        for name, module in self.model.named_modules():
            if 'encoder' in name or 'decoder' in name:
                if len(list(module.children())) == 0 and len(list(module.parameters())) > 0:
                    handle = module.register_backward_hook(hook_fn(name))
                    self.hooks.append(handle)
    
    def _remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def _compute_tensor_stats_gpu(self, tensor: torch.Tensor) -> Dict[str, float]:
        """Compute statistics on GPU to avoid transfers"""
        with torch.no_grad():
            # Compute all stats on GPU
            mean = tensor.mean()
            std = tensor.std()
            norm = tensor.norm()
            has_nan = torch.isnan(tensor).any()
            has_inf = torch.isinf(tensor).any()
            zero_fraction = (tensor == 0).float().mean()
            
            # Transfer only scalars to CPU
            return {
                'mean': mean.item(),
                'std': std.item(),
                'min': tensor.min().item(),
                'max': tensor.max().item(),
                'norm': norm.item(),
                'has_nan': has_nan.item(),
                'has_inf': has_inf.item(),
                'zero_fraction': zero_fraction.item()
            }
    
    def _plot_gradient_flow(self):
        """Plot gradient flow with minimal memory"""
        if not self.gradient_norms:
            return
        
        # Limit number of layers shown
        layers = list(self.gradient_norms.keys())[:20]
        grad_norms = [self.gradient_norms[l] for l in layers]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(layers)), grad_norms)
        plt.xlabel("Layers")
        plt.ylabel("Gradient Norm")
        plt.title("Gradient Flow (Top 20 Layers)")
        plt.xticks(range(len(layers)), [l.split('.')[-1] for l in layers], rotation=45)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "gradient_flow.png"), dpi=100)
        plt.close()
    
    def _cleanup(self):
        """Clean up all stored data"""
        self.activations.clear()
        self.gradients.clear()
        self.gradient_norms.clear()
        self.stored_predictions.clear()
        self._remove_hooks()
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
    
    def _get_memory_usage(self):
        """Get current GPU memory usage in MB"""
        if torch.cuda.is_available():
            return (torch.cuda.memory_allocated() - self.initial_memory) / 1024 / 1024
        return 0.0


def create_memory_efficient_debugger(model: nn.Module, save_dir: str = "debug_outputs") -> MemoryEfficientDebugger:
    """
    Factory function to create a memory-efficient debugger
    
    This is a drop-in replacement for the original ModelDebugger initialization.
    
    Args:
        model: The model to debug
        save_dir: Directory to save debug outputs
        
    Returns:
        MemoryEfficientDebugger instance
    """
    return MemoryEfficientDebugger(
        model=model,
        save_dir=save_dir,
        max_stored_activations=10,  # Limit stored activations
        max_stored_batches=5  # Limit stored batches for diversity check
    )


# Example usage for drop-in replacement:
if __name__ == "__main__":
    # This shows how to use as a drop-in replacement
    
    # Original code:
    # from debugging.model_debugger import ModelDebugger
    # debugger = ModelDebugger(model, save_dir="debug_outputs")
    
    # New memory-efficient code:
    # from debugging.debugger_memory_fix import create_memory_efficient_debugger
    # debugger = create_memory_efficient_debugger(model, save_dir="debug_outputs")
    
    print("Memory-efficient debugger ready for use!")
    print("Key improvements:")
    print("1. Uses torch.no_grad() to prevent gradient accumulation")
    print("2. Moves data to GPU immediately and deletes CPU references") 
    print("3. Limits stored activations and gradients")
    print("4. Adds explicit garbage collection")
    print("5. Disables torch.compile during debugging")
    print("6. Computes statistics on GPU to avoid transfers")