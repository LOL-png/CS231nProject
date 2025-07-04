"""
PyTorch Native Optimization for H200
====================================

This module leverages PyTorch 2.0+ built-in optimizations for maximum performance
without external dependencies or compatibility issues.

Key optimizations:
1. Scaled Dot Product Attention (SDPA) - automatic Flash Attention
2. torch.compile() - kernel fusion and graph optimization  
3. Better memory management with native PyTorch features
4. Optimal settings for H200 GPU
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any
import os

logger = logging.getLogger(__name__)

class PyTorchNativeOptimizer:
    """
    Optimizer that uses only PyTorch native features for best performance
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.pytorch_version = torch.__version__
        self.is_h200 = self._detect_h200()
        
        # Check available optimizations
        self.has_sdpa = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.has_compile = hasattr(torch, 'compile')
        self.has_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
        
        self._log_capabilities()
    
    def _detect_h200(self) -> bool:
        """Detect if running on H200"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return 'H200' in gpu_name or 'H100' in gpu_name
        return False
    
    def _log_capabilities(self):
        """Log available optimizations"""
        logger.info(f"PyTorch version: {self.pytorch_version}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        logger.info(f"Optimizations available:")
        logger.info(f"  - SDPA (Flash Attention): {'✓' if self.has_sdpa else '✗'}")
        logger.info(f"  - torch.compile: {'✓' if self.has_compile else '✗'}")
        logger.info(f"  - BFloat16: {'✓' if self.has_bf16 else '✗'}")
        logger.info(f"  - H200 detected: {'✓' if self.is_h200 else '✗'}")
    
    def optimize_model(self, model: nn.Module, config: Optional[Dict[str, Any]] = None) -> nn.Module:
        """
        Apply all PyTorch native optimizations
        
        Args:
            model: The model to optimize
            config: Optional configuration dict
            
        Returns:
            Optimized model
        """
        config = config or {}
        
        # 1. Enable SDPA optimizations
        model = self._enable_sdpa_optimizations(model)
        
        # 2. Set optimal CUDA settings
        self._configure_cuda_settings()
        
        # 3. Apply torch.compile if available and requested
        if config.get('use_compile', True) and self.has_compile:
            model = self._compile_model(model, config)
        
        # 4. Enable memory optimizations
        self._enable_memory_optimizations(model)
        
        logger.info("✓ PyTorch native optimizations applied successfully")
        return model
    
    def _enable_sdpa_optimizations(self, model: nn.Module) -> nn.Module:
        """Enable SDPA (Scaled Dot Product Attention) optimizations"""
        
        if not self.has_sdpa:
            logger.warning("SDPA not available in this PyTorch version")
            return model
        
        # SDPA is automatically used in MultiheadAttention in PyTorch 2.0+
        # We just need to ensure the right settings
        
        # Count attention layers
        attn_count = sum(1 for _ in model.modules() if isinstance(_, nn.MultiheadAttention))
        
        if attn_count > 0:
            logger.info(f"✓ SDPA enabled for {attn_count} attention layers")
            
            # Enable all SDPA backends for automatic selection
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
                logger.info("  - Enabled all SDPA backends (Flash, MemEfficient, Math)")
        
        return model
    
    def _configure_cuda_settings(self):
        """Configure CUDA settings for optimal performance"""
        
        if not torch.cuda.is_available():
            return
        
        # 1. Enable TF32 for H100/H200
        if self.is_h200:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("✓ TF32 enabled for H200 Tensor Cores")
        
        # 2. Enable cuDNN autotuning
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        logger.info("✓ cuDNN autotuning enabled")
        
        # 3. Set memory allocation settings
        # Larger memory pools for H200's 140GB memory
        if self.is_h200:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
            torch.cuda.set_per_process_memory_fraction(0.95)
            logger.info("✓ Memory allocation optimized for H200")
        
        # 4. Clear cache before starting
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    def _compile_model(self, model: nn.Module, config: Dict[str, Any]) -> nn.Module:
        """Apply torch.compile optimization"""
        
        compile_mode = config.get('compile_mode', 'default')
        
        try:
            # Different modes for different use cases
            if compile_mode == 'max_performance':
                # Maximum optimization, longer compile time
                compiled_model = torch.compile(
                    model,
                    mode='max-autotune',
                    fullgraph=True,
                    dynamic=False
                )
                logger.info("✓ Model compiled with max-autotune mode")
                
            elif compile_mode == 'reduce_overhead':
                # Good for smaller models
                compiled_model = torch.compile(
                    model,
                    mode='reduce-overhead',
                    fullgraph=True
                )
                logger.info("✓ Model compiled with reduce-overhead mode")
                
            else:
                # Default - balanced performance
                compiled_model = torch.compile(model)
                logger.info("✓ Model compiled with default mode")
            
            return compiled_model
            
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
            logger.info("Continuing without compilation")
            return model
    
    def _enable_memory_optimizations(self, model: nn.Module):
        """Enable memory-efficient features"""
        
        # 1. Gradient checkpointing for large models
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        
        if param_count > 500:  # 500M+ parameters
            # Enable gradient checkpointing for transformer layers
            for module in model.modules():
                if hasattr(module, 'gradient_checkpointing_enable'):
                    module.gradient_checkpointing_enable()
                    logger.info(f"✓ Gradient checkpointing enabled for {module.__class__.__name__}")
        
        # 2. Enable channels_last memory format for CNNs
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                module = module.to(memory_format=torch.channels_last)
                logger.info("✓ Channels-last memory format enabled for convolutions")
                break
    
    def create_optimized_trainer(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                               use_amp: bool = True) -> 'OptimizedTrainer':
        """Create an optimized trainer instance"""
        return OptimizedTrainer(model, optimizer, use_amp=use_amp, device=self.device)


class OptimizedTrainer:
    """
    Trainer that uses PyTorch native optimizations
    """
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                 use_amp: bool = True, device: str = 'cuda'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
        # Setup mixed precision
        self.use_amp = use_amp and torch.cuda.is_available()
        if self.use_amp:
            # Use BFloat16 on H100/H200, Float16 on older GPUs
            gpu_capability = torch.cuda.get_device_capability()[0]
            self.amp_dtype = torch.bfloat16 if gpu_capability >= 8 else torch.float16
            logger.info(f"✓ Mixed precision enabled with {self.amp_dtype}")
            
            # GradScaler only needed for Float16
            self.scaler = torch.cuda.amp.GradScaler() if self.amp_dtype == torch.float16 else None
        else:
            self.amp_dtype = None
            self.scaler = None
    
    def train_step(self, batch: Dict[str, torch.Tensor], loss_fn) -> tuple:
        """Single training step with all optimizations"""
        
        # Zero gradients (set_to_none is faster)
        self.optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with mixed precision
        if self.use_amp:
            with torch.amp.autocast(device_type=self.device, dtype=self.amp_dtype):
                outputs = self.model(batch)
                loss = loss_fn(outputs, batch)
        else:
            outputs = self.model(batch)
            loss = loss_fn(outputs, batch)
        
        # Backward pass
        if self.scaler is not None:
            # Float16 needs gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # BFloat16 or no AMP
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        return loss.item(), outputs
    
    @torch.no_grad()
    def eval_step(self, batch: Dict[str, torch.Tensor], loss_fn) -> tuple:
        """Single evaluation step"""
        
        if self.use_amp:
            with torch.amp.autocast(device_type=self.device, dtype=self.amp_dtype):
                outputs = self.model(batch)
                loss = loss_fn(outputs, batch)
        else:
            outputs = self.model(batch)
            loss = loss_fn(outputs, batch)
        
        return loss.item(), outputs


# Convenience functions for notebook users
def optimize_for_h200(model: nn.Module, compile_mode: str = 'default') -> nn.Module:
    """
    One-line optimization for H200 GPU
    
    Args:
        model: Model to optimize
        compile_mode: 'default', 'max_performance', or 'reduce_overhead'
        
    Returns:
        Optimized model
    """
    optimizer = PyTorchNativeOptimizer()
    return optimizer.optimize_model(model, {'compile_mode': compile_mode})


def create_optimized_training_setup(model: nn.Module, learning_rate: float = 1e-4,
                                  weight_decay: float = 0.01, compile_model: bool = True):
    """
    Create complete optimized training setup
    
    Returns:
        model, optimizer, trainer
    """
    # Optimize model
    native_optimizer = PyTorchNativeOptimizer()
    model = native_optimizer.optimize_model(model, {'use_compile': compile_model})
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, 
                                 weight_decay=weight_decay, fused=True)  # Fused optimizer on GPU
    
    # Create trainer
    trainer = native_optimizer.create_optimized_trainer(model, optimizer)
    
    return model, optimizer, trainer


# Example usage and performance test
if __name__ == "__main__":
    print("PyTorch Native Optimization Demo")
    print("=" * 60)
    
    # Create a sample model
    model = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=1024, nhead=16, batch_first=True),
        num_layers=12
    ).cuda()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    # Apply optimizations
    model = optimize_for_h200(model)
    
    # Test performance
    print("\nTesting optimized performance...")
    x = torch.randn(32, 512, 1024, device='cuda')
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
    
    # Benchmark
    import time
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        for _ in range(100):
            _ = model(x)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Average forward pass: {elapsed/100*1000:.2f}ms")
    print(f"Throughput: {32*512*100/elapsed/1e6:.1f}M tokens/sec")