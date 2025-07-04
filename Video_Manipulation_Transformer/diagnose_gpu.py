#!/usr/bin/env python
"""
Diagnose GPU utilization issues
Identifies bottlenecks in the training pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import subprocess
import threading
from datetime import datetime

# CUDA settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

device = torch.device('cuda')


def get_gpu_stats():
    """Get current GPU statistics"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,memory.used,power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            stats = result.stdout.strip().split(', ')
            return {
                'util': float(stats[0]),
                'mem_gb': float(stats[1]) / 1024,
                'power_w': float(stats[2])
            }
    except:
        pass
    return {'util': 0, 'mem_gb': 0, 'power_w': 0}


class GPUMonitor:
    """Monitor GPU stats during operations"""
    def __init__(self):
        self.stats = []
        self.running = False
        self.thread = None
    
    def start(self):
        self.running = True
        self.stats = []
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        return self.stats
    
    def _monitor(self):
        while self.running:
            stats = get_gpu_stats()
            stats['time'] = time.time()
            self.stats.append(stats)
            time.sleep(0.1)  # 100ms intervals


def test_pure_compute():
    """Test pure GPU compute performance"""
    print("\n1. Testing Pure GPU Compute Performance")
    print("-" * 60)
    
    # Large model
    model = nn.Sequential(
        nn.Linear(1024, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 1024)
    ).to(device)
    
    # Compile if available
    if hasattr(torch, 'compile'):
        model = torch.compile(model, mode='max-autotune')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Large batch
    batch_size = 2048
    x = torch.randn(batch_size, 1024, device=device)
    y = torch.randn(batch_size, 1024, device=device)
    
    # Warmup
    for _ in range(10):
        _ = model(x)
    torch.cuda.synchronize()
    
    # Monitor GPU
    monitor = GPUMonitor()
    monitor.start()
    
    # Run compute
    start = time.time()
    for i in range(100):
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        optimizer.step()
        
        if i % 20 == 0:
            torch.cuda.synchronize()
            print(f"  Iter {i}: Loss={loss.item():.4f}")
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    # Stop monitoring
    stats = monitor.stop()
    
    # Analyze
    avg_util = np.mean([s['util'] for s in stats])
    max_util = np.max([s['util'] for s in stats])
    avg_power = np.mean([s['power_w'] for s in stats])
    
    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {100 * batch_size / elapsed:.0f} samples/s")
    print(f"  Avg GPU Util: {avg_util:.1f}%")
    print(f"  Max GPU Util: {max_util:.1f}%")
    print(f"  Avg Power: {avg_power:.0f}W")
    print(f"  Status: {'✓ GOOD' if avg_util > 80 else '✗ LOW'}")
    
    return avg_util


def test_memory_bandwidth():
    """Test GPU memory bandwidth"""
    print("\n2. Testing GPU Memory Bandwidth")
    print("-" * 60)
    
    # Large tensors to saturate memory bandwidth
    size = 1024 * 1024 * 512  # 2GB per tensor
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    c = torch.zeros_like(a)
    
    # Monitor
    monitor = GPUMonitor()
    monitor.start()
    
    # Memory intensive operations
    start = time.time()
    for i in range(50):
        # Copy operations
        c.copy_(a)
        a.copy_(b)
        b.copy_(c)
        
        # Element-wise operations
        c = a + b
        c = a * b
        
        if i % 10 == 0:
            torch.cuda.synchronize()
            print(f"  Iter {i}")
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    stats = monitor.stop()
    
    # Calculate bandwidth
    bytes_moved = 50 * 6 * size * 4  # 6 ops, 4 bytes per float
    bandwidth_gb = bytes_moved / elapsed / 1e9
    
    avg_util = np.mean([s['util'] for s in stats])
    
    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Bandwidth: {bandwidth_gb:.1f} GB/s")
    print(f"  GPU Util: {avg_util:.1f}%")
    print(f"  Status: {'✓ GOOD' if bandwidth_gb > 1000 else '✗ LOW'}")
    
    return bandwidth_gb


def test_kernel_launch_overhead():
    """Test kernel launch overhead"""
    print("\n3. Testing Kernel Launch Overhead")
    print("-" * 60)
    
    # Small operations to test launch overhead
    size = 1024
    tensors = [torch.randn(size, device=device) for _ in range(1000)]
    
    # Many small operations
    monitor = GPUMonitor()
    monitor.start()
    
    start = time.time()
    for i in range(100):
        for t in tensors:
            # Many small operations
            t.add_(1.0)
            t.mul_(2.0)
            t.div_(3.0)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    stats = monitor.stop()
    avg_util = np.mean([s['util'] for s in stats])
    
    # Compare with batched operation
    big_tensor = torch.randn(len(tensors) * size, device=device)
    
    start2 = time.time()
    for i in range(100):
        big_tensor.add_(1.0)
        big_tensor.mul_(2.0)
        big_tensor.div_(3.0)
    
    torch.cuda.synchronize()
    elapsed2 = time.time() - start2
    
    print(f"\nResults:")
    print(f"  Many small ops: {elapsed:.2f}s (GPU: {avg_util:.1f}%)")
    print(f"  Few large ops: {elapsed2:.2f}s")
    print(f"  Overhead ratio: {elapsed/elapsed2:.1f}x")
    print(f"  Status: {'✗ HIGH OVERHEAD' if elapsed/elapsed2 > 5 else '✓ OK'}")
    
    return elapsed / elapsed2


def test_data_transfer():
    """Test CPU-GPU data transfer"""
    print("\n4. Testing CPU-GPU Data Transfer")
    print("-" * 60)
    
    # Test different transfer sizes
    sizes = [1, 10, 100, 1000]  # MB
    
    for size_mb in sizes:
        size = size_mb * 1024 * 1024 // 4  # float32
        cpu_tensor = torch.randn(size)
        
        # Time transfer
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(100):
            gpu_tensor = cpu_tensor.to(device, non_blocking=True)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        bandwidth = 100 * size_mb / elapsed
        print(f"  {size_mb}MB: {bandwidth:.1f} MB/s")
    
    # Test pinned memory
    print("\nWith pinned memory:")
    for size_mb in sizes:
        size = size_mb * 1024 * 1024 // 4
        cpu_tensor = torch.randn(size).pin_memory()
        
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(100):
            gpu_tensor = cpu_tensor.to(device, non_blocking=True)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        bandwidth = 100 * size_mb / elapsed
        print(f"  {size_mb}MB: {bandwidth:.1f} MB/s")


def diagnose_model_performance():
    """Test actual model performance"""
    print("\n5. Testing Actual Model Performance")
    print("-" * 60)
    
    # Import model
    import sys
    sys.path.insert(0, '.')
    from models.encoders.hand_encoder import HandPoseEncoder
    
    # Create model
    model = HandPoseEncoder(
        input_dim=768,
        hidden_dim=2048,
        num_layers=12,
        num_heads=32,
        mlp_dim=8192,
        dropout=0.1
    ).to(device)
    
    if hasattr(torch, 'compile'):
        print("Compiling model...")
        model = torch.compile(model, mode='max-autotune')
    
    # Test data
    batch_sizes = [64, 256, 512, 1024]
    
    for bs in batch_sizes:
        x = torch.randn(bs, 196, 768, device=device)
        
        # Warmup
        for _ in range(5):
            _ = model(x)
        
        # Time
        torch.cuda.synchronize()
        monitor = GPUMonitor()
        monitor.start()
        
        start = time.time()
        for _ in range(20):
            out = model(x)
            loss = out['joints_3d'].mean()
            loss.backward()
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        stats = monitor.stop()
        avg_util = np.mean([s['util'] for s in stats])
        
        throughput = 20 * bs / elapsed
        print(f"  Batch {bs}: {throughput:.0f} samples/s (GPU: {avg_util:.1f}%)")


def main():
    """Run all diagnostics"""
    print("="*60)
    print("GPU UTILIZATION DIAGNOSTICS")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("="*60)
    
    # Initial stats
    stats = get_gpu_stats()
    print(f"\nInitial state:")
    print(f"  GPU Util: {stats['util']}%")
    print(f"  Memory: {stats['mem_gb']:.1f} GB")
    print(f"  Power: {stats['power_w']:.0f}W")
    
    # Run tests
    compute_util = test_pure_compute()
    bandwidth = test_memory_bandwidth()
    overhead = test_kernel_launch_overhead()
    test_data_transfer()
    diagnose_model_performance()
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)
    
    if compute_util < 80:
        print("✗ LOW COMPUTE UTILIZATION")
        print("  - Model may be too small")
        print("  - Try larger batch size or model")
    else:
        print("✓ Good compute utilization")
    
    if bandwidth < 1000:
        print("✗ LOW MEMORY BANDWIDTH")
        print("  - Memory access patterns may be inefficient")
    else:
        print("✓ Good memory bandwidth")
    
    if overhead > 5:
        print("✗ HIGH KERNEL LAUNCH OVERHEAD")
        print("  - Too many small operations")
        print("  - Batch operations together")
    else:
        print("✓ Low kernel overhead")
    
    print("\nRECOMMENDATIONS:")
    print("1. Use larger batch sizes (1024+)")
    print("2. Enable torch.compile with max-autotune")
    print("3. Use GPU-only data pipeline")
    print("4. Scale up model size")
    print("5. Minimize CPU-GPU transfers")


if __name__ == "__main__":
    main()