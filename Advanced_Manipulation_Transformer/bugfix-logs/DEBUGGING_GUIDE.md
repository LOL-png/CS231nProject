# Debugging Guide

This guide helps diagnose and fix common issues when training the Advanced Manipulation Transformer.

## Quick Diagnostics Checklist

Before diving into specific issues, run this diagnostic checklist:

```python
# In a notebook or script
def run_diagnostics(model, data_loader):
    print("=== Model Diagnostics ===")
    
    # 1. Check model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    
    # 2. Check data loading
    batch = next(iter(data_loader))
    print(f"\nBatch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
    
    # 3. Forward pass test
    with torch.no_grad():
        output = model(batch['image'])
    print(f"\nOutput shapes:")
    for k, v in output.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, torch.Tensor):
                    print(f"  {k}.{k2}: {v2.shape}")
    
    # 4. Check for NaN/Inf
    for name, param in model.named_parameters():
        if param.requires_grad:
            if torch.isnan(param).any():
                print(f"WARNING: NaN in {name}")
            if torch.isinf(param).any():
                print(f"WARNING: Inf in {name}")
```

## Common Issues and Solutions

### 1. Mode Collapse (Constant Predictions)

**Symptoms**:
- All predictions are identical
- Standard deviation < 0.001
- Diversity loss → 0

**Diagnostic Code**:
```python
def check_mode_collapse(predictions):
    joints = predictions['hand']['joints_3d']
    
    # Check variance across batch
    batch_std = joints.std(dim=0).mean()
    print(f"Batch std: {batch_std:.6f}")
    
    # Check pairwise distances
    joints_flat = joints.reshape(joints.shape[0], -1)
    dists = torch.cdist(joints_flat, joints_flat)
    avg_dist = dists[~torch.eye(len(dists), dtype=bool)].mean()
    print(f"Avg pairwise distance: {avg_dist:.6f}")
    
    if batch_std < 0.001:
        print("WARNING: Mode collapse detected!")
        return True
    return False
```

**Solutions**:

1. **Check Sigma Reparameterization**:
```python
# Ensure it's enabled
assert model.use_sigma_reparam == True

# Check sigma values
def hook_fn(module, input, output):
    if hasattr(output, 'log_sigma'):
        sigma = torch.exp(output.log_sigma)
        print(f"Sigma range: [{sigma.min():.4f}, {sigma.max():.4f}]")
        
# Add hook to monitor
model.hand_encoder.register_forward_hook(hook_fn)
```

2. **Increase Diversity Loss**:
```python
# In config
loss_weights:
  diversity: 0.05  # Increase from 0.01
  
# Or dynamically
if check_mode_collapse(predictions):
    loss_weights['diversity'] *= 2
```

3. **Add Input Noise**:
```python
# During training
if epoch < 10:  # Early training
    images = images + 0.01 * torch.randn_like(images)
```

4. **Initialize with More Variance**:
```python
# Custom initialization
def init_with_variance(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.02)  # Higher std
        if module.bias is not None:
            nn.init.normal_(module.bias, std=0.01)
            
model.apply(init_with_variance)
```

### 2. High MPJPE (Poor 3D Accuracy)

**Symptoms**:
- MPJPE > 100mm after 20 epochs
- Good 2D projections but bad 3D
- Refinement not helping

**Diagnostic Code**:
```python
def analyze_3d_accuracy(predictions, targets):
    # Basic MPJPE
    joints_pred = predictions['hand']['joints_3d']
    joints_gt = targets['hand_joints_3d']
    mpjpe = (joints_pred - joints_gt).norm(dim=-1).mean()
    
    # Per-joint analysis
    per_joint_error = (joints_pred - joints_gt).norm(dim=-1).mean(dim=0)
    worst_joints = per_joint_error.topk(5)
    print(f"Worst joints: {worst_joints.indices.tolist()}")
    print(f"Their errors: {worst_joints.values.tolist()}")
    
    # Check if refinement helps
    if 'joints_3d_refined' in predictions['hand']:
        refined = predictions['hand']['joints_3d_refined']
        refined_mpjpe = (refined - joints_gt).norm(dim=-1).mean()
        print(f"Initial MPJPE: {mpjpe:.2f}mm")
        print(f"Refined MPJPE: {refined_mpjpe:.2f}mm")
        print(f"Improvement: {mpjpe - refined_mpjpe:.2f}mm")
```

**Solutions**:

1. **Check Camera Parameters**:
```python
# Verify intrinsics are correct
K = batch['camera_intrinsics']
print(f"Focal length: {K[0, 0, 0]:.1f}")
print(f"Principal point: ({K[0, 0, 2]:.1f}, {K[0, 1, 2]:.1f})")

# Test projection-reprojection
joints_3d = predictions['hand']['joints_3d']
joints_2d_proj = project_3d_to_2d(joints_3d, K)
joints_3d_back = unproject_2d_to_3d(joints_2d_proj, K, joints_3d[..., 2:3])
error = (joints_3d - joints_3d_back).norm(dim=-1).mean()
print(f"Projection error: {error:.4f}")
```

2. **Increase Refinement Steps**:
```python
# In config
model:
  num_refinement_steps: 3  # From 2
  
# Or adaptive refinement
if mpjpe > 50:
    model.num_refinement_steps = 4
```

3. **Weight 2D Reprojection More**:
```python
loss_weights:
  hand_2d: 0.5  # Increase from 0.3
  hand_pose_refined: 1.5  # Increase from 1.2
```

4. **Check Feature Alignment**:
```python
# Visualize where features are sampled
def visualize_sampling(image, joints_2d):
    import matplotlib.pyplot as plt
    
    plt.imshow(image.permute(1, 2, 0).cpu())
    plt.scatter(joints_2d[:, 0].cpu(), joints_2d[:, 1].cpu(), c='red')
    plt.title("Feature sampling locations")
    plt.show()
```

### 3. Training Instability

**Symptoms**:
- Loss exploding or NaN
- Gradient norms very high
- Validation metrics oscillating

**Diagnostic Code**:
```python
class GradientMonitor:
    def __init__(self, model):
        self.model = model
        self.grad_history = {}
        
    def log_gradients(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                
                if name not in self.grad_history:
                    self.grad_history[name] = []
                    
                self.grad_history[name].append(grad_norm)
                
                # Check for issues
                if grad_norm > 100:
                    print(f"WARNING: Large gradient in {name}: {grad_norm:.2f}")
                if math.isnan(grad_norm):
                    print(f"ERROR: NaN gradient in {name}")
                    
    def plot_history(self):
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for i, (name, history) in enumerate(self.grad_history.items()):
            if i >= 4:
                break
            axes[i].plot(history)
            axes[i].set_title(name)
            axes[i].set_ylabel("Gradient norm")
            
        plt.tight_layout()
        plt.show()
```

**Solutions**:

1. **Gradient Clipping**:
```python
# Standard clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Adaptive clipping
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
if grad_norm > 50:
    print(f"WARNING: Gradient norm {grad_norm:.2f}, reducing LR")
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.5
```

2. **Learning Rate Warmup**:
```python
class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps=1000):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step_count = 0
        
    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            factor = self.step_count / self.warmup_steps
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self.base_lrs[i] * factor
```

3. **Loss Scaling**:
```python
# Scale losses to similar magnitudes
def balance_losses(losses):
    # Compute exponential moving average of loss magnitudes
    if not hasattr(balance_losses, 'ema'):
        balance_losses.ema = {}
        
    balanced = {}
    for name, loss in losses.items():
        if name == 'total':
            continue
            
        # Update EMA
        if name not in balance_losses.ema:
            balance_losses.ema[name] = loss.item()
        else:
            balance_losses.ema[name] = 0.99 * balance_losses.ema[name] + 0.01 * loss.item()
            
        # Scale to unit magnitude
        scale = 1.0 / (balance_losses.ema[name] + 1e-8)
        balanced[name] = loss * scale
        
    return balanced
```

### 4. Memory Issues

**Symptoms**:
- CUDA out of memory errors
- Training very slow
- GPU utilization low

**Diagnostic Code**:
```python
def profile_memory(model, batch):
    import torch.cuda
    
    torch.cuda.reset_peak_memory_stats()
    
    # Forward pass
    start_mem = torch.cuda.memory_allocated() / 1e9
    output = model(batch['image'])
    forward_mem = torch.cuda.memory_allocated() / 1e9
    
    # Backward pass
    loss = output['hand']['joints_3d'].sum()
    loss.backward()
    backward_mem = torch.cuda.memory_allocated() / 1e9
    
    print(f"Memory usage (GB):")
    print(f"  Start: {start_mem:.2f}")
    print(f"  After forward: {forward_mem:.2f}")
    print(f"  After backward: {backward_mem:.2f}")
    print(f"  Peak: {torch.cuda.max_memory_allocated() / 1e9:.2f}")
```

**Solutions**:

1. **Enable Gradient Checkpointing**:
```python
# In model
def enable_checkpointing(model):
    # For transformer layers
    for module in model.modules():
        if hasattr(module, 'gradient_checkpointing'):
            module.gradient_checkpointing = True
            
# Custom checkpointing
from torch.utils.checkpoint import checkpoint

class CheckpointedLayer(nn.Module):
    def forward(self, x):
        return checkpoint(self.layer, x)
```

2. **Reduce Batch Size with Accumulation**:
```python
# Effective batch size = batch_size * accumulation_steps
batch_size: 8
accumulation_steps: 4  # Effective: 32

# In training loop
for i, batch in enumerate(dataloader):
    loss = compute_loss(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

3. **Mixed Precision Training**:
```python
# Automatic mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast(dtype=torch.bfloat16):  # BF16 for H200
    output = model(batch)
    loss = criterion(output, targets)
    
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 5. Poor Generalization

**Symptoms**:
- Training loss low but validation high
- Model memorizing training data
- Performance drops on new data

**Diagnostic Code**:
```python
def check_overfitting(train_metrics, val_metrics, epoch):
    train_loss = train_metrics['loss']
    val_loss = val_metrics['loss']
    
    gap = val_loss - train_loss
    ratio = val_loss / train_loss
    
    print(f"Epoch {epoch}:")
    print(f"  Train loss: {train_loss:.4f}")
    print(f"  Val loss: {val_loss:.4f}")
    print(f"  Gap: {gap:.4f}")
    print(f"  Ratio: {ratio:.2f}")
    
    if ratio > 2.0:
        print("WARNING: Severe overfitting detected!")
        return True
    return False
```

**Solutions**:

1. **Increase Regularization**:
```python
# Dropout
model:
  dropout: 0.3  # Increase from 0.1
  
# Weight decay
optimizer = torch.optim.AdamW(
    model.parameters(),
    weight_decay=0.05  # Increase
)

# L2 regularization on activations
def activation_reg(model):
    reg_loss = 0
    for module in model.modules():
        if hasattr(module, 'last_activation'):
            reg_loss += 0.01 * module.last_activation.pow(2).mean()
    return reg_loss
```

2. **Data Augmentation**:
```python
# Strong augmentation
class StrongAugmentation:
    def __call__(self, sample):
        # Geometric
        if random.random() < 0.5:
            sample = random_rotation(sample, max_angle=30)
        if random.random() < 0.5:
            sample = random_scale(sample, scale_range=(0.8, 1.2))
            
        # Color
        if random.random() < 0.5:
            sample['image'] = color_jitter(
                sample['image'],
                brightness=0.4,
                contrast=0.4,
                saturation=0.4
            )
            
        # Noise
        if random.random() < 0.3:
            noise = torch.randn_like(sample['image']) * 0.05
            sample['image'] = sample['image'] + noise
            
        return sample
```

3. **Early Stopping**:
```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered!")
                return True
        return False
```

## Advanced Debugging Tools

### 1. Attention Visualization
```python
def visualize_attention(model, image):
    # Hook to capture attention weights
    attention_weights = []
    
    def hook_fn(module, input, output):
        if hasattr(output, 'attentions'):
            attention_weights.append(output.attentions)
            
    # Register hooks
    hooks = []
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            hooks.append(module.register_forward_hook(hook_fn))
    
    # Forward pass
    with torch.no_grad():
        _ = model(image)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Visualize
    import matplotlib.pyplot as plt
    
    for i, attn in enumerate(attention_weights):
        plt.figure(figsize=(10, 8))
        plt.imshow(attn[0, 0].cpu())  # First head, first batch
        plt.colorbar()
        plt.title(f"Attention Layer {i}")
        plt.show()
```

### 2. Feature Map Analysis
```python
def analyze_features(features, name="features"):
    """Analyze feature statistics"""
    print(f"\n{name} Analysis:")
    print(f"  Shape: {features.shape}")
    print(f"  Mean: {features.mean():.4f}")
    print(f"  Std: {features.std():.4f}")
    print(f"  Min: {features.min():.4f}")
    print(f"  Max: {features.max():.4f}")
    
    # Check for dead neurons
    neuron_activations = features.abs().mean(dim=0)
    dead_neurons = (neuron_activations < 1e-6).sum()
    print(f"  Dead neurons: {dead_neurons}/{neuron_activations.numel()}")
    
    # Plot histogram
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.hist(features.flatten().cpu(), bins=100)
    plt.title(f"{name} Distribution")
    plt.xlabel("Activation")
    plt.ylabel("Count")
    plt.show()
```

### 3. Loss Landscape Visualization
```python
def visualize_loss_landscape(model, loss_fn, data, num_points=50):
    """Visualize loss landscape around current parameters"""
    # Save current parameters
    original_params = {name: p.clone() for name, p in model.named_parameters()}
    
    # Choose two random directions
    direction1 = {name: torch.randn_like(p) for name, p in model.named_parameters()}
    direction2 = {name: torch.randn_like(p) for name, p in model.named_parameters()}
    
    # Normalize directions
    norm1 = sum(d.norm()**2 for d in direction1.values()).sqrt()
    norm2 = sum(d.norm()**2 for d in direction2.values()).sqrt()
    for d in direction1.values():
        d /= norm1
    for d in direction2.values():
        d /= norm2
    
    # Compute loss landscape
    alphas = torch.linspace(-1, 1, num_points)
    betas = torch.linspace(-1, 1, num_points)
    losses = torch.zeros(num_points, num_points)
    
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # Perturb parameters
            for name, p in model.named_parameters():
                p.data = original_params[name] + alpha * direction1[name] + beta * direction2[name]
            
            # Compute loss
            with torch.no_grad():
                output = model(data['image'])
                loss = loss_fn(output, data)
                losses[i, j] = loss.item()
    
    # Restore parameters
    for name, p in model.named_parameters():
        p.data = original_params[name]
    
    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.contourf(alphas, betas, losses, levels=50)
    plt.colorbar(label='Loss')
    plt.xlabel('Direction 1')
    plt.ylabel('Direction 2')
    plt.title('Loss Landscape')
    plt.scatter([0], [0], c='red', s=100, marker='x', label='Current')
    plt.legend()
    plt.show()
```

## Quick Fixes Cheatsheet

| Problem | Quick Fix | Config Change |
|---------|-----------|---------------|
| Mode collapse | Increase diversity loss | `diversity: 0.05` |
| High MPJPE | More refinement steps | `num_refinement_steps: 3` |
| NaN loss | Reduce learning rate | `learning_rate: 1e-4` |
| OOM error | Reduce batch size | `batch_size: 16` |
| Overfitting | Increase dropout | `dropout: 0.3` |
| Slow training | Enable mixed precision | `use_bf16: true` |
| Unstable training | Gradient clipping | `grad_clip: 0.5` |

## When to Restart Training

Consider restarting if:
1. Mode collapse persists after 10 epochs
2. NaN losses that can't be recovered
3. Validation metrics haven't improved in 20 epochs
4. Major hyperparameter changes needed

Save checkpoints frequently to avoid losing progress!