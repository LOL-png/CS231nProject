# Advanced Manipulation Transformer - Training Quick Start

## Prerequisites Check
```bash
# Verify environment
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from flash_attn import flash_attn_func; print('FlashAttention: OK')" 2>/dev/null || echo "FlashAttention: Not available"
```

## Option 1: Barebones Training (Simplest)
```bash
cd /home/n231/231nProjectV2/Advanced_Manipulation_Transformer
python train_barebones_debug.py
```
- Uses basic configuration
- Good for debugging
- ~10 minute training

## Option 2: Notebook Training (Recommended)
```bash
jupyter notebook notebooks/train_full_featured.ipynb
```
- Interactive training with visualizations
- All advanced features enabled
- Best for experimentation

## Option 3: Command Line Training
```bash
# Standard training
python train.py

# Advanced training with all features
python train_advanced.py
```

## Configuration Tips

### For H200 GPU (140GB memory)
```yaml
training:
  batch_size: 256  # Can go up to 512
  gradient_accumulation: 1
  fp8_enabled: true  # Enable FP8 on H200
  
data:
  num_workers: 8
  pin_memory: true
  prefetch_factor: 2
```

### For Debugging
```yaml
training:
  batch_size: 32
  num_epochs: 1
  debug_mode: true
  
mode_collapse:
  enabled: false  # Disable for initial tests
```

### For Best Performance
```yaml
training:
  batch_size: 256
  gradient_accumulation: 2  # Effective batch = 512
  fp8_enabled: true
  
mode_collapse:
  enabled: true
  drop_path_rate: 0.3
  feature_noise: 0.05
  
flash_attention:
  enabled: true  # 1.5-2x speedup
```

## Monitoring Training

### Key Metrics to Watch
1. **Loss Convergence**: Should decrease steadily
2. **Hand MPJPE**: Target < 20mm
3. **Object Pose Error**: Target < 5cm, 5°
4. **Gradient Norm**: Should stay < 10
5. **Learning Rate**: Check scheduler is working

### TensorBoard
```bash
tensorboard --logdir runs/
```

### Common Issues

1. **OOM Error**: Reduce batch_size
2. **Slow Training**: Enable FlashAttention, increase batch_size
3. **Mode Collapse**: Check diversity loss > 0
4. **NaN Loss**: Reduce learning rate, enable gradient clipping

## Quick Validation
```python
# Test if model is training properly
from utils.quick_test import validate_training
validate_training('checkpoints/latest.pth')
```

## Full Pipeline Test
```bash
# Run comprehensive test
python test_training_pipeline.py
```

Expected output:
```
✅ Configuration loaded
✅ Model initialized
✅ Dataset loaded
✅ Optimizer created
✅ Training loop works
✅ Checkpoint saved
```

## Tips for Success

1. **Start Simple**: Use barebones training first
2. **Enable Features Gradually**: Add FlashAttention, FP8, etc. one by one
3. **Monitor GPU**: Use `nvidia-smi -l 1` in another terminal
4. **Check Logs**: Look in `runs/` for detailed logs
5. **Save Checkpoints**: Set `save_frequency: 100` for frequent saves

## Need Help?
- Check `bugfix-logs/` for solutions to common errors
- Run test scripts in project root
- Review configuration in `configs/`