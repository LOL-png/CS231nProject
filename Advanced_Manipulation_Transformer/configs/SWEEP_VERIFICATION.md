# Sweep Configuration Verification

## ✅ Sweep-to-Model Mapping Verified

The sweep configuration has been verified to correctly match the model structure. Here's the complete mapping:

### Model Architecture Parameters ✅
- `hidden_dim` → `config.model.hidden_dim`
- `contact_hidden_dim` → `config.model.contact_hidden_dim` 
- `dropout` → `config.model.dropout`
- `num_refinement_steps` → `config.model.num_refinement_steps`
- `freeze_layers` → `config.model.freeze_layers`
- `use_attention_fusion` → `config.model.use_attention_fusion`
- `use_sigma_reparam` → `config.model.use_sigma_reparam`

### Training Parameters ✅
- `learning_rate` → `config.training.learning_rate`
- `batch_size` → `config.training.batch_size`
- `weight_decay` → `config.training.weight_decay`
- `grad_clip` → `config.training.grad_clip`
- `ema_decay` → `config.training.ema_decay`

### Loss Weights ✅ (Correctly Mapped)
The sweep uses `loss_weight_X` format which is correctly mapped to `config.loss.loss_weights.X`:
- `loss_weight_hand_coarse` → `config.loss.loss_weights.hand_coarse`
- `loss_weight_hand_refined` → `config.loss.loss_weights.hand_refined`
- `loss_weight_object_position` → `config.loss.loss_weights.object_position`
- `loss_weight_object_rotation` → `config.loss.loss_weights.object_rotation`
- `loss_weight_contact` → `config.loss.loss_weights.contact`
- `loss_weight_physics` → `config.loss.loss_weights.physics`
- `loss_weight_diversity` → `config.loss.loss_weights.diversity`
- `loss_weight_reprojection` → `config.loss.loss_weights.reprojection`
- `loss_weight_kl` → `config.loss.loss_weights.kl`

### Multi-Rate Learning ✅
- `lr_mult_pretrained` → `config.training.multi_rate.pretrained`
- `lr_mult_new_encoders` → `config.training.multi_rate.new_encoders`

### Scheduler Settings ✅
- `scheduler_T_0` → `config.training.T_0`
- `scheduler_min_lr` → `config.training.min_lr`

### Augmentation Settings ✅
- `aug_rotation_range` → `config.data.augmentation.rotation_range`
- `aug_scale_min/max` → `config.data.augmentation.scale_range`
- `aug_translation_std` → `config.data.augmentation.translation_std`
- `aug_color_jitter` → `config.data.augmentation.color_jitter`
- `aug_joint_noise_std` → `config.data.augmentation.joint_noise_std`

### Memory Optimization ✅
- `gradient_checkpointing` → `config.optimizations.memory.gradient_checkpointing`
- `checkpoint_ratio` → `config.optimizations.memory.checkpoint_ratio`
- `use_amp` → `config.training.use_amp`
- `use_bf16` → `config.training.use_bf16`

## Model Components Used

### UnifiedManipulationTransformer
- Uses DINOv2ImageEncoder with configurable freeze_layers
- MultiCoordinateHandEncoder with hidden_dim
- ObjectPoseEncoder with hidden_dim and max_objects
- ContactEncoder with contact_hidden_dim
- PixelAlignedRefinement with num_refinement_steps

### ComprehensiveLoss
Expects loss weights in the format:
```python
{
    'hand_coarse': 1.0,
    'hand_refined': 1.2,
    'object_position': 1.0,
    'object_rotation': 0.5,
    'contact': 0.3,
    'physics': 0.1,
    'diversity': 0.01,
    'reprojection': 0.5,
    'kl': 0.001
}
```

## Fixed Issues

1. **Contact Hidden Dim Values**: Updated from [256, 384, 512] to [256, 512, 768] to better align with hidden_dim values

2. **Loss Weight Mapping**: The sweep correctly uses `loss_weight_X` format which is properly mapped in `update_config_from_sweep()` function

## Ready to Run

The sweep configuration is now fully aligned with the model structure and ready to use:

```bash
# Start sweep
python run_sweep.py --project advanced-manipulation-transformer --count 50

# Or use the convenience script
./start_sweep.sh --project advanced-manipulation-transformer --num-runs 50
```