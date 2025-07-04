# Video-to-Manipulation Transformer

A transformer-based system for converting monocular video into robot manipulation commands for the Allegro Hand (16 DOF) and Universal Robot (UR) arm (6 DOF).

## Project Overview

This repository contains two main implementations of our video-to-manipulation transformer model:

1. **Video_Manipulation_Transformer**: The original implementation with custom GPU-optimized training pipeline
2. **Advanced_Manipulation_Transformer**: An enhanced implementation using pretrained DINOv2 backbone with state-of-the-art optimizations

## External Dependencies

The following components are borrowed from external sources:

- **dex-ycb-toolkit**: Official toolkit for the DexYCB dataset (hand grasping object dataset)
- **dinov2**: Facebook's pretrained vision transformer model used in the Advanced implementation
1
## Our Contributions

### Video_Manipulation_Transformer

The original implementation featuring:
- Custom multi-encoder architecture (hand pose, object pose, contact detection)
- GPU-only data pipeline for H200 optimization
- Three-stage training strategy
- Differentiable physics integration with MuJoCo

Key files:
- `models/encoders/`: Hand, object, and contact encoders
- `data/gpu_only_dataset.py`: GPU-optimized data loading
- `train_gpu_optimized_final_improved.ipynb`: Latest training notebook with positional embeddings fix

### Advanced_Manipulation_Transformer  

An improved implementation addressing mode collapse issues:
- DINOv2 pretrained backbone (instead of training from scratch)
- �-reparameterization to prevent mode collapse
- FlashAttention-3 and FP8 support for H200 GPUs
- Comprehensive loss functions and regularization
- Hydra configuration system for easy experimentation

Key files:
- `models/unified_model.py`: Main model with �-reparameterization
- `models/encoders/dinov2_encoder.py`: Pretrained visual backbone
- `train_full_featured.ipynb`: Full-featured training notebook
- `train_advanced.py`: Command-line training script

## Dataset

Both implementations use the DexYCB dataset, which provides:
- RGB-D images of hands grasping YCB objects
- 3D hand pose annotations (MANO parameters)
- 6-DoF object poses
- Contact masks and grasp labels

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd 231nProjectV2

# Install dependencies for Advanced implementation
cd Advanced_Manipulation_Transformer
pip install -r requirements.txt

# For Video implementation
cd ../Video_Manipulation_Transformer
# Dependencies are installed as needed in notebooks
```

## Quick Start

### Advanced Manipulation Transformer (Recommended)
```bash
cd Advanced_Manipulation_Transformer

# Run with default settings
python train_advanced.py

# Or use the notebook for interactive development
jupyter notebook notebooks/train_full_featured.ipynb
```

### Video Manipulation Transformer
```bash
cd Video_Manipulation_Transformer

# Use the improved notebook with all fixes
jupyter notebook train_gpu_optimized_final_improved.ipynb
```

## Key Differences Between Implementations

| Feature | Video Transformer | Advanced Transformer |
|---------|------------------|---------------------|
| Visual Backbone | Train from scratch | Pretrained DINOv2 |
| Mode Collapse Prevention | Manual fixes | Built-in �-reparameterization |
| Configuration | Hardcoded | Hydra with CLI |
| GPU Optimization | Custom GPU-only pipeline | Standard + optimizations |
| Training Time | ~50 epochs needed | ~20-30 epochs |