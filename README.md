# 231n Project: Vision-Based Hand-Object Manipulation Transformers

This repository contains our CS231n final project implementation of transformer-based models for understanding and predicting hand-object manipulation from monocular RGB video. We developed two major model iterations that progressively improve 3D understanding and temporal consistency.

## Overview

Our project tackles the challenging problem of understanding hand-object interactions from single RGB images/videos, with applications in robotics, AR/VR, and human-computer interaction. We developed transformer architectures that can:

- Predict 3D hand poses (21 joints) from RGB images
- Estimate 6-DoF object poses for multiple objects
- Detect hand-object contact points and forces
- Generate robot manipulation commands from visual input

## Repository Structure

```
231nProjectV2/
├── Advanced_Manipulation_Transformer/    # Our final model implementation
│   ├── models/                          # Model architectures with σ-reparameterization
│   ├── training/                        # Advanced training pipeline
│   ├── data/                           # GPU-cached dataset loaders
│   ├── solutions/                      # Mode collapse prevention techniques
│   └── notebooks/                      # Training notebooks (used for experiments)
├── Video_Manipulation_Transformer/      # Initial model iteration
│   ├── encoders/                       # Basic transformer encoders
│   ├── train_stage1_notebook.ipynb     # Stage 1 training
│   └── main.py                         # Training script
├── Test_Model/                         # Experimental pairwise transition model
├── dex-ycb-toolkit/                    # [External] DEX-YCB dataset tools
├── dinov2/                             # [External] DINOv2 vision model
└── Mujoco/                             # Physics simulation for evaluation
```

## Model Iterations

### 1. Video Manipulation Transformer (Initial Version)
Our first approach used separate transformer encoders for different tasks:
- **Hand Pose Encoder**: 6 layers, 512 dim → 3D hand joints
- **Object Pose Encoder**: 6 layers, 512 dim → 6-DoF object poses  
- **Contact Encoder**: 4 layers, 256 dim → Contact points

**Issues Encountered:**
- Mode collapse: Model predicted same pose for all inputs
- Poor 3D accuracy: >300mm MPJPE (Mean Per Joint Position Error)
- Training instability

### 2. Advanced Manipulation Transformer (Final Version)
Our improved architecture addresses the fundamental issues:

**Key Innovations:**
- **σ-Reparameterization**: Prevents attention collapse through spectral normalization
- **Multi-Coordinate Representation**: 22 local coordinate frames for richer hand understanding
- **GPU-Cached Dataset**: 100GB+ dataset in GPU memory for 5-20× faster training
- **Comprehensive Loss Design**: 11 loss components with dynamic weighting

**Architecture Highlights:**
- Base: DINOv2-large (1.1B parameters, frozen early layers)
- Total: 516.1M parameters (362.9M trainable)
- Input: 224×224 RGB → Output: 3D poses + contacts
- Performance: 15.8mm MPJPE (20× improvement)

## External Dependencies

This project builds upon several external libraries and datasets:

1. **DEX-YCB Dataset & Toolkit** (`dex-ycb-toolkit/`)
   - Dataset of hand-object manipulation sequences
   - Evaluation metrics and visualization tools
   - Source: https://dex-ycb.github.io/

2. **DINOv2** (`dinov2/`)
   - Self-supervised vision transformer
   - Provides robust visual features
   - Source: https://github.com/facebookresearch/dinov2

3. **MuJoCo** (`Mujoco/`)
   - Physics simulation for evaluation
   - Used for testing physical plausibility
   - Planned, but never used

## Installation

### Prerequisites
- Python 3.12+
- CUDA 12.4+
- PyTorch 2.5.0+
- GPU: NVIDIA H200 (140GB) or A100 (40GB+) recommended

### Setup
```bash
# Clone repository
git clone [repository-url]
cd 231nProjectV2

# Install dependencies for Advanced Manipulation Transformer
cd Advanced_Manipulation_Transformer
pip install -r requirements.txt

# Verify GPU setup
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

## Quick Start

### Training the Advanced Manipulation Transformer

**Option 1: Jupyter Notebook (Recommended)**
```bash
cd Advanced_Manipulation_Transformer
jupyter notebook notebooks/train_full_featured.ipynb
```

**Option 2: Command Line**
```bash
cd Advanced_Manipulation_Transformer
python train_advanced.py
```

### Training the Video Manipulation Transformer
```bash
cd Video_Manipulation_Transformer
python main.py --stage 1 --batch_size 32 --num_epochs 20
```

## Technical Contributions

1. **Mode Collapse Solution**: Novel σ-reparameterization technique that maintains prediction diversity
2. **Multi-Coordinate Hand Representation**: 22 coordinate frames for robust 3D understanding
3. **GPU-Optimized Training**: Complete dataset cached in GPU memory
4. **Comprehensive Loss Design**: Physics-aware losses with dynamic weighting

## Documentation

Each component has detailed documentation:
- `Advanced_Manipulation_Transformer/README.md`: Full model details
- `Advanced_Manipulation_Transformer/ARCHITECTURE.md`: Architecture deep dive
- `Video_Manipulation_Transformer/README.md`: Initial model documentation
- `bugfix-logs/`: Detailed fix documentation

## Authors

Howard Ji and Bryan Dong

## Acknowledgments

- Stanford CS231n course staff for guidance
- DexYCB dataset creators
- DINOv2 team at Meta AI
- PyTorch team for optimization primitives

## License

This project is for educational purposes as part of Stanford CS231n.
External components (dex-ycb-toolkit, dinov2) retain their original licenses.