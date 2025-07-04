# 231nProjectV2 - HOISDF Setup Guide

This repository contains the HOISDF (Hand-Object Interaction with Signed Distance Fields) implementation for 3D hand-object pose estimation from monocular camera signals.

## Quick Setup Guide

### 1. Clone the Repository
```bash
git clone git@github.com:bryandong24/231nProjectV2.git
cd 231nProjectV2
```

### 2. Create Conda Environment
```bash
conda create -n env2.0 python=3.12 -y
conda activate env2.0
```

### 3. Install Dependencies
```bash
cd HOISDF
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install "git+https://github.com/hassony2/libyana.git"
```

### 4. Install MANO Package
```bash
cd manopth
pip install -e .
cd ..
```

### 5. Download Required Files

#### MANO Model Files (Required)
1. Go to https://mano.is.tue.mpg.de/
2. Create an account and download the models
3. Place `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` in `HOISDF/tool/mano_models/`

#### Pre-trained Weights (For Inference)
1. Download from https://zenodo.org/records/11668766
2. Extract and place in `HOISDF/ckpts/`

#### YCB Object Models
1. Download from https://rse-lab.cs.washington.edu/projects/posecnn/
2. Also download simplified models from the Zenodo link above

### 6. Verify Installation
```bash
python test_setup.py
```

### 7. Start Jupyter Notebook
```bash
jupyter notebook
```
Open `HOISDF_Setup_and_Usage.ipynb` for detailed usage instructions.

## What's Included

- Complete HOISDF source code
- Comprehensive Jupyter notebook with examples
- Setup verification script
- Documentation and usage guides

## Notes

- This setup uses PyTorch with CUDA 12.1 support
- The code has been tested with Python 3.12
- Large binary files (models, datasets) are not included and must be downloaded separately

For detailed information, see `HOISDF/SETUP_SUMMARY.md`