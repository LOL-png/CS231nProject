# Video-to-Manipulation Transformer

Implementation of a transformer-based system to convert monocular video into robot manipulation commands for an Allegro Hand (16 DOF) attached to a Universal Robot (UR) arm (6 DOF).

## Setup

### 1. Environment Setup

Make sure you have the DexYCB dataset path set:
```bash
export DEX_YCB_DIR=/home/n231/231nProjectV2/dex-ycb-toolkit/data
```

### 2. Install Dependencies

```bash
# Basic dependencies
pip install opencv-python pillow pyyaml tqdm

# If you haven't already installed the dex-ycb-toolkit dependencies:
cd /home/n231/231nProjectV2/dex-ycb-toolkit/bop_toolkit
pip install -r requirements_updated.txt
```

### 3. Dataset Structure

The DexYCB dataset should be organized as:
```
$DEX_YCB_DIR/
├── 20200709-subject-01/
├── 20200813-subject-02/
├── ...
├── calibration/
└── models/
```

## Running Stage 1 Training

### Option 1: Jupyter Notebook (Recommended for beginners)

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open `train_stage1_notebook.ipynb`

3. Run cells sequentially to:
   - Load DexYCB dataset
   - Visualize samples
   - Train encoders with supervision
   - Save checkpoints

### Option 2: Command Line

```bash
python main.py --stage 1 --batch_size 32 --num_epochs 20
```

## Architecture

The system consists of three specialized encoders:

1. **Hand Pose Encoder**: 6 layers, 512 dim, 8 attention heads
   - Input: RGB patches around detected hands
   - Output: 3D hand joint positions (21 keypoints)

2. **Object Pose Encoder**: 6 layers, 512 dim, 8 attention heads
   - Input: Object regions from detection
   - Output: 6-DoF object poses (position + rotation)

3. **Contact Detection Encoder**: 4 layers, 256 dim, 8 attention heads
   - Input: Hand-object interaction regions
   - Output: Contact points and interaction types

## Common Issues

### 1. "DEX_YCB_DIR not set"
Solution: Set the environment variable:
```bash
export DEX_YCB_DIR=/path/to/dex-ycb/data
```

### 2. "No module named cv2"
Solution: Install OpenCV:
```bash
pip install opencv-python
```

### 3. Memory issues on GPU
Solution: Reduce batch size in config:
```python
config['batch_size'] = 8  # or smaller
```

### 4. "Hand joints all -1"
This is normal - some samples don't have visible hands. The training code handles this automatically.

## Training Stages

1. **Stage 1** (Current): Encoder pre-training with direct supervision
2. **Stage 2**: Frozen encoders + temporal fusion training
3. **Stage 3**: End-to-end fine-tuning with physics simulation

## Monitoring Training

The notebook displays:
- Real-time loss curves
- MPJPE (Mean Per Joint Position Error) for hand poses
- Sample predictions vs ground truth
- Automatic checkpointing

## Next Steps

After Stage 1 training completes:
1. Checkpoints are saved in `checkpoints/stage1/`
2. Proceed to Stage 2 training (temporal fusion)
3. Finally, Stage 3 with physics simulation