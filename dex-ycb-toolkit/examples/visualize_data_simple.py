# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""Simple data visualization without 3D rendering - just shows what's in the dataset."""

import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from dex_ycb_toolkit.factory import get_dataset

def main():
    name = 's0_train'
    dataset = get_dataset(name)
    
    idx = 70
    sample = dataset[idx]
    
    print("=== Dataset Sample Information ===")
    print(f"Sample index: {idx}")
    print(f"Available keys: {list(sample.keys())}")
    print(f"Color file: {sample['color_file']}")
    print(f"Label file: {sample['label_file']}")
    print(f"YCB object IDs: {sample['ycb_ids']}")
    # Define YCB class names
    ycb_classes = {
        1: '002_master_chef_can',
        2: '003_cracker_box',
        3: '004_sugar_box',
        4: '005_tomato_soup_can',
        5: '006_mustard_bottle',
        6: '007_tuna_fish_can',
        7: '008_pudding_box',
        8: '009_gelatin_box',
        9: '010_potted_meat_can',
        10: '011_banana',
        11: '019_pitcher_base',
        12: '021_bleach_cleanser',
        13: '024_bowl',
        14: '025_mug',
        15: '035_power_drill',
        16: '036_wood_block',
        17: '037_scissors',
        18: '040_large_marker',
        19: '051_large_clamp',
        20: '052_extra_large_clamp',
        21: '061_foam_brick',
    }
    print(f"YCB object names: {[ycb_classes.get(i, f'Unknown_{i}') for i in sample['ycb_ids']]}")
    print(f"MANO side: {sample['mano_side']}")
    print(f"Camera intrinsics: fx={sample['intrinsics']['fx']:.2f}, fy={sample['intrinsics']['fy']:.2f}, cx={sample['intrinsics']['ppx']:.2f}, cy={sample['intrinsics']['ppy']:.2f}")
    
    # Load and display the color image
    im_color = cv2.imread(sample['color_file'])
    im_color_rgb = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)
    
    # Load pose data
    label = np.load(sample['label_file'])
    print(f"\nLabel file keys: {list(label.keys())}")
    
    pose_y = label['pose_y']  # YCB object poses
    pose_m = label['pose_m']  # MANO hand pose
    
    print(f"\nYCB poses shape: {pose_y.shape}")
    print(f"MANO pose shape: {pose_m.shape}")
    
    # Check which objects are present
    present_objects = []
    for i, pose in enumerate(pose_y):
        if not np.all(pose == 0.0):
            present_objects.append((i, ycb_classes.get(sample['ycb_ids'][i], f'Unknown_{sample["ycb_ids"][i]}')))
    
    print(f"\nObjects present in scene: {present_objects}")
    print(f"Hand present: {not np.all(pose_m == 0.0)}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Show original image
    axes[0].imshow(im_color_rgb)
    axes[0].set_title(f'Original Image - Sample {idx}')
    axes[0].axis('off')
    
    # Show image with annotations
    axes[1].imshow(im_color_rgb)
    axes[1].set_title('Data Information')
    axes[1].axis('off')
    
    # Add text annotations
    info_text = f"Objects: {', '.join([obj[1] for obj in present_objects])}\n"
    info_text += f"Hand: {'Present' if not np.all(pose_m == 0.0) else 'Not present'}\n"
    info_text += f"Image size: {im_color.shape[1]}x{im_color.shape[0]}"
    
    axes[1].text(10, 30, info_text, fontsize=10, color='yellow', 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
    
    plt.tight_layout()
    output_path = 'data_visualization_simple.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Also save just the original image
    cv2.imwrite('original_image.png', im_color)
    print(f"Original image saved to: original_image.png")
    
    # Print some statistics about the dataset
    print(f"\n=== Dataset Statistics ===")
    print(f"Total samples in {name}: {len(dataset)}")
    print(f"Image dimensions: {dataset.w}x{dataset.h}")


if __name__ == '__main__':
    main()