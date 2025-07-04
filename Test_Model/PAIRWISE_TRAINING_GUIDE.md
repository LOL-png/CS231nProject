# Pairwise Training Guide for Transition Merger

## Overview

This guide explains how to train the transition merger model using pairwise comparisons between HOISDF outputs. The model learns to create smooth, natural transitions between any two hand-object interaction sequences.

## Key Concepts

### 1. Pairwise Learning
Instead of requiring explicit transition examples, the model learns from pairs of sequences:
- Automatically finds compatible sequence pairs based on hand pose similarity
- Creates synthetic ground truth transitions for training
- Learns generalizable transition patterns

### 2. Similarity-Based Pairing
The dataset automatically pairs sequences based on:
- **Hand Pose Similarity**: End pose of sequence 1 vs start pose of sequence 2
- **Object Compatibility**: Spatial proximity of objects
- **Task Compatibility**: Learned through contrastive loss

### 3. Self-Supervised Training
The model learns transitions without explicit supervision:
- Synthetic transitions are generated using smooth interpolation
- Model learns to improve upon these synthetic examples
- Diffusion refinement adds natural motion characteristics

## Dataset Preparation

### Step 1: Extract HOISDF Outputs
```python
# For each video in your dataset
for video_path in video_paths:
    video_frames = load_video(video_path)
    
    # Process through HOISDF
    hoisdf_outputs = hoisdf_model(video_frames)
    
    # Save outputs
    save_data = {
        'mano_params': hoisdf_outputs['mano_params'],
        'hand_sdf': hoisdf_outputs['hand_sdf'],
        'object_sdf': hoisdf_outputs['object_sdf'],
        'contact_points': hoisdf_outputs['contact_points'],
        'contact_frames': hoisdf_outputs['contact_frames'],
        'hand_vertices': hoisdf_outputs['hand_vertices'],
        'object_center': hoisdf_outputs['object_center']
    }
    
    with open(f'data/{video_name}.pkl', 'wb') as f:
        pickle.dump(save_data, f)
```

### Step 2: Dataset Structure
```
data/
├── hoisdf_outputs/
│   ├── pick_phone_001.pkl
│   ├── pick_phone_002.pkl
│   ├── pick_bottle_001.pkl
│   ├── pick_bottle_002.pkl
│   ├── typing_001.pkl
│   └── ...
```

## Training Process

### 1. Dataset Creation
The `HOISDFTransitionDataset` class:
- Loads all HOISDF output files
- Computes pairwise similarities
- Creates training pairs above similarity threshold
- Generates synthetic ground truth transitions

### 2. Similarity Computation
```python
def compute_similarity(seq1, seq2):
    # Compare end of seq1 with start of seq2
    pose1 = seq1['mano_params'][-20:, 3:48].mean(0)
    pose2 = seq2['mano_params'][:20, 3:48].mean(0)
    
    # Cosine similarity
    similarity = F.cosine_similarity(pose1, pose2)
    return similarity
```

### 3. Training Loop
```python
trainer = TransitionTrainer(model, train_dataset, val_dataset, config)
trainer.train(num_epochs=100)
```

## Model Architecture for Pairwise Learning

### Tokenizer Modifications
- Encodes sequences independently
- No assumption about sequence continuity
- Robust to different sequence lengths

### Transformer Enhancements
- Video embeddings distinguish source sequences
- Attention learns transition patterns
- Boundary detection identifies optimal transition points

### Diffusion Refinement
- Conditions on both sequences
- Learns natural motion priors
- Maintains contact constraints

## Loss Functions for Pairwise Training

### 1. Reconstruction Loss
- Compares predicted transitions to synthetic ground truth
- Weighted by pair compatibility score

### 2. Contrastive Loss
- Ensures similar poses have similar embeddings
- Distinguishes between different tasks

### 3. Smoothness Loss
- Penalizes jerky transitions
- Enforces physical plausibility

### 4. Contact Consistency
- Maintains contact during object interaction
- Prevents hand-object penetration

## Training Tips

### 1. Data Diversity
- Include various hand poses and tasks
- Ensure good coverage of pose space
- Balance sequence types

### 2. Similarity Threshold
- Start with lower threshold (0.2-0.3)
- Increase as model improves
- Monitor pair quality

### 3. Batch Composition
- Mix easy and hard pairs
- Include diverse transition types
- Balance sequence lengths

### 4. Curriculum Learning
```python
# Start with similar sequences
dataset.similarity_threshold = 0.5

# Gradually include harder pairs
for epoch in range(num_epochs):
    if epoch % 20 == 0:
        dataset.similarity_threshold -= 0.1
```

## Evaluation

### 1. Quantitative Metrics
- MANO parameter error
- Smoothness (velocity/acceleration)
- Contact consistency
- Transition quality score

### 2. Qualitative Evaluation
- Visual inspection of transitions
- Physical plausibility
- Natural motion characteristics

## Advanced Features

### 1. Multi-Modal Transitions
Train separate models for different transition types:
- Object-to-object transitions
- Same-object different tasks
- Free-space transitions

### 2. Conditional Generation
Condition transitions on:
- Task labels
- Object properties
- Desired transition style

### 3. Few-Shot Adaptation
Fine-tune on specific transition types with limited data

## Troubleshooting

### Problem: Too few pairs created
**Solution**: Lower similarity threshold or add more diverse sequences

### Problem: Unnatural transitions
**Solution**: Increase smoothness loss weight, add more diffusion steps

### Problem: Contact violations
**Solution**: Increase contact loss weight, add contact-aware sampling

## Example Results

After training, the model can:
1. Generate transitions between any two sequences
2. Maintain physical plausibility
3. Respect contact constraints
4. Create natural, human-like motions

The pairwise training approach enables learning from any collection of HOISDF outputs without requiring explicit transition examples!