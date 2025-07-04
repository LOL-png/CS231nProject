# Configuration guide for the enhanced SimpleJointLoss
# Add this to Cell 7 in train_full_featured.ipynb

config = OmegaConf.create({
    # ... other config ...
    
    'loss': {
        'loss_weights': {
            # Original weights
            'hand_coarse': 1.0,           # Weight for per-joint L2 distance
            'object_position': 1.0,       # Weight for object position L2 distance
            
            # New weights for enhanced simple loss
            'object_size': 0.5,           # Weight for object size/dimensions error
            'pairwise_distances': 0.3,    # Weight for hand joint pairwise distances
            
            # These are ignored by SimpleJointLoss but kept for compatibility
            'hand_refined': 1.0,
            'object_rotation': 0.5,
            'contact': 0.3,
            'physics': 0.1,
            'diversity': 0.01,
            'reprojection': 0.5,
            'kl': 0.001
        },
        'per_joint_weighting': True,      # Enable fingertip weighting
        'fingertip_weight': 1.5,          # Fingertips get 1.5x weight
    },
    
    # ... rest of config ...
})