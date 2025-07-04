# Modified Cell 17 for train_full_featured.ipynb to use simple joint loss
# Copy this code to replace the loss setup section in Cell 17

# Import the simple loss instead of ComprehensiveLoss
from training.simple_joint_loss import SimpleJointLoss

# Comment out the original loss
# from training.losses import ComprehensiveLoss
# manipulation_trainer.criterion = ComprehensiveLoss(config.loss)

# Use the simple joint loss instead
manipulation_trainer.criterion = SimpleJointLoss(config.loss)
print("✓ Using SimpleJointLoss - only per-joint L2 distance and object position")
print(f"  Hand joint weight: {config.loss.loss_weights.hand_coarse}")
print(f"  Object position weight: {config.loss.loss_weights.object_position}")
print(f"  Fingertip weighting: {'Enabled' if config.loss.per_joint_weighting else 'Disabled'}")

# The rest of Cell 17 remains the same...