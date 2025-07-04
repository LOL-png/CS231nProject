import jax
import jax.numpy as jnp
import torch
import numpy as np
from model import RobotArmAllegro
from mujoco import mjx

# Optional: for automatic JAX-PyTorch conversion
# pip install jax2torch
from jax2torch import jax2torch

def create_pickup_error_function(obj_filename=None, obj_initial_pos=[0.3, 0.2, 0.5]):
    """Create a pickup error function with fixed object"""
    
    # Initialize robot with object
    robot = RobotArmAllegro(
        obj_filename=obj_filename,
        obj_pos=obj_initial_pos,
        obj_scale=[1.0, 1.0, 1.0]
    )
    
    initial_height = obj_initial_pos[2]
    
    @jax.jit
    def compute_pickup_error(hand_target_pos, allegro_joint_angles):
        """
        Inputs:
            hand_target_pos: (3,) array - desired hand position [x, y, z]
            allegro_joint_angles: (16,) array - joint angles for Allegro hand
        
        Returns:
            error: scalar - differentiable pickup error
        """
        # Create full joint state
        current_joints = jnp.zeros(22)
        
        # Solve IK for arm to reach hand_target_pos
        arm_joints = robot.inverse_kinematics_numeric(
            target_pos=hand_target_pos,
            target_quat=jnp.array([0.7071, 0, 0.7071, 0]),  # Pointing down
            initial_joints=current_joints,
            max_iter=30
        )
        
        # Combine arm IK solution with provided hand joints
        full_joints = jnp.concatenate([arm_joints, allegro_joint_angles])
        
        # Create robot state
        data = robot.data.replace(qpos=data.qpos.at[:22].set(full_joints))
        
        # Simulate for a few steps to settle
        for _ in range(20):
            data = mjx.step(robot.model, data)
        
        # Compute pickup error
        metrics = robot.multi_construct_pickup_error(data, initial_height)
        
        # Return total error (single scalar for backprop)
        return metrics['total_error']
    
    return compute_pickup_error, robot


# === EXAMPLE USAGE ===

# 1. Create the error function
error_fn, robot = create_pickup_error_function(
    obj_filename=None,  # or "path/to/object.obj"
    obj_initial_pos=[0.3, 0.2, 0.5]
)

# 2. Convert to PyTorch-compatible function
pickup_error_torch = jax2torch(error_fn)

# 3. Use with PyTorch tensors
# Your inputs
hand_location = torch.tensor([0.3, 0.2, 0.45], requires_grad=True)
allegro_joints = torch.tensor([
    0.1, 0.8, 0.6, 0.4,  # Index
    0.0, 0.8, 0.6, 0.4,  # Middle  
    0.0, 0.8, 0.6, 0.4,  # Ring
    0.2, 0.7, 0.5, 0.3   # Thumb
], requires_grad=True)

# Compute error
error = pickup_error_torch(hand_location, allegro_joints)
print(f"Pickup error: {error.item():.3f}")

# Compute gradients
error.backward()
print(f"Hand position gradient: {hand_location.grad}")
print(f"Joint angles gradient: {allegro_joints.grad[:4]}...")  # First 4

# === OPTIMIZATION EXAMPLE ===
print("\n=== PyTorch Optimization ===")

# Reset gradients
hand_location = torch.tensor([0.35, 0.25, 0.5], requires_grad=True)
allegro_joints = torch.tensor(np.zeros(16), requires_grad=True)

optimizer = torch.optim.Adam([hand_location, allegro_joints], lr=0.01)

for i in range(20):
    optimizer.zero_grad()
    
    # Compute error
    error = pickup_error_torch(hand_location, allegro_joints)
    
    # Backward pass
    error.backward()
    
    # Update
    optimizer.step()
    
    if i % 5 == 0:
        print(f"Step {i}: error = {error.item():.3f}, hand_pos = {hand_location.data.numpy()}")

print(f"\nOptimized hand position: {hand_location.data.numpy()}")
print(f"Optimized joint config: {allegro_joints.data.numpy()[:4]}...")


# === ALTERNATIVE: Manual conversion (if jax2torch not available) ===
def manual_pytorch_wrapper(hand_pos_torch, joints_torch):
    """Manually convert between PyTorch and JAX"""
    # Convert to JAX
    hand_pos_jax = jnp.array(hand_pos_torch.detach().numpy())
    joints_jax = jnp.array(joints_torch.detach().numpy())
    
    # Compute in JAX
    error_jax = error_fn(hand_pos_jax, joints_jax)
    
    # Convert back to PyTorch
    error_torch = torch.tensor(float(error_jax), requires_grad=True)
    
    # For gradients, you'd need to use JAX's grad and convert
    grad_fn = jax.grad(error_fn, argnums=(0, 1))
    grads = grad_fn(hand_pos_jax, joints_jax)
    
    # Set up backward pass
    def backward_hook(grad_output):
        hand_grad = torch.tensor(np.array(grads[0])) * grad_output
        joints_grad = torch.tensor(np.array(grads[1])) * grad_output
        return hand_grad, joints_grad
    
    error_torch.register_hook(backward_hook)
    
    return error_torch


# === BATCHED VERSION ===
@jax.jit
def batched_pickup_error(hand_positions, joint_configs):
    """Process multiple configurations at once"""
    # vmap over batch dimension
    vmapped_error = jax.vmap(error_fn)
    return vmapped_error(hand_positions, joint_configs)

# Convert for batch processing
batched_error_torch = jax2torch(batched_pickup_error)

# Batch of configurations
batch_size = 10
hand_positions = torch.randn(batch_size, 3, requires_grad=True)
joint_angles = torch.randn(batch_size, 16, requires_grad=True)

errors = batched_error_torch(hand_positions, joint_angles)
print(f"\nBatch errors shape: {errors.shape}")
print(f"Mean error: {errors.mean().item():.3f}")