import os
os.environ['JAX_DISABLE_JIT'] = '1'  # Disable JIT to debug

import jax
import jax.numpy as jnp
import torch
import numpy as np
from model import RobotArmAllegro
from mujoco import mjx
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

def test_deterministic_pickup():
    """Test deterministic pickup error function with gradients"""
    
    print("=" * 60)
    print("DETERMINISTIC PICKUP TEST (DEBUG MODE - JIT DISABLED)")
    print("=" * 60)
    
    # 1. Create robot with cube
    print("\n1. Creating robot with cube...")
    robot = RobotArmAllegro(
        obj_filename="cube.obj",
        obj_pos=[0.3, 0.2, 0.5],
        obj_scale=[0.04, 0.04, 0.04]  # 4cm cube
    )
    
    initial_height = robot.initial_object_pos[2]
    model = robot.model
    initial_data = robot.data
    object_body_id = robot.object_body_id
    palm_id = robot.mj_model.body('palm').id
    
    print(f"   ✓ Robot created")
    print(f"   ✓ Cube at position {robot.initial_object_pos}")
    
    # 2. Define deterministic error function
    def compute_error(hand_pos, hand_joints):
        """Deterministic error with smooth gradients"""
        print(f"   Computing error for hand_pos shape: {hand_pos.shape}, hand_joints shape: {hand_joints.shape}")
        
        # Simple geometric IK for arm
        dx = hand_pos[0]
        dy = hand_pos[1]
        dz = hand_pos[2] - 0.1
        
        r = jnp.sqrt(dx**2 + dy**2)
        joint1 = jnp.arctan2(dy, dx)
        
        # 2-link arm IK
        l1, l2 = 0.45, 0.45
        target_dist = jnp.sqrt(r**2 + dz**2)
        target_dist = jnp.clip(target_dist, 0.1, 0.89)
        
        cos_elbow = (target_dist**2 - l1**2 - l2**2) / (2 * l1 * l2)
        cos_elbow = jnp.clip(cos_elbow, -0.99, 0.99)
        elbow_angle = jnp.pi - jnp.arccos(cos_elbow)
        
        alpha = jnp.arctan2(dz, r)
        cos_beta = (l1**2 + target_dist**2 - l2**2) / (2 * l1 * target_dist)
        cos_beta = jnp.clip(cos_beta, -0.99, 0.99)
        beta = jnp.arccos(cos_beta)
        shoulder_angle = alpha + beta
        
        arm_joints = jnp.array([joint1, -shoulder_angle, -elbow_angle, 0., 0., 0.])
        full_joints = jnp.concatenate([arm_joints, hand_joints])
        
        # Update state
        data = initial_data.replace(qpos=initial_data.qpos.at[:22].set(full_joints))
        data = data.replace(ctrl=full_joints)
        
        # Simulate - reduced steps for debugging
        for i in range(3):
            print(f"      Sim step {i+1}/3")
            data = mjx.step(model, data)
        
        # Compute error
        object_pos = data.xpos[object_body_id]
        palm_pos = data.xpos[palm_id]
        
        # Error components
        lift_reward = jnp.exp(-(object_pos[2] - initial_height - 0.1)**2)
        distance_error = jnp.linalg.norm(palm_pos - object_pos)
        finger_closure = jnp.mean(jnp.abs(hand_joints))
        
        total_error = (1.0 - lift_reward) + 2.0 * distance_error + 0.5 * (1.0 - jnp.tanh(finger_closure))
        
        return total_error
    
    # 3. Test JAX gradients
    print("\n2. Testing JAX gradients (NO JIT)...")
    
    hand_pos_jax = jnp.array([0.3, 0.2, 0.45])
    hand_joints_jax = jnp.zeros(16)
    
    # Compute value
    print("   Computing initial error...")
    error_val = compute_error(hand_pos_jax, hand_joints_jax)
    print(f"   Initial error: {error_val:.3f}")
    
    # Compute gradients
    print("   Computing gradients...")
    grad_fn = jax.grad(compute_error, argnums=(0, 1))
    pos_grad, joint_grad = grad_fn(hand_pos_jax, hand_joints_jax)
    
    print(f"   Position gradient: {pos_grad}")
    print(f"   Joint gradient (first 4): {joint_grad[:4]}")
    
    # 4. Limited optimization
    print("\n3. Limited optimization (3 steps only)...")
    
    hand_pos = jnp.array([0.35, 0.25, 0.5])
    hand_joints = jnp.zeros(16)
    learning_rate = 0.01
    
    errors = []
    for i in range(3):  # Only 3 steps for debugging
        print(f"\n   Step {i}:")
        error = compute_error(hand_pos, hand_joints)
        errors.append(float(error))
        
        pos_grad, joint_grad = grad_fn(hand_pos, hand_joints)
        
        # Update
        hand_pos = hand_pos - learning_rate * pos_grad
        hand_joints = hand_joints - learning_rate * joint_grad
        hand_joints = jnp.clip(hand_joints, -0.47, 1.6)
        
        print(f"      Error = {error:.3f}, hand_z = {hand_pos[2]:.3f}")
    
    print(f"\n   Final position: {hand_pos}")
    print(f"   Final joints (first 4): {hand_joints[:4]}")
    
    # 5. Skip PyTorch for now
    print("\n4. Skipping PyTorch integration for debugging...")
    
    # 6. Simple plot
    if len(errors) > 1:
        plt.figure(figsize=(8, 5))
        plt.plot(errors, 'b-o', linewidth=2)
        plt.xlabel('Optimization Step')
        plt.ylabel('Pickup Error')
        plt.title('Deterministic Optimization Progress (Debug Mode)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('deterministic_optimization_debug.png')
        print("\n   ✓ Saved plot to 'deterministic_optimization_debug.png'")
    
    print("\n" + "=" * 60)
    print("DEBUG TEST COMPLETE!")
    print("=" * 60)
    
    return errors

if __name__ == "__main__":
    errors = test_deterministic_pickup()
    if len(errors) > 1:
        print(f"\nError reduction: {errors[0]:.3f} -> {errors[-1]:.3f}")