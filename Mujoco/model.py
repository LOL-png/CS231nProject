import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
from typing import Tuple, Dict
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

class RobotArmAllegro:
    """Generic 6-DOF robot arm with Allegro hand in MuJoCo JAX"""
    
    def __init__(self, obj_filename=None, obj_pos=None, obj_scale=None):
        self.model_xml = self._create_model_xml(obj_filename, obj_pos, obj_scale)
        self.mj_model = mujoco.MjModel.from_xml_string(self.model_xml)
        
        # Try to create MJX model, handle collision compatibility issues
        try:
            self.model = mjx.put_model(self.mj_model)
        except NotImplementedError as e:
            if "collisions not implemented" in str(e):
                print("Warning: MJX collision compatibility issue detected. Disabling contacts.")
                # Recreate model with contacts disabled
                self.model_xml = self.model_xml.replace('contact="enable"', 'contact="disable"')
                self.mj_model = mujoco.MjModel.from_xml_string(self.model_xml)
                self.model = mjx.put_model(self.mj_model)
            else:
                raise e
        
        self.data = mjx.make_data(self.model)
        
        # Control dimensions
        self.n_arm_joints = 6
        self.n_hand_joints = 16  # Allegro hand has 16 actuated joints
        self.n_total_joints = self.n_arm_joints + self.n_hand_joints
        
        # Object tracking
        self.object_body_id = self.mj_model.body('target_object').id
        self.initial_object_pos = obj_pos if obj_pos else [0.3, 0.2, 0.5]
        
        # Setup kinematics
        self.setup_kinematics()
        self.setup_hand_kinematics()
        
    def _create_model_xml(self, obj_filename=None, obj_pos=None, obj_scale=None) -> str:
        """Create XML string defining robot arm + Allegro hand + optional object"""
        
        # Object configuration
        if obj_pos is None:
            obj_pos = [0.3, 0.2, 0.5]
        if obj_scale is None:
            obj_scale = [1.0, 1.0, 1.0]
            
        # Object XML section
        object_xml = ""
        if obj_filename:
            object_xml = f'''
        <!-- Target object from OBJ file -->
        <body name="target_object" pos="{obj_pos[0]} {obj_pos[1]} {obj_pos[2]}">
            <joint type="free"/>
            <geom name="target_geom" type="mesh" mesh="object_mesh" rgba="1 0.5 0 1" mass="0.1" contype="1" conaffinity="1"/>
            <site name="object_center" pos="0 0 0" size="0.01"/>
        </body>'''
        else:
            # Default sphere object
            object_xml = f'''
        <!-- Target object (default sphere) -->
        <body name="target_object" pos="{obj_pos[0]} {obj_pos[1]} {obj_pos[2]}">
            <joint type="free"/>
            <geom name="target_geom" type="sphere" size="0.04" rgba="1 0.5 0 1" mass="0.1" contype="1" conaffinity="1"/>
            <site name="object_center" pos="0 0 0" size="0.01"/>
        </body>'''
        
        # Asset section with mesh
        asset_section = '''
    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0 0 0" width="100" height="100"/>
        <texture name="grid" type="2d" builtin="checker" rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4" width="300" height="300"/>
        <material name="grid" texture="grid" texrepeat="5 5" reflectance="0.2"/>'''
        
        if obj_filename:
            asset_section += f'''
        <mesh name="object_mesh" file="{obj_filename}" scale="{obj_scale[0]} {obj_scale[1]} {obj_scale[2]}"/>'''
        
        asset_section += '''
    </asset>'''
        
        # Return full XML with object
        return f'''
<mujoco model="arm_allegro">
    <compiler angle="radian" meshdir="." autolimits="true"/>
    
    <option gravity="0 0 -9.81" timestep="0.002">
        <flag contact="enable"/>
        <!-- Note: Set contact="disable" if MJX has collision compatibility issues -->
    </option>
    
    <default>
        <joint limited="true" damping="0.1" armature="0.01"/>
        <geom contype="0" conaffinity="0" friction="1 0.5 0.5"/>
        <motor ctrllimited="true" ctrlrange="-1 1"/>
    </default>
    
{asset_section}
    
    <worldbody>
        <!-- Ground -->
        <geom name="floor" pos="0 0 0" size="2 2 0.01" type="plane" material="grid" contype="1" conaffinity="1"/>
        
        <!-- Robot base -->
        <body name="base">
            <geom name="base_geom" type="cylinder" size="0.1 0.05" rgba="0.2 0.2 0.2 1" pos="0 0 0.05"/>
            
            <!-- Link 1 -->
            <body name="link1" pos="0 0 0.1">
                <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                <geom name="link1_geom" type="cylinder" size="0.05 0.1" rgba="0.8 0.2 0.2 1"/>
                
                <!-- Link 2 -->
                <body name="link2" pos="0 0 0.2">
                    <joint name="joint2" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                    <geom name="link2_geom" type="cylinder" size="0.04 0.15" rgba="0.2 0.8 0.2 1"/>
                    
                    <!-- Link 3 -->
                    <body name="link3" pos="0 0 0.3">
                        <joint name="joint3" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                        <geom name="link3_geom" type="cylinder" size="0.04 0.15" rgba="0.2 0.2 0.8 1"/>
                        
                        <!-- Link 4 -->
                        <body name="link4" pos="0 0 0.3">
                            <joint name="joint4" type="hinge" axis="1 0 0" range="-3.14 3.14"/>
                            <geom name="link4_geom" type="cylinder" size="0.03 0.1" rgba="0.8 0.8 0.2 1"/>
                            
                            <!-- Link 5 -->
                            <body name="link5" pos="0 0 0.2">
                                <joint name="joint5" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                                <geom name="link5_geom" type="cylinder" size="0.03 0.08" rgba="0.8 0.2 0.8 1"/>
                                
                                <!-- Link 6 (wrist) -->
                                <body name="link6" pos="0 0 0.16">
                                    <joint name="joint6" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                                    <geom name="link6_geom" type="cylinder" size="0.025 0.05" rgba="0.2 0.8 0.8 1"/>
                                    
                                    <!-- Allegro Hand Mount -->
                                    <body name="hand_mount" pos="0 0 0.1">
                                        <geom name="mount_plate" type="box" size="0.04 0.04 0.01" rgba="0.5 0.5 0.5 1"/>
                                        
                                        <!-- Allegro Hand (simplified version) -->
                                        <!-- Palm -->
                                        <body name="palm" pos="0 0 0.02">
                                            <geom name="palm_geom" type="box" size="0.045 0.07 0.03" rgba="0.3 0.3 0.3 1" contype="1" conaffinity="1"/>
                                            
                                            <!-- Index Finger -->
                                            <body name="index_base" pos="0.02 0.05 0">
                                                <joint name="index_joint1" type="hinge" axis="1 0 0" range="-0.47 0.47"/>
                                                <geom name="index_link1" type="capsule" fromto="0 0 0 0 0.03 0" size="0.008" rgba="0.7 0.7 0.7 1"/>
                                                
                                                <body name="index_middle" pos="0 0.03 0">
                                                    <joint name="index_joint2" type="hinge" axis="1 0 0" range="-0.3 1.6"/>
                                                    <geom name="index_link2" type="capsule" fromto="0 0 0 0 0.025 0" size="0.007" rgba="0.7 0.7 0.7 1"/>
                                                    
                                                    <body name="index_distal" pos="0 0.025 0">
                                                        <joint name="index_joint3" type="hinge" axis="1 0 0" range="-0.3 1.6"/>
                                                        <geom name="index_link3" type="capsule" fromto="0 0 0 0 0.02 0" size="0.006" rgba="0.7 0.7 0.7 1"/>
                                                        
                                                        <body name="index_tip" pos="0 0.02 0">
                                                            <joint name="index_joint4" type="hinge" axis="1 0 0" range="-0.3 1.6"/>
                                                            <geom name="index_tip_geom" type="sphere" size="0.008" rgba="0.8 0.6 0.6 1" contype="1" conaffinity="1"/>
                                                            <site name="index_tip_site" pos="0 0 0" size="0.005"/>
                                                        </body>
                                                    </body>
                                                </body>
                                            </body>
                                            
                                            <!-- Middle Finger -->
                                            <body name="middle_base" pos="0 0.05 0">
                                                <joint name="middle_joint1" type="hinge" axis="1 0 0" range="-0.47 0.47"/>
                                                <geom name="middle_link1" type="capsule" fromto="0 0 0 0 0.03 0" size="0.008" rgba="0.7 0.7 0.7 1"/>
                                                
                                                <body name="middle_middle" pos="0 0.03 0">
                                                    <joint name="middle_joint2" type="hinge" axis="1 0 0" range="-0.3 1.6"/>
                                                    <geom name="middle_link2" type="capsule" fromto="0 0 0 0 0.025 0" size="0.007" rgba="0.7 0.7 0.7 1"/>
                                                    
                                                    <body name="middle_distal" pos="0 0.025 0">
                                                        <joint name="middle_joint3" type="hinge" axis="1 0 0" range="-0.3 1.6"/>
                                                        <geom name="middle_link3" type="capsule" fromto="0 0 0 0 0.02 0" size="0.006" rgba="0.7 0.7 0.7 1"/>
                                                        
                                                        <body name="middle_tip" pos="0 0.02 0">
                                                            <joint name="middle_joint4" type="hinge" axis="1 0 0" range="-0.3 1.6"/>
                                                            <geom name="middle_tip_geom" type="sphere" size="0.008" rgba="0.8 0.6 0.6 1" contype="1" conaffinity="1"/>
                                                            <site name="middle_tip_site" pos="0 0 0" size="0.005"/>
                                                        </body>
                                                    </body>
                                                </body>
                                            </body>
                                            
                                            <!-- Ring Finger -->
                                            <body name="ring_base" pos="-0.02 0.05 0">
                                                <joint name="ring_joint1" type="hinge" axis="1 0 0" range="-0.47 0.47"/>
                                                <geom name="ring_link1" type="capsule" fromto="0 0 0 0 0.03 0" size="0.008" rgba="0.7 0.7 0.7 1"/>
                                                
                                                <body name="ring_middle" pos="0 0.03 0">
                                                    <joint name="ring_joint2" type="hinge" axis="1 0 0" range="-0.3 1.6"/>
                                                    <geom name="ring_link2" type="capsule" fromto="0 0 0 0 0.025 0" size="0.007" rgba="0.7 0.7 0.7 1"/>
                                                    
                                                    <body name="ring_distal" pos="0 0.025 0">
                                                        <joint name="ring_joint3" type="hinge" axis="1 0 0" range="-0.3 1.6"/>
                                                        <geom name="ring_link3" type="capsule" fromto="0 0 0 0 0.02 0" size="0.006" rgba="0.7 0.7 0.7 1"/>
                                                        
                                                        <body name="ring_tip" pos="0 0.02 0">
                                                            <joint name="ring_joint4" type="hinge" axis="1 0 0" range="-0.3 1.6"/>
                                                            <geom name="ring_tip_geom" type="sphere" size="0.008" rgba="0.8 0.6 0.6 1" contype="1" conaffinity="1"/>
                                                            <site name="ring_tip_site" pos="0 0 0" size="0.005"/>
                                                        </body>
                                                    </body>
                                                </body>
                                            </body>
                                            
                                            <!-- Thumb -->
                                            <body name="thumb_base" pos="0.035 0 0" euler="0 0 1.57">
                                                <joint name="thumb_joint1" type="hinge" axis="1 0 0" range="-0.47 0.47"/>
                                                <geom name="thumb_link1" type="capsule" fromto="0 0 0 0 0.025 0" size="0.009" rgba="0.7 0.7 0.7 1"/>
                                                
                                                <body name="thumb_middle" pos="0 0.025 0">
                                                    <joint name="thumb_joint2" type="hinge" axis="1 0 0" range="-0.3 1.6"/>
                                                    <geom name="thumb_link2" type="capsule" fromto="0 0 0 0 0.025 0" size="0.008" rgba="0.7 0.7 0.7 1"/>
                                                    
                                                    <body name="thumb_distal" pos="0 0.025 0">
                                                        <joint name="thumb_joint3" type="hinge" axis="1 0 0" range="-0.3 1.6"/>
                                                        <geom name="thumb_link3" type="capsule" fromto="0 0 0 0 0.02 0" size="0.007" rgba="0.7 0.7 0.7 1"/>
                                                        
                                                        <body name="thumb_tip" pos="0 0.02 0">
                                                            <joint name="thumb_joint4" type="hinge" axis="1 0 0" range="-0.3 1.6"/>
                                                            <geom name="thumb_tip_geom" type="sphere" size="0.009" rgba="0.8 0.6 0.6 1" contype="1" conaffinity="1"/>
                                                            <site name="thumb_tip_site" pos="0 0 0" size="0.005"/>
                                                        </body>
                                                    </body>
                                                </body>
                                            </body>
                                        </body>
                                    </body>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
        
{object_xml}
    </worldbody>
    
    <actuator>
        <!-- Arm actuators -->
        <motor name="arm_motor1" joint="joint1" gear="50"/>
        <motor name="arm_motor2" joint="joint2" gear="50"/>
        <motor name="arm_motor3" joint="joint3" gear="50"/>
        <motor name="arm_motor4" joint="joint4" gear="30"/>
        <motor name="arm_motor5" joint="joint5" gear="30"/>
        <motor name="arm_motor6" joint="joint6" gear="20"/>
        
        <!-- Hand actuators -->
        <motor name="index_motor1" joint="index_joint1" gear="10"/>
        <motor name="index_motor2" joint="index_joint2" gear="10"/>
        <motor name="index_motor3" joint="index_joint3" gear="10"/>
        <motor name="index_motor4" joint="index_joint4" gear="10"/>
        
        <motor name="middle_motor1" joint="middle_joint1" gear="10"/>
        <motor name="middle_motor2" joint="middle_joint2" gear="10"/>
        <motor name="middle_motor3" joint="middle_joint3" gear="10"/>
        <motor name="middle_motor4" joint="middle_joint4" gear="10"/>
        
        <motor name="ring_motor1" joint="ring_joint1" gear="10"/>
        <motor name="ring_motor2" joint="ring_joint2" gear="10"/>
        <motor name="ring_motor3" joint="ring_joint3" gear="10"/>
        <motor name="ring_motor4" joint="ring_joint4" gear="10"/>
        
        <motor name="thumb_motor1" joint="thumb_joint1" gear="10"/>
        <motor name="thumb_motor2" joint="thumb_joint2" gear="10"/>
        <motor name="thumb_motor3" joint="thumb_joint3" gear="10"/>
        <motor name="thumb_motor4" joint="thumb_joint4" gear="10"/>
    </actuator>
    
    <sensor>
        <!-- End effector position -->
        <framepos name="ee_pos" objtype="body" objname="palm"/>
        <framequat name="ee_quat" objtype="body" objname="palm"/>
        
        <!-- Contact sensors on fingertips -->
        <touch name="index_touch" site="index_tip_site"/>
        <touch name="middle_touch" site="middle_tip_site"/>
        <touch name="ring_touch" site="ring_tip_site"/>
        <touch name="thumb_touch" site="thumb_tip_site"/>
        
        <!-- Object sensors -->
        <framepos name="object_pos" objtype="site" objname="object_center"/>
        <framelinvel name="object_vel" objtype="site" objname="object_center"/>
    </sensor>
</mujoco>
        '''
    
    @jax.jit
    def step(self, data: mjx.Data, action: jnp.ndarray) -> mjx.Data:
        """Single simulation step"""
        data = data.replace(ctrl=action)
        data = mjx.step(self.model, data)
        return data
    
    @jax.jit
    def rollout(self, initial_data: mjx.Data, actions: jnp.ndarray) -> mjx.Data:
        """Rollout trajectory given sequence of actions"""
        def scan_fn(data, action):
            data = self.step(data, action)
            return data, data
        
        _, trajectory = jax.lax.scan(scan_fn, initial_data, actions)
        return trajectory
    
    def demo_rl_reward(self, data: mjx.Data) -> float:
        """Example reward function for RL using pickup metrics"""
        # Get object state
        object_state = self.get_object_state(data)
        object_pos = object_state['position']
        
        # Use the new pickup reward function
        reward = self.pickup_reward(data, self.initial_object_pos[2])
        
        return reward
    
    @jax.jit
    def get_ee_pose(self, data: mjx.Data) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get end-effector position and orientation"""
        palm_id = self.mj_model.body('palm').id
        pos = data.xpos[palm_id]
        quat = data.xquat[palm_id]
        return pos, quat
    
    def setup_kinematics(self):
        """Setup kinematic parameters for the 6-DOF arm"""
        # DH parameters for the arm (a, alpha, d, theta_offset)
        # These are approximate based on the model geometry
        self.dh_params = jnp.array([
            [0.0,  0.0,   0.3,  0.0],    # Joint 1
            [0.0,  jnp.pi/2, 0.0,  0.0],  # Joint 2  
            [0.3,  0.0,   0.0,  0.0],    # Joint 3
            [0.0,  jnp.pi/2, 0.3,  0.0],  # Joint 4
            [0.0, -jnp.pi/2, 0.0,  0.0],  # Joint 5
            [0.0,  jnp.pi/2, 0.26, 0.0]   # Joint 6
        ])
        
        # Joint limits
        self.joint_limits = jnp.array([
            [-3.14, 3.14],    # Joint 1
            [-1.57, 1.57],    # Joint 2
            [-1.57, 1.57],    # Joint 3
            [-3.14, 3.14],    # Joint 4
            [-1.57, 1.57],    # Joint 5
            [-3.14, 3.14]     # Joint 6
        ])
    
    @jax.jit
    def dh_matrix(self, a: float, alpha: float, d: float, theta: float) -> jnp.ndarray:
        """Compute DH transformation matrix"""
        ct = jnp.cos(theta)
        st = jnp.sin(theta)
        ca = jnp.cos(alpha)
        sa = jnp.sin(alpha)
        
        return jnp.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,   sa,     ca,    d],
            [0,   0,      0,     1]
        ])
    
    @jax.jit
    def forward_kinematics(self, joint_angles: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute forward kinematics for 6-DOF arm"""
        T = jnp.eye(4)
        
        # Compute transformation for each joint
        for i in range(6):
            a, alpha, d, theta_offset = self.dh_params[i]
            theta = joint_angles[i] + theta_offset
            T_i = self.dh_matrix(a, alpha, d, theta)
            T = T @ T_i
        
        # Extract position and rotation matrix
        position = T[:3, 3]
        rotation = T[:3, :3]
        
        # Convert rotation matrix to quaternion
        quaternion = self.rotation_to_quaternion(rotation)
        
        return position, quaternion
    
    @jax.jit
    def rotation_to_quaternion(self, R: jnp.ndarray) -> jnp.ndarray:
        """Convert rotation matrix to quaternion"""
        trace = jnp.trace(R)
        
        # Compute quaternion components
        w = jnp.sqrt(1 + trace) / 2
        x = (R[2, 1] - R[1, 2]) / (4 * w + 1e-10)
        y = (R[0, 2] - R[2, 0]) / (4 * w + 1e-10)
        z = (R[1, 0] - R[0, 1]) / (4 * w + 1e-10)
        
        quat = jnp.array([w, x, y, z])
        return quat / (jnp.linalg.norm(quat) + 1e-10)
    
    @jax.jit
    def quaternion_to_rotation(self, quat: jnp.ndarray) -> jnp.ndarray:
        """Convert quaternion to rotation matrix"""
        w, x, y, z = quat
        
        return jnp.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ])
    
    @jax.jit
    def compute_jacobian(self, joint_angles: jnp.ndarray) -> jnp.ndarray:
        """Compute 6x6 Jacobian matrix analytically"""
        J = jnp.zeros((6, 6))
        T = jnp.eye(4)
        transforms = []
        
        # Forward pass to get all transformations
        for i in range(6):
            a, alpha, d, theta_offset = self.dh_params[i]
            theta = joint_angles[i] + theta_offset
            T_i = self.dh_matrix(a, alpha, d, theta)
            T = T @ T_i
            transforms.append(T)
        
        # End-effector position
        p_ee = transforms[-1][:3, 3]
        
        # Compute Jacobian columns
        for i in range(6):
            if i == 0:
                T_prev = jnp.eye(4)
            else:
                T_prev = transforms[i-1]
            
            # Joint axis in world frame
            z_i = T_prev[:3, 2]
            p_i = T_prev[:3, 3]
            
            # Linear velocity component (revolute joint)
            J = J.at[:3, i].set(jnp.cross(z_i, p_ee - p_i))
            
            # Angular velocity component
            J = J.at[3:, i].set(z_i)
        
        return J
    
    @jax.jit
    def inverse_kinematics_numeric(self, target_pos: jnp.ndarray, 
                                  target_quat: jnp.ndarray,
                                  initial_joints: jnp.ndarray,
                                  max_iter: int = 50,
                                  tolerance: float = 1e-3) -> jnp.ndarray:
        """Numerical IK using damped least squares (Levenberg-Marquardt)"""
        joint_angles = initial_joints[:6]  # Only arm joints
        damping = 0.01
        
        def ik_iteration(carry, _):
            joint_angles = carry
            
            # Forward kinematics
            current_pos, current_quat = self.forward_kinematics(joint_angles)
            
            # Position error
            pos_error = target_pos - current_pos
            
            # Orientation error (quaternion difference)
            quat_error = self.quaternion_error(current_quat, target_quat)
            
            # Combined error vector
            error = jnp.concatenate([pos_error, quat_error])
            
            # Check convergence
            error_norm = jnp.linalg.norm(error)
            
            # Compute Jacobian
            J = self.compute_jacobian(joint_angles)
            
            # Damped least squares solution
            JtJ = J.T @ J
            damped_JtJ = JtJ + damping * jnp.eye(6)
            delta_theta = jnp.linalg.solve(damped_JtJ, J.T @ error)
            
            # Update joint angles with scaling
            alpha = 0.5  # Step size
            new_angles = joint_angles + alpha * delta_theta
            
            # Apply joint limits
            new_angles = jnp.clip(new_angles, self.joint_limits[:, 0], self.joint_limits[:, 1])
            
            return new_angles, error_norm
        
        # Run iterations
        final_angles, _ = jax.lax.scan(ik_iteration, joint_angles, jnp.arange(max_iter))
        
        return final_angles
    
    @jax.jit
    def quaternion_error(self, q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
        """Compute error between two quaternions as axis-angle"""
        # Quaternion difference
        q_diff = self.quaternion_multiply(q2, self.quaternion_conjugate(q1))
        
        # Convert to axis-angle
        angle = 2 * jnp.arccos(jnp.clip(q_diff[0], -1, 1))
        axis = q_diff[1:] / (jnp.sin(angle/2) + 1e-10)
        
        # Return as rotation vector
        return angle * axis
    
    @jax.jit
    def quaternion_multiply(self, q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return jnp.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    @jax.jit
    def quaternion_conjugate(self, q: jnp.ndarray) -> jnp.ndarray:
        """Compute quaternion conjugate"""
        return jnp.array([q[0], -q[1], -q[2], -q[3]])
    
    @jax.jit
    def inverse_kinematics_ccd(self, target_pos: jnp.ndarray,
                              target_quat: jnp.ndarray,
                              initial_joints: jnp.ndarray,
                              max_iter: int = 10) -> jnp.ndarray:
        """Cyclic Coordinate Descent IK - simpler but effective"""
        joint_angles = initial_joints[:6]
        
        def ccd_iteration(carry, _):
            angles = carry
            
            # Work backwards from end-effector
            for i in range(5, -1, -1):
                # Get current end-effector position
                ee_pos, _ = self.forward_kinematics(angles)
                
                # Get position of current joint
                angles_to_joint = angles.at[i].set(0)
                joint_pos, _ = self.forward_kinematics(angles_to_joint)
                
                # Vectors from joint to end-effector and target
                v_to_ee = ee_pos - joint_pos
                v_to_target = target_pos - joint_pos
                
                # Compute angle between vectors
                v_to_ee = v_to_ee / (jnp.linalg.norm(v_to_ee) + 1e-10)
                v_to_target = v_to_target / (jnp.linalg.norm(v_to_target) + 1e-10)
                
                cos_angle = jnp.dot(v_to_ee, v_to_target)
                angle = jnp.arccos(jnp.clip(cos_angle, -1, 1))
                
                # Update joint angle
                delta = 0.5 * angle  # Damping factor
                angles = angles.at[i].add(delta)
                
                # Apply joint limits
                angles = angles.at[i].set(
                    jnp.clip(angles[i], self.joint_limits[i, 0], self.joint_limits[i, 1])
                )
            
            return angles, None
        
        final_angles, _ = jax.lax.scan(ccd_iteration, joint_angles, jnp.arange(max_iter))
        return final_angles
    
    @jax.jit
    def ik_with_nullspace(self, target_pos: jnp.ndarray,
                         target_quat: jnp.ndarray, 
                         initial_joints: jnp.ndarray,
                         secondary_task: callable = None) -> jnp.ndarray:
        """IK with nullspace optimization for secondary tasks"""
        joint_angles = initial_joints[:6]
        
        # Primary task: reach target
        J = self.compute_jacobian(joint_angles)
        current_pos, current_quat = self.forward_kinematics(joint_angles)
        
        # Task space error
        pos_error = target_pos - current_pos
        rot_error = self.quaternion_error(current_quat, target_quat)
        error = jnp.concatenate([pos_error, rot_error])
        
        # Pseudoinverse solution
        J_pinv = jnp.linalg.pinv(J)
        primary_motion = J_pinv @ error
        
        # Nullspace projection for secondary task
        if secondary_task is not None:
            # Nullspace projector
            P_null = jnp.eye(6) - J_pinv @ J
            
            # Secondary task gradient (e.g., joint limit avoidance)
            secondary_grad = secondary_task(joint_angles)
            nullspace_motion = P_null @ secondary_grad
            
            # Combined motion
            delta_theta = primary_motion + 0.1 * nullspace_motion
        else:
            delta_theta = primary_motion
        
        # Update with step size
        new_angles = joint_angles + 0.5 * delta_theta
        
        # Apply limits
        return jnp.clip(new_angles, self.joint_limits[:, 0], self.joint_limits[:, 1])
    
    def setup_hand_kinematics(self):
        """Setup kinematic parameters for Allegro hand"""
        # Finger segment lengths (in meters)
        self.finger_lengths = jnp.array([
            0.03,   # Proximal segment
            0.025,  # Middle segment  
            0.02,   # Distal segment
            0.015   # Tip segment (to fingertip center)
        ])
        
        # Joint ranges for each finger joint (radians)
        self.finger_joint_limits = jnp.array([
            [-0.47, 0.47],    # Base joint (abduction/adduction)
            [-0.3, 1.6],      # Proximal joint
            [-0.3, 1.6],      # Middle joint
            [-0.3, 1.6]       # Distal joint
        ])
        
        # Finger base positions relative to palm (for each finger)
        self.finger_base_positions = jnp.array([
            [0.02, 0.05, 0.0],    # Index
            [0.0, 0.05, 0.0],     # Middle
            [-0.02, 0.05, 0.0],   # Ring
            [0.035, 0.0, 0.0]     # Thumb (different orientation)
        ])
    
    @jax.jit
    def finger_forward_kinematics(self, finger_joints: jnp.ndarray, 
                                 finger_id: int) -> jnp.ndarray:
        """Forward kinematics for a single finger"""
        # Get base position for this finger
        base_pos = self.finger_base_positions[finger_id]
        
        # Special handling for thumb orientation
        if finger_id == 3:  # Thumb
            base_rotation = jnp.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        else:
            base_rotation = jnp.eye(3)
        
        positions = [base_pos]
        current_pos = base_pos
        current_rot = base_rotation
        
        # Forward kinematics through finger joints
        for i, (joint_angle, length) in enumerate(zip(finger_joints, self.finger_lengths)):
            if i == 0:  # Base joint (abduction/adduction)
                rot_axis = jnp.array([1, 0, 0])
            else:  # Other joints (flexion)
                rot_axis = jnp.array([1, 0, 0])
            
            # Rotation matrix for this joint
            c = jnp.cos(joint_angle)
            s = jnp.sin(joint_angle)
            rot_matrix = jnp.array([
                [1, 0, 0],
                [0, c, -s],
                [0, s, c]
            ])
            
            current_rot = current_rot @ rot_matrix
            current_pos = current_pos + current_rot @ jnp.array([0, length, 0])
            positions.append(current_pos)
        
        return jnp.array(positions)
    
    @jax.jit 
    def finger_inverse_kinematics(self, target_pos: jnp.ndarray,
                                 finger_id: int,
                                 initial_joints: jnp.ndarray = None) -> jnp.ndarray:
        """IK for single finger to reach target position"""
        if initial_joints is None:
            initial_joints = jnp.zeros(4)
        
        finger_joints = initial_joints
        
        # Simplified IK using geometric approach
        base_pos = self.finger_base_positions[finger_id]
        target_vec = target_pos - base_pos
        distance = jnp.linalg.norm(target_vec)
        
        # Check reachability
        max_reach = jnp.sum(self.finger_lengths)
        distance = jnp.clip(distance, 0.01, max_reach * 0.95)
        
        # Analytical solution for planar 3-link chain (simplified)
        # This is approximate - you'd implement full 3D IK for production
        
        # Base joint: point toward target
        base_angle = jnp.arctan2(target_vec[0], target_vec[1])
        base_angle = jnp.clip(base_angle, *self.finger_joint_limits[0])
        
        # Remaining joints: distribute to reach distance
        remaining_dist = distance
        joint_angles = jnp.array([base_angle, 0.0, 0.0, 0.0])
        
        # Simple heuristic: distribute angles proportionally
        for i in range(1, 4):
            desired_angle = remaining_dist / (4 - i) 
            joint_angles = joint_angles.at[i].set(
                jnp.clip(desired_angle, *self.finger_joint_limits[i])
            )
        
        return joint_angles
    
    @jax.jit
    def compute_grasp_matrix(self, data: mjx.Data) -> jnp.ndarray:
        """Compute grasp matrix for current hand configuration"""
        # Get contact points and normals for each finger
        fingertip_positions = []
        contact_normals = []
        
        for finger_id in range(4):
            # Get fingertip position (simplified)
            finger_joints = data.qpos[6 + finger_id*4 : 6 + (finger_id+1)*4]
            positions = self.finger_forward_kinematics(finger_joints, finger_id)
            fingertip_positions.append(positions[-1])
            
            # Approximate contact normal (pointing inward)
            normal = -positions[-1] / (jnp.linalg.norm(positions[-1]) + 1e-6)
            contact_normals.append(normal)
        
        # Build grasp matrix (simplified version)
        G = jnp.zeros((6, 12))  # 6 DOF wrench, 3 forces per finger
        
        for i, (pos, normal) in enumerate(zip(fingertip_positions, contact_normals)):
            # Force contribution
            G = G.at[0:3, i*3:(i+1)*3].set(jnp.eye(3))
            
            # Torque contribution (cross product matrix)
            skew = jnp.array([
                [0, -pos[2], pos[1]],
                [pos[2], 0, -pos[0]],
                [-pos[1], pos[0], 0]
            ])
            G = G.at[3:6, i*3:(i+1)*3].set(skew)
        
        return G
    
    def create_advanced_grasp_controller(self):
        """Advanced grasp controller with multiple grasp types"""
        
        @jax.jit
        def power_grasp(t: float) -> jnp.ndarray:
            """Power grasp: all fingers close uniformly"""
            progress = jnp.clip(t, 0, 1)
            angles = jnp.zeros(16)
            
            # All fingers curl inward
            for finger in range(4):
                base_idx = finger * 4
                angles = angles.at[base_idx].set(0.1 * progress)      # Slight spread
                angles = angles.at[base_idx + 1].set(1.2 * progress)  # Proximal
                angles = angles.at[base_idx + 2].set(1.0 * progress)  # Middle
                angles = angles.at[base_idx + 3].set(0.8 * progress)  # Distal
            
            return angles
        
        @jax.jit
        def pinch_grasp(t: float) -> jnp.ndarray:
            """Precision pinch: thumb + index"""
            progress = jnp.clip(t, 0, 1)
            angles = jnp.zeros(16)
            
            # Index finger
            angles = angles.at[0].set(-0.2 * progress)   # Adduct slightly
            angles = angles.at[1].set(0.8 * progress)    # Curl
            angles = angles.at[2].set(0.6 * progress)
            angles = angles.at[3].set(0.4 * progress)
            
            # Thumb
            angles = angles.at[12].set(0.3 * progress)   # Oppose
            angles = angles.at[13].set(0.7 * progress)   # Curl
            angles = angles.at[14].set(0.5 * progress)
            angles = angles.at[15].set(0.3 * progress)
            
            return angles
        
        @jax.jit
        def tripod_grasp(t: float) -> jnp.ndarray:
            """Tripod grasp: thumb + index + middle"""
            progress = jnp.clip(t, 0, 1)
            angles = jnp.zeros(16)
            
            # Index and middle fingers
            for finger in [0, 1]:
                base_idx = finger * 4
                angles = angles.at[base_idx].set((-0.1 if finger == 0 else 0.1) * progress)
                angles = angles.at[base_idx + 1].set(0.9 * progress)
                angles = angles.at[base_idx + 2].set(0.7 * progress)
                angles = angles.at[base_idx + 3].set(0.5 * progress)
            
            # Thumb opposes both
            angles = angles.at[12].set(0.2 * progress)
            angles = angles.at[13].set(0.8 * progress)
            angles = angles.at[14].set(0.6 * progress)
            angles = angles.at[15].set(0.4 * progress)
            
            return angles
        
        @jax.jit
        def adaptive_grasp(object_width: float) -> jnp.ndarray:
            """Adapt grasp to object size"""
            # Estimate finger spread based on object width
            spread = jnp.clip(object_width / 0.1, 0.2, 1.0)
            
            angles = jnp.zeros(16)
            
            # Adjust finger positions
            for finger in range(3):  # Index, middle, ring
                base_idx = finger * 4
                angles = angles.at[base_idx].set((finger - 1) * 0.2 * spread)
                angles = angles.at[base_idx + 1].set(1.0 - 0.3 * spread)
                angles = angles.at[base_idx + 2].set(0.8 - 0.2 * spread)
                angles = angles.at[base_idx + 3].set(0.6)
            
            # Thumb
            angles = angles.at[12].set(0.3 * spread)
            angles = angles.at[13].set(0.9 - 0.2 * spread)
            angles = angles.at[14].set(0.7)
            angles = angles.at[15].set(0.5)
            
            return angles
        
        return {
            'power': power_grasp,
            'pinch': pinch_grasp,
            'tripod': tripod_grasp,
            'adaptive': adaptive_grasp
        }
    
    @jax.jit
    def hand_jacobian(self, data: mjx.Data) -> jnp.ndarray:
        """Compute Jacobian for all fingertips"""
        # Stack Jacobians for each fingertip
        J_hand = []
        
        for finger_id in range(4):
            finger_joints = data.qpos[6 + finger_id*4 : 6 + (finger_id+1)*4]
            
            # Numerical Jacobian for this finger
            eps = 1e-4
            J_finger = jnp.zeros((3, 4))
            
            for joint_idx in range(4):
                # Perturb joint
                joints_plus = finger_joints.at[joint_idx].add(eps)
                joints_minus = finger_joints.at[joint_idx].add(-eps)
                
                # Forward kinematics
                pos_plus = self.finger_forward_kinematics(joints_plus, finger_id)[-1]
                pos_minus = self.finger_forward_kinematics(joints_minus, finger_id)[-1]
                
                # Finite difference
                J_finger = J_finger.at[:, joint_idx].set((pos_plus - pos_minus) / (2 * eps))
            
            J_hand.append(J_finger)
        
        return J_hand
    
    @jax.jit
    def grasp_quality_metric(self, data: mjx.Data, object_pos: jnp.ndarray = None) -> float:
        """Compute grasp quality metric"""
        # Use tracked object position if not provided
        if object_pos is None:
            object_pos = data.xpos[self.object_body_id]
            
        # Get palm position for relative calculations
        palm_id = self.mj_model.body('palm').id
        palm_pos = data.xpos[palm_id]
        
        # Get fingertip positions
        fingertip_positions = []
        for finger_id in range(4):
            finger_joints = data.qpos[6 + finger_id*4 : 6 + (finger_id+1)*4]
            positions = self.finger_forward_kinematics(finger_joints, finger_id)
            # Convert to world coordinates
            fingertip_world = palm_pos + positions[-1]
            fingertip_positions.append(fingertip_world)
        
        fingertip_positions = jnp.array(fingertip_positions)
        
        # Distance from fingertips to object
        distances = jnp.linalg.norm(fingertip_positions - object_pos[None, :], axis=1)
        avg_distance = jnp.mean(distances)
        
        # Grasp polygon area (simplified)
        # Compute convex hull area of contact points projected to x-y plane
        hull_area = self._compute_hull_area(fingertip_positions[:, :2])
        
        # Force closure approximation
        # Check if fingertips surround the object
        relative_positions = fingertip_positions - object_pos
        angles = jnp.arctan2(relative_positions[:, 1], relative_positions[:, 0])
        angle_spread = jnp.max(angles) - jnp.min(angles)
        
        # Quality metric combines multiple factors
        quality = (1.0 / (avg_distance + 0.01)) * hull_area * (angle_spread / jnp.pi)
        
        return quality
    
    @jax.jit
    def multi_construct_pickup_error(self, data: mjx.Data, 
                                   initial_object_height: float = None) -> Dict:
        """
        Multi-construct error function to determine if object was picked up.
        Returns dictionary with individual errors and overall success score.
        
        Measures:
        1. Object lift height
        2. Grasp stability 
        3. Palm-object proximity
        4. Fingertip contacts
        5. Object velocity (should be low if grasped well)
        6. Hand-object relative motion
        """
        if initial_object_height is None:
            initial_object_height = self.initial_object_pos[2]
            
        # Get object state
        object_pos = data.xpos[self.object_body_id]
        object_vel = data.qvel[self.object_body_id*6:(self.object_body_id*6)+3]  # Linear velocity
        
        # Get palm state
        palm_id = self.mj_model.body('palm').id
        palm_pos = data.xpos[palm_id]
        palm_vel = data.qvel[palm_id*6:(palm_id*6)+3]
        
        # 1. Lift error (negative = good, object lifted)
        current_height = object_pos[2]
        lift_error = -(current_height - initial_object_height)
        lift_success = current_height > initial_object_height + 0.05  # 5cm threshold
        
        # 2. Palm-object distance error
        palm_object_dist = jnp.linalg.norm(palm_pos - object_pos)
        proximity_error = palm_object_dist
        proximity_success = palm_object_dist < 0.08  # Within 8cm
        
        # 3. Fingertip contact errors
        fingertip_distances = []
        finger_names = ['index', 'middle', 'ring', 'thumb']
        
        for i, finger_name in enumerate(finger_names):
            # Get fingertip position using FK
            finger_joints = data.qpos[6 + i*4 : 6 + (i+1)*4]
            fingertip_pos = self.finger_forward_kinematics(finger_joints, i)[-1]
            
            # Add palm position since FK is relative
            fingertip_world_pos = palm_pos + fingertip_pos
            dist = jnp.linalg.norm(fingertip_world_pos - object_pos)
            fingertip_distances.append(dist)
        
        fingertip_distances = jnp.array(fingertip_distances)
        avg_fingertip_dist = jnp.mean(fingertip_distances)
        min_fingertip_dist = jnp.min(fingertip_distances)
        
        # At least 3 fingers should be close for good grasp
        close_fingers = jnp.sum(fingertip_distances < 0.06)
        contact_success = close_fingers >= 3
        
        # 4. Grasp quality score
        grasp_quality = self.grasp_quality_metric(data)
        quality_success = grasp_quality > 2.0
        
        # 5. Object stability (low velocity = stable grasp)
        object_speed = jnp.linalg.norm(object_vel)
        stability_error = object_speed
        stability_success = object_speed < 0.5  # m/s threshold
        
        # 6. Relative motion (object should move with hand)
        relative_velocity = jnp.linalg.norm(object_vel - palm_vel)
        relative_motion_error = relative_velocity
        coupled_motion = relative_velocity < 0.3
        
        # 7. Object orientation stability (bonus metric)
        # Check if object is tumbling by looking at angular velocity
        object_angvel = data.qvel[(self.object_body_id*6)+3:(self.object_body_id*6)+6]
        angular_speed = jnp.linalg.norm(object_angvel)
        rotation_stable = angular_speed < 2.0  # rad/s
        
        # Overall pickup success (all conditions must be met)
        pickup_success = jnp.logical_and.reduce(jnp.array([
            lift_success,
            proximity_success,
            contact_success,
            quality_success,
            stability_success,
            coupled_motion
        ]))
        
        # Combined error for optimization (weighted sum)
        total_error = (
            1.0 * jnp.maximum(0, -lift_error - 0.05) +  # Penalize if not lifted enough
            2.0 * proximity_error +                       # Palm should be close
            1.0 * avg_fingertip_dist +                   # Fingers should contact
            0.5 * stability_error +                       # Object should be stable
            0.5 * relative_motion_error +                 # Move together
            0.1 * angular_speed                          # Minimize tumbling
        )
        
        return {
            # Individual metrics
            'lift_height': current_height - initial_object_height,
            'lift_error': lift_error,
            'proximity_error': proximity_error,
            'avg_fingertip_distance': avg_fingertip_dist,
            'min_fingertip_distance': min_fingertip_dist,
            'close_fingers_count': close_fingers,
            'grasp_quality': grasp_quality,
            'object_speed': object_speed,
            'relative_velocity': relative_velocity,
            'angular_speed': angular_speed,
            
            # Success flags
            'lift_success': lift_success,
            'proximity_success': proximity_success,
            'contact_success': contact_success,
            'quality_success': quality_success,
            'stability_success': stability_success,
            'coupled_motion_success': coupled_motion,
            'rotation_stable': rotation_stable,
            
            # Overall results
            'pickup_success': pickup_success,
            'total_error': total_error,
            
            # Useful for debugging
            'object_pos': object_pos,
            'palm_pos': palm_pos,
            'fingertip_distances': fingertip_distances
        }
    
    @jax.jit
    def pickup_reward(self, data: mjx.Data, initial_object_height: float = None) -> float:
        """
        Simplified reward function for RL based on pickup error.
        Returns positive reward for successful pickup behaviors.
        """
        metrics = self.multi_construct_pickup_error(data, initial_object_height)
        
        # Reward components
        lift_reward = 10.0 * jnp.maximum(0, metrics['lift_height'])
        proximity_reward = 2.0 * jnp.exp(-metrics['proximity_error'] / 0.05)
        contact_reward = 1.0 * metrics['close_fingers_count']
        quality_reward = metrics['grasp_quality']
        stability_reward = 5.0 * jnp.exp(-metrics['object_speed'] / 0.1)
        
        # Big bonus for successful pickup
        success_bonus = 50.0 * metrics['pickup_success'].astype(jnp.float32)
        
        total_reward = (
            lift_reward + 
            proximity_reward + 
            contact_reward + 
            quality_reward + 
            stability_reward + 
            success_bonus
        )
        
    def _compute_hull_area(self, points_2d: jnp.ndarray) -> float:
        """Compute area of convex hull (simplified)"""
        # Simple approximation using shoelace formula
        n = len(points_2d)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points_2d[i, 0] * points_2d[j, 1]
            area -= points_2d[j, 0] * points_2d[i, 1]
        return jnp.abs(area) / 2.0
    
    def demo_trajectory_optimization(self, target_pos: jnp.ndarray = None, num_steps: int = 100):
        """Optimize trajectory to reach target position (or object if not specified)"""
        
        # If no target specified, use object position
        if target_pos is None:
            target_pos = self.data.xpos[self.object_body_id]
        
        @jax.jit
        def trajectory_cost(actions: jnp.ndarray) -> float:
            """Cost function for trajectory optimization"""
            # Initial data
            data = self.data
            
            # Rollout trajectory
            trajectory = self.rollout(data, actions)
            
            # Final position error
            final_ee_pos, _ = self.get_ee_pose(trajectory[-1])
            pos_error = jnp.sum((final_ee_pos - target_pos) ** 2)
            
            # Control effort
            control_effort = 0.01 * jnp.sum(actions ** 2)
            
            # Smoothness penalty
            smoothness = 0.1 * jnp.sum((actions[1:] - actions[:-1]) ** 2)
            
            return pos_error + control_effort + smoothness
        
        # Initialize with zeros
        actions = jnp.zeros((num_steps, self.n_total_joints))
        
        # Optimize using gradient descent
        learning_rate = 0.01
        for i in range(50):
            loss, grads = jax.value_and_grad(trajectory_cost)(actions)
            actions = actions - learning_rate * grads
            
            if i % 10 == 0:
                print(f"Iteration {i}, Loss: {loss:.4f}")
        
        return actions
    
    def demo_ik_methods(self, target_pos: jnp.ndarray, target_quat: jnp.ndarray = None):
        """Demonstrate different IK methods"""
        if target_quat is None:
            target_quat = jnp.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        
        # Initial joint configuration
        initial_joints = jnp.zeros(self.n_total_joints)
        
        print("\n=== IK Method Comparison ===")
        
        # Method 1: Numerical IK (Damped Least Squares)
        print("\n1. Numerical IK (Damped Least Squares):")
        start_time = time.time()
        joints_numeric = self.inverse_kinematics_numeric(
            target_pos, target_quat, initial_joints
        )
        numeric_time = time.time() - start_time
        
        # Check result
        pos_numeric, quat_numeric = self.forward_kinematics(joints_numeric)
        error_numeric = jnp.linalg.norm(pos_numeric - target_pos)
        print(f"   Time: {numeric_time:.4f}s")
        print(f"   Position error: {error_numeric:.6f}")
        print(f"   Joint angles: {joints_numeric}")
        
        # Method 2: CCD IK
        print("\n2. Cyclic Coordinate Descent IK:")
        start_time = time.time()
        joints_ccd = self.inverse_kinematics_ccd(
            target_pos, target_quat, initial_joints
        )
        ccd_time = time.time() - start_time
        
        pos_ccd, quat_ccd = self.forward_kinematics(joints_ccd)
        error_ccd = jnp.linalg.norm(pos_ccd - target_pos)
        print(f"   Time: {ccd_time:.4f}s")
        print(f"   Position error: {error_ccd:.6f}")
        print(f"   Joint angles: {joints_ccd}")
        
        # Method 3: IK with nullspace optimization
        print("\n3. IK with Nullspace Optimization:")
        
        # Define secondary task: minimize joint velocities
        def joint_center_task(joints):
            """Secondary task: keep joints near center of range"""
            joint_centers = (self.joint_limits[:, 0] + self.joint_limits[:, 1]) / 2
            return -2 * (joints - joint_centers)  # Gradient toward centers
        
        start_time = time.time()
        joints_nullspace = self.ik_with_nullspace(
            target_pos, target_quat, initial_joints, joint_center_task
        )
        nullspace_time = time.time() - start_time
        
        pos_null, quat_null = self.forward_kinematics(joints_nullspace)
        error_null = jnp.linalg.norm(pos_null - target_pos)
        print(f"   Time: {nullspace_time:.4f}s")
        print(f"   Position error: {error_null:.6f}")
        print(f"   Joint angles: {joints_nullspace}")
        
        return joints_numeric, joints_ccd, joints_nullspace
    
    @jax.jit
    def cartesian_trajectory_tracking(self, waypoints: jnp.ndarray, 
                                    orientations: jnp.ndarray = None,
                                    time_steps: int = 100) -> jnp.ndarray:
        """Generate joint trajectory to follow Cartesian waypoints"""
        n_waypoints = waypoints.shape[0]
        
        if orientations is None:
            # Default orientation (pointing down)
            orientations = jnp.tile(jnp.array([0.7071, 0.0, 0.7071, 0.0]), (n_waypoints, 1))
        
        # Interpolate between waypoints
        t = jnp.linspace(0, n_waypoints - 1, time_steps)
        indices = jnp.floor(t).astype(jnp.int32)
        fractions = t - indices
        
        # Handle last index
        indices = jnp.clip(indices, 0, n_waypoints - 2)
        
        # Interpolate positions
        positions = waypoints[indices] * (1 - fractions[:, None]) + \
                   waypoints[indices + 1] * fractions[:, None]
        
        # Interpolate orientations (SLERP for quaternions)
        quats = []
        for i in range(time_steps):
            idx = indices[i]
            frac = fractions[i]
            q1 = orientations[idx]
            q2 = orientations[idx + 1]
            
            # SLERP
            dot = jnp.dot(q1, q2)
            q2 = jnp.where(dot < 0, -q2, q2)
            dot = jnp.abs(dot)
            
            theta = jnp.arccos(jnp.clip(dot, -1, 1))
            sin_theta = jnp.sin(theta)
            
            w1 = jnp.where(sin_theta > 1e-5, 
                          jnp.sin((1 - frac) * theta) / sin_theta,
                          1 - frac)
            w2 = jnp.where(sin_theta > 1e-5,
                          jnp.sin(frac * theta) / sin_theta,
                          frac)
            
            q_interp = w1 * q1 + w2 * q2
            quats.append(q_interp / jnp.linalg.norm(q_interp))
        
        quats = jnp.array(quats)
        
        # Solve IK for each waypoint
        joint_trajectory = []
        current_joints = jnp.zeros(self.n_total_joints)
        
        for i in range(time_steps):
            # Use previous solution as initial guess
            arm_joints = self.inverse_kinematics_numeric(
                positions[i], quats[i], current_joints, max_iter=20
            )
            
            # Keep hand configuration unchanged
            current_joints = current_joints.at[:6].set(arm_joints)
            joint_trajectory.append(current_joints)
        
        return jnp.array(joint_trajectory)
    
    def demo_manipulation_task(self):
        """Demo: Pick and place task with IK"""
        print("\n=== Manipulation Task Demo ===")
        
        # Define task waypoints
        # 1. Home position
        home_pos = jnp.array([0.3, 0.0, 0.6])
        # 2. Above object
        above_object = jnp.array([0.3, 0.2, 0.6])
        # 3. Grasp position
        grasp_pos = jnp.array([0.3, 0.2, 0.5])
        # 4. Lift position
        lift_pos = jnp.array([0.3, 0.2, 0.7])
        # 5. Above target
        above_target = jnp.array([0.1, -0.2, 0.7])
        # 6. Place position
        place_pos = jnp.array([0.1, -0.2, 0.5])
        
        waypoints = jnp.array([
            home_pos, above_object, grasp_pos, lift_pos,
            above_target, place_pos, above_target, home_pos
        ])
        
        # Generate trajectory
        print("Generating manipulation trajectory...")
        trajectory = self.cartesian_trajectory_tracking(waypoints, time_steps=200)
        
        # Add hand control
        grasp_ctrl = self.create_advanced_grasp_controller()
        
        # Modify trajectory to include grasping
        for i in range(len(trajectory)):
            t = i / len(trajectory)
            
            # Close hand when near grasp position
            if 0.2 < t < 0.7:  # During grasp and transport
                hand_ctrl = grasp_ctrl['power'](1.0)  # Closed
            else:
                hand_ctrl = grasp_ctrl['power'](0.0)  # Open
            
            trajectory = trajectory.at[i, 6:].set(hand_ctrl)
        
        return trajectory, grasp_ctrl
    
    def demo_grasp_controller(self):
        """Demo basic grasp controller"""
        grasp_ctrl = self.create_advanced_grasp_controller()
        
        # Open hand
        open_hand = grasp_ctrl['power'](0.0)
        print(f"\nOpen hand control: {open_hand[:4]}...")  # Show first 4 values
        
        # Close hand
        close_hand = grasp_ctrl['power'](1.0)
        print(f"Close hand control: {close_hand[:4]}...")
        
        return grasp_ctrl


# Example usage
if __name__ == "__main__":
    # Create robot with custom object
    # Option 1: Default sphere
    robot = RobotArmAllegro()
    
    # Option 2: Load from OBJ file
    # robot = RobotArmAllegro(
    #     obj_filename="path/to/your/object.obj",
    #     obj_pos=[0.3, 0.2, 0.5],
    #     obj_scale=[0.01, 0.01, 0.01]  # Scale down if needed
    # )
    
    # Test basic simulation
    print("Model loaded successfully!")
    print(f"Number of joints: {robot.model.nq}")
    print(f"Number of actuators: {robot.model.nu}")
    print(f"  - Arm joints: {robot.n_arm_joints}")
    print(f"  - Hand joints: {robot.n_hand_joints}")
    print(f"Object initial position: {robot.initial_object_pos}")
    
    # Initialize data
    key = jax.random.PRNGKey(0)
    data = robot.data
    
    # ========== PICKUP ERROR TESTING ==========
    print("\n=== Multi-Construct Pickup Error Testing ===")
    
    # Get initial object state
    initial_obj_state = robot.get_object_state(data)
    initial_height = initial_obj_state['position'][2]
    print(f"Initial object height: {initial_height:.3f}m")
    
    # Test 1: Check initial state (no pickup)
    metrics = robot.multi_construct_pickup_error(data, initial_height)
    print("\n1. Initial State Metrics:")
    print(f"   Lift height: {metrics['lift_height']:.3f}m")
    print(f"   Palm-object distance: {metrics['proximity_error']:.3f}m")
    print(f"   Avg fingertip distance: {metrics['avg_fingertip_distance']:.3f}m")
    print(f"   Grasp quality: {metrics['grasp_quality']:.2f}")
    print(f"   Pickup success: {metrics['pickup_success']}")
    print(f"   Total error: {metrics['total_error']:.3f}")
    
    # Test 2: Simulate reaching and grasping
    print("\n2. Simulating Pickup Attempt...")
    
    # Move arm to object
    object_pos = initial_obj_state['position']
    arm_joints = robot.inverse_kinematics_numeric(
        object_pos + jnp.array([0, 0, -0.05]),  # Slightly below object
        jnp.array([0.7071, 0, 0.7071, 0]),      # Pointing down
        data.qpos[:robot.n_total_joints]
    )
    
    # Close hand
    grasp_ctrl = robot.create_advanced_grasp_controller()
    hand_joints = grasp_ctrl['adaptive'](0.08)  # 8cm object
    
    # Apply control
    control = jnp.concatenate([arm_joints, hand_joints])
    data = data.replace(ctrl=control)
    
    # Simulate for 100 steps
    for _ in range(100):
        data = mjx.step(robot.model, data)
    
    # Check pickup metrics after grasp
    metrics = robot.multi_construct_pickup_error(data, initial_height)
    print("\n   After Grasp Attempt:")
    print(f"   Lift height: {metrics['lift_height']:.3f}m")
    print(f"   Close fingers: {metrics['close_fingers_count']}/4")
    print(f"   Object speed: {metrics['object_speed']:.3f}m/s")
    print(f"   Pickup success: {metrics['pickup_success']}")
    
    # Test 3: Attempt to lift
    print("\n3. Attempting to Lift...")
    
    # Move arm up
    lift_target = object_pos + jnp.array([0, 0, 0.2])  # 20cm up
    arm_joints = robot.inverse_kinematics_numeric(
        lift_target,
        jnp.array([0.7071, 0, 0.7071, 0]),
        data.qpos[:robot.n_total_joints]
    )
    
    control = control.at[:6].set(arm_joints)
    data = data.replace(ctrl=control)
    
    # Simulate lift
    for _ in range(100):
        data = mjx.step(robot.model, data)
    
    # Final pickup assessment
    metrics = robot.multi_construct_pickup_error(data, initial_height)
    print("\n   After Lift Attempt:")
    print(f"   Lift height: {metrics['lift_height']:.3f}m")
    print(f"   Proximity error: {metrics['proximity_error']:.3f}m")
    print(f"   Grasp quality: {metrics['grasp_quality']:.2f}")
    print(f"   Object stability: {metrics['stability_success']}")
    print(f"   Coupled motion: {metrics['coupled_motion_success']}")
    print(f"   PICKUP SUCCESS: {metrics['pickup_success']}")
    print(f"   Total error: {metrics['total_error']:.3f}")
    
    # Show individual success criteria
    print("\n   Success Criteria Breakdown:")
    print(f"   ✓ Lifted 5cm+: {metrics['lift_success']}")
    print(f"   ✓ Palm close (<8cm): {metrics['proximity_success']}")
    print(f"   ✓ 3+ fingers contact: {metrics['contact_success']}")
    print(f"   ✓ Good grasp quality: {metrics['quality_success']}")
    print(f"   ✓ Object stable: {metrics['stability_success']}")
    print(f"   ✓ Moving together: {metrics['coupled_motion_success']}")
    
    # ========== RL TRAINING EXAMPLE ==========
    print("\n=== RL Training with Pickup Error ===")
    
    @jax.jit
    def rl_step_with_pickup(data, action, initial_height):
        """Single RL step optimizing for pickup"""
        # Action: [arm_delta_pos (3), grasp_closure (1)]
        ee_pos, ee_quat = robot.get_ee_pose(data)
        
        # Arm control
        target_pos = ee_pos + action[:3] * 0.05
        arm_joints = robot.inverse_kinematics_numeric(
            target_pos, ee_quat, data.qpos, max_iter=10
        )
        
        # Hand control
        closure = jnp.clip(action[3], 0, 1)
        hand_joints = grasp_ctrl['adaptive'](0.08 * (1 - closure))  # Close as action increases
        
        # Apply control
        control = jnp.concatenate([arm_joints, hand_joints])
        data = data.replace(ctrl=control)
        data = mjx.step(robot.model, data)
        
        # Compute reward
        reward = robot.pickup_reward(data, initial_height)
        
        # Get observation
        obj_state = robot.get_object_state(data)
        obs = jnp.concatenate([
            data.qpos[:22],              # Joint positions
            obj_state['position'],        # Object position
            obj_state['linear_velocity'], # Object velocity
            ee_pos                        # End-effector position
        ])
        
        return data, reward, obs
    
    # Example RL action
    action = jnp.array([0.0, 0.0, -0.1, 0.8])  # Move down and close gripper
    data, reward, obs = rl_step_with_pickup(data, action, initial_height)
    print(f"\nRL step reward: {reward:.2f}")
    
    # ========== GRADIENT-BASED OPTIMIZATION ==========
    print("\n=== Gradient-Based Pickup Optimization ===")
    
    @jax.jit
    def pickup_loss(control_sequence, initial_data, target_height):
        """Loss function for optimizing pickup sequence"""
        data = initial_data
        
        # Apply control sequence
        for ctrl in control_sequence:
            data = data.replace(ctrl=ctrl)
            data = mjx.step(robot.model, data)
        
        # Compute pickup error
        metrics = robot.multi_construct_pickup_error(data, target_height)
        
        # Loss is negative reward (minimize error, maximize success)
        loss = metrics['total_error'] - 10.0 * metrics['pickup_success']
        
        return loss
    
    # Optimize control sequence
    print("Optimizing control sequence for pickup...")
    initial_data = robot.data
    control_seq = jnp.zeros((50, 22))  # 50 timesteps
    
    # Gradient descent
    for i in range(5):
        loss, grads = jax.value_and_grad(pickup_loss)(
            control_seq, initial_data, initial_height
        )
        control_seq = control_seq - 0.001 * grads
        print(f"   Iteration {i}: Loss = {loss:.3f}")
    
    print("\n=== Summary ===")
    print("The multi-construct pickup error provides:")
    print("- Comprehensive pickup success detection")
    print("- Individual error components for debugging")
    print("- Differentiable metrics for gradient-based optimization")
    print("- Ready-to-use reward function for RL")
    print("- Support for custom OBJ file objects")
    print("\nAll metrics are computed in parallel using JAX!")