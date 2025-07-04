import os
os.environ['JAX_DISABLE_JIT'] = '1'
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import time
from typing import List, Dict, Tuple, Optional
import cv2
import os
from datetime import datetime
from model import RobotArmAllegro

# Set matplotlib backend for headless environments
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')  # Use non-interactive backend

# Import the model (adjust path as needed)
# from robot_arm_allegro import RobotArmAllegro

class RobotArmVisualizationSuite:
    """Comprehensive visualization and testing suite for RobotArmAllegro"""
    
    def __init__(self, robot: 'RobotArmAllegro', save_dir: str = "test_results", headless: bool = None):
        self.robot = robot
        self.save_dir = save_dir
        self.viewer = None
        self.fig = None
        self.renderer = None
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Auto-detect headless mode if not specified
        if headless is None:
            headless = not self._check_display_available()
        
        self.headless = headless
        
        if self.headless:
            print("Running in headless mode - real-time viewer and video rendering disabled")
            print("Matplotlib visualizations will still work")
        else:
            try:
                # Initialize renderer for offline rendering
                self.renderer = mujoco.Renderer(self.robot.mj_model)
            except Exception as e:
                print(f"Warning: Could not initialize renderer: {e}")
                print("Video rendering will be disabled, but other visualizations will work")
                self.renderer = None
    
    def _check_display_available(self):
        """Check if display is available for rendering"""
        return os.environ.get('DISPLAY') is not None
        
    def setup_viewer(self):
        """Initialize MuJoCo viewer for real-time visualization"""
        if self.headless:
            print("Viewer not available in headless mode")
            return
            
        if self.viewer is None:
            try:
                self.viewer = mujoco.viewer.launch_passive(self.robot.mj_model, self.robot.mj_model.data)
                # Set camera position for better view
                self.viewer.cam.azimuth = 135
                self.viewer.cam.elevation = -20
                self.viewer.cam.distance = 2.0
                self.viewer.cam.lookat[:] = [0.2, 0.0, 0.5]
            except Exception as e:
                print(f"Warning: Could not launch viewer: {e}")
                self.viewer = None
    
    def update_viewer(self, data: mjx.Data):
        """Update viewer with new data"""
        if self.headless or self.viewer is None:
            return
            
        try:
            # Convert MJX data to MuJoCo data
            mj_data = mujoco.MjData(self.robot.mj_model)
            mj_data.qpos[:] = np.array(data.qpos)
            mj_data.qvel[:] = np.array(data.qvel)
            mj_data.ctrl[:] = np.array(data.ctrl)
            mujoco.mj_forward(self.robot.mj_model, mj_data)
            
            self.viewer.sync()
        except Exception as e:
            print(f"Warning: Could not update viewer: {e}")
            
    def render_frame(self, data: mjx.Data, camera_name: str = None) -> np.ndarray:
        """Render a single frame to image"""
        if self.renderer is None:
            return None
            
        try:
            # Convert MJX data to MuJoCo data
            mj_data = mujoco.MjData(self.robot.mj_model)
            mj_data.qpos[:] = np.array(data.qpos)
            mj_data.qvel[:] = np.array(data.qvel)
            mj_data.ctrl[:] = np.array(data.ctrl)
            mujoco.mj_forward(self.robot.mj_model, mj_data)
            
            # Update renderer
            self.renderer.update_scene(mj_data, camera_name)
            pixels = self.renderer.render()
            return pixels
        except Exception as e:
            print(f"Warning: Could not render frame: {e}")
            return None
    
    def save_trajectory_video(self, trajectory: List[mjx.Data], filename: str, fps: int = 30):
        """Save trajectory as video file"""
        if self.renderer is None:
            print("Video rendering not available in current environment")
            return
            
        if not trajectory:
            return
            
        try:
            # Get frame dimensions
            first_frame = self.render_frame(trajectory[0])
            if first_frame is None:
                print("Could not render frames for video")
                return
                
            height, width = first_frame.shape[:2]
            
            # Setup video writer
            filepath = os.path.join(self.save_dir, f"{filename}_{self.timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
            
            # Write frames
            for data in trajectory:
                frame = self.render_frame(data)
                if frame is not None:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
            
            out.release()
            print(f"Video saved to: {filepath}")
        except Exception as e:
            print(f"Warning: Could not save video: {e}")
    
    def get_object_state(self, data: mjx.Data) -> Dict:
        """Get current object state (helper method for missing function in robot)"""
        object_pos = data.xpos[self.robot.object_body_id]
        object_quat = data.xquat[self.robot.object_body_id]
        
        # Get velocities (6 DOF for free joint)
        # Find the qvel index for the object body
        # For a free joint, velocities are stored as [lin_vel(3), ang_vel(3)]
        # The index depends on how many joints come before the object
        
        # Count DOFs before object (6 arm joints = 6 DOF)
        qvel_start = 6  # After arm joints
        
        object_lin_vel = data.qvel[qvel_start:qvel_start+3]
        object_ang_vel = data.qvel[qvel_start+3:qvel_start+6]
        
        return {
            'position': object_pos,
            'orientation': object_quat,
            'linear_velocity': object_lin_vel,
            'angular_velocity': object_ang_vel
        }
    
    def _show_or_close(self):
        """Show plot in interactive mode or close in headless mode"""
        if not self.headless:
            plt.show()
        else:
            plt.close()
    
    def test_forward_kinematics_visualization(self):
        """Test and visualize forward kinematics"""
        print("\n=== Forward Kinematics Visualization Test ===")
        
        fig = plt.figure(figsize=(15, 5))
        
        # Test different joint configurations
        test_configs = [
            ("Home", jnp.zeros(6)),
            ("Extended", jnp.array([0, -0.5, -0.5, 0, -0.5, 0])),
            ("Folded", jnp.array([0, 1.0, 1.0, 0, 0.5, 0])),
            ("Twisted", jnp.array([1.57, 0.5, -0.5, 1.57, -0.5, 1.57]))
        ]
        
        for i, (name, joints) in enumerate(test_configs):
            ax = fig.add_subplot(1, 4, i+1, projection='3d')
            
            # Compute FK
            pos, quat = self.robot.forward_kinematics(joints)
            
            # Visualize kinematic chain
            self._plot_kinematic_chain(ax, joints)
            
            ax.set_title(f"{name}\nEE: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            self._set_equal_aspect_3d(ax)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"fk_test_{self.timestamp}.png"))
        self._show_or_close()
    
    def test_ik_comparison_visualization(self):
        """Visualize and compare different IK methods"""
        print("\n=== IK Methods Comparison Visualization ===")
        
        # Define test targets
        test_targets = [
            jnp.array([0.4, 0.2, 0.5]),
            jnp.array([0.2, -0.3, 0.6]),
            jnp.array([0.5, 0.0, 0.3]),
        ]
        
        fig = plt.figure(figsize=(15, 10))
        
        for i, target_pos in enumerate(test_targets):
            target_quat = jnp.array([0.7071, 0, 0.7071, 0])  # Pointing down
            initial_joints = jnp.zeros(22)
            
            # Run IK methods
            joints_numeric = self.robot.inverse_kinematics_numeric(
                target_pos, target_quat, initial_joints
            )
            joints_ccd = self.robot.inverse_kinematics_ccd(
                target_pos, target_quat, initial_joints
            )
            
            # Visualize results
            methods = [
                ("Numeric IK", joints_numeric),
                ("CCD IK", joints_ccd)
            ]
            
            for j, (method_name, joints) in enumerate(methods):
                ax = fig.add_subplot(len(test_targets), 2, i*2 + j + 1, projection='3d')
                
                # Plot kinematic chain
                self._plot_kinematic_chain(ax, joints)
                
                # Compute actual position
                actual_pos, _ = self.robot.forward_kinematics(joints)
                error = jnp.linalg.norm(actual_pos - target_pos)
                
                # Plot target
                ax.scatter(*target_pos, color='red', s=100, marker='*', label='Target')
                ax.scatter(*actual_pos, color='green', s=100, marker='o', label='Actual')
                
                ax.set_title(f"{method_name}\nTarget {i+1}, Error: {error:.4f}m")
                ax.legend()
                self._set_equal_aspect_3d(ax)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"ik_comparison_{self.timestamp}.png"))
        self._show_or_close()
    
    def test_grasp_controller_visualization(self):
        """Visualize different grasp types"""
        print("\n=== Grasp Controller Visualization ===")
        
        grasp_ctrl = self.robot.create_advanced_grasp_controller()
        grasp_types = ['power', 'pinch', 'tripod']
        
        fig = plt.figure(figsize=(15, 5))
        
        for i, grasp_type in enumerate(grasp_types):
            # Create subplot
            ax = fig.add_subplot(1, 3, i+1)
            
            # Generate grasp sequence
            t_values = jnp.linspace(0, 1, 20)
            joint_values = []
            
            for t in t_values:
                angles = grasp_ctrl[grasp_type](t)
                joint_values.append(angles)
            
            joint_values = jnp.array(joint_values)
            
            # Plot joint trajectories
            for finger in range(4):
                finger_joints = joint_values[:, finger*4:(finger+1)*4]
                for joint in range(4):
                    ax.plot(t_values, finger_joints[:, joint], 
                           label=f"F{finger}J{joint}" if i == 0 else "")
            
            ax.set_title(f"{grasp_type.capitalize()} Grasp")
            ax.set_xlabel("Grasp Progress")
            ax.set_ylabel("Joint Angle (rad)")
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"grasp_controllers_{self.timestamp}.png"))
        self._show_or_close()
    
    def test_pickup_task_visualization(self):
        """Visualize complete pickup task with metrics"""
        print("\n=== Pickup Task Visualization ===")
        
        # Initialize
        data = self.robot.data
        initial_obj_state = self.get_object_state(data)
        initial_height = initial_obj_state['position'][2]
        
        # Setup real-time viewer if available
        if not self.headless:
            self.setup_viewer()
        
        # Trajectory storage
        trajectory = []
        metrics_history = []
        
        # Phase 1: Approach
        print("Phase 1: Approaching object...")
        object_pos = initial_obj_state['position']
        approach_pos = object_pos + jnp.array([0, 0, 0.1])  # 10cm above
        
        arm_joints = self.robot.inverse_kinematics_numeric(
            approach_pos,
            jnp.array([0.7071, 0, 0.7071, 0]),
            data.qpos[:self.robot.n_total_joints]
        )
        
        # Open hand
        grasp_ctrl = self.robot.create_advanced_grasp_controller()
        hand_joints = grasp_ctrl['adaptive'](0.1)  # Open for 10cm object
        
        control = jnp.concatenate([arm_joints, hand_joints * 0])  # Start open
        
        # Simulate approach
        for step in range(50):
            data = data.replace(ctrl=control)
            data = mjx.step(self.robot.model, data)
            trajectory.append(data)
            metrics = self.robot.multi_construct_pickup_error(data, initial_height)
            metrics_history.append(metrics)
            if not self.headless:
                self.update_viewer(data)
                time.sleep(0.02)
        
        # Phase 2: Descend and grasp
        print("Phase 2: Descending and grasping...")
        grasp_pos = object_pos + jnp.array([0, 0, -0.02])  # Slightly below
        
        arm_joints = self.robot.inverse_kinematics_numeric(
            grasp_pos,
            jnp.array([0.7071, 0, 0.7071, 0]),
            data.qpos[:self.robot.n_total_joints]
        )
        
        # Gradually close hand
        for step in range(100):
            progress = step / 100.0
            hand_joints = grasp_ctrl['adaptive'](0.08 * (1 - progress))  # Close gradually
            control = jnp.concatenate([arm_joints, hand_joints])
            
            data = data.replace(ctrl=control)
            data = mjx.step(self.robot.model, data)
            trajectory.append(data)
            metrics = self.robot.multi_construct_pickup_error(data, initial_height)
            metrics_history.append(metrics)
            if not self.headless:
                self.update_viewer(data)
                time.sleep(0.02)
        
        # Phase 3: Lift
        print("Phase 3: Lifting object...")
        lift_pos = grasp_pos + jnp.array([0, 0, 0.3])  # 30cm up
        
        arm_joints = self.robot.inverse_kinematics_numeric(
            lift_pos,
            jnp.array([0.7071, 0, 0.7071, 0]),
            data.qpos[:self.robot.n_total_joints]
        )
        
        control = control.at[:6].set(arm_joints)  # Keep hand closed
        
        for step in range(100):
            data = data.replace(ctrl=control)
            data = mjx.step(self.robot.model, data)
            trajectory.append(data)
            metrics = self.robot.multi_construct_pickup_error(data, initial_height)
            metrics_history.append(metrics)
            if not self.headless:
                self.update_viewer(data)
                time.sleep(0.02)
        
        # Save video if possible
        if not self.headless:
            self.save_trajectory_video(trajectory, "pickup_task")
        
        # Plot metrics over time
        self._plot_pickup_metrics(metrics_history)
        
        # Create 3D trajectory plot as alternative visualization
        self._plot_3d_trajectory(trajectory)
        
        # Final success assessment
        final_metrics = metrics_history[-1]
        print(f"\n=== Final Pickup Assessment ===")
        print(f"Pickup Success: {final_metrics['pickup_success']}")
        print(f"Lift Height: {final_metrics['lift_height']:.3f}m")
        print(f"Grasp Quality: {final_metrics['grasp_quality']:.2f}")
        
        return trajectory, metrics_history
    
    def _plot_3d_trajectory(self, trajectory):
        """Plot 3D trajectory of end-effector and object"""
        if not trajectory:
            return
            
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract positions over time
        ee_positions = []
        object_positions = []
        
        for data in trajectory[::5]:  # Sample every 5th frame
            # Get end-effector position
            ee_pos, _ = self.robot.get_ee_pose(data)
            ee_positions.append(ee_pos)
            
            # Get object position
            obj_state = self.get_object_state(data)
            object_positions.append(obj_state['position'])
        
        ee_positions = jnp.array(ee_positions)
        object_positions = jnp.array(object_positions)
        
        # Plot trajectories
        ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 
                'b-', linewidth=2, label='End-Effector')
        ax.plot(object_positions[:, 0], object_positions[:, 1], object_positions[:, 2], 
                'r-', linewidth=2, label='Object')
        
        # Mark key points
        ax.scatter(*ee_positions[0], color='green', s=100, marker='o', label='EE Start')
        ax.scatter(*ee_positions[-1], color='darkgreen', s=100, marker='o', label='EE End')
        ax.scatter(*object_positions[0], color='orange', s=100, marker='^', label='Object Start')
        ax.scatter(*object_positions[-1], color='red', s=100, marker='^', label='Object End')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Pickup Task Trajectory')
        ax.legend()
        
        self._set_equal_aspect_3d(ax)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"pickup_trajectory_3d_{self.timestamp}.png"))
        if not self.headless:
            plt.show()
        else:
            plt.close()
    
    def test_workspace_visualization(self):
        """Visualize robot workspace and reachability"""
        print("\n=== Workspace Visualization ===")
        
        # Sample workspace
        n_samples = 1000
        key = jax.random.PRNGKey(42)
        
        # Random joint configurations
        joint_samples = jax.random.uniform(
            key, 
            (n_samples, 6),
            minval=self.robot.joint_limits[:, 0],
            maxval=self.robot.joint_limits[:, 1]
        )
        
        # Compute end-effector positions
        ee_positions = []
        for joints in joint_samples:
            pos, _ = self.robot.forward_kinematics(joints)
            ee_positions.append(pos)
        
        ee_positions = jnp.array(ee_positions)
        
        # Create 3D scatter plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot workspace points
        scatter = ax.scatter(
            ee_positions[:, 0],
            ee_positions[:, 1], 
            ee_positions[:, 2],
            c=ee_positions[:, 2],  # Color by height
            cmap='viridis',
            alpha=0.6,
            s=1
        )
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Height (m)')
        
        # Plot robot base
        ax.scatter(0, 0, 0, color='red', s=200, marker='^', label='Robot Base')
        
        # Plot object position
        obj_pos = self.robot.initial_object_pos
        ax.scatter(*obj_pos, color='orange', s=100, marker='*', label='Object')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Robot Workspace Analysis\n(1000 random configurations)')
        ax.legend()
        
        self._set_equal_aspect_3d(ax)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"workspace_{self.timestamp}.png"), dpi=300)
        self._show_or_close()
        
        # Compute workspace statistics
        print(f"\nWorkspace Statistics:")
        print(f"X range: [{ee_positions[:, 0].min():.3f}, {ee_positions[:, 0].max():.3f}]")
        print(f"Y range: [{ee_positions[:, 1].min():.3f}, {ee_positions[:, 1].max():.3f}]")
        print(f"Z range: [{ee_positions[:, 2].min():.3f}, {ee_positions[:, 2].max():.3f}]")
        print(f"Max reach: {jnp.linalg.norm(ee_positions, axis=1).max():.3f}m")
    
    def test_jacobian_visualization(self):
        """Visualize Jacobian and manipulability"""
        print("\n=== Jacobian Analysis Visualization ===")
        
        # Test configurations
        test_configs = [
            ("Singular", jnp.array([0, 0, 0, 0, 0, 0])),
            ("Extended", jnp.array([0, -1.57, 0, 0, 0, 0])),
            ("General", jnp.array([0.5, 0.5, -0.5, 0.3, -0.3, 0.2]))
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for i, (name, joints) in enumerate(test_configs):
            # Compute Jacobian
            J = self.robot.compute_jacobian(joints)
            
            # Manipulability measure
            manipulability = jnp.sqrt(jnp.linalg.det(J @ J.T))
            
            # Singular values
            U, S, Vt = jnp.linalg.svd(J[:3, :])  # Position part only
            
            # Plot Jacobian heatmap
            ax1 = axes[0, i]
            im1 = ax1.imshow(np.array(J), cmap='RdBu', aspect='auto')
            ax1.set_title(f"{name} Configuration\nJacobian Matrix")
            ax1.set_xlabel("Joint")
            ax1.set_ylabel("Task Space")
            plt.colorbar(im1, ax=ax1)
            
            # Plot singular values
            ax2 = axes[1, i]
            ax2.bar(range(len(S)), S)
            ax2.set_title(f"Singular Values\nManipulability: {manipulability:.3f}")
            ax2.set_xlabel("Index")
            ax2.set_ylabel("Value")
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"jacobian_analysis_{self.timestamp}.png"))
        self._show_or_close()
    
    def test_trajectory_optimization_visualization(self):
        """Visualize trajectory optimization process"""
        print("\n=== Trajectory Optimization Visualization ===")
        
        # Target position
        target_pos = self.robot.initial_object_pos + jnp.array([0, 0, 0.1])
        
        # Initial trajectory (straight line in joint space)
        num_steps = 50
        initial_joints = jnp.zeros(6)
        target_joints = self.robot.inverse_kinematics_numeric(
            target_pos,
            jnp.array([0.7071, 0, 0.7071, 0]),
            initial_joints
        )
        
        # Linear interpolation as initial guess
        t = jnp.linspace(0, 1, num_steps)[:, None]
        initial_trajectory = initial_joints[None, :] * (1 - t) + target_joints[None, :] * t
        
        # Optimize trajectory
        @jax.jit
        def trajectory_cost(joint_trajectory):
            """Cost function for smooth trajectory"""
            cost = 0.0
            
            for i, joints in enumerate(joint_trajectory):
                # Position error at final step
                if i == num_steps - 1:
                    pos, _ = self.robot.forward_kinematics(joints)
                    cost += 10.0 * jnp.sum((pos - target_pos) ** 2)
                
                # Smoothness
                if i > 0:
                    cost += 0.1 * jnp.sum((joints - joint_trajectory[i-1]) ** 2)
                
                # Joint limits
                cost += 0.01 * jnp.sum(jnp.maximum(0, joints - self.robot.joint_limits[:, 1]) ** 2)
                cost += 0.01 * jnp.sum(jnp.maximum(0, self.robot.joint_limits[:, 0] - joints) ** 2)
            
            return cost
        
        # Optimize
        trajectory = initial_trajectory
        costs = []
        
        print("Optimizing trajectory...")
        for iter in range(100):
            cost, grad = jax.value_and_grad(trajectory_cost)(trajectory)
            trajectory = trajectory - 0.01 * grad
            costs.append(float(cost))
            
            if iter % 20 == 0:
                print(f"  Iteration {iter}: Cost = {cost:.4f}")
        
        # Visualize optimization
        fig = plt.figure(figsize=(15, 5))
        
        # Plot 1: Cost over iterations
        ax1 = fig.add_subplot(131)
        ax1.plot(costs)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Cost")
        ax1.set_title("Optimization Progress")
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Joint trajectories
        ax2 = fig.add_subplot(132)
        for j in range(6):
            ax2.plot(trajectory[:, j], label=f"Joint {j+1}")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Joint Angle (rad)")
        ax2.set_title("Optimized Joint Trajectories")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: End-effector path
        ax3 = fig.add_subplot(133, projection='3d')
        ee_positions = []
        for joints in trajectory:
            pos, _ = self.robot.forward_kinematics(joints)
            ee_positions.append(pos)
        ee_positions = jnp.array(ee_positions)
        
        ax3.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 'b-', linewidth=2)
        ax3.scatter(*ee_positions[0], color='green', s=100, marker='o', label='Start')
        ax3.scatter(*ee_positions[-1], color='red', s=100, marker='*', label='End')
        ax3.scatter(*target_pos, color='orange', s=100, marker='^', label='Target')
        
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.set_title("End-Effector Path")
        ax3.legend()
        self._set_equal_aspect_3d(ax3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"trajectory_optimization_{self.timestamp}.png"))
        self._show_or_close()
        
        return trajectory
    
    def test_grasp_quality_heatmap(self):
        """Visualize grasp quality as a heatmap around object"""
        print("\n=== Grasp Quality Heatmap ===")
        
        # Get object position
        data = self.robot.data
        object_pos = data.xpos[self.robot.object_body_id]
        
        # Create grid of approach positions
        n_points = 20
        radius = 0.15  # 15cm around object
        angles = jnp.linspace(0, 2*jnp.pi, n_points)
        heights = jnp.linspace(object_pos[2] - 0.1, object_pos[2] + 0.1, n_points)
        
        quality_map = np.zeros((n_points, n_points))
        
        print("Computing grasp quality map...")
        grasp_ctrl = self.robot.create_advanced_grasp_controller()
        
        for i, angle in enumerate(angles):
            for j, height in enumerate(heights):
                # Approach position
                approach_pos = object_pos + jnp.array([
                    radius * jnp.cos(angle),
                    radius * jnp.sin(angle),
                    height - object_pos[2]
                ])
                
                # Compute approach orientation (pointing toward object)
                direction = object_pos - approach_pos
                direction = direction / jnp.linalg.norm(direction)
                
                # Simple orientation (this could be improved)
                quat = jnp.array([0.7071, 0, 0.7071, 0])
                
                # Try to reach this position
                try:
                    arm_joints = self.robot.inverse_kinematics_numeric(
                        approach_pos, quat, data.qpos[:22], max_iter=20
                    )
                    
                    # Apply grasp
                    hand_joints = grasp_ctrl['adaptive'](0.08)
                    control = jnp.concatenate([arm_joints, hand_joints])
                    
                    # Simulate briefly
                    test_data = data.replace(ctrl=control)
                    for _ in range(10):
                        test_data = mjx.step(self.robot.model, test_data)
                    
                    # Compute grasp quality
                    quality = self.robot.grasp_quality_metric(test_data, object_pos)
                    quality_map[i, j] = float(quality)
                    
                except:
                    quality_map[i, j] = 0.0
        
        # Visualize heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Cylindrical heatmap
        im = ax1.imshow(quality_map.T, origin='lower', aspect='auto', cmap='hot')
        ax1.set_xlabel('Angle (rad)')
        ax1.set_ylabel('Height (m)')
        ax1.set_title('Grasp Quality Heatmap\n(Cylindrical coordinates around object)')
        
        # Set proper tick labels
        angle_labels = [f"{a:.1f}" for a in np.linspace(0, 2*np.pi, 5)]
        height_labels = [f"{h:.2f}" for h in np.linspace(heights[0], heights[-1], 5)]
        ax1.set_xticks(np.linspace(0, n_points-1, 5))
        ax1.set_xticklabels(angle_labels)
        ax1.set_yticks(np.linspace(0, n_points-1, 5))
        ax1.set_yticklabels(height_labels)
        
        plt.colorbar(im, ax=ax1, label='Grasp Quality')
        
        # 3D visualization
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Convert to 3D points
        X, Y, Z = [], [], []
        colors = []
        
        for i, angle in enumerate(angles):
            for j, height in enumerate(heights):
                x = object_pos[0] + radius * np.cos(angle)
                y = object_pos[1] + radius * np.sin(angle)
                z = height
                
                X.append(x)
                Y.append(y)
                Z.append(z)
                colors.append(quality_map[i, j])
        
        scatter = ax2.scatter(X, Y, Z, c=colors, cmap='hot', s=20, alpha=0.6)
        
        # Plot object
        ax2.scatter(*object_pos, color='blue', s=200, marker='*', label='Object')
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('Grasp Quality in 3D Space')
        self._set_equal_aspect_3d(ax2)
        
        plt.colorbar(scatter, ax=ax2, label='Grasp Quality')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"grasp_quality_heatmap_{self.timestamp}.png"))
        self._show_or_close()
        
        # Find best grasp position
        best_idx = np.unravel_index(np.argmax(quality_map), quality_map.shape)
        best_angle = angles[best_idx[0]]
        best_height = heights[best_idx[1]]
        print(f"\nBest grasp position:")
        print(f"  Angle: {best_angle:.2f} rad ({np.degrees(best_angle):.1f}°)")
        print(f"  Height: {best_height:.3f} m")
        print(f"  Quality: {quality_map[best_idx]:.3f}")
    
    def test_multi_finger_coordination(self):
        """Visualize multi-finger coordination during grasping"""
        print("\n=== Multi-Finger Coordination Test ===")
        
        # Initialize
        data = self.robot.data
        grasp_ctrl = self.robot.create_advanced_grasp_controller()
        
        # Different coordination patterns
        patterns = {
            'sequential': lambda t: jnp.array([
                grasp_ctrl['power'](jnp.clip(t*4, 0, 1)),      # Index first
                grasp_ctrl['power'](jnp.clip(t*4-1, 0, 1)),    # Then middle
                grasp_ctrl['power'](jnp.clip(t*4-2, 0, 1)),    # Then ring
                grasp_ctrl['power'](jnp.clip(t*4-3, 0, 1))     # Finally thumb
            ]).reshape(-1),
            
            'synchronized': lambda t: grasp_ctrl['power'](t),
            
            'opposition': lambda t: jnp.concatenate([
                grasp_ctrl['power'](t)[:12],      # Three fingers
                grasp_ctrl['power'](t*1.5)[12:]   # Thumb faster
            ])
        }
        
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        
        for idx, (pattern_name, pattern_fn) in enumerate(patterns.items()):
            print(f"\nTesting {pattern_name} pattern...")
            
            # Generate trajectory
            trajectory = []
            fingertip_positions = []
            
            for t in jnp.linspace(0, 1, 50):
                hand_joints = pattern_fn(t)
                
                # Update only hand joints
                control = jnp.concatenate([data.qpos[:6], hand_joints])
                test_data = data.replace(qpos=control)
                
                # Get fingertip positions
                palm_pos = test_data.xpos[self.robot.mj_model.body('palm').id]
                finger_tips = []
                
                for finger_id in range(4):
                    finger_joints = hand_joints[finger_id*4:(finger_id+1)*4]
                    tip_pos = self.robot.finger_forward_kinematics(finger_joints, finger_id)[-1]
                    finger_tips.append(palm_pos + tip_pos)
                
                fingertip_positions.append(finger_tips)
                trajectory.append(hand_joints)
            
            trajectory = jnp.array(trajectory)
            fingertip_positions = jnp.array(fingertip_positions)
            
            # Plot joint angles
            ax1 = axes[idx, 0]
            for finger in range(4):
                finger_data = trajectory[:, finger*4:(finger+1)*4]
                for joint in range(4):
                    ax1.plot(finger_data[:, joint], 
                            label=f"F{finger}J{joint}" if idx == 0 and finger < 2 else "")
            
            ax1.set_title(f"{pattern_name.capitalize()} Pattern - Joint Angles")
            ax1.set_xlabel("Time Step")
            ax1.set_ylabel("Joint Angle (rad)")
            ax1.grid(True, alpha=0.3)
            if idx == 0:
                ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Plot fingertip trajectories
            ax2 = axes[idx, 1]
            colors = ['red', 'green', 'blue', 'orange']
            labels = ['Index', 'Middle', 'Ring', 'Thumb']
            
            for finger in range(4):
                # Distance to object over time
                distances = jnp.linalg.norm(
                    fingertip_positions[:, finger] - data.xpos[self.robot.object_body_id],
                    axis=1
                )
                ax2.plot(distances, color=colors[finger], label=labels[finger])
            
            ax2.set_title(f"{pattern_name.capitalize()} Pattern - Fingertip Distances")
            ax2.set_xlabel("Time Step")
            ax2.set_ylabel("Distance to Object (m)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"finger_coordination_{self.timestamp}.png"))
        self._show_or_close()
    
    def run_full_test_suite(self):
        """Run all visualization tests"""
        print("="*60)
        print("ROBOT ARM ALLEGRO VISUALIZATION TEST SUITE")
        print("="*60)
        
        tests = [
            ("Forward Kinematics", self.test_forward_kinematics_visualization),
            ("IK Comparison", self.test_ik_comparison_visualization),
            ("Grasp Controllers", self.test_grasp_controller_visualization),
            ("Workspace Analysis", self.test_workspace_visualization),
            ("Jacobian Analysis", self.test_jacobian_visualization),
            ("Trajectory Optimization", self.test_trajectory_optimization_visualization),
            ("Grasp Quality Heatmap", self.test_grasp_quality_heatmap),
            ("Multi-Finger Coordination", self.test_multi_finger_coordination),
            ("Pickup Task", self.test_pickup_task_visualization)
        ]
        
        results = []
        
        for test_name, test_fn in tests:
            print(f"\n{'='*60}")
            print(f"Running: {test_name}")
            print(f"{'='*60}")
            
            try:
                start_time = time.time()
                test_fn()
                duration = time.time() - start_time
                results.append((test_name, "PASSED", duration))
                print(f"\n✓ {test_name} completed in {duration:.2f}s")
            except Exception as e:
                results.append((test_name, "FAILED", str(e)))
                print(f"\n✗ {test_name} failed: {str(e)}")
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        for test_name, status, info in results:
            if status == "PASSED":
                print(f"✓ {test_name:<30} {info:.2f}s")
            else:
                print(f"✗ {test_name:<30} {info}")
        
        passed = sum(1 for _, status, _ in results if status == "PASSED")
        print(f"\nTotal: {passed}/{len(tests)} tests passed")
        
        # Close viewer if open
        if self.viewer is not None:
            try:
                self.viewer.close()
            except:
                pass
    
    # Helper methods
    def _plot_kinematic_chain(self, ax, joint_angles):
        """Plot the kinematic chain for given joint angles"""
        # Compute forward kinematics for each link
        positions = [jnp.array([0, 0, 0])]  # Base
        
        T = jnp.eye(4)
        for i in range(6):
            a, alpha, d, theta_offset = self.robot.dh_params[i]
            theta = joint_angles[i] + theta_offset
            T_i = self.robot.dh_matrix(a, alpha, d, theta)
            T = T @ T_i
            positions.append(T[:3, 3])
        
        positions = jnp.array(positions)
        
        # Plot links
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'b-o', linewidth=2, markersize=8)
        
        # Plot base
        ax.scatter(0, 0, 0, color='red', s=100, marker='^')
        
        # Plot end-effector
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                  color='green', s=100, marker='o')
        
        # Set limits
        ax.set_xlim([-0.8, 0.8])
        ax.set_ylim([-0.8, 0.8])
        ax.set_zlim([0, 1.2])
    
    def _set_equal_aspect_3d(self, ax):
        """Set equal aspect ratio for 3D plot"""
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()
        
        max_range = max(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]) / 2
        
        mid_x = (xlim[1] + xlim[0]) / 2
        mid_y = (ylim[1] + ylim[0]) / 2
        mid_z = (zlim[1] + zlim[0]) / 2
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    def _plot_pickup_metrics(self, metrics_history):
        """Plot pickup metrics over time"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Extract metric arrays
        times = np.arange(len(metrics_history))
        lift_heights = [m['lift_height'] for m in metrics_history]
        proximity_errors = [m['proximity_error'] for m in metrics_history]
        grasp_qualities = [m['grasp_quality'] for m in metrics_history]
        object_speeds = [m['object_speed'] for m in metrics_history]
        close_fingers = [m['close_fingers_count'] for m in metrics_history]
        pickup_success = [float(m['pickup_success']) for m in metrics_history]
        
        # Plot 1: Lift height
        ax = axes[0, 0]
        ax.plot(times, lift_heights, 'b-', linewidth=2)
        ax.axhline(y=0.05, color='g', linestyle='--', label='Success threshold')
        ax.set_title('Object Lift Height')
        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Height (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Palm-object distance
        ax = axes[0, 1]
        ax.plot(times, proximity_errors, 'r-', linewidth=2)
        ax.axhline(y=0.08, color='g', linestyle='--', label='Success threshold')
        ax.set_title('Palm-Object Distance')
        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Distance (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Grasp quality
        ax = axes[0, 2]
        ax.plot(times, grasp_qualities, 'g-', linewidth=2)
        ax.axhline(y=2.0, color='g', linestyle='--', label='Good grasp')
        ax.set_title('Grasp Quality Score')
        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Quality')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Object speed
        ax = axes[1, 0]
        ax.plot(times, object_speeds, 'm-', linewidth=2)
        ax.axhline(y=0.5, color='g', linestyle='--', label='Stability threshold')
        ax.set_title('Object Speed')
        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Speed (m/s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Fingers in contact
        ax = axes[1, 1]
        ax.plot(times, close_fingers, 'c-', linewidth=2)
        ax.axhline(y=3, color='g', linestyle='--', label='Min required')
        ax.set_title('Fingers in Contact')
        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Count')
        ax.set_ylim(0, 4.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Overall success
        ax = axes[1, 2]
        ax.plot(times, pickup_success, 'k-', linewidth=3)
        ax.fill_between(times, 0, pickup_success, alpha=0.3, color='green')
        ax.set_title('Pickup Success')
        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Success (0/1)')
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"pickup_metrics_{self.timestamp}.png"))
        self._show_or_close()


# Example usage and test execution
if __name__ == "__main__":
    # Import the robot model (assuming it's saved as robot_arm_allegro.py)
    # Note: You'll need to save the robot model code in a file named robot_arm_allegro.py
    # or adjust the import accordingly
    from model import RobotArmAllegro
    
    # Create robot
    print("Creating robot model...")
    robot = RobotArmAllegro()
    
    # Create visualization suite
    # For headless environments (no display), it will automatically detect and adapt
    # You can also explicitly set headless=True
    viz_suite = RobotArmVisualizationSuite(robot, save_dir="visualization_results")
    
    if viz_suite.headless:
        print("\nRunning in headless mode - visualizations will be saved to files only")
        print("Real-time viewer and video rendering are disabled")
    
    # Run specific tests or full suite
    print("\nSelect test to run:")
    print("1. Forward Kinematics Visualization")
    print("2. IK Methods Comparison")
    print("3. Grasp Controller Visualization")
    print("4. Workspace Analysis")
    print("5. Jacobian Analysis")
    print("6. Trajectory Optimization")
    print("7. Grasp Quality Heatmap")
    print("8. Multi-Finger Coordination")
    print("9. Pickup Task Demo")
    print("0. Run Full Test Suite")
    
    choice = input("\nEnter choice (0-9): ")
    
    if choice == "1":
        viz_suite.test_forward_kinematics_visualization()
    elif choice == "2":
        viz_suite.test_ik_comparison_visualization()
    elif choice == "3":
        viz_suite.test_grasp_controller_visualization()
    elif choice == "4":
        viz_suite.test_workspace_visualization()
    elif choice == "5":
        viz_suite.test_jacobian_visualization()
    elif choice == "6":
        viz_suite.test_trajectory_optimization_visualization()
    elif choice == "7":
        viz_suite.test_grasp_quality_heatmap()
    elif choice == "8":
        viz_suite.test_multi_finger_coordination()
    elif choice == "9":
        viz_suite.test_pickup_task_visualization()
    elif choice == "0":
        viz_suite.run_full_test_suite()
    else:
        print("Invalid choice. Running full test suite...")
        viz_suite.run_full_test_suite()
    
    print("\nAll tests completed! Check the visualization_results folder for outputs.")