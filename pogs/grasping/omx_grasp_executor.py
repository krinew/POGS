"""
Integrate POGS grasp generation with 4-DOF OpenManipulator X control.

Handles:
1. 6-DOF grasp from grasp net (via POGS pipeline)
2. Project to 4-DOF pose (fixed roll/pitch, free yaw)
3. Compute IK using Pinocchio (via LeRobot)
4. Execute joint commands on robot
"""

import numpy as np
import math
from typing import Optional, Dict, Tuple
from scipy.spatial.transform import Rotation as R


def project_pose_to_4dof(
    pose_6dof: np.ndarray,
    fixed_roll: float = 0.0,
    fixed_pitch: float | None = None,
    pitch_range: Tuple[float, float] = (-1.57, 1.57),
) -> np.ndarray:
    """
    Project 6-DoF SE(3) pose to 4-DoF by constraining roll/pitch.
    
    Args:
        pose_6dof: 4x4 SE(3) matrix
        fixed_roll: Roll angle (default 0)
        fixed_pitch: Pitch angle (if None, clamp to pitch_range)
        pitch_range: (min, max) pitch bounds
    
    Returns:
        4x4 pose with constrained orientation
    """
    pose_6dof = np.array(pose_6dof)
    t = pose_6dof[:3, 3]
    rot = pose_6dof[:3, :3]
    yaw, pitch, _roll = R.from_matrix(rot).as_euler('zyx')
    
    if fixed_pitch is None:
        pitch = float(np.clip(pitch, pitch_range[0], pitch_range[1]))
    else:
        pitch = float(fixed_pitch)
    
    new_rot = R.from_euler('zyx', [yaw, pitch, fixed_roll]).as_matrix()
    pose_4dof = np.eye(4)
    pose_4dof[:3, :3] = new_rot
    pose_4dof[:3, 3] = t
    return pose_4dof


class OMXGraspExecutor:
    """
    Execute POGS grasps on OpenManipulator X using LeRobot IK.
    
    Workflow:
        1. Receive 6-DOF grasp from grasp net
        2. Project to 4-DOF (respects OMX constraints)
        3. Compute IK using Pinocchio kinematics
        4. Send joint commands to robot
    """
    
    # OMX link lengths (meters)
    L1 = 0.077   # Base to shoulder_lift
    L2 = 0.130   # shoulder_lift to elbow_flex
    L3 = 0.124   # elbow_flex to wrist_flex
    L4 = 0.126   # wrist_flex to TCP
    
    def __init__(self, robot=None, use_pinocchio: bool = True):
        """
        Args:
            robot: LeRobot OmxFollower instance (for actual control)
            use_pinocchio: If True, use Pinocchio IK; else use geometric fallback
        """
        self.robot = robot
        self.use_pinocchio = use_pinocchio and self._has_pinocchio()
        
        if self.use_pinocchio:
            self._init_pinocchio()
        else:
            print("[OMXGraspExecutor] Pinocchio unavailable; using geometric IK fallback")
    
    def _has_pinocchio(self) -> bool:
        """Check if Pinocchio is available."""
        try:
            import pinocchio
            return True
        except ImportError:
            return False
    
    def _init_pinocchio(self):
        """Initialize Pinocchio with OMX URDF model."""
        try:
            import pinocchio as pin
            self.pin = pin
            # Load OMX URDF from LeRobot (if available)
            # Fallback: use hardcoded DH parameters
            self._build_pinocchio_model()
        except Exception as e:
            print(f"[OMXGraspExecutor] Pinocchio init failed: {e}")
            self.use_pinocchio = False
    
    def _build_pinocchio_model(self):
        """Build OMX kinematics model using Pinocchio."""
        try:
            import pinocchio as pin
            # Create a simple chain for OMX (4 joints)
            self.model = pin.Model()
            self.data = self.model.createData()
            
            # Add joints with DH params
            # This is simplified; real model uses URDF
            # For now, set up frame for IK reference
            self.tcp_frame_id = None  # Will use end-effector as reference
        except Exception as e:
            print(f"[OMXGraspExecutor] Pinocchio model build failed: {e}")
    
    def grasp_to_joints(
        self,
        grasp_6dof: np.ndarray,
        current_joints: Optional[np.ndarray] = None,
        fixed_pitch: float = 1.57,
    ) -> Optional[np.ndarray]:
        """
        Convert 6-DOF grasp to 4-DOF joint commands.
        
        Args:
            grasp_6dof: 4x4 SE(3) grasp pose from grasp net
            current_joints: Current joint state [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex]
            fixed_pitch: Desired pitch angle (down=-1.57, up=1.57)
        
        Returns:
            4-element joint array or None if IK fails
        """
        # Step 1: Project to 4-DOF
        pose_4dof = project_pose_to_4dof(grasp_6dof, fixed_pitch=fixed_pitch)
        
        # Step 2: Solve IK
        if self.use_pinocchio:
            joints = self._ik_pinocchio(pose_4dof, current_joints)
        else:
            joints = self._ik_geometric(pose_4dof)
        
        return joints
    
    def _ik_pinocchio(
        self,
        target_pose: np.ndarray,
        current_joints: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Inverse kinematics using Pinocchio.
        
        Args:
            target_pose: 4x4 target SE(3) pose
            current_joints: Initial guess for IK
        
        Returns:
            4-element joint array or None
        """
        try:
            import pinocchio as pin
            
            # Placeholder: Real implementation needs proper Pinocchio setup
            # For now, fall back to geometric IK
            return self._ik_geometric(target_pose)
        except Exception as e:
            print(f"[OMXGraspExecutor] Pinocchio IK failed: {e}")
            return self._ik_geometric(target_pose)
    
    def _ik_geometric(self, target_pose: np.ndarray) -> Optional[np.ndarray]:
        """
        Geometric inverse kinematics for 4-DOF OMX.
        
        Based on arm configuration: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex.
        """
        x = target_pose[0, 3]
        y = target_pose[1, 3]
        z = target_pose[2, 3]
        
        # Joint 1: yaw (shoulder_pan)
        q1 = math.atan2(y, x)
        
        # Project to (r, z) plane
        r = math.sqrt(x*x + y*y)
        
        # Extract pitch from target rotation
        ax = target_pose[0, 2]  # approach vector (Z-axis)
        ay = target_pose[1, 2]
        az = target_pose[2, 2]
        ar = ax * math.cos(q1) + ay * math.sin(q1)
        phi = math.atan2(az, ar)  # end-effector pitch
        
        # Wrist (Joint 4) position
        rw = r - self.L4 * math.cos(phi)
        zw = z - self.L4 * math.sin(phi)
        
        # Joint 2 is at (0, L1) in (r, z) frame
        ro, zo = 0, self.L1
        
        # Vector from shoulder_lift to wrist
        dr = rw - ro
        dz = zw - zo
        d_sq = dr*dr + dz*dz
        d = math.sqrt(d_sq)
        
        if d > (self.L2 + self.L3) or d < abs(self.L2 - self.L3):
            return None  # Out of reach
        
        # Law of cosines for Joint 3 (elbow)
        cos_q3 = (d_sq - self.L2**2 - self.L3**2) / (2 * self.L2 * self.L3)
        cos_q3 = np.clip(cos_q3, -1.0, 1.0)
        q3 = math.acos(cos_q3)
        
        # Joint 2 calculation
        beta = math.atan2(dz, dr)
        psi = math.atan2(self.L3 * math.sin(q3), self.L2 + self.L3 * math.cos(q3))
        q2 = beta - psi
        
        # Joint 4 (wrist)
        q4 = phi - q2 - q3
        
        # Convert to OMX convention (q2 ref: upright = pi/2 - calculated angle)
        q2_omx = (math.pi / 2) - q2
        q3_omx = -q3
        
        return np.array([q1, q2_omx, q3_omx, q4])
    
    def execute_grasp(
        self,
        grasp_6dof: np.ndarray,
        current_joints: Optional[np.ndarray] = None,
        fixed_pitch: float = 1.57,
        vel: float = 1.0,
        acc: float = 0.1,
    ) -> bool:
        """
        Execute grasp on robot.
        
        Args:
            grasp_6dof: 6-DOF grasp pose from grasp net
            current_joints: Current joint state
            fixed_pitch: Desired pitch angle
            vel: Motion speed (0-1)
            acc: Motion acceleration
        
        Returns:
            True if successful, False otherwise
        """
        if self.robot is None:
            print("[OMXGraspExecutor] Robot not connected; dry-run mode")
            return False
        
        joints = self.grasp_to_joints(grasp_6dof, current_joints, fixed_pitch)
        if joints is None:
            print("[OMXGraspExecutor] IK failed for grasp pose")
            return False
        
        try:
            self.robot.move_joint(joints, vel=vel, acc=acc)
            return True
        except Exception as e:
            print(f"[OMXGraspExecutor] Robot motion failed: {e}")
            return False
    
    def batch_grasps(
        self,
        grasps_6dof: np.ndarray,
        fixed_pitch: float = 1.57,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert batch of 6-DOF grasps to 4-DOF joints.
        
        Args:
            grasps_6dof: (N, 4, 4) array of SE(3) poses
            fixed_pitch: Desired pitch
        
        Returns:
            (joints, mask) where:
            - joints: (N, 4) array of joint configs
            - mask: (N,) bool array indicating successful IK solutions
        """
        n_grasps = grasps_6dof.shape[0]
        joints = np.full((n_grasps, 4), np.nan)
        mask = np.zeros(n_grasps, dtype=bool)
        
        for i, grasp in enumerate(grasps_6dof):
            sol = self.grasp_to_joints(grasp, fixed_pitch=fixed_pitch)
            if sol is not None:
                joints[i] = sol
                mask[i] = True
        
        return joints, mask
