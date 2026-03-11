#!/usr/bin/env python3
"""
Execute grasps from POGS/ContactGraspNet on OpenManipulator X.

This script:
1. Loads generated grasps from ContactGraspNet output
2. Transforms from camera frame back to POGS world frame
3. Projects to 4-DOF for OMX constraints
4. Computes IK and executes on robot

Usage:
    python execute_omx_grasp.py --grasp-dir outputs/my_scene/pogs/2026-01-25_223408/grasps --dry-run
    python execute_omx_grasp.py --grasp-dir outputs/my_scene/pogs/2026-01-25_223408/grasps --execute
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

# Add POGS to path
POGS_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(POGS_ROOT))


def load_grasps(grasp_dir: Path):
    """Load grasps from ContactGraspNet output."""
    grasp_poses_file = grasp_dir / "grasp_poses.npy"
    grasp_scores_file = grasp_dir / "grasp_scores.npy"
    
    if not grasp_poses_file.exists():
        print(f"[ERROR] Grasp poses file not found: {grasp_poses_file}")
        return None, None
    
    grasp_poses = np.load(grasp_poses_file)
    grasp_scores = np.load(grasp_scores_file) if grasp_scores_file.exists() else None
    
    print(f"[INFO] Loaded {len(grasp_poses)} grasps")
    if grasp_scores is not None:
        print(f"[INFO] Score range: [{grasp_scores.min():.3f}, {grasp_scores.max():.3f}]")
    
    return grasp_poses, grasp_scores


def transform_to_robot_frame(grasp_cam: np.ndarray, scale_factor: float = 0.0126) -> np.ndarray:
    """
    Transform grasp from ContactGraspNet camera frame to robot base frame.
    
    ContactGraspNet outputs grasps in a camera frame where:
    - Point cloud was centered at origin
    - Scaled to ~0.5m extent
    - Z is depth (forward from camera)
    
    For OpenManipulator X:
    - Workspace is ~0.3m radius from base
    - Z is up
    - Assumes camera looking down at tabletop scene
    """
    # Extract position from grasp pose (in camera frame)
    # Camera frame: X right, Y down, Z forward (depth)
    x_cam = grasp_cam[0, 3]
    y_cam = grasp_cam[1, 3]  
    z_cam = grasp_cam[2, 3]
    
    # The ContactGraspNet grasps are in the processed camera frame:
    # - Z was shifted to [0.6, 0.8] range
    # - Scene was scaled to ~0.5m extent
    # We need to map this to the OMX workspace
    
    # OMX workspace parameters (meters)
    OMX_REACH = 0.35  # max reach from base center
    OMX_TABLE_Z = 0.0  # table height relative to robot base (0 = at base level)
    
    # Map camera Z (depth) to robot X (forward)
    # Camera Z was in range ~[0.6, 0.8], map to robot forward reach
    x_robot = (z_cam - 0.6) * 0.5 + 0.25  # Scale to [0.15, 0.35]m forward
    
    # Camera X (right) maps to robot Y (left/right)
    y_robot = -x_cam  # Flip X to Y
    
    # Camera Y (down) maps to robot Z (up)  
    z_robot = -y_cam + 0.15  # Flip and offset for grasp height
    
    # Clamp to OMX workspace limits
    x_robot = np.clip(x_robot, 0.15, OMX_REACH)
    y_robot = np.clip(y_robot, -OMX_REACH/2, OMX_REACH/2)
    z_robot = np.clip(z_robot, 0.0, 0.25)  # Reasonable grasp height
    
    print(f"[DEBUG] Camera frame: x={x_cam:.3f}, y={y_cam:.3f}, z={z_cam:.3f}")
    print(f"[DEBUG] Robot frame: x={x_robot:.3f}, y={y_robot:.3f}, z={z_robot:.3f}")
    
    # Create robot frame pose with gripper pointing down
    grasp_robot = np.eye(4)
    grasp_robot[:3, 3] = [x_robot, y_robot, z_robot]
    
    # Gripper orientation: pointing down (Z axis down in gripper frame)
    # For OMX with end-effector, typical down-grasp orientation
    # Use a proper right-handed rotation (rotate 180 deg about X axis)
    grasp_robot[:3, :3] = np.array([
        [1, 0,  0],   # X unchanged
        [0, -1, 0],   # Y flipped
        [0, 0, -1]    # Z flipped (pointing down)
    ])
    
    return grasp_robot


def connect_robot(dry_run: bool = True):
    """Connect to OpenManipulator X."""
    if dry_run:
        print("[DRY RUN] Simulating robot connection")
        return None
    
    try:
        from pogs.controller.open_manipulator import OpenManipulatorLeRobot
        
        robot = OpenManipulatorLeRobot(
            port="/dev/ttyUSB0",
            robot_id="lead",
            use_leader_ids=True,
            include_gripper=True,
        )
        print("[INFO] Robot connected successfully")
        return robot
    except Exception as e:
        print(f"[ERROR] Failed to connect to robot: {e}")
        return None


def execute_grasp_sequence(robot, grasp_pose: np.ndarray, dry_run: bool = True):
    """
    Execute a pick operation:
    1. Move to pre-grasp position (above target)
    2. Open gripper
    3. Move down to grasp position
    4. Close gripper
    5. Lift up
    """
    from pogs.grasping.omx_grasp_executor import OMXGraspExecutor, project_pose_to_4dof
    
    executor = OMXGraspExecutor(robot=robot)
    
    # Pre-grasp: 5cm above the grasp
    pre_grasp = grasp_pose.copy()
    pre_grasp[2, 3] += 0.05  # 5cm above
    
    # Lift position: 10cm above grasp
    lift_pose = grasp_pose.copy()
    lift_pose[2, 3] += 0.10
    
    print(f"\n[INFO] Grasp sequence:")
    print(f"  Grasp position: {grasp_pose[:3, 3]}")
    print(f"  Pre-grasp position: {pre_grasp[:3, 3]}")
    print(f"  Lift position: {lift_pose[:3, 3]}")
    
    # Project to 4-DOF and compute IK
    pose_4dof = project_pose_to_4dof(grasp_pose, fixed_pitch=1.57)  # Gripper down
    print(f"\n[INFO] 4-DOF projected pose:\n{pose_4dof}")
    
    joints = executor.grasp_to_joints(grasp_pose, fixed_pitch=1.57)
    if joints is not None:
        print(f"[INFO] IK solution (radians): {joints}")
        print(f"[INFO] IK solution (degrees): {np.degrees(joints)}")
    else:
        print("[WARNING] IK failed for this grasp pose")
        return False
    
    if dry_run:
        print("\n[DRY RUN] Would execute:")
        print("  1. Move to pre-grasp position")
        print("  2. Open gripper")
        print("  3. Move to grasp position")
        print("  4. Close gripper")
        print("  5. Lift up")
        return True
    
    # Actually execute on robot
    try:
        print("\n[EXECUTING] Moving to pre-grasp...")
        pre_joints = executor.grasp_to_joints(pre_grasp, fixed_pitch=1.57)
        if pre_joints is not None:
            robot.move_joint(pre_joints)
            time.sleep(1.0)
        
        print("[EXECUTING] Opening gripper...")
        robot.gripper.open()
        time.sleep(0.5)
        
        print("[EXECUTING] Moving to grasp position...")
        robot.move_joint(joints)
        time.sleep(1.0)
        
        print("[EXECUTING] Closing gripper...")
        robot.gripper.close()
        time.sleep(0.5)
        
        print("[EXECUTING] Lifting...")
        lift_joints = executor.grasp_to_joints(lift_pose, fixed_pitch=1.57)
        if lift_joints is not None:
            robot.move_joint(lift_joints)
            time.sleep(1.0)
        
        print("[SUCCESS] Grasp executed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Grasp execution failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Execute grasps on OpenManipulator X")
    parser.add_argument("--grasp-dir", type=str, required=True,
                        help="Directory containing ContactGraspNet output")
    parser.add_argument("--grasp-idx", type=int, default=0,
                        help="Index of grasp to execute (default: best grasp)")
    parser.add_argument("--execute", action="store_true",
                        help="Actually execute on robot (default: dry run)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Dry run mode (no robot motion)")
    parser.add_argument("--scale-factor", type=float, default=0.0126,
                        help="Scale factor used during grasp generation")
    args = parser.parse_args()
    
    grasp_dir = Path(args.grasp_dir)
    dry_run = not args.execute or args.dry_run
    
    print("="*60)
    print("POGS Grasp Execution for OpenManipulator X")
    print("="*60)
    print(f"Grasp directory: {grasp_dir}")
    print(f"Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
    print()
    
    # Load grasps
    grasp_poses, grasp_scores = load_grasps(grasp_dir)
    if grasp_poses is None:
        return 1
    
    # Select grasp
    if grasp_scores is not None:
        sorted_indices = np.argsort(grasp_scores)[::-1]
        grasp_idx = sorted_indices[args.grasp_idx]
        print(f"\n[INFO] Selected grasp #{grasp_idx} (rank {args.grasp_idx+1})")
        print(f"       Score: {grasp_scores[grasp_idx]:.4f}")
    else:
        grasp_idx = args.grasp_idx
    
    grasp_cam = grasp_poses[grasp_idx]
    print(f"\n[INFO] Grasp pose (camera frame):\n{grasp_cam}")
    
    # Transform to robot frame
    grasp_robot = transform_to_robot_frame(grasp_cam, scale_factor=args.scale_factor)
    print(f"\n[INFO] Grasp pose (robot frame):\n{grasp_robot}")
    
    # Connect to robot
    robot = connect_robot(dry_run=dry_run)
    
    # Execute grasp
    import time
    success = execute_grasp_sequence(robot, grasp_robot, dry_run=dry_run)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
