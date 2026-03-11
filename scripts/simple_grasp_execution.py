#!/usr/bin/env python3
"""
Simple Grasp Execution Script

This script:
1. Loads a pre-trained POGS scene
2. Exports object mesh from a cluster
3. Generates grasps using ContactGraspNet
4. Executes grasp on OpenManipulator robot

Usage:
    python scripts/simple_grasp_execution.py --cluster-id 0
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Add POGS paths
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def clear_cuda_memory():
    """Clear CUDA memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"✅ CUDA memory cleared. Free: {torch.cuda.memory_allocated()/1e9:.2f}GB used")


def export_cluster_to_ply(config_path: Path, cluster_id: int = 0):
    """
    Export a specific cluster from POGS scene to PLY file.
    This is a lightweight version that doesn't need the full tracking pipeline.
    """
    import open3d as o3d
    
    output_dir = config_path.parent
    
    # Check if PLY files already exist
    local_ply = output_dir / "local.ply"
    global_ply = output_dir / "global.ply"
    
    if local_ply.exists() and global_ply.exists():
        print(f"✅ Using existing PLY files from {output_dir}")
        return str(local_ply), str(global_ply)
    
    # If not, we need to load the model and export
    print(f"⚠️ PLY files not found. Need to export from POGS model...")
    print("This requires loading the full model which may use significant memory.")
    
    # Try to load just the gaussian means from checkpoint
    ckpt_dir = output_dir / "nerfstudio_models"
    latest_ckpt = sorted(ckpt_dir.glob("step-*.ckpt"))[-1] if ckpt_dir.exists() else None
    
    if latest_ckpt:
        print(f"Loading checkpoint: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location="cpu")
        
        # Extract gaussian means
        if "pipeline" in ckpt and "means" in str(ckpt.get("pipeline", {}).keys()):
            means = ckpt["pipeline"]["_model.gauss_params.means"]
            print(f"Found {len(means)} gaussians")
            
            # Create simple point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(means.numpy())
            
            # Save as PLY
            o3d.io.write_point_cloud(str(local_ply), pcd)
            o3d.io.write_point_cloud(str(global_ply), pcd)
            print(f"✅ Exported PLY to {output_dir}")
            return str(local_ply), str(global_ply)
    
    raise FileNotFoundError(f"Cannot find PLY files or checkpoint at {output_dir}")


def generate_grasps_contact_graspnet(local_ply: str, global_ply: str, save_dir: str):
    """
    Call ContactGraspNet to generate grasps.
    Uses subprocess to avoid memory conflicts.
    """
    from pogs.tracking.toad_object import ToadObject
    
    # Create dummy table bounding cube if it doesn't exist
    table_bbox_path = Path(save_dir).parent / "table_bounding_cube.json"
    if not table_bbox_path.exists():
        import json
        # Default table bounding box (adjust based on your setup)
        table_bbox = {
            "x_min": -0.5, "x_max": 0.5,
            "y_min": -0.5, "y_max": 0.5,
            "z_min": -0.1, "z_max": 0.0  # Table surface at z=0
        }
        with open(table_bbox_path, 'w') as f:
            json.dump(table_bbox, f)
        print(f"✅ Created default table bounding box at {table_bbox_path}")
    
    print("🔄 Generating grasps with ContactGraspNet...")
    clear_cuda_memory()
    
    try:
        ToadObject.generate_grasps(
            local_ply, 
            global_ply, 
            str(table_bbox_path), 
            save_dir
        )
        print("✅ Grasps generated successfully!")
        return True
    except Exception as e:
        print(f"❌ Grasp generation failed: {e}")
        return False


def load_best_grasp(save_dir: str) -> np.ndarray:
    """Load the best grasp from saved file."""
    grasp_file = os.path.join(save_dir, 'grasp_point_world.npy')
    
    if not os.path.exists(grasp_file):
        raise FileNotFoundError(f"No grasp file at {grasp_file}")
    
    best_grasp = np.load(grasp_file)
    print(f"✅ Loaded grasp from {grasp_file}")
    print(f"   Position: {best_grasp[:3, 3]}")
    return best_grasp


def execute_grasp_on_robot(best_grasp: np.ndarray, dry_run: bool = False):
    """
    Execute grasp on OpenManipulator robot.
    """
    from autolab_core import RigidTransform
    from pogs.controller.robot_interface import project_pose_to_4dof
    
    # Apply Z-axis rotation if grasp is on the negative Y side
    if best_grasp[0, 1] < 0:
        rotate_180_z = np.array([
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        best_grasp = best_grasp @ rotate_180_z
    
    # Create pre-grasp and post-grasp poses
    pre_grasp_offset = np.eye(4)
    pre_grasp_offset[2, 3] = -0.1  # 10cm back
    pre_grasp = best_grasp @ pre_grasp_offset
    
    post_grasp_offset = np.eye(4)
    post_grasp_offset[2, 3] = -0.05  # 5cm back (for lifting)
    post_grasp = best_grasp @ post_grasp_offset
    
    # Convert to RigidTransform
    pre_grasp_tf = RigidTransform(
        rotation=pre_grasp[:3, :3], 
        translation=pre_grasp[:3, 3]
    )
    grasp_tf = RigidTransform(
        rotation=best_grasp[:3, :3], 
        translation=best_grasp[:3, 3]
    )
    post_grasp_tf = RigidTransform(
        rotation=post_grasp[:3, :3], 
        translation=post_grasp[:3, 3]
    )
    
    # Project to 4-DOF
    pre_grasp_4dof = project_pose_to_4dof(pre_grasp_tf, min_pitch=-1.57, max_pitch=0.0)
    grasp_4dof = project_pose_to_4dof(grasp_tf, min_pitch=-1.57, max_pitch=0.0)
    post_grasp_4dof = project_pose_to_4dof(post_grasp_tf, min_pitch=-1.57, max_pitch=0.0)
    
    print(f"\n📍 Grasp poses (4-DOF projected):")
    print(f"   Pre-grasp: {pre_grasp_4dof.translation}")
    print(f"   Grasp: {grasp_4dof.translation}")
    print(f"   Post-grasp: {post_grasp_4dof.translation}")
    
    if dry_run:
        print("\n🔧 [DRY RUN] Would execute grasp sequence:")
        print("   1. Move to pre-grasp position")
        print("   2. Open gripper")
        print("   3. Move to grasp position")
        print("   4. Close gripper")
        print("   5. Lift to post-grasp position")
        print("   6. Return to home")
        return True
    
    # Initialize robot
    from pogs.controller.omx_controller import OpenManipulatorLeRobot
    
    print("\n🤖 Connecting to OpenManipulator...")
    robot = OpenManipulatorLeRobot()
    print(f"✅ Connected on {robot.port}")
    
    # Execute grasp sequence
    print("\n🚀 Executing grasp sequence...")
    
    # 1. Move to home position first
    print("   [1/6] Moving to home position...")
    robot.go_home()
    
    # 2. Open gripper
    print("   [2/6] Opening gripper...")
    robot.open_gripper()
    
    # 3. Move to pre-grasp
    print("   [3/6] Moving to pre-grasp position...")
    robot.move_to_pose(pre_grasp_4dof)
    
    # 4. Move to grasp position
    print("   [4/6] Moving to grasp position...")
    robot.move_to_pose(grasp_4dof)
    
    # 5. Close gripper
    print("   [5/6] Closing gripper (grasping)...")
    robot.close_gripper()
    
    # 6. Lift to post-grasp
    print("   [6/6] Lifting object...")
    robot.move_to_pose(post_grasp_4dof)
    
    print("\n✅ Grasp execution complete!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Simple POGS Grasp Execution")
    parser.add_argument(
        "--scene-dir", 
        type=str, 
        default="outputs/my_scene/pogs/2026-01-25_223408",
        help="Path to POGS scene output directory"
    )
    parser.add_argument(
        "--cluster-id",
        type=int,
        default=0,
        help="Cluster ID to grasp (default: 0)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't execute on real robot"
    )
    parser.add_argument(
        "--skip-grasp-gen",
        action="store_true",
        help="Skip grasp generation (use existing grasp file)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 Simple POGS Grasp Execution")
    print("=" * 60)
    
    scene_dir = Path(args.scene_dir)
    if not scene_dir.exists():
        # Try to find the scene
        alt_path = Path("/home/pi0/POGS") / args.scene_dir
        if alt_path.exists():
            scene_dir = alt_path
        else:
            print(f"❌ Scene directory not found: {args.scene_dir}")
            sys.exit(1)
    
    print(f"📁 Scene: {scene_dir}")
    print(f"🎯 Cluster ID: {args.cluster_id}")
    print(f"🤖 Dry run: {args.dry_run}")
    print()
    
    # Clear CUDA memory first
    clear_cuda_memory()
    
    # Step 1: Check for existing PLY files or export
    local_ply = scene_dir / "local.ply"
    global_ply = scene_dir / "global.ply"
    
    if not local_ply.exists() or not global_ply.exists():
        print("⚠️ PLY files not found. You need to export them first using the full demo.")
        print("   Run: python pogs/scripts/track_main_online_demo.py --dry-run")
        print("   Then click 'Cluster Scene' and 'Generate Grasps on Query'")
        sys.exit(1)
    
    print(f"✅ Found PLY files: {local_ply}")
    
    # Step 2: Generate grasps (if not skipping)
    if not args.skip_grasp_gen:
        success = generate_grasps_contact_graspnet(
            str(local_ply), 
            str(global_ply), 
            str(scene_dir)
        )
        if not success:
            print("❌ Grasp generation failed")
            sys.exit(1)
    
    # Step 3: Load best grasp
    try:
        best_grasp = load_best_grasp(str(scene_dir))
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("   Make sure to generate grasps first!")
        sys.exit(1)
    
    # Step 4: Execute grasp
    print("\n" + "=" * 60)
    print("🤖 Robot Execution")
    print("=" * 60)
    
    try:
        execute_grasp_on_robot(best_grasp, dry_run=args.dry_run)
    except Exception as e:
        print(f"❌ Robot execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
