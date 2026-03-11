#!/usr/bin/env python3
"""
Standalone script to export point cloud from trained POGS model and generate grasps.
This bypasses the heavy tracking pipeline for simpler grasp execution.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np

# Add POGS to path
POGS_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(POGS_ROOT))


def export_pogs_pointcloud(config_path: Path, output_ply: Path) -> bool:
    """Load POGS model and export full scene point cloud."""
    import torch
    import open3d as o3d
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.models.splatfacto import SH2RGB
    
    print(f"[INFO] Loading POGS model from {config_path}")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config, pipeline, checkpoint_path, step = eval_setup(
        config_path,
        eval_num_rays_per_chunk=None,
        test_mode="inference",
    )
    pipeline.to(device)
    pipeline.eval()
    
    print(f"[INFO] Model loaded from checkpoint: {checkpoint_path}")
    
    # Extract gaussian parameters
    model = pipeline.model
    means = model.gauss_params["means"].detach().cpu().numpy()
    features_dc = model.gauss_params["features_dc"].detach().cpu().numpy()
    opacities = torch.sigmoid(model.gauss_params["opacities"]).detach().cpu().numpy().squeeze()
    
    print(f"[INFO] Total gaussians: {len(means)}")
    
    # Filter by opacity
    opacity_threshold = 0.1
    valid_mask = opacities > opacity_threshold
    means = means[valid_mask]
    features_dc = features_dc[valid_mask]
    
    print(f"[INFO] After opacity filter (>{opacity_threshold}): {len(means)} gaussians")
    
    # Convert SH to RGB
    if model.config.sh_degree > 0:
        colors = SH2RGB(features_dc)
    else:
        colors = 1 / (1 + np.exp(-features_dc))  # sigmoid
    
    # Normalize colors to [0, 1]
    colors = np.clip(colors, 0, 1)
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    
    # Save
    o3d.io.write_point_cloud(str(output_ply), pcd)
    print(f"[INFO] Saved point cloud to {output_ply}")
    
    # Clean up GPU memory
    del pipeline, model
    torch.cuda.empty_cache()
    
    return True


def run_contact_graspnet(ply_path: Path, output_dir: Path) -> bool:
    """Run ContactGraspNet on the point cloud."""
    contact_graspnet_dir = POGS_ROOT / "pogs/dependencies/contact_graspnet"
    checkpoint_dir = contact_graspnet_dir / "checkpoints/scene_test_2048_bs3_hor_sigma_001"
    
    if not checkpoint_dir.exists():
        print(f"[ERROR] ContactGraspNet checkpoint not found at {checkpoint_dir}")
        return False
    
    python_exe = Path.home() / "miniconda3/envs/contact_graspnet_env/bin/python"
    if not python_exe.exists():
        print(f"[ERROR] ContactGraspNet Python not found at {python_exe}")
        return False
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the no-viz version that doesn't require mayavi
    cmd = [
        str(python_exe),
        str(contact_graspnet_dir / "contact_graspnet/inference_no_viz.py"),
        "--input_path", str(ply_path.resolve()),  # Use absolute path
        "--ckpt_dir", str(checkpoint_dir),
        "--forward_passes", "5",
        "--output_dir", str(output_dir.resolve()),  # Use absolute path
    ]
    
    print(f"[INFO] Running ContactGraspNet...")
    print(f"[CMD] {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(contact_graspnet_dir))
    
    if result.returncode != 0:
        print(f"[ERROR] ContactGraspNet failed:")
        print(result.stderr)
        return False
    
    print(result.stdout)
    return True


def load_grasps(grasp_dir: Path):
    """Load generated grasps from ContactGraspNet output."""
    grasp_poses_file = grasp_dir / "grasp_poses.npy"
    grasp_scores_file = grasp_dir / "grasp_scores.npy"
    
    if not grasp_poses_file.exists():
        # Try alternate file patterns
        npy_files = list(grasp_dir.glob("*.npy"))
        print(f"[INFO] Found .npy files: {npy_files}")
        
        for f in npy_files:
            if "poses" in f.name.lower() or "grasp" in f.name.lower():
                data = np.load(f, allow_pickle=True)
                print(f"[INFO] {f.name}: shape={data.shape if hasattr(data, 'shape') else 'dict'}")
        return None, None
    
    grasp_poses = np.load(grasp_poses_file)
    grasp_scores = np.load(grasp_scores_file) if grasp_scores_file.exists() else None
    
    return grasp_poses, grasp_scores


def execute_grasp_on_robot(grasp_pose: np.ndarray, dry_run: bool = True):
    """Execute a grasp on the OpenManipulator robot."""
    from pogs.manipulation.robot_interface import RobotInterface
    
    print(f"[INFO] Executing grasp at position: {grasp_pose[:3, 3]}")
    
    if dry_run:
        print("[DRY RUN] Would execute grasp (skipping actual robot motion)")
        return True
    
    try:
        robot = RobotInterface()
        
        # Move to pre-grasp position (above the grasp)
        pre_grasp = grasp_pose.copy()
        pre_grasp[2, 3] += 0.05  # 5cm above
        
        robot.move_to_pose(pre_grasp)
        robot.open_gripper()
        
        # Move to grasp position
        robot.move_to_pose(grasp_pose)
        robot.close_gripper()
        
        # Lift
        lift_pose = grasp_pose.copy()
        lift_pose[2, 3] += 0.1  # 10cm up
        robot.move_to_pose(lift_pose)
        
        print("[SUCCESS] Grasp executed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Robot execution failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export POGS point cloud and generate grasps")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to POGS config.yml")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for point cloud and grasps")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip point cloud export (use existing)")
    parser.add_argument("--skip-grasp-gen", action="store_true",
                        help="Skip grasp generation (use existing)")
    parser.add_argument("--execute", action="store_true",
                        help="Execute best grasp on robot")
    parser.add_argument("--dry-run", action="store_true",
                        help="Dry run mode (no actual robot motion)")
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        return 1
    
    # Setup output directory
    output_dir = Path(args.output_dir) if args.output_dir else config_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ply_path = output_dir / "scene.ply"
    grasp_output_dir = output_dir / "grasps"
    
    # Step 1: Export point cloud
    if not args.skip_export:
        print("\n" + "="*60)
        print("STEP 1: Exporting point cloud from POGS model")
        print("="*60)
        if not export_pogs_pointcloud(config_path, ply_path):
            print("[ERROR] Point cloud export failed")
            return 1
    else:
        print(f"[INFO] Skipping export, using existing: {ply_path}")
        if not ply_path.exists():
            print(f"[ERROR] PLY file not found: {ply_path}")
            return 1
    
    # Step 2: Generate grasps
    if not args.skip_grasp_gen:
        print("\n" + "="*60)
        print("STEP 2: Generating grasps with ContactGraspNet")
        print("="*60)
        if not run_contact_graspnet(ply_path, grasp_output_dir):
            print("[ERROR] Grasp generation failed")
            return 1
    else:
        print(f"[INFO] Skipping grasp generation, using existing: {grasp_output_dir}")
    
    # Step 3: Load grasps
    print("\n" + "="*60)
    print("STEP 3: Loading generated grasps")
    print("="*60)
    grasp_poses, grasp_scores = load_grasps(grasp_output_dir)
    
    if grasp_poses is not None and len(grasp_poses) > 0:
        print(f"[INFO] Loaded {len(grasp_poses)} grasps")
        
        # Get best grasp
        if grasp_scores is not None and len(grasp_scores) > 0:
            best_idx = np.argmax(grasp_scores)
            print(f"[INFO] Best grasp idx={best_idx}, score={grasp_scores[best_idx]:.3f}")
        else:
            best_idx = 0
        
        best_grasp = grasp_poses[best_idx]
        print(f"[INFO] Best grasp position: {best_grasp[:3, 3]}")
        
        # Save best grasp
        np.save(output_dir / "best_grasp.npy", best_grasp)
        print(f"[INFO] Saved best grasp to {output_dir / 'best_grasp.npy'}")
        
        # Step 4: Execute grasp
        if args.execute:
            print("\n" + "="*60)
            print("STEP 4: Executing grasp on robot")
            print("="*60)
            execute_grasp_on_robot(best_grasp, dry_run=args.dry_run)
    else:
        print("[WARNING] No grasps loaded - check output directory")
    
    print("\n[DONE]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
