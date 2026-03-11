"""
Demo: POGS Grasp -> 4-DOF OpenManipulator X Execution

Pipeline:
1. Capture scene with POGS (tracking + segmentation)
2. Generate grasps with grasp net (6-DOF)
3. Project to 4-DOF + compute IK
4. Execute on OMX robot

Usage:
    python demo_omx_grasp_execution.py --scene data/my_scene --output outputs/my_scene
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Add POGS paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from pogs.grasping.omx_grasp_executor import OMXGraspExecutor, project_pose_to_4dof
from pogs.controller.robot_interface import RobotInterface


def load_grasps_from_graspnet(
    scene_dir: str,
    n_forward_passes: int = 5,
    filter_grasps: bool = True,
) -> tuple:
    """
    Generate grasps from grasp net on scene.
    
    Returns:
        (pred_grasps_6dof, scores, contact_points)
        - pred_grasps_6dof: (N, 4, 4) SE(3) poses
        - scores: (N,) confidence scores
        - contact_points: (N, 3) contact locations
    """
    try:
        from pogs.grasping.generate_grasps_ply import generate_grasps
        from pogs.contact_graspnet_wrapper.prime_config_utils import load_config
        # Paths (adapt to your setup)
        seg_np_path = f"{scene_dir}/segmented_gaussians.npy"
        full_np_path = f"{scene_dir}/full_gaussians.npy"
        bbox_path = f"{scene_dir}/bbox.json"
        ckpt_dir = "path/to/graspnet/checkpoint"  # Update this
        print(f"[Pipeline] Generating grasps from {scene_dir}...")
        pred_grasps, scores, contact_pts, _, _ = generate_grasps(
            seg_np_path=seg_np_path,
            full_np_path=full_np_path,
            pc_bounding_box_path=bbox_path,
            ckpt_dir=ckpt_dir,
            z_range=[0.01, 0.5],
            K=None,
            local_regions=True,
            filter_grasps=filter_grasps,
            skip_border_objects=True,
            forward_passes=n_forward_passes,
            segmap_id=None,
            arg_configs={},
            save_dir=scene_dir,
        )
        return pred_grasps, scores, contact_pts
    except Exception as e:
        print(f"[Pipeline] Grasp generation failed: {e}")
        raise RuntimeError("GraspNet failed. No mock fallback. Exiting.")


def generate_mock_grasps(n_grasps: int = 5) -> tuple:
    """Generate mock grasps for testing."""
    grasps = []
    scores = []
    
    for i in range(n_grasps):
        # Random grasp in workspace
        x = 0.25 + np.random.uniform(-0.1, 0.1)
        y = np.random.uniform(-0.2, 0.2)
        z = 0.15 + np.random.uniform(-0.05, 0.1)
        
        # Random orientation (downward approach)
        roll = np.random.uniform(-0.2, 0.2)
        pitch = -1.57 + np.random.uniform(-0.3, 0.3)  # Down-ish
        yaw = np.random.uniform(0, 2*np.pi)
        
        pose = np.eye(4)
        from scipy.spatial.transform import Rotation as R
        pose[:3, :3] = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
        pose[:3, 3] = [x, y, z]
        
        grasps.append(pose)
        scores.append(np.random.uniform(0.5, 1.0))
    
    contact_pts = np.array([[g[0, 3], g[1, 3], g[2, 3]] for g in grasps])
    return np.array(grasps), np.array(scores), contact_pts


def main():
    parser = argparse.ArgumentParser(
        description="Execute POGS grasps on OpenManipulator X"
    )
    parser.add_argument("--scene", default="data/my_scene", help="Scene directory")
    parser.add_argument("--output", default="outputs/my_scene", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Dry-run (no robot)")
    parser.add_argument("--top-k", type=int, default=3, help="Execute top-K grasps")
    parser.add_argument("--pitch", type=float, default=1.57, help="Fixed pitch angle")
    
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # ===== Step 1: Initialize Robot =====
    print("[Main] Initializing robot...")
    robot_interface = RobotInterface()
    if not args.dry_run and not robot_interface.connect():
        print("[Main] Warning: Robot not available; using dry-run mode")
        args.dry_run = True
    
    # ===== Step 2: Initialize Grasp Executor =====
    print("[Main] Initializing OMX grasp executor...")
    executor = OMXGraspExecutor(
        robot=robot_interface.robot if not args.dry_run else None,
        use_pinocchio=True,
    )
    
    # ===== Step 3: Generate Grasps =====
    print("[Main] Running POGS + grasp net...")
    grasps_6dof, scores, contact_pts = load_grasps_from_graspnet(args.scene)
    print(f"[Main] Generated {len(grasps_6dof)} grasps")
    
    # ===== Step 4: Filter and Project =====
    print("[Main] Projecting 6-DOF -> 4-DOF and solving IK...")
    joints, ik_success = executor.batch_grasps(grasps_6dof, fixed_pitch=args.pitch)
    
    n_success = np.sum(ik_success)
    print(f"[Main] IK success: {n_success}/{len(grasps_6dof)}")
    
    if n_success == 0:
        print("[Main] No feasible grasps; exiting")
        return
    
    # ===== Step 5: Execute Top-K Grasps =====
    valid_idx = np.where(ik_success)[0]
    valid_scores = scores[valid_idx]
    top_idx = valid_idx[np.argsort(-valid_scores)[:args.top_k]]
    
    print(f"[Main] Executing top-{len(top_idx)} grasps...")
    for rank, idx in enumerate(top_idx):
        print(f"\n[Execution] Grasp {rank+1}/{len(top_idx)} (score={scores[idx]:.3f})")
        
        success = executor.execute_grasp(
            grasps_6dof[idx],
            fixed_pitch=args.pitch,
            vel=0.5,
            acc=0.1,
        )
        
        if success:
            print(f"[Execution] Grasp executed successfully")
            print(f"[Execution] Joints: {joints[idx]}")
            
            # In real scenario: grab, retract, place, etc.
            # For now just log the result
            result = {
                "grasp_idx": int(idx),
                "score": float(scores[idx]),
                "joints": joints[idx].tolist(),
                "pose_6dof": grasps_6dof[idx].tolist(),
                "contact_point": contact_pts[idx].tolist(),
            }
            
            import json
            result_path = Path(args.output) / f"grasp_{idx:03d}.json"
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"[Execution] Saved result to {result_path}")
        else:
            print(f"[Execution] Grasp execution failed")
    
    print("\n[Main] Demo complete!")


if __name__ == "__main__":
    main()
