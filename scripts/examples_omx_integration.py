"""
Quick integration examples showing how to use OMXGraspExecutor with POGS.
Reference/helper file - not required for operation.
"""

import numpy as np
from pathlib import Path
from pogs.grasping.omx_grasp_executor import OMXGraspExecutor
from pogs.controller.open_manipulator import OpenManipulatorLeRobot


# ============================================================================
# EXAMPLE 1: Simple execution from stored grasps
# ============================================================================

def example_execute_stored_grasps():
    """Execute pre-computed grasps from POGS output."""
    
    # Load grasps saved by grasp net (typically .npy format)
    grasps_6dof = np.load("outputs/my_scene/pred_grasps.npy")  # Shape: (N, 4, 4)
    scores = np.load("outputs/my_scene/scores.npy")
    
    # Initialize robot
    robot = OpenManipulatorLeRobot(port="/dev/ttyUSB0", use_leader_ids=False)
    
    # Create executor
    executor = OMXGraspExecutor(robot=robot.bus, use_pinocchio=True)
    
    # Process grasps
    joints, feasible = executor.batch_grasps(grasps_6dof)
    
    # Execute top-3 feasible grasps
    feasible_idx = np.where(feasible)[0]
    top_idx = feasible_idx[np.argsort(-scores[feasible_idx])[:3]]
    
    for idx in top_idx:
        print(f"Executing grasp {idx} (score: {scores[idx]:.3f})")
        robot.move_joint(joints[idx], vel=0.5)
        # TODO: Actually grasp here (close gripper, retract, etc.)


# ============================================================================
# EXAMPLE 2: Integrate with POGS perception pipeline
# ============================================================================

def example_pogs_pipeline_integration():
    """Full pipeline: perception → grasp generation → execution."""
    
    from pogs.pogs import POGS
    from pogs.grasping.generate_grasps_ply import generate_grasps
    
    # 1. Run POGS perception on scene
    scene_dir = "data/my_scene"
    pogs_model = POGS.load_from_checkpoint("path/to/pogs/ckpt")
    scene_pcd = pogs_model.infer_scene(scene_dir)
    
    # 2. Generate grasps
    grasps, scores, contacts = generate_grasps(
        seg_np_path=f"{scene_dir}/segmented.npy",
        full_np_path=f"{scene_dir}/full.npy",
        pc_bounding_box_path=f"{scene_dir}/bbox.json",
        ckpt_dir="path/to/graspnet/ckpt",
        z_range=[0.01, 0.5],
        K=None,
        local_regions=True,
        filter_grasps=True,
        skip_border_objects=True,
        forward_passes=5,
        segmap_id=None,
        arg_configs={},
        save_dir=scene_dir,
    )
    
    # 3. Execute on robot
    robot = OpenManipulatorLeRobot()
    executor = OMXGraspExecutor(robot=robot.bus)
    
    joints, feasible = executor.batch_grasps(grasps)
    
    # Pick best feasible grasp
    feasible_idx = np.where(feasible)[0]
    best_idx = feasible_idx[np.argmax(scores[feasible_idx])]
    
    print(f"Executing best grasp (idx={best_idx}, score={scores[best_idx]:.3f})")
    robot.move_joint(joints[best_idx])


# ============================================================================
# EXAMPLE 3: Dry-run testing (no hardware needed)
# ============================================================================

def example_dry_run():
    """Test grasp execution without robot hardware."""
    
    # Create random 6-DOF grasps for testing
    n_test_grasps = 5
    test_grasps = []
    
    for _ in range(n_test_grasps):
        pose = np.eye(4)
        pose[0, 3] = np.random.uniform(0.15, 0.35)  # x: workspace bounds
        pose[1, 3] = np.random.uniform(-0.25, 0.25)  # y
        pose[2, 3] = np.random.uniform(0.10, 0.30)   # z
        
        # Random orientation
        from scipy.spatial.transform import Rotation as R
        pose[:3, :3] = R.random().as_matrix()
        
        test_grasps.append(pose)
    
    test_grasps = np.array(test_grasps)
    
    # Executor without robot (dry-run)
    executor = OMXGraspExecutor(robot=None, use_pinocchio=False)
    
    joints, feasible = executor.batch_grasps(test_grasps)
    
    print(f"Feasibility: {np.sum(feasible)}/{len(test_grasps)} grasps reachable")
    print(f"Joint solutions:\n{joints}")


# ============================================================================
# EXAMPLE 4: Custom IK with different pitch angles
# ============================================================================

def example_custom_pitch_angles():
    """Try different pitch angles to optimize grasp approach."""
    
    grasps = np.load("outputs/my_scene/pred_grasps.npy")
    
    executor = OMXGraspExecutor(robot=None, use_pinocchio=False)
    
    # Try multiple pitch angles
    pitch_angles = {
        "approach_down": 1.57,      # Gripper pointing down
        "approach_side": 0.0,       # Gripper pointing sideways
        "approach_up": -1.57,       # Gripper pointing up
    }
    
    results = {}
    for approach_name, pitch in pitch_angles.items():
        joints, feasible = executor.batch_grasps(grasps, fixed_pitch=pitch)
        results[approach_name] = {
            "pitch": pitch,
            "feasible_count": np.sum(feasible),
            "total": len(feasible),
        }
        print(f"{approach_name:20s}: {np.sum(feasible):2d}/{len(feasible)} feasible")
    
    return results


if __name__ == "__main__":
    print("POGS OMX Integration Examples")
    print("=" * 50)
    
    # Choose which example to run
    print("\n1. Dry-run test (no hardware):")
    example_dry_run()
    
    # print("\n2. Custom pitch angles:")
    # example_custom_pitch_angles()
