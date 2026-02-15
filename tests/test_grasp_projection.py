
import numpy as np
from scipy.spatial.transform import Rotation as R
from pogs.controller.robot_interface import project_pose_to_4dof

def test_projection():
    print("Testing 6-DoF to 4-DoF projection with pitch clamping...\n")

    # Case 1: Pure Top-Down Grasp (Pitch = pi/2 approx)
    # Roll=0, Pitch=90deg (1.57), Yaw=45deg (0.785)
    print("--- Test Case 1: Standard Top-Down Grasp ---")
    
    # Create rotation from euler (using scipy convention)
    # Note: scipy might output euler differently, so we construct explicitly to be sure
    # Using 'zyx' usually maps to yaw, pitch, roll
    r_target = R.from_euler('zyx', [0.785, 1.57, 0.2]) # some small roll noise
    rot_matrix = r_target.as_matrix()
    
    pose_6dof = np.eye(4)
    pose_6dof[:3, :3] = rot_matrix
    pose_6dof[:3, 3] = [0.2, 0.1, 0.3] # x,y,z translation

    # Project with clamping (-90 to +90 degrees)
    pose_4dof = project_pose_to_4dof(pose_6dof, fixed_roll=0.0, pitch_range=(-1.57, 1.57))
    
    # Verify
    r_res = R.from_matrix(pose_4dof[:3, :3])
    yaw, pitch, roll = r_res.as_euler('zyx')
    
    print(f"Original Euler (z,y,x): [0.785, 1.57, 0.2]")
    print(f"Projected Euler (z,y,x): [{yaw:.3f}, {pitch:.3f}, {roll:.3f}]")
    print(f"Roll aligned to 0? {np.isclose(roll, 0.0, atol=1e-3)}")
    print(f"Pitch preserved (~1.57)? {np.isclose(pitch, 1.57, atol=0.1)}")
    print(f"Position retained? {np.allclose(pose_4dof[:3, 3], [0.2, 0.1, 0.3])}")
    print("Result: PASS" if np.isclose(roll, 0) else "Result: FAIL")
    print("\n")


    # Case 2: Side Grasp (Pitch ~ 0)
    print("--- Test Case 2: Side Grasp (Pitch=0) ---")
    r_target = R.from_euler('zyx', [0.0, 0.1, 0.0]) # Nearly horizontal grasp
    pose_6dof[:3, :3] = r_target.as_matrix()
    
    pose_4dof = project_pose_to_4dof(pose_6dof, fixed_roll=0.0, pitch_range=(-1.57, 1.57))
    yaw, pitch, roll = R.from_matrix(pose_4dof[:3, :3]).as_euler('zyx')
    
    print(f"Original Pitch: 0.1")
    print(f"Projected Pitch: {pitch:.3f}")
    print(f"Preserved? {np.isclose(pitch, 0.1, atol=1e-3)}")
    print("\n")


    # Case 3: Out of Range Pitch (e.g., Upside down or too steep)
    print("--- Test Case 3: Pitch Clamping (Input=2.0 rad > 1.57 limit) ---")
    # Pitch = 2.0 rad (~114 deg), should be clamped to 1.57 (90 deg)
    r_target = R.from_euler('zyx', [0.0, 2.0, 0.0]) 
    pose_6dof[:3, :3] = r_target.as_matrix()
    
    pose_4dof = project_pose_to_4dof(pose_6dof, fixed_roll=0.0, pitch_range=(-1.57, 1.57))
    yaw, pitch, roll = R.from_matrix(pose_4dof[:3, :3]).as_euler('zyx')
    
    print(f"Original Pitch: 2.0")
    print(f"Projected Pitch: {pitch:.3f}")
    print(f"Clamped to ~1.57? {np.isclose(pitch, 1.57, atol=1e-3)}")
    print("\n")

if __name__ == "__main__":
    test_projection()
