import time
import numpy as np
import sys
import os

# Add relevant paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pogs.controller.robot_interface import RobotInterface, project_pose_to_4dof
from pogs.controller.memory_policy import MemoryGraspPolicy

def mock_detect_grasp():
    """
    Simulates finding a grasp using POGS/GraspNet.
    Returns a random valid 6-DoF pose in the robot workspace.
    """
    print("[Perception] Scanning scene with POGS...")
    time.sleep(1.0)
    print("[Perception] Object 'mug' found at (0.3, 0.0, 0.2).")
    
    # Create a dummy 6-DoF pose
    pose = np.eye(4)
    pose[0, 3] = 0.3  # x
    pose[1, 3] = 0.0  # y
    pose[2, 3] = 0.2  # z (table height approx)
    
    # Add some random rotation (which we will strip out)
    # 45 deg roll
    pose[:3, :3] = [[1, 0, 0], [0, 0.707, -0.707], [0, 0.707, 0.707]]
    
    print(f"[Perception] Raw 6-DoF Grasp:\n{pose}")
    return pose

def main():
    # 1. Initialize Robot
    robot = RobotInterface()
    if not robot.connect():
        print("Warning: Robot not connected. Running in Dry-Run mode.")

    # 2. Initialize Policy
    policy = MemoryGraspPolicy(target_query="mug")
    
    # 3. Perception Phase (The 'Scan')
    # In a real app, you might move the robot here to look around
    raw_grasp_6dof = mock_detect_grasp()
    
    # 4. Projection Phase (6-DoF -> 4-DoF)
    # Using our new utility with default pitch=1.57 (Down)
    target_pose_4dof = project_pose_to_4dof(raw_grasp_6dof, fixed_pitch=1.57)
    
    # 5. Lock Target (Memory Phase)
    policy.set_target(target_pose_4dof)
    
    # 6. Execution Loop (The 'Blind' Approach)
    print("Starting Control Loop (Ctrl+C to stop)...")
    try:
        while policy.state != "DONE":
            # Simulate grabbing a frame (would come from RealSense)
            dummy_color = np.zeros((480, 640, 3), dtype=np.uint8)
            dummy_depth = np.zeros((480, 640), dtype=np.float32)
            
            # Ask policy for next move
            cmd = policy.propose_command(dummy_color, dummy_depth)
            
            # Execute
            if cmd.type == "pose":
                # Note: We don't need to project here anymore, 
                # because the policy holds the already-projected target!
                robot.move_pose(cmd.pose)
            elif cmd.type == "wait":
                time.sleep(0.1)
                
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("Stopped by user.")

    print("Task Complete.")

if __name__ == "__main__":
    main()
