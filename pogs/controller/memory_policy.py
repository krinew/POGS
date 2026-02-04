from typing import Optional, Tuple
import numpy as np
import time
from .commands import Command
from .robot_interface import project_pose_to_4dof

class MemoryGraspPolicy:
    """
    A policy that locks onto a target grasp pose and executes it blindly/open-loop
    to handle occlusion during the approach phase.
    
    States:
    0: SEARCH - Waiting for a valid target/grasp to be detected.
    1: LOCK   - A valid grasp has been found and saved.
    2: APPROACH - Moving towards the pre-grasp pose.
    3: GRASP - Moving to grasp pose and closing gripper.
    4: LIFT - Lifting the object.
    """
    
    def __init__(self, grasp_estimator=None, target_query: str = None):
        self.state = "SEARCH"
        self.target_pose_world_4dof = None # The locked target pose (4x4)
        self.target_query = target_query
        self.start_time = 0
        self.grasp_estimator = grasp_estimator # Optional: Link to GraspNet instance

    def set_target(self, pose_4dof: np.ndarray):
        """Externally set the target grasp (e.g. from POGS semantics)."""
        self.target_pose_world_4dof = pose_4dof
        self.state = "LOCK"
        print(f"[MemoryPolicy] Target Locked! Pose:\n{pose_4dof}")

    def propose_command(self, color: np.ndarray, depth: np.ndarray) -> Command:
        """
        State machine execution.
        Note: This policy largely ignores 'color' and 'depth' once in LOCK state,
        relying on the 'memory' of target_pose_world_4dof.
        """
        
        if self.state == "SEARCH":
            # Just hover or wait. 
            # In a real integration, this is where we'd ask POGS/GraspNet "Where is the mug?"
            # For now, we return None (do nothing) until set_target is called.
            return Command(type="wait")

        elif self.state == "LOCK":
            # Transition to Approach
            print("[MemoryPolicy] Starting Approach...")
            self.state = "APPROACH"
            self.start_time = time.time()
            
            # Generate Pre-grasp (e.g. 10cm above)
            pre_grasp = self.target_pose_world_4dof.copy()
            pre_grasp[2, 3] += 0.10 # Move Z up by 10cm (assuming Z is world Up)
            
            return Command(type="pose", pose=pre_grasp)

        elif self.state == "APPROACH":
            # Rudimentary open-loop timing. In real robot, we'd check "is_at_pose()"
            if time.time() - self.start_time > 3.0: 
                print("[MemoryPolicy] Approach complete. Going for Grasp.")
                self.state = "GRASP"
                self.start_time = time.time()
                return Command(type="pose", pose=self.target_pose_world_4dof)
            else:
                return Command(type="wait") # waiting for robot to move

        elif self.state == "GRASP":
            if time.time() - self.start_time > 2.0:
                print("[MemoryPolicy] Grasp reached. Closing Gripper.")
                self.state = "LIFT"
                self.start_time = time.time()
                # Assuming 'joints' command can trigger gripper or special command type
                # For now, we just stay at the pose.
                return Command(type="pose", pose=self.target_pose_world_4dof)
            else:
                return Command(type="wait")

        elif self.state == "LIFT":
            if time.time() - self.start_time > 1.0:
                # Lift up
                lift_pose = self.target_pose_world_4dof.copy()
                lift_pose[2, 3] += 0.15
                print("[MemoryPolicy] Lifting...")
                self.state = "DONE"
                return Command(type="pose", pose=lift_pose)

        return Command(type="wait")
