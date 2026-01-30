from typing import Optional
import numpy as np


class RobotInterface:
    """A thin wrapper to abstract sending commands to a UR5 robot if available.

    If `ur5py` is not installed/available, the interface will operate in dry-run mode
    and print commands instead of sending them to hardware.
    """

    def __init__(self):
        self.robot = None
        self.connected = False

    def connect(self) -> bool:
        try:
            from ur5py.ur5 import UR5Robot
            # instantiate with gripper enabled if supported
            self.robot = UR5Robot(gripper=1)
            self.connected = True
        except Exception:
            self.robot = None
            self.connected = False
        return self.connected

    def move_pose(self, pose: list, vel: float = 0.3, acc: float = 0.1):
        """Move robot to given 4x4 pose matrix. If UR5Robot isn't available this is a dry-run.

        `pose` is expected as a 4x4 nested list or numpy array.
        """
        mat = np.array(pose)
        if self.robot is None:
            print(f"[RobotInterface] DRY-RUN move_pose: \\n+                  pose=\n{mat}\n vel={vel} acc={acc}")
            return False

        # Try to call robot.move_pose; adapt if robot API expects another type
        try:
            if hasattr(self.robot, 'move_pose'):
                # many integrations expect a RigidTransform; try passing raw matrix first
                try:
                    self.robot.move_pose(mat, vel=vel, acc=acc)
                except Exception:
                    # fallback: try passing matrix as-is without kwargs
                    self.robot.move_pose(mat)
                return True
            else:
                print("[RobotInterface] robot object has no move_pose method — dry-run")
                return False
        except Exception as e:
            print(f"[RobotInterface] Error sending move_pose: {e}")
            return False

    def move_joints(self, joints: list, vel: float = 0.3, acc: float = 0.1):
        if self.robot is None:
            print(f"[RobotInterface] DRY-RUN move_joints: {joints} vel={vel} acc={acc}")
            return False
        try:
            if hasattr(self.robot, 'move_joint'):
                self.robot.move_joint(joints, vel=vel, acc=acc)
                return True
            else:
                print("[RobotInterface] robot has no move_joint — dry-run")
                return False
        except Exception as e:
            print(f"[RobotInterface] Error sending joints: {e}")
            return False
