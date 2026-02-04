from typing import Optional
import numpy as np

def project_pose_to_4dof(pose_6dof, fixed_roll=0.0, fixed_pitch=1.57):
    """
    Project a 6-DoF SE(3) pose (4x4 matrix) to a 4-DoF pose by fixing roll and pitch, keeping x, y, z, and yaw.
    Args:
        pose_6dof: 4x4 numpy array or nested list (SE(3) transform)
        fixed_roll: value (rad) to fix roll (default 0)
        fixed_pitch: value (rad) to fix pitch (default 1.57 or ~pi/2 for Down)
    Returns:
        4x4 numpy array with only yaw, x, y, z preserved
    """
    import math
    from scipy.spatial.transform import Rotation as R
    pose_6dof = np.array(pose_6dof)
    t = pose_6dof[:3, 3]
    rot = pose_6dof[:3, :3]
    # Extract yaw from original rotation
    yaw = R.from_matrix(rot).as_euler('zyx')[0]
    # Compose new rotation with fixed roll/pitch, original yaw
    new_rot = R.from_euler('zyx', [yaw, fixed_pitch, fixed_roll]).as_matrix()
    pose_4dof = np.eye(4)
    pose_4dof[:3, :3] = new_rot
    pose_4dof[:3, 3] = t
    return pose_4dof


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
