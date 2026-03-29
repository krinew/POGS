from ur5py.ur5 import UR5Robot
import numpy as np
from autolab_core import RigidTransform

WRIST_TO_CAM = RigidTransform.load("/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/wrist_to_cam.tf")

class Motion:
    def __init__(self, robot: UR5Robot):
        self.robot = robot
        # robot = UR5Robot(gripper=1)
        self.clear_tcp()
        
        home_joints = np.array([0.30947089195251465, -1.2793572584735315, -2.035713497792379, -1.388848606740133, 1.5713528394699097, 0.34230729937553406])
        robot.move_joint(home_joints,vel=1.0,acc=0.1)
        world_to_wrist = robot.get_pose()
        world_to_wrist.from_frame = "wrist"
        world_to_cam = world_to_wrist * WRIST_TO_CAM
        proper_world_to_cam_rotation = np.array([[0,1,0],[1,0,0],[0,0,-1]])
        self.home_world_to_cam = RigidTransform(rotation=proper_world_to_cam_rotation,translation=world_to_cam.translation,from_frame='cam',to_frame='world')
        self.home_world_to_wrist = self.home_world_to_cam * WRIST_TO_CAM.inverse()
            
        self.robot.move_pose(self.home_world_to_wrist,vel=1.0,acc=0.1)
        
        
    def clear_tcp(self):
        tool_to_wrist = RigidTransform()
        tool_to_wrist.translation = np.array([0, 0, 0])
        tool_to_wrist.from_frame = "tool"
        tool_to_wrist.to_frame = "wrist"
        self.robot.set_tcp(tool_to_wrist)
        
    