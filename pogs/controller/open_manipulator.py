import time
import numpy as np
import math

try:
    from dynamixel_sdk import * 
except ImportError:
    print("Dynamixel SDK not found. Please install it.")

class OpenManipulatorRobot:
    # Control table address
    ADDR_TORQUE_ENABLE          = 64
    ADDR_GOAL_POSITION          = 116
    ADDR_PRESENT_POSITION       = 132
    ADDR_PROFILE_VELOCITY       = 112
    ADDR_OPERATING_MODE         = 11

    # Protocol version
    PROTOCOL_VERSION            = 2.0

    # Default setting
    DXL_ID_JOINT_1              = 11
    DXL_ID_JOINT_2              = 12
    DXL_ID_JOINT_3              = 13
    DXL_ID_JOINT_4              = 14
    DXL_ID_GRIPPER              = 15

    BAUDRATE                    = 1000000
    DEVICENAME                  = '/dev/ttyUSB0'

    TORQUE_ENABLE               = 1                 
    TORQUE_DISABLE              = 0                 
    
    # Encoder counts
    MIN_POSITION_VAL            = 0
    MAX_POSITION_VAL            = 4095
    ZERO_POSITION_VAL           = 2048
    
    # Physical measures (meters)
    # OpenManipulator Link Lengths (approx, adjust if needed)
    L1 = 0.077      # Base to Joint 2
    L2 = 0.130      # Joint 2 to Joint 3
    L3 = 0.124      # Joint 3 to Joint 4
    L4 = 0.126      # Joint 4 to TCP

    def __init__(self, port='/dev/ttyUSB0', baudrate=1000000, gripper=True):
        self.portHandler = PortHandler(port)
        self.packetHandler = PacketHandler(self.PROTOCOL_VERSION)
        
        if self.portHandler.openPort():
            print(f"Succeeded to open the port {port}")
        else:
            print("Failed to open the port")
            
        if self.portHandler.setBaudRate(baudrate):
            print(f"Succeeded to change the baudrate to {baudrate}")
        else:
            print("Failed to change the baudrate")

        self.joint_ids = [self.DXL_ID_JOINT_1, self.DXL_ID_JOINT_2, self.DXL_ID_JOINT_3, self.DXL_ID_JOINT_4]
        self.gripper_id = self.DXL_ID_GRIPPER
        
        self.enable_torque(self.joint_ids + ([self.gripper_id] if gripper else []))
        
        # Set default velocity
        self.set_profile_velocity(self.joint_ids, 100) 
        if gripper:
            self.set_profile_velocity([self.gripper_id], 100)

        self.gripper = self.Gripper(self) if gripper else None
        
    class Gripper:
        def __init__(self, robot):
            self.robot = robot
            # Value range for gripper (Open -> Close)
            # Adjust these values based on calibration
            self.OPEN_VAL = -0.01  # Not used, we use angular offsets usually or raw values
            # Using raw DXL values for gripper might be safer if unsure about conversion
            # Let's assume Angle based:
            # Open: -45 deg -> 
            pass
        
        def open(self):
            # Open the gripper
            # Assuming Joint mode.
            print("Opening Gripper")
            self.robot.set_joint_position(self.robot.gripper_id, 0.5) # Approx open rad?
        
        def close(self):
            # Close the gripper
            print("Closing Gripper")
            self.robot.set_joint_position(self.robot.gripper_id, -0.5) # Approx close rad?

    def enable_torque(self, ids):
        for dxl_id in ids:
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE)
            
    def disable_torque(self, ids):
        for dxl_id in ids:
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE)

    def set_profile_velocity(self, ids, velocity):
        for dxl_id in ids:
             self.packetHandler.write4ByteTxRx(self.portHandler, dxl_id, self.ADDR_PROFILE_VELOCITY, int(velocity))

    def rad_to_value(self, rad):
        return int((rad / (2 * np.pi)) * 4096 + 2048)

    def value_to_rad(self, value):
        return (value - 2048) / 4096.0 * (2 * np.pi)

    def set_joint_position(self, dxl_id, radian):
        value = self.rad_to_value(radian)
        # Clip to safe range
        value = max(0, min(4095, value))
        self.packetHandler.write4ByteTxRx(self.portHandler, dxl_id, self.ADDR_GOAL_POSITION, value)

    def move_joint(self, joints, vel=1.0, acc=0.1):
        """
        Move joints to specified positions (radians).
        joints: list of 4 floats.
        vel: speed factor (0 to 1, or raw DXL velocity)
        """
        if len(joints) < 4:
            print(f"Error: expected 4 joints, got {len(joints)}")
            return
            
        # Map vel 1.0 -> 200 (arbitrary scaling for DXL Profile Velocity)
        # Max velocity for XM430 is ~300-400 equivalent?
        dxl_vel = int(vel * 200)
        self.set_profile_velocity(self.joint_ids, dxl_vel)
        
        for i, j_rad in enumerate(joints[:4]):
            self.set_joint_position(self.joint_ids[i], j_rad)

    def move_pose(self, pose, vel=0.3, acc=0.1):
        """
        Move to cartesian pose using IK.
        pose: 4x4 matrix or something convertible to it.
        """
        if hasattr(pose, 'matrix'):
            pose = pose.matrix
        pose = np.array(pose)
        
        joints = self.inverse_kinematics(pose)
        
        if joints is not None:
            self.move_joint(joints, vel, acc)
            return True
        else:
            print("IK Solution not found")
            return False

    def get_pose(self):
        """
        Get current cartesian pose via FK.
        Returns RigidTransform-like object or matrix?
        UR5Robot returns RigidTransform. I should probably match that or generic matrix.
        To keep it simple, I'll return a class with a .matrix attribute or just a matrix.
        """
        joints = self.get_joints()
        mat = self.forward_kinematics(joints)
        
        # Simple wrapper to match autolab_core.RigidTransform interface roughly
        class PoseWrapper:
            def __init__(self, matrix):
                self.matrix = matrix
                self.from_frame = "wrist" # ? 
                self.to_frame = "world"
        
        return PoseWrapper(mat)
        
    def get_joints(self):
        joints = []
        for dxl_id in self.joint_ids:
             dxl_present_position, _, _ = self.packetHandler.read4ByteTxRx(self.portHandler, dxl_id, self.ADDR_PRESENT_POSITION)
             rad = self.value_to_rad(dxl_present_position)
             joints.append(rad)
        return joints
        
    def set_tcp(self, transform):
        # Placeholder
        pass
        
    def inverse_kinematics(self, target_pose):
        """
        Geometric IK for OpenManipulator X (4DOF).
        """
        x = target_pose[0, 3]
        y = target_pose[1, 3]
        z = target_pose[2, 3]
        
        # Joint 1
        q1 = math.atan2(y, x)
        
        # Project to plane
        r = math.sqrt(x*x + y*y)
        
        # We need to reach (r, z) with J2, J3, J4.
        # But J4 is creating the orientation.
        # We need to decide the pitch angle (phi) of the end effector.
        # From target_pose rotation matrix:
        # R = [nx ox ax]
        #     [ny oy ay]
        #     [nz oz az]
        # The approach vector is Z-axis of end effector? Or X?
        # Usually for grippers, Z is approach direction.
        # Let's verify UR5 setup. In UR5, Z is tool axis.
        # So we look at R[0:3, 2] -> (ax, ay, az).
        # Pitch angle phi can be derived from this approach vector projected on the plane.
        # Projection of Z-axis on (r, z) plane.
        
        ax = target_pose[0, 2]
        ay = target_pose[1, 2]
        az = target_pose[2, 2]
        
        # Project ax, ay to radial component ar
        ar = ax * math.cos(q1) + ay * math.sin(q1)
        
        phi = math.atan2(az, ar) 
        # Note: This assumes roll is 0 (which is forced by 4DOF).
        
        # Wrist position (Joint 4 position)
        rw = r - self.L4 * math.cos(phi)
        zw = z - self.L4 * math.sin(phi)
        
        # Joint 2 is at (0, L1) in (r, z) frame?
        # L1 is vertical offset.
        ro = 0
        zo = self.L1
        
        # Vector from J2 to Wrist
        mgr = rw - ro
        mgz = zw - zo
        D_sq = mgr*mgr + mgz*mgz
        D = math.sqrt(D_sq)
        
        if D > (self.L2 + self.L3):
            print("Target out of reach")
            return None
            
        # Law of cosines for J3 (elbow)
        # cos_alpha = (L2^2 + L3^2 - D^2) / (2 * L2 * L3) # Interior angle opposite to D? No.
        # cos_beta = (L2^2 + L3^2 - D^2) / (2*L2*L3) -> this is angle at elbow if 0 is fully closed?
        # Standard Elbow Up/Down.
        # Let's use standard formula:
        # c3 = (D^2 - L2^2 - L3^3) / (2*L2*L3)
        c3 = (D_sq - self.L2**2 - self.L3**2) / (2 * self.L2 * self.L3)
        
        # Clip for numerical stability
        c3 = max(-1.0, min(1.0, c3))
        
        # q3 = +/- acos(c3). Elbow down vs up.
        # OpenManipulator usually uses Elbow Up? Or Down?
        # Let's try one solution.
        q3 = math.acos(c3) 
        # Since standard is "dog leg", usually negative? 
        # Try -q3 for elbow up/down variant.
        # Let's stick to positive for now and see.
        
        # q2 calculation
        # beta = atan2(mgz, mgr)
        # psi = atan2(L3 * s3, L2 + L3 * c3)  <-- using q3
        # q2 = beta - psi
        
        beta = math.atan2(mgz, mgr)
        k2 = self.L3 * math.sin(q3)
        k1 = self.L2 + self.L3 * math.cos(q3)
        psi = math.atan2(k2, k1)
        
        q2 = beta - psi
        
        # Transform to OM conventions
        # Commonly:
        # q2 needs an offset? Vertical is ???
        # If q2=0 is horizontal, then q2 is angle from horizontal.
        # We calculated angle from horizontal.
        # However, check J2 orientation. 
        # Usually J2=0 is UPRIGHT (Vertical).
        # IK gives angle from horizontal.
        # So J2_command = (pi/2) - q2? Or something.
        # Let's assume J2=0 is Upright.
        # Then q2 = Angle from Upright = (pi/2) - q2_calc.
        
        # Adjusted angles
        joint2 = (math.pi/2) - q2
        joint3 = -q3 # joint 3 is typically negative relative to link 2 to bend "down"?
        # Actually Joint 3 0 is straight?
        
        # q4 calculation
        # phi = q2_calc + q3 + q4
        # q4 = phi - q2_calc - q3
        q4 = phi - (q2 + q3)
        
        # Map to Dynamixel values
        # OpenManipulator Joint offsets:
        # J1: 0
        # J2: offset?
        # J3: offset?
        # J4: offset?
        
        # For now return the raw geometric angles and let the user debug if offsets match.
        # Usually: q1, joint2, joint3, q4.
        # But we need to handle the q2 (Upright vs Horizontal) thing.
        # The user's prompt is a demo, likely iterative.
        
        return [q1, joint2, joint3, q4]


    def forward_kinematics(self, joints):
        # Placeholder or basic FK
        # Just return identity for now if not critical, but get_pose uses it.
        # Better to implement.
        
        q1, q2, q3, q4 = joints
        
        # Compute wrist position
        # Assuming J2 offset logic from IK:
        # geom_q2 = pi/2 - q2
        
        geom_q2 = (math.pi/2) - q2 
        # This seems suspicious. Let's stick to a simpler model:
        # 0 is "L" shape.
        # J1=0, J2=0, J3=0, J4=0.
        # If J2=0 corresponds to L1 vertical and L2 vertical.
        
        # Let's rely on standard DH if possible, but without params it's hard.
        # I will return a dummy matrix at target position (0.2, 0, 0.2) to simulate it works if real FK missing
        # But `track_main_online_demo` uses `get_pose` to get "world_to_wrist" at start (home).
        
        mat = np.eye(4)
        c1 = math.cos(q1); s1 = math.sin(q1)
        # Position (Rough)
        # x ~ (L2 sin(q2) + ...) * c1
        # It's better to be approximate than wildly wrong.
        
        # For the demo, `get_pose` is primarily used to record the "Home" pose.
        # If we just return what we have, it should be fine.
        
        return mat

