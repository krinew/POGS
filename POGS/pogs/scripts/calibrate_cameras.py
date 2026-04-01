from ur5py.ur5 import UR5Robot
import cv2
from pogs.camera.capture_utils import estimate_cam2rob
import time
import numpy as np
from autolab_core import CameraIntrinsics, PointCloud, RigidTransform, Point
import matplotlib.pyplot as plt
import subprocess
import pyzed.sl as sl
from tqdm import tqdm
import pdb
import os
from scipy.spatial.transform import Rotation as R
import pathlib
import yaml

# Set up paths for saving calibration results
file_path = os.path.dirname(os.path.realpath(__file__))
calibration_save_path = os.path.join(file_path,'../calibration_outputs')
wrist_to_zed_mini_path = os.path.join(file_path,'../calibration_outputs/wrist_to_zed_mini.tf')

# Load camera configuration from YAML file
config_filepath = os.path.join(file_path,'../configs/camera_config.yaml')
if not os.path.exists(calibration_save_path):
    os.makedirs(calibration_save_path)

def find_corners(img, sx, sy, SB=True):
    """
    Detect checkerboard corners in an image.
    
    Args:
        img: Input image containing a checkerboard
        sx: Number of internal corners in the x direction
        sy: Number of internal corners in the y direction
        SB: Boolean flag to use OpenCV's advanced findChessboardCornersSB algorithm
        
    Returns:
        Array of detected corner coordinates or None if detection fails
    """
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(sx-1,sy-1,0)
    objp = np.zeros((sx * sy, 3), np.float32)
    objp[:, :2] = np.mgrid[0:sx, 0:sy].T.reshape(-1, 2)

    # Arrays to store object points and image points
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane

    # Convert image to grayscale for corner detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners using either standard or enhanced algorithm
    if SB:
        ret, corners = cv2.findChessboardCornersSB(gray, (sx, sy), None)
    else:
        ret, corners = cv2.findChessboardCorners(gray, (sx, sy), None)
        
    # If corners are found, refine them for higher accuracy
    if ret:
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        if corners is not None:
            return corners.squeeze()
    return None


def rvec_tvec_to_transform(rvec, tvec, to_frame):
    """
    Convert OpenCV rotation vector (rvec) and translation vector (tvec) to a RigidTransform object.
    
    Args:
        rvec: Rotation vector from OpenCV's solvePnP
        tvec: Translation vector from OpenCV's solvePnP
        to_frame: Target coordinate frame name
        
    Returns:
        RigidTransform object representing the pose
    """
    if rvec is None or tvec is None:
        return None

    # Convert Rodrigues rotation vector to rotation matrix
    R = cv2.Rodrigues(rvec)[0]
    t = tvec
    return RigidTransform(R, t, from_frame="tag", to_frame=to_frame)

def clear_tcp(robot):
    """
    Reset the tool center point (TCP) for the robot.
    
    Args:
        robot: UR5Robot object
    """
    tool_to_wrist = RigidTransform()
    tool_to_wrist.translation = np.array([0, 0, 0])
    tool_to_wrist.from_frame = "tool"
    tool_to_wrist.to_frame = "wrist"
    robot.set_tcp(tool_to_wrist)

def pose_estimation(
    frame,
    aruco_dict_type,
    matrix_coefficients,
    distortion_coefficients,
    tag_length,
    visualize=False,
):
    """
    Estimate the pose of an ArUco marker in the given frame.
    
    Args:
        frame: Input image frame containing ArUco markers
        aruco_dict_type: ArUco dictionary type for marker detection
        matrix_coefficients: Camera intrinsic matrix
        distortion_coefficients: Camera distortion coefficients
        tag_length: Physical size of the ArUco marker in meters
        visualize: Flag to enable visualization of detected markers
        
    Returns:
        Tuple containing (annotated frame, rotation vector, translation vector) or None if no markers detected
    """
    # Convert frame to grayscale for marker detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Set up ArUco detector with specified dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Detect markers in the image
    corners, ids, _ = detector.detectMarkers(gray)

    if len(corners) == 0 or len(ids) == 0:
        print("No markers found")
        return None

    # If markers are detected, estimate their 3D pose
    rvec, tvec = None, None
    if len(corners) > 0:
        # Define 3D coordinates of marker corners in marker coordinate system
        obj_points = np.array([[-tag_length / 2, tag_length / 2, 0],
                              [tag_length / 2, tag_length / 2, 0],
                              [tag_length / 2, -tag_length / 2, 0],
                              [-tag_length / 2, -tag_length / 2, 0]], dtype=np.float32)
        
        # Process each detected marker
        for i in range(0, len(ids)):
            img_points = corners[i].reshape((4, 2))
            
            # Solve perspective-n-point problem to get marker pose
            success, rvec, tvec = cv2.solvePnP(obj_points, img_points, matrix_coefficients, distortion_coefficients)
            
            if success:
                # Visualize the coordinate axes if requested
                frame_3 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_3 = cv2.drawFrameAxes(
                    frame_3, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.1
                )
                
                if visualize:
                    cv2.imshow("img", frame_3)
                    cv2.waitKey(0)
                    
                return frame, rvec, tvec
    return None

def point_at(cam_t, obstacle_t, extra_R=np.eye(3)):
    """
    Compute rotation matrix to point camera at a target.
    
    Args:
        cam_t: 3D position of camera/gripper
        obstacle_t: 3D position of target to point at
        extra_R: Additional rotation to apply (default: identity)
        
    Returns:
        Rotation matrix to orient camera toward target
    """
    # Calculate direction vector from camera to target
    direction = obstacle_t - cam_t
    z_axis = direction / np.linalg.norm(direction)
    
    # Create orthogonal coordinate system
    x_axis_dir = -np.cross(np.array((0, 0, 1)), z_axis)
    if np.linalg.norm(x_axis_dir) < 1e-10:
        # Handle singularity when z_axis is parallel to world z-axis
        x_axis_dir = np.array((0, 1, 0))
    x_axis = x_axis_dir / np.linalg.norm(x_axis_dir)
    
    y_axis_dir = np.cross(z_axis, x_axis)
    y_axis = y_axis_dir / np.linalg.norm(y_axis_dir)

    # Create rotation matrix from orthogonal axes
    R = RigidTransform.rotation_from_axes(x_axis, y_axis, z_axis)
    return R


def register_webcam():
    """
    Main function to calibrate camera-robot system. This function:
    1. Initializes the robot and cameras
    2. Collects calibration data by moving robot to different poses
    3. Estimates transformations between camera, robot, and world frames
    4. Saves calibration results for later use
    """
    # Initialize UR5 robot
    port_num = 0
    ur = UR5Robot(gripper=1)
    time.sleep(1)
    ur.gripper.open()
    clear_tcp(ur)
    
    # Define home position joints for starting the calibration
    home_joints = np.array([-1.433847729359762, -1.6635258833514612, -0.8512895742999476, -3.7683952490436, -1.4371045271502894, 3.1419787406921387])
    ur.move_joint(home_joints, vel=1.0, acc=0.1)
    
    # Initialize ZED cameras
    from pogs.camera.zed_stereo import Zed
    with open(config_filepath, 'r') as file:
        camera_parameters = yaml.safe_load(file)

    # Initialize cameras with desired settings from config file
    zed_mini = Zed(flip_mode=camera_parameters['wrist_mounted_zed']['flip_mode'],
                  resolution=camera_parameters['wrist_mounted_zed']['resolution'],
                  fps=camera_parameters['wrist_mounted_zed']['fps'],
                  cam_id=camera_parameters['wrist_mounted_zed']['id'])
                  
    extrinsic_zed = Zed(flip_mode=camera_parameters['third_view_zed']['flip_mode'],
                       resolution=camera_parameters['third_view_zed']['resolution'],
                       fps=camera_parameters['third_view_zed']['fps'],
                       cam_id=camera_parameters['third_view_zed']['id'])

    # Allow cameras to initialize
    time.sleep(1.0)
    
    # Print camera settings for verification
    print("Zed mini Exposure is set to: ",
        zed_mini.cam.get_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE),
    )
    print("Zed mini Gain is set to: ",
        zed_mini.cam.get_camera_settings(sl.VIDEO_SETTINGS.GAIN),
    )
    print("Zed mini fps set to: ",
            zed_mini.cam.get_camera_information().camera_configuration.fps)
    
    print("Extrinsic Zed Exposure is set to: ",
        extrinsic_zed.cam.get_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE),
    )
    print("Extrinsic Zed Gain is set to: ",
        extrinsic_zed.cam.get_camera_settings(sl.VIDEO_SETTINGS.GAIN),
    )
    print("Extrinsic Zed fps set to: ",
            extrinsic_zed.cam.get_camera_information().camera_configuration.fps)
    
    # Set teaching mode flag and initialize data collection arrays
    teach_mode = False
    saved_joints = []

    # Initialize transform data structures
    H_WRIST = RigidTransform(translation=[0, 0, 0]).as_frames("rob", "rob")
    ur.set_tcp(H_WRIST)
    H_chess_cams = []  # Camera to checkerboard transforms
    H_rob_worlds = []  # Robot to world transforms
    world_to_zed_extrinsic_rvecs = []  # Rotation vectors for extrinsic camera
    world_to_zed_extrinsic_tvecs = []  # Translation vectors for extrinsic camera
    
    world_to_wrists = []  # World to robot wrist transforms
    zed_mini_to_arucos = []  # Wrist-mounted camera to ArUco marker transforms
    zed_extrinsic_to_arucos = []  # External camera to ArUco marker transforms
        
    # Define center point for trajectories
    center = np.array((0, -0.5, 0))
    trajectory_path = pathlib.Path(calibration_save_path + "/prime_centered_trajectory.npy")
    traj = None
    automatic_path = False
    
    # Try to load pre-recorded trajectory if available
    if trajectory_path.exists() and not teach_mode:
        traj = np.load(trajectory_path)
        automatic_path = True
    else:
        # Otherwise, create a new trajectory through teaching
        num_poses = input("How many poses do you want to save?")
        num_poses = int(num_poses)
        traj = [0] * num_poses
    
    # Iterate through trajectory poses for data collection
    for p in tqdm(traj):
        if not automatic_path:
            # In teaching mode, let user manually position robot
            ur.start_teach()  
            input("Enter to take picture")
        else:
            # In automatic mode, move robot to pre-recorded pose
            ur.move_joint(p, vel=1.0, acc=0.1)
            time.sleep(0.1)
            
        # Capture images from both cameras
        img_zed_mini = zed_mini.get_frame()[0]
        img_zed_mini = img_zed_mini.detach().cpu().numpy()
        
        img_zed_extrinsic = extrinsic_zed.get_frame()[0]
        img_zed_extrinsic = img_zed_extrinsic.detach().cpu().numpy()
        
        # Get current robot pose
        H_rob_world = ur.get_pose()
        print("Robot joints: " + str(ur.get_joints()))
        
        # Get camera intrinsic matrices
        k_zed_mini = zed_mini.get_K()
        k_zed_extrinsic = extrinsic_zed.get_K()
        
        # Define distortion coefficients (assuming calibrated cameras)
        d = np.array([0.0, 0, 0, 0, 0])
        
        # Define ArUco tag dimensions (in meters)
        l = 0.170 
        
        # Initialize output variables
        out_zed_mini = None
        out_zed_extrinsic = None
        
        if automatic_path:
            visualize_zed_mini = teach_mode
            # Detect ArUco markers and estimate poses for both cameras
            out_zed_mini = pose_estimation(img_zed_mini, cv2.aruco.DICT_6X6_50, k_zed_mini, d, l, visualize_zed_mini)
            out_zed_extrinsic = pose_estimation(img_zed_extrinsic, cv2.aruco.DICT_6X6_50, k_zed_extrinsic, d, l, False)
            
            # Process wrist camera results if marker was detected
            if(out_zed_mini is not None):
                output_zed_mini, rvec_zed_mini, tvec_zed_mini = out_zed_mini
                zed_mini_to_aruco = rvec_tvec_to_transform(rvec_zed_mini, tvec_zed_mini, to_frame="zed_mini")
                world_to_wrist = H_rob_world.as_frames("wrist", "world")
                
                # Store transforms for calibration calculation
                world_to_wrists.append(world_to_wrist)
                zed_mini_to_arucos.append(zed_mini_to_aruco)
                H_chess_cams.append(zed_mini_to_aruco.as_frames("cb", "cam"))
                H_rob_worlds.append(H_rob_world.as_frames("rob", "world"))
            
            # Process external camera results if marker was detected
            if(out_zed_extrinsic is not None):
                output_zed_extrinsic, rvec_zed_extrinsic, tvec_zed_extrinsic = out_zed_extrinsic
                zed_extrinsic_to_aruco = rvec_tvec_to_transform(rvec_zed_extrinsic, tvec_zed_extrinsic, to_frame="zed_extrinsic")
                zed_extrinsic_to_arucos.append(zed_extrinsic_to_aruco)
        else:
            # In teaching mode, retry until marker is detected in wrist camera
            while out_zed_mini is None:
                out_zed_mini = pose_estimation(img_zed_mini, cv2.aruco.DICT_6X6_50, k_zed_mini, d, l, True)
                out_zed_extrinsic = pose_estimation(img_zed_extrinsic, cv2.aruco.DICT_6X6_50, k_zed_extrinsic, d, l, False)
                
                # If no marker found, prompt for new position
                if out_zed_mini is None:
                    input("Enter to take picture")
                    img_zed_mini = zed_mini.get_frame()[0]
                    img_zed_mini = img_zed_mini.detach().cpu().numpy()
                    
                    img_zed_extrinsic = extrinsic_zed.get_frame()[0]
                    img_zed_extrinsic = img_zed_extrinsic.detach().cpu().numpy()
                    H_rob_world = ur.get_pose()
                    print("Robot joints: " + str(ur.get_joints()))
                    k_zed_mini = zed_mini.get_K()
                    k_zed_extrinsic = extrinsic_zed.get_K()
                    d = np.array([0.0, 0, 0, 0, 0])
                    l = 0.170
                    
            # Process and store results if marker was detected in wrist camera
            if(out_zed_mini is not None):
                output_zed_mini, rvec_zed_mini, tvec_zed_mini = out_zed_mini
                zed_mini_to_aruco = rvec_tvec_to_transform(rvec_zed_mini, tvec_zed_mini, to_frame="zed_mini")
                world_to_wrist = H_rob_world.as_frames("wrist", "world")
                
                # Store transforms for calibration calculation
                world_to_wrists.append(world_to_wrist)
                zed_mini_to_arucos.append(zed_mini_to_aruco)
                H_chess_cams.append(zed_mini_to_aruco.as_frames("cb", "cam"))
                H_rob_worlds.append(H_rob_world.as_frames("rob", "world"))
                saved_joints.append(ur.get_joints())
            
            # Process external camera results if marker was detected
            if(out_zed_extrinsic is not None):
                output_zed_extrinsic, rvec_zed_extrinsic, tvec_zed_extrinsic = out_zed_extrinsic
                zed_extrinsic_to_aruco = rvec_tvec_to_transform(rvec_zed_extrinsic, tvec_zed_extrinsic, to_frame="zed_extrinsic")
                zed_extrinsic_to_arucos.append(zed_extrinsic_to_aruco)
    
    # Save collected joint positions if in teaching mode
    if(teach_mode):
        np.save(calibration_save_path + "/calibrate_extrinsics_trajectory.npy", np.array(saved_joints))
      
    # Estimate camera-to-robot transform using collected data
    H_cam_rob, H_chess_world = estimate_cam2rob(H_chess_cams, H_rob_worlds)
    # Remove the pre-specified wrist transform to get final result
    H_cam_rob = H_WRIST * H_cam_rob
    print("Estimated cam2rob:")
    print(H_cam_rob)
    print()
    print(H_chess_world)
    
    # Store the wrist-to-camera transform
    wrist_to_zed_mini = H_cam_rob
    
    # Save calibration results if user confirms
    if "n" not in input("Save? [y]/n"):
        H_cam_rob.to_frame = 'wrist'
        H_cam_rob.from_frame = 'zed_mini'
        H_cam_rob.save(calibration_save_path + "/wrist_to_zed_mini.tf")
    
    # Process external camera calibration by averaging detected poses
    zed_extrinsic_to_aruco_translations = []
    zed_extrinsic_to_aruco_eulers = []
    
    # Extract translations and Euler angles from all detected external camera poses
    for zed_extrinsic_to_aruco in zed_extrinsic_to_arucos:
        zed_extrinsic_to_aruco_translations.append(zed_extrinsic_to_aruco.translation)
        zed_extrinsic_to_aruco_eulers.append(R.from_matrix(zed_extrinsic_to_arucos[0].rotation).as_euler("xyz"))
    
    # Calculate mean translation and rotation
    mean_zed_extrinsic_to_aruco_translation = np.mean(zed_extrinsic_to_aruco_translations, axis=0)
    mean_zed_extrinsic_to_aruco_euler = np.mean(zed_extrinsic_to_aruco_eulers, axis=0)
    mean_zed_extrinsic_to_aruco_rotation = R.from_euler("xyz", mean_zed_extrinsic_to_aruco_euler).as_matrix()
    
    # Create transform from external camera to ArUco marker
    zed_extrinsic_to_aruco = RigidTransform(
        rotation=mean_zed_extrinsic_to_aruco_rotation,
        translation=mean_zed_extrinsic_to_aruco_translation,
        from_frame="tag",
        to_frame='zed_extrinsic'
    )
    
    # Calculate world-to-external-camera transform using known transformations
    for(world_to_wrist, zed_mini_to_aruco) in zip(world_to_wrists, zed_mini_to_arucos):  
        # Chain transforms: world → wrist → zed_mini → aruco → zed_extrinsic
        world_to_zed_extrinsic = world_to_wrist * wrist_to_zed_mini * zed_mini_to_aruco * zed_extrinsic_to_aruco.inverse()
        
        # Convert to rotation vector and translation for averaging
        world_to_zed_extrinsic_rvec, _ = cv2.Rodrigues(world_to_zed_extrinsic.rotation)
        world_to_zed_extrinsic_tvec = world_to_zed_extrinsic.translation
        
        # Store results
        world_to_zed_extrinsic_rvecs.append(world_to_zed_extrinsic_rvec)
        world_to_zed_extrinsic_tvecs.append(world_to_zed_extrinsic_tvec)
    
    # Calculate average translation and rotation for final transform
    world_to_zed_extrinsic_translation = np.mean(np.array(world_to_zed_extrinsic_tvecs), axis=0)
    world_to_zed_extrinsic_rotation, _ = cv2.Rodrigues(np.mean(np.array(world_to_zed_extrinsic_rvecs), axis=0))
    
    # Create final world-to-external-camera transform
    world_to_zed_extrinsic_rigid_tf = RigidTransform(
        rotation=world_to_zed_extrinsic_rotation,
        translation=world_to_zed_extrinsic_translation,
        from_frame="zed_extrinsic",
        to_frame="world"
    )
    
    print("Estimated cam2rob:")
    print(world_to_zed_extrinsic_rigid_tf)
    
    # Save external camera calibration if user confirms
    if "n" not in input("Save? [y]/n"):
        world_to_zed_extrinsic_rigid_tf.save(calibration_save_path + "/world_to_extrinsic_zed.tf")


if __name__ == "__main__":
    register_webcam()