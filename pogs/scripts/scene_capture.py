import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from ur5py.ur5 import UR5Robot
from pogs.camera.zed_stereo import Zed
from autolab_core import RigidTransform, DepthImage, CameraIntrinsics, PointCloud, RgbCloud
import open3d as o3d
import json
from tqdm import tqdm
import viser
import pdb
import pathlib
import pyvista as pv
import argparse
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import yaml
import pyzed.sl as sl

# ============================================================================
# Path and Configuration Setup
# ============================================================================

# Get directory of the current script
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# Define the home directory for storing datasets
HOME_DIR = os.path.join(DIR_PATH,'../data/utils/datasets')

# Load calibration data for the camera system
calibration_save_path = os.path.join(DIR_PATH,'../calibration_outputs')
wrist_to_cam = RigidTransform.load(os.path.join(DIR_PATH,'../calibration_outputs/wrist_to_zed_mini.tf'))
world_to_extrinsic_zed = RigidTransform.load(os.path.join(DIR_PATH,'../calibration_outputs/world_to_extrinsic_zed.tf'))

# Transformation matrix to convert between NeRF coordinate frame and image coordinate frame
nerf_frame_to_image_frame = np.array([[1,0,0,0],
                                        [0,-1,0,0],
                                        [0,0,-1,0],
                                        [0,0,0,1]])

# ============================================================================
# Helper Functions for Visualization and Transformations
# ============================================================================

def visualize_poses(poses, radius=0.1):
    """
    Visualize a list of poses as 3D coordinate frames with colored axes.
    
    Args:
        poses (list): List of RigidTransform objects to visualize
        radius (float): Size of the visualization
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for pose in poses:
        translation = pose.translation
        rotation = pose.rotation
        x_axis = rotation[:, 0]
        y_axis = rotation[:, 1]
        z_axis = rotation[:, 2]

        # Plot the X, Y, Z axes of the frame
        ax.quiver(
            translation[0],
            translation[1],
            translation[2],
            x_axis[0],
            x_axis[1],
            x_axis[2],
            color="r",
            length=0.01,
            normalize=True,
        )
        ax.quiver(
            translation[0],
            translation[1],
            translation[2],
            y_axis[0],
            y_axis[1],
            y_axis[2],
            color="g",
            length=0.01,
            normalize=True,
        )
        ax.quiver(
            translation[0],
            translation[1],
            translation[2],
            z_axis[0],
            z_axis[1],
            z_axis[2],
            color="b",
            length=0.01,
            normalize=True,
        )

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    plt.show()


def clear_tcp(robot):
    """
    Clear the TCP (Tool Center Point) settings of the robot.
    
    Args:
        robot: UR5Robot object
    """
    tool_to_wrist = RigidTransform()
    tool_to_wrist.translation = np.array([0, 0, 0])
    tool_to_wrist.from_frame = "tool"
    tool_to_wrist.to_frame = "wrist"
    robot.set_tcp(tool_to_wrist)


def get_hemi_translations(
    phi_min, phi_max, theta_min, theta_max, table_center, phi_div, theta_div, R
):
    """
    Generate positions on a hemisphere for robot camera positions.
    
    Args:
        phi_min, phi_max: Min/max elevation angles in degrees
        theta_min, theta_max: Min/max azimuth angles in degrees
        table_center: Center point of the table [x,y,z]
        phi_div, theta_div: Number of divisions in elevation and azimuth
        R: Radius of the hemisphere
        
    Returns:
        rel_pos: Array of 3D positions on the hemisphere
    """
    sin, cos = lambda x: np.sin(np.deg2rad(x)), lambda x: np.cos(np.deg2rad(x))
    rel_pos = np.zeros((phi_div * theta_div, 3))
    for i, phi in enumerate(np.linspace(phi_min, phi_max, phi_div)):
        tmp_pose = []
        for j, theta in enumerate(np.linspace(theta_min, theta_max, theta_div)):
            tmp_pose.append(
                np.array(
                    [R * sin(phi) * cos(theta), R * sin(phi) * sin(theta), R * cos(phi)]
                )
            )
        if i % 2 == 1:
            tmp_pose.reverse()
        for k, pose in enumerate(tmp_pose):
            rel_pos[i * theta_div + k] = pose

    return rel_pos + table_center


def get_rotation(point, center):
    """
    Calculate rotation matrix to point from center to point.
    
    Args:
        point: Target point [x,y,z]
        center: Origin point [x,y,z]
        
    Returns:
        R: Rotation matrix
    """
    direction = point - center
    z_axis = direction / np.linalg.norm(direction)

    x_axis_dir = -np.cross(np.array((0, 0, 1)), z_axis)
    if np.linalg.norm(x_axis_dir) < 1e-10:
        x_axis_dir = np.array((0, 1, 0))
    x_axis = x_axis_dir / np.linalg.norm(x_axis_dir)
    y_axis_dir = np.cross(z_axis, x_axis)
    y_axis = y_axis_dir / np.linalg.norm(y_axis_dir)

    R = RigidTransform.rotation_from_axes(x_axis, y_axis, z_axis)
    return R


def point_at(cam_t, obstacle_t, extra_R=np.eye(3)):
    """
    Calculate rotation matrix to point camera at a target.
    
    Args:
        cam_t: Camera position [x,y,z]
        obstacle_t: Target position [x,y,z]
        extra_R: Additional rotation to apply
        
    Returns:
        R: Rotation matrix
    """
    direction = obstacle_t - cam_t
    z_axis = direction / np.linalg.norm(direction)
    x_axis_dir = -np.cross(np.array((0, 0, 1)), z_axis)
    if np.linalg.norm(x_axis_dir) < 1e-10:
        x_axis_dir = np.array((0, 1, 0))
    x_axis = x_axis_dir / np.linalg.norm(x_axis_dir)
    y_axis_dir = np.cross(z_axis, x_axis)
    y_axis = y_axis_dir / np.linalg.norm(y_axis_dir)

    # Postmultiply the extra rotation to rotate the camera WRT itself
    R = RigidTransform.rotation_from_axes(x_axis, y_axis, z_axis)
    return R


def save_pose(base_to_wrist, i=0, save_dir=None):
    """
    Save camera pose information for NeRF dataset.
    
    Args:
        base_to_wrist: Transform from robot base to wrist
        i: Frame index
        save_dir: Directory to save pose files
        
    Returns:
        cam_pose.matrix: Camera pose as a 4x4 matrix
    """
    base_to_wrist.from_frame = "wrist"
    base_to_wrist.to_frame = "base"

    wrist_to_cam.from_frame = "cam"
    wrist_to_cam.to_frame = "wrist"

    cam_to_nerfcam = RigidTransform(
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
        np.zeros(3),
        from_frame="nerf_cam",
        to_frame="cam",
    )

    # Calculate camera pose in NeRF coordinate system
    cam_pose = base_to_wrist * wrist_to_cam * cam_to_nerfcam

    if save_dir is not None:
        np.savetxt(os.path.join(save_dir, f"{i:03d}.txt"), cam_pose.matrix)

    return cam_pose.matrix


def set_up_dirs(scene_name):
    """
    Create and set up directory structure for storing dataset.
    
    Args:
        scene_name: Name of the scene/dataset
        
    Returns:
        Dictionary of directory paths
    """
    img_save_dir = f"{HOME_DIR}/{scene_name}/img"
    img_r_save_dir = f"{HOME_DIR}/{scene_name}/img_r"
    depth_save_dir = f"{HOME_DIR}/{scene_name}/depth"
    depth_viz_save_dir = f"{HOME_DIR}/{scene_name}/depth_png"
    pose_save_dir = f"{HOME_DIR}/{scene_name}/poses"

    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(img_r_save_dir, exist_ok=True)
    os.makedirs(depth_save_dir, exist_ok=True)
    os.makedirs(depth_viz_save_dir, exist_ok=True)
    os.makedirs(pose_save_dir, exist_ok=True)
    return {
        "img": img_save_dir,
        "img_r": img_r_save_dir,
        "depth": depth_save_dir,
        "depth_png": depth_viz_save_dir,
        "poses": pose_save_dir,
    }


def save_imgs(img_l, img_r, depth, i, save_dirs, flip_table=False):
    """
    Save RGB and depth images to disk.
    
    Args:
        img_l: Left RGB image
        img_r: Right RGB image
        depth: Depth image
        i: Frame index
        save_dirs: Dictionary of directory paths
        flip_table: Flag for table flipping operation
    """
    img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
    img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
    i = i + 1
    cv2.imwrite(os.path.join(save_dirs["img"], f"frame_{i:05d}.png"), img_l)
    cv2.imwrite(os.path.join(save_dirs["img_r"], f"frame_{i:05d}.png"), img_r)
    np.save(os.path.join(save_dirs["depth"], f"frame_{i:05d}.npy"), depth)
    plt.imsave(os.path.join(save_dirs["depth_png"], f"frame_{i:05d}.png"), depth, cmap='jet')


def table_rejection_depth(depth, camera_intr, transform):
    """
    Process depth image to remove table points.
    
    Args:
        depth: Depth image
        camera_intr: Camera intrinsics
        transform: Camera pose transform
        
    Returns:
        depth_im: Processed depth image
        mask: Segmentation mask
    """
    # Convert depth to point cloud
    depth_im = DepthImage(depth, frame="zed")
    point_cloud = camera_intr.deproject(depth_im)
    point_cloud = PointCloud(point_cloud.data, frame="zed")
    
    # Transform point cloud to world frame
    tsfm = RigidTransform(
        *RigidTransform.rotation_and_translation_from_matrix(transform),
        from_frame="zed",
        to_frame="base",
    )
    point_cloud = tsfm * point_cloud

    # Use RANSAC to segment the plane (table)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud.data.T)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=1000
    )
    
    # Remove table points and filter by height
    point = np.delete(point_cloud.data.T, inliers, axis=0)
    point = np.delete(point, np.where(point[:, 2] < -0.05), axis=0)

    # Project filtered points back to depth image
    pc = tsfm.inverse() * PointCloud(point[:, :3].T, frame="base")
    depth_im = camera_intr.project_to_image(pc).raw_data[:, :, 0]

    # Create mask and handle special cases
    mask = np.zeros_like(depth_im)
    neg_depth = np.where(depth_im <= 0)
    depth_im[neg_depth] = 0.6
    mask[neg_depth] = 1

    hi_depth = np.where(depth_im > 0.8)
    depth_im[hi_depth] = 0.6
    mask[hi_depth] = 1

    cloud = DepthImage(depth_im, "zed").point_normal_cloud(camera_intr).point_cloud
    return depth_im, np.array(mask, dtype=bool)


def isolateTable(cam, proper_world_to_cam):
    """
    Identify and isolate table boundaries from point cloud data.
    
    Args:
        cam: ZED camera object
        proper_world_to_cam: Transform from world to camera
        
    Returns:
        Table boundary coordinates and height
    """
    # Get RGB and depth data
    img_l, img_r = cam.get_rgb()
    depth, points, rgbs = cam.get_depth_image_and_pointcloud(img_l, img_r, from_frame="zed_mini")
    
    # Use Open3D to segment the plane (table)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.data.T)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=1000
    )
    
    # Cluster points on the table plane
    table_points = points.data.T[inliers]
    db = DBSCAN(eps=0.02, min_samples=20).fit(table_points)
    label_set = set(db.labels_)
    label_set.discard(-1)

    print(label_set)
    
    # Find the largest cluster and determine the table boundaries
    max_size = 0
    x_min_cam = 0
    x_max_cam = 0
    y_min_cam = 0
    y_max_cam = 0
    z_min_cam = 0
    z_max_cam = 0
    
    for label in label_set:
        filtered_table_point_mask = db.labels_ == label
        filtered_table_pointcloud = points.data.T[inliers][filtered_table_point_mask]
        
        min_bounding_cube_camera_frame = np.array([
            np.min(filtered_table_pointcloud[:,0]),
            np.min(filtered_table_pointcloud[:,1]),
            np.min(points.data.T[:,2]),1
        ]).reshape(-1,1)
        
        max_bounding_cube_camera_frame = np.array([
            np.max(filtered_table_pointcloud[:,0]),
            np.max(filtered_table_pointcloud[:,1]),
            np.max(filtered_table_pointcloud[:,2]),1
        ]).reshape(-1,1)
        
        x_min_cam_cluster = min_bounding_cube_camera_frame[0,0]
        y_min_cam_cluster = min_bounding_cube_camera_frame[1,0]
        z_min_cam_cluster = min_bounding_cube_camera_frame[2,0]
        x_max_cam_cluster = max_bounding_cube_camera_frame[0,0]
        y_max_cam_cluster = max_bounding_cube_camera_frame[1,0]
        z_max_cam_cluster = max_bounding_cube_camera_frame[2,0]
        
        if(len(filtered_table_pointcloud) > max_size):
            max_size = len(filtered_table_pointcloud)
            x_min_cam = x_min_cam_cluster
            x_max_cam = x_max_cam_cluster
            y_min_cam = y_min_cam_cluster
            y_max_cam = y_max_cam_cluster
            z_min_cam = z_min_cam_cluster
            z_max_cam = z_max_cam_cluster

    # Transform table bounds to world coordinates
    global_pointcloud = proper_world_to_cam.apply(points)
    
    min_bounding_cube_camera_frame = np.array([x_min_cam, y_min_cam, z_min_cam, 1]).reshape(-1, 1)
    max_bounding_cube_camera_frame = np.array([x_max_cam, y_max_cam, z_max_cam, 1]).reshape(-1, 1)
    table_height_camera_frame = np.array([0, 0, -plane_model[3], 1]).reshape(-1, 1)
    
    min_bounding_cube_world = proper_world_to_cam.matrix @ min_bounding_cube_camera_frame
    max_bounding_cube_world = proper_world_to_cam.matrix @ max_bounding_cube_camera_frame
    table_height_world_frame = proper_world_to_cam.matrix @ table_height_camera_frame
    
    # Extract coordinates from transformation matrices
    x_min_world = min_bounding_cube_world[0,0]
    y_min_world = min_bounding_cube_world[1,0]
    z_min_world = min_bounding_cube_world[2,0]
    x_max_world = max_bounding_cube_world[0,0]
    y_max_world = max_bounding_cube_world[1,0]
    z_max_world = max_bounding_cube_world[2,0]
    
    # Ensure min values are smaller than max values
    if(x_min_world > x_max_world):
        temp = x_min_world
        x_min_world = x_max_world
        x_max_world = temp
    if(y_min_world > y_max_world):
        temp = y_min_world
        y_min_world = y_max_world
        y_max_world = temp
    if(z_min_world > z_max_world):
        temp = z_min_world
        z_min_world = z_max_world
        z_max_world = temp
        
    table_height = table_height_world_frame[2,0]
    return x_min_world, x_max_world, y_min_world, y_max_world, z_min_world, z_max_world, table_height
    
    
def convert_pointcloud_to_image(points, rgbs, K, image_width, image_height):
    """
    Project a point cloud to a 2D image with RGB values.
    
    Args:
        points: Point cloud data
        rgbs: RGB values for each point
        K: Camera intrinsic matrix
        image_width, image_height: Dimensions of output image
        
    Returns:
        image_inpainted: RGB image with holes filled
        depth_inpainted: Depth image with holes filled
    """
    # Prepare points for projection
    ones = np.ones((points.data.T.shape[0], 1))
    homogenous_points = np.hstack((points.data.T, ones))
    projected_points = K @ homogenous_points.T
    projected_pixels = projected_points[:2] / projected_points[2]
    projected_pixels = projected_pixels.T
    projected_pixels = np.round(projected_pixels).astype('int')
    
    # Initialize output images
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    image_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    depth_image = np.zeros((image_height, image_width), dtype='float32')
    
    # Fill images with projected points
    for i in range(projected_pixels.shape[0]):
        x, y = int(projected_pixels[i, 0]), int(projected_pixels[i, 1])
        if 0 <= x < image_width and 0 <= y < image_height:
            image[y, x] = rgbs.data.T[i]
            image_mask[y, x] = 255
            depth_image[y, x] = points.data.T[i, 2]

    # Fill holes in the images
    image_mask_inverted = cv2.bitwise_not(image_mask)
    image_inpainted = cv2.inpaint(image, image_mask_inverted, 3, cv2.INPAINT_TELEA)
    depth_inpainted = cv2.inpaint(depth_image, image_mask_inverted, 3, cv2.INPAINT_TELEA)
    
    return image_inpainted, depth_inpainted


def save_json(data, filename):
    """
    Save data to a JSON file, overwriting if it exists.
    
    Args:
        data: Data to save
        filename: Output file path
    """
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
        
        
def save_poses_with_diff_cameras(poses_dir, intrinsics_dict_list, single_cam=False):
    """
    Save camera poses and intrinsics in NeRF-compatible format.
    
    Args:
        poses_dir: Directory containing pose files
        intrinsics_dict_list: List of camera intrinsics dictionaries
        single_cam: Flag for single camera setup
    """
    img_dir = 'img'
    depth_dir = 'depth'
    extrinsics_dicts = []
    
    print(os.listdir(poses_dir))
    num_files = len(os.listdir(poses_dir))
    assert num_files == len(intrinsics_dict_list), "Make sure you save an intrinsics dictionary for each image frame"
    print(num_files)
    
    for i in range(num_files):    
        frame_dict = intrinsics_dict_list[i]
        if single_cam:
            i += 1
        transform_mat = np.loadtxt(os.path.join(poses_dir, f"{i:03d}.txt")) 
        frame_dict['file_path'] = os.path.join(img_dir, f"frame_{i+1:05d}.png")
        frame_dict['depth_file_path'] = os.path.join(depth_dir, f"frame_{i+1:05d}.npy")
        frame_dict['transform_matrix'] = transform_mat.tolist()
        extrinsics_dicts.append(frame_dict)
        
    final_dict = {
        'frames': extrinsics_dicts,
        "ply_file_path": "sparse_pc.ply"
    }
    save_json(final_dict, os.path.join(poses_dir, "..", "transforms.json"))
    
def prime_sphere_main(scene_name, single_image=False, flip_table=False):
    """
    Main function for collecting multi-view images, depth maps, and point clouds of a scene.
    Uses a UR5 robot arm with mounted ZED camera to capture data from multiple viewpoints.
    
    Args:
        scene_name (str): Name of the scene for organizing saved data
        single_image (bool): If True, only captures one image instead of following trajectory
        flip_table (bool): If True, flips the table orientation in saved images
        
    Returns:
        int: Status code (1 for success)
    """
    debug = False  # Toggle debug visualization
    save_dirs = set_up_dirs(scene_name)  # Create directory structure for saving data
    use_robot, use_cam, use_2_cams = True, True, True  # Configure which hardware components to use
    save_nerfstudio_intrinsics_per_frame_list = []  # Store camera intrinsics for NeRFStudio format
    
    if use_robot:
        # Initialize robot and set to home position
        robot = UR5Robot(gripper=1)
        clear_tcp(robot)  # Clear tool center point configuration
        
        # Set and move to home joint configuration
        home_joints = np.array([-1.433847729359762, -1.6635258833514612, -0.8512895742999476, -3.7683952490436, -1.4371045271502894, 3.1419787406921387])
        robot.move_joint(home_joints, vel=1.0, acc=0.1)
        
        # Get current pose and transform to camera frame
        world_to_wrist = robot.get_pose()
        world_to_wrist.from_frame = "wrist"
        world_to_cam = world_to_wrist * wrist_to_cam
        proper_world_to_wrist = world_to_cam * wrist_to_cam.inverse()
        
        # Move robot to corrected pose
        robot.move_pose(proper_world_to_wrist, vel=1.0, acc=0.1)
        robot.gripper.open()

    if use_cam:
        # Initialize camera(s) with parameters from config file
        save_joints = False
        saved_joints = []
        file_path = os.path.dirname(os.path.realpath(__file__))
        config_filepath = os.path.join(file_path, '../configs/camera_config.yaml')
        with open(config_filepath, 'r') as file:
            camera_parameters = yaml.safe_load(file)
    
        # Initialize wrist-mounted ZED camera
        cam = Zed(
            flip_mode=camera_parameters['wrist_mounted_zed']['flip_mode'],
            resolution=camera_parameters['wrist_mounted_zed']['resolution'],
            fps=camera_parameters['wrist_mounted_zed']['fps'],
            cam_id=camera_parameters['wrist_mounted_zed']['id']
        )
        time.sleep(1.0)  # Allow camera to initialize
        
        # Print camera settings for debugging
        print("Zed mini Exposure is set to: ",
            cam.cam.get_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE),
        )
        print("Zed mini Gain is set to: ",
            cam.cam.get_camera_settings(sl.VIDEO_SETTINGS.GAIN),
        )
        print("Zed mini fps set to: ",
            cam.cam.get_camera_information().camera_configuration.fps
        )
        camera_parameters['wrist_mounted_zed']['gain'] = cam.cam.get_camera_settings(sl.VIDEO_SETTINGS.GAIN)[1]
        camera_parameters['wrist_mounted_zed']['exposure'] = cam.cam.get_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE)[1]
        # Initialize external (third-view) ZED camera if using dual camera setup
        extrinsic_zed = None
        if use_2_cams:
            extrinsic_zed = Zed(
                flip_mode=camera_parameters['third_view_zed']['flip_mode'],
                resolution=camera_parameters['third_view_zed']['resolution'],
                fps=camera_parameters['third_view_zed']['fps'],
                cam_id=camera_parameters['third_view_zed']['id']
            )
            time.sleep(1.0)  # Allow camera to initialize
            
            # Print camera settings for debugging
            print("Extrinsic Zed Exposure is set to: ",
                extrinsic_zed.cam.get_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE),
            )
            print("Extrinsic Zed Gain is set to: ",
                extrinsic_zed.cam.get_camera_settings(sl.VIDEO_SETTINGS.GAIN),
            )
            print("Extrinsic Zed fps set to: ",
                extrinsic_zed.cam.get_camera_information().camera_configuration.fps
            )
            camera_parameters['third_view_zed']['gain'] = extrinsic_zed.cam.get_camera_settings(sl.VIDEO_SETTINGS.GAIN)[1]
            camera_parameters['third_view_zed']['exposure'] = extrinsic_zed.cam.get_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE)[1]

        with open(config_filepath, 'w') as file:
            yaml.dump(camera_parameters,file,default_flow_style=False)
    # Initialize global point cloud variables
    global_pointcloud = None
    global_rgbcloud = None
    
    # Capture initial point cloud from external camera if available
    if extrinsic_zed is not None:
        # Get RGB stereo pair
        img_l, img_r = extrinsic_zed.get_rgb()
        
        # Get depth and point cloud from stereo images
        depth, points, rgbs = extrinsic_zed.get_depth_image_and_pointcloud(img_l, img_r, from_frame="zed_extrinsic")
        
        # Set up camera parameters for projection
        K = np.array([[cam.f_, 0, cam.cx_, 0], [0, cam.f_, cam.cy_, 0], [0, 0, 1, 0]])
        image_width, image_height = cam.width_, cam.height_

        # Convert point cloud to image representation
        image_inpainted, depth_inpainted = convert_pointcloud_to_image(points, rgbs, K, image_width, image_height)
        # Transform points to world coordinate frame
        points_world_frame = world_to_extrinsic_zed.apply(points)
        global_pointcloud = points_world_frame.data.T
        global_rgbcloud = rgbs.data.T
        
        # Visualize point clouds in debug mode
        if debug:
            debug_server = viser.ViserServer()
            debug_server.add_point_cloud('extrinsic_pc', points=global_pointcloud, colors=global_rgbcloud, point_size=0.001)
            
            # Get additional point cloud from wrist-mounted camera for debugging
            img_l, img_r = cam.get_rgb()
            wrist_pose = robot.get_pose()
            wrist_pose.from_frame = 'wrist'
            print(wrist_pose)
            cam_pose = wrist_pose * wrist_to_cam
            depth, points, rgbs = cam.get_depth_image_and_pointcloud(img_l, img_r, from_frame="cam")
            points_world_frame = cam_pose.apply(points)
            debug_server.add_point_cloud('top_down_pc', points=points_world_frame.data.T, colors=rgbs.data.T, point_size=0.001)
        
        # Transform and save first camera pose
        world_to_extrinsic_zed_image_frame = world_to_extrinsic_zed.matrix @ nerf_frame_to_image_frame
        world_to_extrinsic_zed_image_rigid_tf = RigidTransform(
            rotation=world_to_extrinsic_zed_image_frame[:3, :3],
            translation=world_to_extrinsic_zed_image_frame[:3, 3],
            from_frame="zed_extrinsic",
            to_frame="world"
        )
        np.savetxt(os.path.join(save_dirs["poses"], "000.txt"), world_to_extrinsic_zed_image_rigid_tf.matrix)
        
        # Save first image set
        save_imgs(
            image_inpainted,
            image_inpainted,
            depth_inpainted,
            0,
            save_dirs,
            flip_table=flip_table,
        ) 
        
        # Save camera intrinsics for NeRFStudio (We can't yet support different camera sizes so we are downsizing the 1080p photo to 720p)
        save_nerfstudio_intrinsics_per_frame_list.append(cam.get_ns_intrinsics())
    
    # Reset tool center point to wrist joint
    tool_to_wrist = RigidTransform()
    tool_to_wrist.translation = np.array([0, 0, 0])
    tool_to_wrist.from_frame = "tool"
    tool_to_wrist.to_frame = "wrist"
    robot.set_tcp(tool_to_wrist)
    
    # Isolate table region in the scene for later point cloud processing
    x_min_world, x_max_world, y_min_world, y_max_world, z_min_world, z_max_world, table_height = isolateTable(cam, world_to_cam)
    
    # Load predefined robot trajectory for capturing multiple viewpoints
    trajectory_path = pathlib.Path(calibration_save_path + "/prime_centered_trajectory.npy")
    joints = np.load(str(trajectory_path))
    
    # Initialize lists to store images and poses
    left_images = []
    right_images = []
    world_to_images = []
    
    # Execute trajectory and collect data at each point
    if use_robot:
        for i, joint in enumerate(tqdm(joints)):
            # Additional waypoint for smooth motion for the second position
            if i == 1:
                joint_waypoint = np.array([-0.029453579579488576, -1.678246800099508, -0.8632600943194788, -3.762667957936422, -1.3856657187091272, 3.1419308185577393])
                robot.move_joint(joint_waypoint, vel=0.7, acc=0.15)
                time.sleep(1.0)
            
            # Move to trajectory point
            robot.move_joint(joint, vel=0.7, acc=0.15)
            time.sleep(1.0)  # Allow robot to stabilize
            
            # Get current robot pose
            wrist_pose = robot.get_pose()
            print(wrist_pose)
            
            # Save camera pose (transform from wrist to camera)
            cam_pose = save_pose(wrist_pose, i+1, save_dirs["poses"])
            
            # Transform camera pose to image frame for NeRF compatibility
            world_to_image_frame = cam_pose @ nerf_frame_to_image_frame
            print("From frame: " + str("image_frame_"+str(i+1)))
            
            # Create rigid transform representation
            world_to_image_rigid_tf = RigidTransform(
                rotation=world_to_image_frame[:3, :3],
                translation=world_to_image_frame[:3, 3],
                from_frame="image_frame_"+str(i+1),
                to_frame="world"
            )
            world_to_images.append(world_to_image_rigid_tf)
            
            # Capture images if camera is enabled
            if use_cam:
                img_l, img_r = cam.get_rgb()
                left_images.append(img_l)
                right_images.append(img_r)
                
                # Note: Table rejection depth code is commented out
                # This would filter depth maps to exclude the table surface
                # cam_intr = cam.get_intr()
                # no_t_depth, seg_mask = table_rejection_depth(
                #     depth,
                #     cam_intr,
                #     cam_pose,
                # )
                # save_imgs(
                #     img_l,
                #     img_r,
                #     depth,
                #     # no_t_depth,
                #     # seg_mask,
                #     i,
                #     save_dirs,
                #     flip_table=flip_table,
                # ) 
            
            time.sleep(0.05)  # Short delay between trajectory points
    
    # Process all collected images to build combined point cloud
    # Note: First image (index 0) comes from external ZED2 camera, so we start at i=1
    i = 1
    for (left_image, right_image, world_to_image) in zip(left_images, right_images, world_to_images):
        # Generate point cloud from stereo images
        depth, points, rgbs = cam.get_depth_image_and_pointcloud(left_image, right_image, from_frame="image_frame_"+str(i))
        
        # Transform point cloud to world frame
        pointcloud_world_frame = world_to_image.apply(points)
        
        # Append to global point cloud
        if global_pointcloud is None:
            global_pointcloud = pointcloud_world_frame.data.T
            global_rgbcloud = rgbs.data.T
        else:
            # Optionally visualize individual point clouds in debug mode
            if debug:
                debug_server.add_point_cloud(
                    'hemisphere_photo_'+str(i-1),
                    pointcloud_world_frame.data.T,
                    rgbs.data.T,
                    visible=False,
                    point_size=0.001
                )
                import pdb
                pdb.set_trace()
            
            # Stack points and colors to build complete point cloud
            global_pointcloud = np.vstack((global_pointcloud, pointcloud_world_frame.data.T))
            global_rgbcloud = np.vstack((global_rgbcloud, rgbs.data.T))
        
        # Save images and depth maps
        save_imgs(
            left_image,
            right_image,
            depth,
            i,
            save_dirs,
            flip_table=flip_table,
        )
        
        # Save camera intrinsics for NeRFStudio
        save_nerfstudio_intrinsics_per_frame_list.append(cam.get_ns_intrinsics())
        print("Made depth image " + str(i) + "/" + str(len(left_images)))
        i += 1
    
    # Save camera intrinsics and combined camera poses
    if use_cam:
        camera_intr = cam.get_intr()
        camera_intr.save(f"{HOME_DIR}/{scene_name}/zed.intr")
        save_poses_with_diff_cameras(
            save_dirs["poses"],
            save_nerfstudio_intrinsics_per_frame_list,
            single_cam=(extrinsic_zed == None)
        )
    
    # Filter point cloud to focus on relevant region (near table)
    close_pointcloud = global_pointcloud[
        (global_pointcloud[:, 0] >= x_min_world) & (global_pointcloud[:, 0] <= x_max_world) &
        (global_pointcloud[:, 1] >= y_min_world) & (global_pointcloud[:, 1] <= y_max_world) &
        (global_pointcloud[:, 2] >= z_min_world) & (global_pointcloud[:, 2] <= z_max_world)
    ]
    close_rgbcloud = global_rgbcloud[
        (global_pointcloud[:, 0] >= x_min_world) & (global_pointcloud[:, 0] <= x_max_world) &
        (global_pointcloud[:, 1] >= y_min_world) & (global_pointcloud[:, 1] <= y_max_world) &
        (global_pointcloud[:, 2] >= z_min_world) & (global_pointcloud[:, 2] <= z_max_world)
    ]
    
    # Subsample point cloud for efficient processing
    num_gaussians_initialization = 200000
    
    # Randomly sample from close region (near table)
    close_gaussian_indices = np.random.choice(close_pointcloud.shape[0], num_gaussians_initialization, replace=False)
    subsampled_close_pointcloud = close_pointcloud[close_gaussian_indices]
    subsampled_close_rgbcloud = close_rgbcloud[close_gaussian_indices]
    
    # Remove outliers using DBSCAN clustering
    db = DBSCAN(eps=0.005, min_samples=20)
    labels = db.fit_predict(subsampled_close_pointcloud)
    subsampled_close_pointcloud = subsampled_close_pointcloud[labels != -1]
    subsampled_close_rgbcloud = subsampled_close_rgbcloud[labels != -1]
    
    # Process points outside the close region (background)
    not_close_pointcloud = global_pointcloud[
        ~((global_pointcloud[:, 0] >= x_min_world) & (global_pointcloud[:, 0] <= x_max_world) &
        (global_pointcloud[:, 1] >= y_min_world) & (global_pointcloud[:, 1] <= y_max_world) &
        (global_pointcloud[:, 2] >= z_min_world) & (global_pointcloud[:, 2] <= z_max_world))
    ]
    not_close_rgbcloud = global_rgbcloud[
        ~((global_pointcloud[:, 0] >= x_min_world) & (global_pointcloud[:, 0] <= x_max_world) &
        (global_pointcloud[:, 1] >= y_min_world) & (global_pointcloud[:, 1] <= y_max_world) &
        (global_pointcloud[:, 2] >= z_min_world) & (global_pointcloud[:, 2] <= z_max_world))
    ]
    
    # Sample from background proportionally
    not_close_gaussian_initialization = num_gaussians_initialization * not_close_pointcloud.shape[0] / close_pointcloud.shape[0]
    not_close_gaussian_indices = np.random.choice(
        not_close_pointcloud.shape[0],
        int(not_close_gaussian_initialization),
        replace=False
    )
    subsampled_not_close_pointcloud = not_close_pointcloud[not_close_gaussian_indices]
    subsampled_not_close_rgbcloud = not_close_rgbcloud[not_close_gaussian_indices]
    
    # Combine foreground and background into final point cloud
    full_subsampled_pointcloud = np.vstack((subsampled_close_pointcloud, subsampled_not_close_pointcloud))
    full_subsampled_rgbcloud = np.vstack((subsampled_close_rgbcloud, subsampled_not_close_rgbcloud))
    
    # Final subsampling to desired point count
    final_indices = np.random.choice(
        full_subsampled_pointcloud.shape[0],
        min(full_subsampled_pointcloud.shape[0], num_gaussians_initialization),
        replace=False
    )
    final_pointcloud = full_subsampled_pointcloud[final_indices]
    final_rgbcloud = full_subsampled_rgbcloud[final_indices]
    
    # Visualize final point cloud
    server = viser.ViserServer()
    server.add_point_cloud(
        name="full_pointcloud",
        points=final_pointcloud,
        colors=final_rgbcloud,
        point_size=0.001
    )
    
    # Save point cloud in Open3D format (PLY)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_pointcloud)
    pcd.colors = o3d.utility.Vector3dVector(final_rgbcloud / 255.)
    o3d.io.write_point_cloud(os.path.join(save_dirs['poses'], '..', 'sparse_pc.ply'), pcd)
    
    # Save table boundaries for later use
    table_bounding_cube = {
        'x_min': x_min_world,
        'x_max': x_max_world,
        'y_min': y_min_world,
        'y_max': y_max_world,
        'z_min': z_min_world,
        'z_max': z_max_world,
        'table_height': table_height
    }
    with open(os.path.join(save_dirs['poses'], '..', 'table_bounding_cube.json'), 'w') as json_file:
        json.dump(table_bounding_cube, json_file, indent=4)
    
    # Note: Robot return to home position code is commented out
    # collection_finish_joints = np.array(robot.get_joints())
    # collection_finish_joints[-1] = -np.pi / 2
    # collection_finish_joints[-2] = np.pi / 2
    # robot.move_joint(collection_finish_joints)
    # ...

    # Disconnect robot controller
    robot.ur_c.disconnect()
    
    # Wait for user input before closing visualization
    input("Kill pointcloud?")
    return 1


def table_paste_dir(scene_name):
    """
    Process depth maps for a collected scene, potentially applying table rejection.
    
    This function loads depth maps from a scene directory, processes them, and
    saves visualization images of the depth maps.
    
    Args:
        scene_name (str): Path to the scene directory containing transforms.json
    """
    # Load camera transformation data
    with open(os.path.join(scene_name, "transforms.json")) as r:
        transform_json = json.load(r)
        print(transform_json)
    
    # Load camera intrinsics
    cam_intrinsics = CameraIntrinsics.load("zed.intr")
    
    # Process each frame in the dataset
    for i, frame_dict in enumerate(tqdm(transform_json["frames"])):
        # Load depth map
        path = os.path.join(
            scene_name, "depth", frame_dict["file_path"].split("/")[-1][:-3] + "npy"
        )
        depth = np.load(path)
        
        # Note: Table rejection commented out
        # new_depth, mask = table_rejection_depth(
        #     depth, cam_intrinsics, np.array(frame_dict["transform_matrix"])
        # )
        
        # Save visualization of depth map
        plt.imshow(depth)
        plt.savefig(f"{HOME_DIR}/{scene_name}/depth_png/{i:03d}.png")
        
        # Save processed depth (in this case, just the original depth)
        np.save(path, depth)
        # np.save(os.path.join(scene_name, "seg_mask", path.split("/")[-1]), mask)


if __name__ == "__main__":
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--scene", type=str, required=True)
    args = argparser.parse_args()
    scene_name = args.scene
    
    # Run main function to collect scene data
    prime_sphere_main(scene_name)