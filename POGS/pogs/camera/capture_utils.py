import numpy as np
from autolab_core import RigidTransform, CameraIntrinsics
from autolab_core.transformations import euler_matrix, euler_from_matrix
from PIL import Image
import json
from scipy.optimize import minimize
from typing import List
                 
def estimate_cam2rob(H_chess_cams: List[RigidTransform], H_rob_worlds: List[RigidTransform]):
    '''
    Estimates transform between camera and robot end-effector frame using least-squares optimization.
    This implements the hand-eye calibration procedure using multiple observations.
    
    Parameters:
    -----------
    H_chess_cams : List[RigidTransform]
        List of transformations from chessboard frame to camera frame for each observation
    H_rob_worlds : List[RigidTransform]
        List of transformations from robot end-effector frame to world frame for each observation
        
    Returns:
    --------
    H_cam_rob : RigidTransform
        Estimated transformation from camera frame to robot end-effector frame
    H_chess_world : RigidTransform
        Estimated transformation from chessboard frame to world frame
    '''
    def residual(x):
        '''
        Computes the residual error for the current estimate of transformations.
        
        Parameters:
        -----------
        x : numpy.ndarray
            Parameter vector formatted as [x, y, z, rx, ry, rz] for chessboard-to-world 
            followed by [x, y, z, rx, ry, rz] for camera-to-robot
            
        Returns:
        --------
        float
            Sum of position and orientation errors across all observations
        '''
        err = 0
        # Extract transformations from parameter vector
        H_chess_world = RigidTransform(translation=x[:3], 
                                      rotation=euler_matrix(x[3], x[4], x[5])[:3, :3],
                                      from_frame='chess', to_frame='world')
        H_cam_rob = RigidTransform(translation=x[6:9], 
                                  rotation=euler_matrix(x[9], x[10], x[11])[:3, :3],
                                  from_frame='cam', to_frame='rob')
        
        # Compute error across all observations
        for H_chess_cam, H_rob_world in zip(H_chess_cams, H_rob_worlds):
            # Estimate chessboard-to-world transform using current parameters
            H_chess_world_est = H_rob_world * H_cam_rob * H_chess_cam
            
            # Compute translation error
            err += np.linalg.norm(H_chess_world.translation - H_chess_world_est.translation)
            
            # Compute rotation error using Euler angles
            rot_diff = H_chess_world.rotation @ np.linalg.inv(H_chess_world_est.rotation)
            eul_diff = euler_from_matrix(rot_diff)
            err += np.linalg.norm(eul_diff)
            
        print(err)
        return err
    
    # Initialize parameters with zeros
    x0 = np.zeros(12)
    
    # Perform optimization using Sequential Least Squares Programming
    res = minimize(residual, x0, method='SLSQP')
    print(res)
    
    # Alert if optimization did not converge
    if not res.success:
        input("Optimization was not successful, press enter to acknowledge")
    
    # Extract optimized parameters
    x = res.x
    H_chess_world = RigidTransform(translation=x[:3], 
                                  rotation=euler_matrix(x[3], x[4], x[5])[:3, :3],
                                  from_frame='chess', to_frame='world')
    H_cam_rob = RigidTransform(translation=x[6:9], 
                              rotation=euler_matrix(x[9], x[10], x[11])[:3, :3],
                              from_frame='cam', to_frame='rob')
    
    return H_cam_rob, H_chess_world
    

def point_at(cam_t, obstacle_t, extra_R=np.eye(3)):
    '''
    Computes a transformation that orients a camera to point at a specific 3D point.
    
    Parameters:
    -----------
    cam_t : numpy.ndarray
        3D position of the camera/gripper in world coordinates
    obstacle_t : numpy.ndarray
        3D position of the target location to point the camera at
    extra_R : numpy.ndarray, optional
        Additional rotation matrix to apply to the camera frame, useful for fine-tuning orientation
        
    Returns:
    --------
    RigidTransform
        Transformation representing camera pose pointing at the target
    '''
    # Compute the direction vector from camera to target
    dir = obstacle_t - cam_t
    z_axis = dir / np.linalg.norm(dir)
    
    # Compute the x-axis perpendicular to z-axis and world z-axis
    # Note: change the sign if camera positioning is difficult
    x_axis_dir = -np.cross(np.array((0, 0, 1)), z_axis)
    
    # Handle special case when camera direction is aligned with world z-axis
    if np.linalg.norm(x_axis_dir) < 1e-10:
        x_axis_dir = np.array((0, 1, 0))
    x_axis = x_axis_dir / np.linalg.norm(x_axis_dir)
    
    # Complete the right-handed coordinate system with y-axis
    y_axis_dir = np.cross(z_axis, x_axis)
    y_axis = y_axis_dir / np.linalg.norm(y_axis_dir)
    
    # Create rotation matrix from axes and apply extra rotation
    # Post-multiply to rotate the camera with respect to itself
    R = RigidTransform.rotation_from_axes(x_axis, y_axis, z_axis) @ extra_R
    
    # Create and return the complete transformation
    H = RigidTransform(translation=cam_t, rotation=R, from_frame='camera', to_frame='base_link')
    return H

def save_data(imgs, poses, savedir, intr: CameraIntrinsics):
    '''
    Saves a collection of images and camera poses in the NeRF dataset format.
    
    Parameters:
    -----------
    imgs : List[numpy.ndarray]
        List of images captured from each viewpoint
    poses : List[RigidTransform]
        List of camera poses corresponding to each image
    savedir : str
        Directory path to save the dataset
    intr : CameraIntrinsics
        Camera intrinsic parameters
        
    Notes:
    ------
    The NeRF format includes:
    - Individual image files
    - A transforms.json file containing camera parameters and poses
    - Special conventions for coordinate systems (flipped y and z axes)
    '''
    import os
    os.makedirs(savedir, exist_ok=True)
    
    # Initialize data dictionary for transforms.json
    data_dict = dict()
    data_dict['frames'] = []
    
    # Add camera intrinsic parameters
    data_dict['fl_x'] = intr.fx
    data_dict['fl_y'] = intr.fy
    data_dict['cx'] = intr.cx
    data_dict['cy'] = intr.cy
    data_dict['h'] = imgs[0].shape[0]
    data_dict['w'] = imgs[0].shape[1]
    
    # NeRF-specific parameters
    data_dict['aabb_scale'] = 2
    data_dict['scale'] = 1.2
    
    pil_images = []
    for i, (im, p) in enumerate(zip(imgs, poses)):
        # Remove alpha channel if present
        if im.shape[2] == 4:
            im = im[..., :3]
        
        # Save image file
        img = Image.fromarray(im)
        pil_images.append(img)
        img.save(f'{savedir}/img{i}.jpg')
        
        # Convert pose to NeRF format (flip y and z axes)
        mat = p.matrix
        mat[:3, 1] *= -1
        mat[:3, 2] *= -1
        
        # Add frame information to data dictionary
        frame = {'file_path': f'img{i}.jpg', 'transform_matrix': mat.tolist()}
        data_dict['frames'].append(frame)
    
    # Save transforms.json file
    with open(f"{savedir}/transforms.json", 'w') as fp:
        json.dump(data_dict, fp)

def load_data(savedir):
    '''
    Loads a NeRF dataset from a directory.
    
    Parameters:
    -----------
    savedir : str
        Directory path containing the NeRF dataset
        
    Returns:
    --------
    Tuple[List[numpy.ndarray], List[RigidTransform]]
        Tuple containing (images, camera_poses)
        
    Notes:
    ------
    This function reverses the coordinate system conversions applied in save_data
    to restore the original camera poses.
    '''
    import os
    if not os.path.exists(savedir):
        raise FileNotFoundError(f'{savedir} does not exist')
    
    # Load transforms.json file
    with open(f"{savedir}/transforms.json", 'r') as fp:
        data_dict = json.load(fp)
    
    poses = []
    imgs = []
    
    # Process each frame in the dataset
    for frame in data_dict['frames']:
        # Load image from file
        img = Image.open(f'{savedir}/{frame["file_path"]}')
        imgs.append(np.array(img))
        
        # Convert pose matrix from NeRF format back to original format
        mat = np.array(frame['transform_matrix'])
        mat[:3, 1] *= -1
        mat[:3, 2] *= -1
        
        # Create RigidTransform from matrix
        poses.append(RigidTransform(*RigidTransform.rotation_and_translation_from_matrix(mat)))
    
    return imgs, poses