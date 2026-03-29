import os
import sys
import argparse
import numpy as np
import time
import glob
import cv2
from sklearn.cluster import DBSCAN
from autolab_core import RigidTransform
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(DIR_PATH,'../dependencies/contact_graspnet/contact_graspnet'))
sys.path.append(os.path.join(BASE_DIR))
import prime_config_utils as config_utils
import json
from contact_grasp_estimator import GraspEstimator
from prime_visualization_utils import visualize_grasps, show_image
from PIL import Image

def load_graspnet_data(rgb_image_path):
    
    """
    Loads data from the GraspNet-1Billion dataset
    # https://graspnet.net/

    :param rgb_image_path: .png file path to depth image in graspnet dataset
    :returns: (depth, rgb, segmap, K)
    """
    
    depth = np.array(Image.open(rgb_image_path))/1000. # m to mm
    segmap = np.array(Image.open(rgb_image_path.replace('depth', 'label')))
    rgb = np.array(Image.open(rgb_image_path.replace('depth', 'rgb')))

    # graspnet images are upside down, rotate for inference
    # careful: rotate grasp poses back for evaluation
    depth = np.rot90(depth,2)
    segmap = np.rot90(segmap,2)
    rgb = np.rot90(rgb,2)
    
    if 'kinect' in rgb_image_path:
        # Kinect azure:
        K=np.array([[631.54864502 ,  0.    ,     638.43517329],
                    [  0.    ,     631.20751953, 366.49904066],
                    [  0.    ,       0.    ,       1.        ]])
    else:
        # Realsense:
        K=np.array([[616.36529541 ,  0.    ,     310.25881958],
                    [  0.    ,     616.20294189, 236.59980774],
                    [  0.    ,       0.    ,       1.        ]])

    return depth, rgb, segmap, K

def load_available_input_data(p,bounding_box_path=None,K=None, world_to_cam_tf=None):
    """
    Load available data from input file path. 
    
    Numpy files .npz/.npy should have keys
    'depth' + 'K' + (optionally) 'segmap' + (optionally) 'rgb'
    or for point clouds:
    'xyz' + (optionally) 'xyz_color'
    
    png files with only depth data (in mm) can be also loaded.
    If the image path is from the GraspNet dataset, corresponding rgb, segmap and intrinic are also loaded.
      
    :param p: .png/.npz/.npy file path that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
    :param K: 3x3 Camera Matrix with intrinsics
    :returns: All available data among segmap, rgb, depth, cam_K, pc_full, pc_colors
    """
    
    segmap, rgb, depth, pc_full, pc_colors = None, None, None, None, None
    if '.ply' in p:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(p)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        # NOTE: KARIM CHANGED THIS DEBUGGING FOR INHAND CODE
        if bounding_box_path:
            bounding_box_dict = None
            with open(bounding_box_path, 'r') as json_file:
                # Step 2: Load the contents of the file into a Python dictionary
                bounding_box_dict = json.load(json_file)
            cropped_indices = (points[:, 0] >= bounding_box_dict['x_min']) & (points[:, 0] <= bounding_box_dict['x_max']) & (points[:, 1] >= bounding_box_dict['y_min']) & (points[:, 1] <= bounding_box_dict['y_max']) & (points[:, 2] >= bounding_box_dict['z_min']) & (points[:, 2] <= bounding_box_dict['z_max'])
            points = points[cropped_indices]
            colors = colors[cropped_indices]
        ones = np.ones((points.shape[0],1))
        if world_to_cam_tf is None: # Karim Change
            import pdb
            pdb.set_trace()
            world_to_cam_tf = RigidTransform.load('/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/world_to_extrinsic_zed_for_grasping.tf').matrix
        homogenous_points_world = np.hstack((points,ones))
        cam_to_world_tf = np.linalg.inv(world_to_cam_tf)
        homogenous_points_cam = cam_to_world_tf @ homogenous_points_world.T
        points_cam = homogenous_points_cam[:3,:] / homogenous_points_cam[3,:][np.newaxis,:]
        pc_full = points_cam.T
        pc_colors = colors

        
        return segmap, rgb, depth, K, pc_full, pc_colors
    if K is not None:
        if isinstance(K,str):
            cam_K = eval(K)
        cam_K = np.array(K).reshape(3,3)

    if '.np' in p:
        data = np.load(p, allow_pickle=True)
        if '.npz' in p:
            keys = data.files
        else:
            keys = []
            if len(data.shape) == 0:
                data = data.item()
                keys = data.keys()
            elif data.shape[-1] == 3:
                pc_full = data
            else:
                depth = data

        if 'depth' in keys:
            depth = data['depth']
            if K is None and 'K' in keys:
                cam_K = data['K'].reshape(3,3)
            if 'segmap' in keys:    
                segmap = data['segmap']
            if 'seg' in keys:    
                segmap = data['seg']
            if 'rgb' in keys:    
                rgb = data['rgb']
                rgb = np.array(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        elif 'xyz' in keys:
            pc_full = np.array(data['xyz']).reshape(-1,3)
            if 'xyz_color' in keys:
                pc_colors = data['xyz_color']
    elif '.png' in p:
        if os.path.exists(p.replace('depth', 'label')):
            # graspnet data
            depth, rgb, segmap, K = load_graspnet_data(p)
        elif os.path.exists(p.replace('depths', 'images').replace('npy', 'png')):
            rgb = np.array(Image.open(p.replace('depths', 'images').replace('npy', 'png')))
        else:
            depth = np.array(Image.open(p))
    else:
        raise ValueError('{} is neither png nor npz/npy file'.format(p))
    
    return segmap, rgb, depth, cam_K, pc_full, pc_colors

def inference(global_config, checkpoint_dir, input_paths, K=None, local_regions=True, skip_border_objects=False, filter_grasps=True, segmap_id=None, z_range=[0.2,1.8], forward_passes=1):
    """
    Predict 6-DoF grasp distribution for given model and input data
    
    :param global_config: config.yaml from checkpoint directory
    :param checkpoint_dir: checkpoint directory
    :param input_paths: .png/.npz/.npy file paths that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
    :param K: Camera Matrix with intrinsics to convert depth to point cloud
    :param local_regions: Crop 3D local regions around given segments. 
    :param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
    :param filter_grasps: Filter and assign grasp contacts according to segmap.
    :param segmap_id: only return grasps from specified segmap_id.
    :param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
    :param forward_passes: Number of forward passes to run on each point cloud. Default: 1
    """
    
    # Build the model
    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(save_relative_paths=True)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    # Load weights
    grasp_estimator.load_weights(sess, saver, checkpoint_dir, mode='test')
    
    os.makedirs('results', exist_ok=True)

    # Process example test scenes
    for p in glob.glob(input_paths):
        print('Loading ', p)

        pc_segments = {}
        segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(p, K=K)
        
        if segmap is None and (local_regions or filter_grasps):
            raise ValueError('Need segmentation map to extract local regions or filter grasps')

        if pc_full is None:
            print('Converting depth to point cloud(s)...')
            pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgb,
                                                                                    skip_border_objects=skip_border_objects, z_range=z_range)

        print('Generating Grasps...')
        pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(sess, pc_full, pc_segments=pc_segments, 
                                                                                          local_regions=local_regions, filter_grasps=filter_grasps, forward_passes=forward_passes)  

        # Save results
        np.savez('results/predictions_{}'.format(os.path.basename(p.replace('png','npz').replace('npy','npz'))), 
                  pred_grasps_cam=pred_grasps_cam, scores=scores, contact_pts=contact_pts)

        # Visualize results          
        show_image(rgb, segmap)
        visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)
        
    if not glob.glob(input_paths):
        print('No files found: ', input_paths)

def filter_noise(points, colors=None):
    eps = 0.005  # Maximum distance between two samples to be considered as neighbors
    min_samples = 10  # Minimum number of samples in a neighborhood for a point to be a core point
    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)

    # Filter out the noise points (label = -1)
    filtered_pointcloud = points[labels != -1]
    if colors is not None:
        filtered_colors = colors[labels != -1]
    else:
        filtered_colors = None
    return filtered_pointcloud, filtered_colors
    
def load_ply(path, world_to_cam_tf=None):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    # eps = 0.005  # Maximum distance between two samples to be considered as neighbors
    # min_samples = 10  # Minimum number of samples in a neighborhood for a point to be a core point

    # # Apply DBSCAN
    # dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    # labels = dbscan.fit_predict(points)

    # # Filter out the noise points (label = -1)
    # points = points[labels != -1]
    # colors = colors[labels != -1]
    
    ones = np.ones((points.shape[0],1))
    if world_to_cam_tf is None: # Karim Change
        world_to_cam_tf = RigidTransform.load('/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/world_to_extrinsic_zed_for_grasping.tf').matrix
    homogenous_points_world = np.hstack((points,ones))
    cam_to_world_tf = np.linalg.inv(world_to_cam_tf)
    homogenous_points_cam = cam_to_world_tf @ homogenous_points_world.T
    points_cam = homogenous_points_cam[:3,:] / homogenous_points_cam[3,:][np.newaxis,:]
    pc = points_cam.T
    pc_colors = colors
    return pc

def get_pc_full_without_segment(pc_full, pc_segment, pc_colors):
    """
    Find the indices of points in pointcloud1 that are also in pointcloud2.

    Parameters:
    pointcloud1 (numpy.ndarray): The first point cloud (larger one).
    pointcloud2 (numpy.ndarray): The second point cloud.

    Returns:
    list: Indices of points in pointcloud1 that need to be removed.
    """
    # Convert point clouds to sets of tuples
    # pc_full_sorted = np.argsort(pc_full, axis=0)
    # ypos = np.searchsorted(x[xsorted], y)
    # indices = xsorted[ypos]
    
    # pc_full_without_segment = full_set - segment_set
    
    set1 = set(map(tuple, pc_full))
    set2 = set(map(tuple, pc_segment))
    # Find common points
    common_points = set1.intersection(set2)
    
    # Find indices of common points in pointcloud1
    idxs = [i for i, point in enumerate(map(tuple, pc_full)) if point in common_points]
    
    pc_full_without_segment = np.delete(pc_full, idxs, axis=0)
    pc_colors_without_segment = np.delete(pc_colors, idxs, axis=0)
    return pc_full_without_segment, pc_colors_without_segment

def inference_from_angle(grasp_estimator,saver,config,sess,global_config, checkpoint_dir, seg_path,full_path,bounding_box_path,world_to_cam_tf_path, K=None, local_regions=True, skip_border_objects=False, filter_grasps=True, segmap_id=None, z_range=[0.2,1.8], forward_passes=1,debug=True):
    # panda_grasp_point_to_robotiq_grasp_point = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-0.04],[0,0,0,1]]) # -0.06
    panda_grasp_point_to_robotiq_grasp_point = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.0],[0,0,0,1]]) 
    # is_down = False
    # is_forward = False
    # if('down' in world_to_cam_tf_path):
    #     is_down = True
    # elif('forward' in world_to_cam_tf_path):
    #     is_forward = True
    # else:
    #     raise ValueError('Need to be grasping down or forward')
    world_to_cam_tf = RigidTransform.load(world_to_cam_tf_path).matrix
    # Process example test scenes
    for p in glob.glob(full_path):
        print('Loading ', p)
        
        segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(p,bounding_box_path, K=K, world_to_cam_tf=world_to_cam_tf) # Karim Change
        
        queried_pc = load_ply(seg_path, world_to_cam_tf=world_to_cam_tf)
        pc_segments = {0:queried_pc}
        # delete the segment from the full point cloud and dbscan both of them individually and put them back together
        pc_full_without_segment, pc_colors_without_segment = get_pc_full_without_segment(pc_full, queried_pc, pc_colors)
        pc_full_without_segment_filtered, pc_colors_without_seg_filtered = filter_noise(pc_full_without_segment, pc_colors_without_segment)
        pc_segment_filtered, _ = filter_noise(queried_pc)
        # pc_segment_filtered = queried_pc
        pc_full_filtered = np.concatenate((pc_full_without_segment_filtered, pc_segment_filtered), axis=0)
        pc_segment = {0:pc_segment_filtered}
        
        #if segmap is None and (local_regions or filter_grasps):
        #    raise ValueError('Need segmentation map to extract local regions or filter grasps')

        if pc_full is None:
            print('Converting depth to point cloud(s)...')
            pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgb,
                                                                                    skip_border_objects=skip_border_objects, z_range=z_range)

        print('Generating Grasps...')
        pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(sess, pc_full_filtered, pc_segments=pc_segment, local_regions=local_regions, filter_grasps=filter_grasps, forward_passes=forward_passes)  

        # Save results
        np.savez('results/predictions_{}'.format(os.path.basename(p.replace('png','npz').replace('npy','npz'))), pred_grasps_cam=pred_grasps_cam, scores=scores, contact_pts=contact_pts)

        # Calculate mean and standard deviation of the scores
        # mean_score = np.mean(scores[0])
        # std_dev_score = np.std(scores[0])

        # # Define the threshold as mean + 1 standard deviation
        # threshold = mean_score #+ (1 * std_dev_score)

        # # Find indices where scores are at least 1 standard deviation above the mean
        # indices = scores[0] >= threshold

        # # Use these indices to filter pred_grasps_cam
        # pred_grasps_cam[0] = pred_grasps_cam[0][indices]
        # if(is_down):
        #     scores[0] = pred_grasps_cam[0][:,0,3]
        # elif(is_forward):
        #     scores[0] = pred_grasps_cam[0][:,2,3]
        # else:
        #     raise ValueError('Need to be grasping down or forward')

        # Visualize results          
        #show_image(rgb, segmap)
        pred_grasps_world_np = []
        for i in range(len(pred_grasps_cam[0])):
            grasp = world_to_cam_tf @ pred_grasps_cam[0][i] @ panda_grasp_point_to_robotiq_grasp_point
            pred_grasps_world_np.append(grasp)
        pred_grasps_world_np = np.array(pred_grasps_world_np)

        ones = np.ones((pc_full.shape[0],1))
        homogenous_points_cam = np.hstack((pc_full,ones))
        homogenous_points_world = world_to_cam_tf @ homogenous_points_cam.T
        points_world = homogenous_points_world[:3,:] / homogenous_points_world[3,:][np.newaxis,:]
        points_world = points_world.T
        pred_grasps_world = {}
        pred_grasps_world[0] = pred_grasps_world_np
        if(debug):
            print("Debug grasps")
            visualize_grasps(points_world, pred_grasps_world, scores, plot_opencv_cam=True, pc_colors=pc_colors)
    
        return pred_grasps_world, scores, contact_pts, points_world,pc_colors
    

def modified_inference(global_config, checkpoint_dir, seg_path,full_path,bounding_box_path, K=None, local_regions=True, skip_border_objects=False, filter_grasps=True, segmap_id=None, z_range=[0.2,1.8], forward_passes=1,debug=True):
    """
    Predict 6-DoF grasp distribution for given model and input data
    
    :param global_config: config.yaml from checkpoint directory
    :param checkpoint_dir: checkpoint directory
    :param seg_path: .png/.npz/.npy file paths that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
    :param K: Camera Matrix with intrinsics to convert depth to point cloud
    :param local_regions: Crop 3D local regions around given segments. 
    :param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
    :param filter_grasps: Filter and assign grasp contacts according to segmap.
    :param segmap_id: only return grasps from specified segmap_id.
    :param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
    :param forward_passes: Number of forward passes to run on each point cloud. Default: 1
    """
    # Build the model
    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(save_relative_paths=True)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    # Load weights
    grasp_estimator.load_weights(sess, saver, checkpoint_dir, mode='test')
    
    
    os.makedirs('results', exist_ok=True)

    world_to_cam_tf_path = os.path.dirname(os.path.realpath(__file__)) + '/../calibration_outputs/world_to_extrinsic_zed_for_grasping_down.tf'
    pred_grasps_world_down, scores_down, contact_pts_down, points_world,pc_colors = inference_from_angle(grasp_estimator=grasp_estimator,saver=saver,config=config,sess=sess,global_config=global_config, checkpoint_dir=checkpoint_dir, seg_path=seg_path,full_path=full_path,bounding_box_path=bounding_box_path,world_to_cam_tf_path=world_to_cam_tf_path, K=K, local_regions=local_regions, skip_border_objects=skip_border_objects, filter_grasps=filter_grasps, segmap_id=segmap_id, z_range=z_range, forward_passes=forward_passes,debug=debug)

    #world_to_cam_tf_path = os.path.dirname(os.path.realpath(__file__)) + '/../../ur5_interface/ur5_interface/calibration_outputs/world_to_extrinsic_zed_for_grasping_forward.tf'
    #pred_grasps_world_forward, scores_forward, contact_pts_forward, _,_ = inference_from_angle(grasp_estimator=grasp_estimator,saver=saver,config=config,sess=sess,global_config=global_config, checkpoint_dir=checkpoint_dir, seg_path=seg_path,full_path=full_path,bounding_box_path=bounding_box_path,world_to_cam_tf_path=world_to_cam_tf_path, K=K, local_regions=local_regions, skip_border_objects=skip_border_objects, filter_grasps=filter_grasps, segmap_id=segmap_id, z_range=z_range, forward_passes=forward_passes,debug=debug)
    pred_grasps_world = {}
    #pred_grasps_world[0] = np.concatenate((pred_grasps_world_down[0],pred_grasps_world_forward[0]),axis=0)
    pred_grasps_world[0] = pred_grasps_world_down[0]
    scores = scores_down
    # #scores[0] = np.concatenate((scores_down[0],scores_forward[0]),axis=0)
    # scores[0] = scores_down[0]
    # scores[0] = 1 - (scores[0] / np.max(scores[0]))
    contact_pts = {}
    contact_pts[0] = contact_pts_down[0]
    #contact_pts[0] = np.concatenate((contact_pts_down[0],contact_pts_forward[0]),axis=0)
    return pred_grasps_world, scores, contact_pts, points_world, pc_colors
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='checkpoints/scene_test_2048_bs3_hor_sigma_001', help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]')
    parser.add_argument('--np_path', default='test_data/7.npy', help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
    parser.add_argument('--png_path', default='', help='Input data: depth map png in meters')
    parser.add_argument('--K', default=None, help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
    parser.add_argument('--z_range', default=[0.2,1.8], help='Z value threshold to crop the input point cloud')
    parser.add_argument('--local_regions', action='store_true', default=False, help='Crop 3D local regions around given segments.')
    parser.add_argument('--filter_grasps', action='store_true', default=False,  help='Filter grasp contacts according to segmap.')
    parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
    parser.add_argument('--forward_passes', type=int, default=1,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    parser.add_argument('--segmap_id', type=int, default=0,  help='Only return grasps of the given object id')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    FLAGS = parser.parse_args()

    global_config = config_utils.load_config(FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs)
    
    print(str(global_config))
    print('pid: %s'%(str(os.getpid())))

    inference(global_config, FLAGS.ckpt_dir, FLAGS.np_path if not FLAGS.png_path else FLAGS.png_path, z_range=eval(str(FLAGS.z_range)),
                K=FLAGS.K, local_regions=FLAGS.local_regions, filter_grasps=FLAGS.filter_grasps, segmap_id=FLAGS.segmap_id, 
                forward_passes=FLAGS.forward_passes, skip_border_objects=FLAGS.skip_border_objects)

