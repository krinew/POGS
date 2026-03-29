import torch
import viser
import viser.transforms as vtf
import time
import pyzed.sl as sl
import numpy as np
import tyro
from pathlib import Path
from autolab_core import RigidTransform
from pogs.tracking.tri_zed import Zed
from pogs.tracking.optim import Optimizer
import warp as wp
from pogs.encoders.openclip_encoder import OpenCLIPNetworkConfig, OpenCLIPNetwork
from pogs.tracking.toad_object import ToadObject
import yaml
import os
from ur5py.ur5 import UR5Robot
import open3d as o3d

# Path to the directory containing this script
dir_path = os.path.dirname(os.path.realpath(__file__))

# Pre-calibrated transforms between coordinate frames
WORLD_TO_ZED2 = RigidTransform.load(dir_path+"/../calibration_outputs/world_to_extrinsic_zed.tf")
WRIST_TO_CAM = RigidTransform.load(dir_path + "/../calibration_outputs/wrist_to_zed_mini.tf")

DEVICE = 'cuda:0'

def clear_tcp(robot):
    """
    Reset the Tool Center Point (TCP) of the robot to the default position.
    
    Args:
        robot: UR5Robot instance to reset TCP for
    """
    tool_to_wrist = RigidTransform()
    tool_to_wrist.translation = np.array([0, 0, 0])
    tool_to_wrist.from_frame = "tool"
    tool_to_wrist.to_frame = "wrist"
    robot.set_tcp(tool_to_wrist)

def create_mesh_box(width, height, depth, dx=0, dy=0, dz=0):
    """
    Create a 3D box mesh for visualization purposes.
    
    Args:
        width (float): Width of the box
        height (float): Height of the box
        depth (float): Depth of the box
        dx (float): X offset for the box position
        dy (float): Y offset for the box position
        dz (float): Z offset for the box position
    
    Returns:
        open3d.geometry.TriangleMesh: Box mesh with specified dimensions
    """
    box = o3d.geometry.TriangleMesh()
    vertices = np.array([[0,0,0],
                        [width,0,0],
                        [0,0,depth],
                        [width,0,depth],
                        [0,height,0],
                        [width,height,0],
                        [0,height,depth],
                        [width,height,depth]])
    vertices[:,0] += dx
    vertices[:,1] += dy
    vertices[:,2] += dz
    triangles = np.array([[4,7,5],[4,6,7],[0,2,4],[2,6,4],
                        [0,1,2],[1,3,2],[1,5,7],[1,7,3],
                        [2,3,7],[2,7,6],[0,4,1],[1,4,5]])
    box.vertices = o3d.utility.Vector3dVector(vertices)
    box.triangles = o3d.utility.Vector3iVector(triangles)
    return box

def plot_gripper_pro_max(center, R, width, depth, score=1, color=None):
    """
    Create a visualizable mesh of a robotic gripper.
    
    Args:
        center (numpy.ndarray): Target point (3,) as gripper center
        R (numpy.ndarray): Rotation matrix (3,3) of gripper
        width (float): Gripper width (distance between fingers)
        depth (float): Depth of the gripper fingers
        score (float): Grasp quality score from 0 to 1
        color (tuple): Optional RGB color tuple to override score-based coloring
    
    Returns:
        tuple: (open3d.geometry.TriangleMesh, numpy.ndarray) Gripper mesh and its color
    """
    x, y, z = center
    height = 0.004
    finger_width = 0.004
    tail_length = 0.04
    depth_base = 0.02
    
    if color is not None:
        color_r, color_g, color_b = color
    else:
        color_r = 1 - score  # red for low score
        color_g = score  # green for high score
        color_b = 0 
    
    # Create the component parts of the gripper
    left = create_mesh_box(depth+depth_base+finger_width, finger_width, height)
    right = create_mesh_box(depth+depth_base+finger_width, finger_width, height)
    bottom = create_mesh_box(finger_width, width, height)
    tail = create_mesh_box(tail_length, finger_width, height)

    # Position the left finger
    left_points = np.array(left.vertices)
    left_triangles = np.array(left.triangles)
    left_points[:,0] -= depth_base + finger_width
    left_points[:,1] -= width/2 + finger_width
    left_points[:,2] -= height/2

    # Position the right finger
    right_points = np.array(right.vertices)
    right_triangles = np.array(right.triangles) + 8
    right_points[:,0] -= depth_base + finger_width
    right_points[:,1] += width/2
    right_points[:,2] -= height/2

    # Position the bottom connecting piece
    bottom_points = np.array(bottom.vertices)
    bottom_triangles = np.array(bottom.triangles) + 16
    bottom_points[:,0] -= finger_width + depth_base
    bottom_points[:,1] -= width/2
    bottom_points[:,2] -= height/2

    # Position the tail (gripper base)
    tail_points = np.array(tail.vertices)
    tail_triangles = np.array(tail.triangles) + 24
    tail_points[:,0] -= tail_length + finger_width + depth_base
    tail_points[:,1] -= finger_width / 2
    tail_points[:,2] -= height/2

    # Combine all parts into a single mesh
    vertices = np.concatenate([left_points, right_points, bottom_points, tail_points], axis=0)
    vertices = np.dot(R, vertices.T).T + center
    triangles = np.concatenate([left_triangles, right_triangles, bottom_triangles, tail_triangles], axis=0)
    colors = np.array([[color_r,color_g,color_b] for _ in range(len(vertices))])

    gripper = o3d.geometry.TriangleMesh()
    gripper.vertices = o3d.utility.Vector3dVector(vertices)
    gripper.triangles = o3d.utility.Vector3iVector(triangles)
    gripper.vertex_colors = o3d.utility.Vector3dVector(colors)
    return gripper, colors[0]

def main(
    config_path: Path = Path("/home/lifelong/pogs/pogs/data/utils/datasets/outputs/20250305_prime_drill/pogs/2025-03-05_180006/config.yml"),
):
    """
    Main function for the POGS (Perception for Object Grasping System) demo.
    Implements interactive object tracking, language querying, and grasp execution.
    
    Args:
        config_path: Path to the configuration file containing model parameters and settings.
    """
    # Initialize visualization server
    server = viser.ViserServer()
    wp.init()
    
    # Create UI controls
    opt_init_handle = server.add_gui_button("Set initial frame", disabled=True)
    
    # Initialize CLIP model for language understanding
    clip_encoder = OpenCLIPNetworkConfig(
            clip_model_type="ViT-B-16", 
            clip_model_pretrained="laion2b_s34b_b88k", 
            clip_n_dims=512, 
            device='cuda:0'
                ).setup()
    assert isinstance(clip_encoder, OpenCLIPNetwork)
    
    # Add UI elements for user interaction
    text_handle = server.add_gui_text("Positives", "", disabled=True)
    query_handle = server.add_gui_button("Query", disabled=True)
    generate_grasps_handle = server.add_gui_button("Generate Grasps on Query", disabled=True)
    execute_grasp_handle = server.add_gui_button("Execute Grasp for Query", disabled=True)
    
    # Load camera configuration from YAML
    config_filepath = os.path.join(dir_path, '../configs/camera_config.yaml')
    with open(config_filepath, 'r') as file:
        camera_parameters = yaml.safe_load(file)

    # Initialize ZED camera with parameters from config
    zed = Zed(flip_mode=camera_parameters['third_view_zed']['flip_mode'],
              resolution=camera_parameters['third_view_zed']['resolution'],
              fps=camera_parameters['third_view_zed']['fps'],
              cam_id=camera_parameters['third_view_zed']['id'])
    
    # Apply camera settings from capture session to ensure consistency
    zed.cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, camera_parameters['third_view_zed']['exposure'])
    zed.cam.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, camera_parameters['third_view_zed']['gain'])

    time.sleep(1.0)  # Allow camera to initialize
    
    # Initialize robot arm
    robot = UR5Robot(gripper=1)
    clear_tcp(robot)
    
    # Move robot to home position
    home_joints = np.array([-1.433847729359762, -1.6635258833514612, -0.8512895742999476, -3.7683952490436, -1.4371045271502894, 3.1419787406921387])
    robot.move_joint(home_joints, vel=1.0, acc=0.1)
    world_to_wrist = robot.get_pose()
    world_to_wrist.from_frame = "wrist"

    # Get camera transformation
    camera_tf = WORLD_TO_ZED2
            
    # Add camera visualization to the scene
    camera_frame = server.add_frame(
        "camera",
        position=camera_tf.translation,
        wxyz=camera_tf.quaternion,
        show_axes=True,
        axes_length=0.1,
        axes_radius=0.005,
    )
    server.add_mesh_trimesh(
        "camera/mesh",
        mesh=zed.zed_mesh,
        scale=0.001,
        position=zed.cam_to_zed.translation,
        wxyz=zed.cam_to_zed.quaternion,
    )

    # Get initial frame from camera
    l, _, depth = zed.get_frame(depth=True)
    
    # Initialize the neural object tracker
    toad_opt = Optimizer(
        config_path,
        zed.get_K(),
        l.shape[1],
        l.shape[0], 
        init_cam_pose=torch.from_numpy(
            vtf.SE3(
                wxyz_xyz=np.array([*camera_frame.wxyz, *camera_frame.position])
            ).as_matrix()[None, :3, :]
        ).float(),
    )

    @opt_init_handle.on_click
    def _(_):
        """
        Callback for initializing the tracking optimization.
        Gets current frame from camera and initializes object pose.
        """
        assert (zed is not None) and (toad_opt is not None)
        opt_init_handle.disabled = True
        l, _, depth = zed.get_frame(depth=True)
        toad_opt.set_frame(l, toad_opt.cam2world_ns, depth)
        with zed.raft_lock:
            toad_opt.init_obj_pose()
        query_handle.disabled = False
        
    opt_init_handle.disabled = False
    text_handle.disabled = False
    
    @query_handle.on_click
    def _(_):
        """
        Callback for processing language queries.
        Uses CLIP to identify objects matching the query text.
        """
        text_positives = text_handle.value
        
        clip_encoder.set_positives(text_positives.split(";"))
        if len(clip_encoder.positives) > 0:
            # Calculate relevancy scores based on CLIP embeddings
            relevancy = toad_opt.get_clip_relevancy(clip_encoder)
            group_masks = toad_opt.optimizer.group_masks

            # Find object with highest relevancy to query
            relevancy_avg = []
            for mask in group_masks:
                relevancy_avg.append(torch.mean(relevancy[:,0:1][mask]))
            relevancy_avg = torch.tensor(relevancy_avg)
            toad_opt.max_relevancy_label = torch.argmax(relevancy_avg).item()
            toad_opt.max_relevancy_text = text_positives
            generate_grasps_handle.disabled = False
            
        else:
            print("No language query provided")
    
    @generate_grasps_handle.on_click
    def _(_):
        """
        Callback for generating grasp poses.
        Exports object meshes and uses grasp planning to find optimal grasps.
        """
        print("Enter generate grasps")
        # Export mesh of the queried object
        toad_opt.state_to_ply(toad_opt.max_relevancy_label)
        local_ply_filename = str(toad_opt.config_path.parent.joinpath("local.ply"))
        global_ply_filename = str(toad_opt.config_path.parent.joinpath("global.ply"))
        table_bounding_cube_filename = str(toad_opt.pipeline.datamanager.get_datapath().joinpath("table_bounding_cube.json"))
        save_dir = str(toad_opt.config_path.parent)
        print("Starting generate grasps")
        # Run grasp planning algorithm
        ToadObject.generate_grasps(local_ply_filename, global_ply_filename, table_bounding_cube_filename, save_dir)
        print("End generate grasps")
        
        # Optionally visualize table bounding box for collision avoidance
        vis_table_bounding_cube = False
        if vis_table_bounding_cube:
            import json
            with open(table_bounding_cube_filename, 'r') as json_file:
                bounding_box_dict = json.load(json_file)

            def create_box(box_data, name: str, server, color = (1.0, 1.0, 1.0)):
                """Helper function to create a bounding box visualization"""
                # Extract min and max values from the dictionary
                x_min = box_data["x_min"]
                x_max = box_data["x_max"]
                y_min = box_data["y_min"]
                y_max = box_data["y_max"]
                z_min = box_data["z_min"]
                z_max = box_data["z_max"]

                # Calculate dimensions (width, height, depth)
                width = x_max - x_min
                height = y_max - y_min
                depth = z_max - z_min

                # Calculate the position (center of the box)
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                center_z = (z_min + z_max) / 2
                # Calculate the position (center of the box)
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                center_z = (z_min + z_max) / 2
                
                server.add_box(
                    name=name,
                    color=color,
                    dimensions=(width, height, depth),
                    position=(center_x, center_y, center_z),
                    visible=True,
                    )
            create_box(bounding_box_dict, name="table_boundaries", server=server)
   
        # Load and visualize the best grasp pose
        grasp_point_world = np.load(os.path.join(save_dir,'grasp_point_world.npy'))
        
        center = grasp_point_world[:3,3]
        rotation_matrix = grasp_point_world[:3,:3]
        correction_rot = np.array([[0,-1,0], [0,0,-1], [1,0,0]])  # Correction for visualization purposes
        vis_rotation_matrix = rotation_matrix @ correction_rot
        grasp_mesh, grasp_color = plot_gripper_pro_max(center, R=vis_rotation_matrix, width=0.085, depth=0.1016)

        # Add the grasp visualization to the scene
        server.add_mesh_simple(
            name="grasp",
            vertices=np.asarray(grasp_mesh.vertices),
            faces=np.asarray(grasp_mesh.triangles),
            color=grasp_color
        )
        
        # Enable execute grasp button after planning is complete
        execute_grasp_handle.disabled = False

    @execute_grasp_handle.on_click
    def _(_):
        """
        Callback for executing the planned grasp with the robot.
        Plans and executes a pre-grasp, grasp, and post-grasp trajectory.
        """
        save_dir = str(toad_opt.config_path.parent)
        best_grasp = np.load(os.path.join(save_dir,'grasp_point_world.npy'))
        
        # Apply Z-axis rotation if grasp is on the negative Y side of the workspace
        if(best_grasp[0,1] < 0):
            rotate_180_z = np.array([[-1,0,0,0],
                                     [0,-1,0,0],
                                     [0,0,1,0],
                                     [0,0,0,1]])
            best_grasp = best_grasp @ rotate_180_z
            
        # Create visualization frames for the grasp poses
        grasp_point_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        grasp_point_world.transform(best_grasp)
        
        # Create pre-grasp position (offset along Z-axis)
        pre_grasp_tf = np.array([[1,0,0,0],
                                [0,1,0,0],
                                [0,0,1,-0.1],  # 10cm offset in Z direction
                                [0,0,0,1]])
        pre_grasp_world_frame = best_grasp @ pre_grasp_tf
        pre_grasp_point_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        pre_grasp_point_world.transform(pre_grasp_world_frame)
        
        # Create post-grasp position (smaller offset for lifting object)
        post_grasp_tf = np.array([[1,0,0,0],
                                [0,1,0,0],
                                [0,0,1,-0.05],  # 5cm offset in Z direction for lifting
                                [0,0,0,1]])
        post_grasp_world_frame = best_grasp @ post_grasp_tf
        
        # Convert to RigidTransform for robot control
        post_grasp_rigid_tf = RigidTransform(rotation=post_grasp_world_frame[:3,:3], translation=post_grasp_world_frame[:3,3])
        pre_grasp_rigid_tf = RigidTransform(rotation=pre_grasp_world_frame[:3,:3], translation=pre_grasp_world_frame[:3,3])
        
        # Execute the grasp sequence
        robot.gripper.open()  # Open gripper
        time.sleep(1)
        robot.move_pose(pre_grasp_rigid_tf, vel=0.3, acc=0.1)  # Move to pre-grasp position
        time.sleep(1)
        final_grasp_rigid_tf = RigidTransform(rotation=best_grasp[:3,:3], translation=best_grasp[:3,3])
        robot.move_pose(final_grasp_rigid_tf, vel=0.3, acc=0.1)  # Move to grasp position
        time.sleep(1)
        robot.gripper.close()  # Close gripper to grasp object
        time.sleep(1)
        robot.move_pose(post_grasp_rigid_tf, vel=0.3, acc=0.1)  # Lift object
        time.sleep(1)

    # Lists to store frames for debugging or recording
    real_frames = []
    rendered_rgb_frames = []

    # Initialize object labels list
    obj_label_list = [None for _ in range(toad_opt.num_groups)]
    
    # Main tracking and visualization loop
    while True:
        if zed is not None:
            start_time = time.time()
            # Get new frame from camera
            left, right, depth = zed.get_frame()
            
            assert isinstance(toad_opt, Optimizer)
            if toad_opt.initialized:
                start_time3 = time.time()
                # Update tracker with new observation
                toad_opt.set_observation(left, toad_opt.cam2world_ns, depth)
                
                # Run optimization iterations
                n_opt_iters = 25
                with zed.raft_lock:
                    outputs = toad_opt.step_opt(niter=n_opt_iters)

                # Add current camera image to visualization
                server.add_image(
                    "cam/zed_left",
                    left.cpu().detach().numpy(),
                    render_width=left.shape[1]/2500,
                    render_height=left.shape[0]/2500,
                    position = (0.5, 0.5, 0.5),
                    wxyz=(0, -0.7071068, -0.7071068, 0),
                    visible=True
                )
                real_frames.append(left.cpu().detach().numpy())
                
                # Add rendered RGB from neural tracking to visualization
                server.add_image(
                    "cam/gs_render",
                    outputs["rgb"].cpu().detach().numpy(),
                    render_width=left.shape[1]/2500,
                    render_height=left.shape[0]/2500,
                    position = (0.5, -0.5, 0.5),
                    wxyz=(0, -0.7071068, -0.7071068, 0),
                    visible=True
                )
                rendered_rgb_frames.append(outputs["rgb"].cpu().detach().numpy())
                
                # Update object transforms and meshes in visualization
                tf_list = toad_opt.get_parts2world()
                for idx, tf in enumerate(tf_list):
                    server.add_frame(
                        f"object/group_{idx}",
                        position=tf.translation(),
                        wxyz=tf.rotation().wxyz,
                        show_axes=True,
                        axes_length=0.05,
                        axes_radius=.001
                    )
                    mesh = toad_opt.toad_object.meshes[idx]
                    server.add_mesh_trimesh(
                        f"object/group_{idx}/mesh",
                        mesh=mesh,
                    )
                    # Add label to the queried object
                    if idx == toad_opt.max_relevancy_label:
                        obj_label_list[idx] = server.add_label(
                        f"object/group_{idx}/label",
                        text=toad_opt.max_relevancy_text,
                        position = (0,0,0.05),
                        )
                    else:
                        if obj_label_list[idx] is not None:
                            obj_label_list[idx].remove()

            # Visualize 3D point cloud from depth image
            K = torch.from_numpy(zed.get_K()).float().cuda()
            assert isinstance(left, torch.Tensor) and isinstance(depth, torch.Tensor)
            points, colors = Zed.project_depth(left, depth, K, depth_threshold=1.0, subsample=100)
            server.add_point_cloud(
                "camera/points",
                points=points,
                colors=colors,
                point_size=0.001,
            )

        else:
            time.sleep(1)
        
if __name__ == "__main__":
    tyro.cli(main)