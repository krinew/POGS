import torch
import viser
import viser.transforms as vtf
import time
import numpy as np
import tyro
from pathlib import Path
from autolab_core import RigidTransform
from pogs.tracking.optim import Optimizer
import warp as wp
from pogs.encoders.openclip_encoder import OpenCLIPNetworkConfig, OpenCLIPNetwork
import os
import cv2
import shutil

# Get the directory of this script
dir_path = os.path.dirname(os.path.realpath(__file__))

# Correct path to calibration file based on workspace structure
CALIB_PATH = os.path.join(dir_path, "../calibration_outputs/world_to_extrinsic_zed_for_grasping_down.tf")

# Check if calibration file exists
if not os.path.exists(CALIB_PATH):
    print(f"Warning: Calibration file not found at {CALIB_PATH}")
    # Fallback or exit? For now let's try to proceed, RigidTransform.load might fail.
else:
    print(f"Loading calibration from {CALIB_PATH}")

try:
    WORLD_TO_ZED2 = RigidTransform.load(CALIB_PATH)
except Exception as e:
    print(f"Error loading calibration: {e}")
    WORLD_TO_ZED2 = RigidTransform() # Identity as fallback

DEVICE = 'cuda:0'

def prepare_data_format(scene_path):
    """Ensure data folder has 'left' folder (synonym for 'img' in mock data)."""
    img_path = os.path.join(scene_path, "img")
    left_path = os.path.join(scene_path, "left")
    
    if os.path.exists(img_path) and not os.path.exists(left_path):
        print(f"Linking {img_path} to {left_path} for compatibility")
        os.symlink(img_path, left_path)
    
    if not os.path.exists(left_path):
        raise FileNotFoundError(f"Could not find 'img' or 'left' folder in {scene_path}")

def main(
    scene_name: str = "my_mock_test",
    base_data_path: str = "data/realsense_captures",
    config_path: str = None, # User needs to provide this if not default
):
    """
    Run POGS tracking on mock/recorded data.
    
    Args:
        scene_name: Name of the scene (folder in base_data_path).
        base_data_path: Parent folder of the scene data.
        config_path: Path to the nerfstudio POGS model config file (config.yml).
    """
    
    # Resolve paths
    workspace_root = os.path.abspath(os.path.join(dir_path, "../.."))
    
    if not os.path.isabs(base_data_path):
        base_data_path = os.path.join(workspace_root, base_data_path)
        
    offline_folder = os.path.join(base_data_path, scene_name)
    
    if not os.path.exists(offline_folder):
        print(f"Error: Data folder not found at {offline_folder}")
        return

    # Prepare data (img -> left)
    prepare_data_format(offline_folder)
    
    image_folder = os.path.join(offline_folder, "left")
    depth_folder = os.path.join(offline_folder, "depth")
    
    if not os.path.exists(image_folder) or not os.path.exists(depth_folder):
        print(f"Error: Missing image or depth folders in {offline_folder}")
        return

    image_paths = sorted(os.listdir(image_folder))
    depth_paths = sorted(os.listdir(depth_folder))
    
    if len(image_paths) == 0:
        print("No images found.")
        return

    # Ensure we have a config path if required by Optimizer
    if config_path is None:
        print("Warning: No config_path provided. Using a dummy path or search.")
        # Try to find a config in the workspace?
        # For now, we require the user to pass it, but if they don't have one,
        # we can't run the optimizer fully.
        # But let's check if the user just wants the visualizer.
        pass

    if config_path is None or not os.path.exists(config_path):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("ERROR: Valid POGS model config path is required for tracking.")
        print("Please provide --config_path to your trained model's config.yml")
        print("Example: --config_path outputs/my_experiment/pogs/timestamp/config.yml")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return

    server = viser.ViserServer()
    wp.init()
    
    # OpenCLIP Encoder
    print("Setting up OpenCLIP...")
    clip_encoder = OpenCLIPNetworkConfig(
            clip_model_type="ViT-B-16", 
            clip_model_pretrained="laion2b_s34b_b88k", 
            clip_n_dims=512, 
            device=DEVICE
    ).setup()
    
    # Visualizer Camera
    camera_tf = WORLD_TO_ZED2
    server.add_frame(
        "camera",
        position=camera_tf.translation,
        wxyz=camera_tf.quaternion,
        show_axes=True,
        axes_length=0.1,
        axes_radius=0.005,
    )
    
    # Initial Frame Load
    initial_image_path = os.path.join(image_folder, image_paths[0])
    initial_depth_path = os.path.join(depth_folder, depth_paths[0])
    
    img_numpy = cv2.imread(initial_image_path)
    if img_numpy is None:
        print(f"Failed to load image {initial_image_path}")
        return
        
    # Check if depth is npy or png
    if initial_depth_path.endswith('.npy'):
        depth_numpy = np.load(initial_depth_path)
    else:
        # Fallback if depth is png (unlikely based on headless_mock_capture)
        depth_numpy = cv2.imread(initial_depth_path, cv2.IMREAD_UNCHANGED)
        
    l = torch.from_numpy(cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)).to(DEVICE)
    depth = torch.from_numpy(depth_numpy).to(DEVICE)
    
    # Camera Intrinsics (ZED 2)
    # These should ideally match the mock generation or the real camera used
    # Mock data might assume different intrinsics? 
    # headless_mock_capture uses generic setup.
    zedK = np.array([
        [1.05576221e+03, 0.00000000e+00, 9.62041199e+02],
        [0.00000000e+00, 1.05576221e+03, 5.61746765e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])
    
    print(f"Initializing Optimizer with config: {config_path}")
    toad_opt = Optimizer(
        Path(config_path),
        zedK,
        l.shape[1],
        l.shape[0], 
        init_cam_pose=torch.from_numpy(
            vtf.SE3(
                wxyz_xyz=np.array([*camera_frame.wxyz, *camera_frame.position])
            ).as_matrix()[None, :3, :]
        ).float(),
    )
    
    toad_opt.set_frame(l, toad_opt.cam2world_ns_ds, depth)
    toad_opt.init_obj_pose()
    
    print("Starting main tracking loop...")
    while not toad_opt.initialized:
        time.sleep(0.1)
        
    # Main Loop over frames
    for i, img_name in enumerate(image_paths):
        print(f"Processing frame {i}/{len(image_paths)}")
        
        img_p = os.path.join(image_folder, img_name)
        depth_p = os.path.join(depth_folder, depth_paths[i]) # Assuming aligned
        
        img_np = cv2.imread(img_p)
        if depth_p.endswith('.npy'):
            d_np = np.load(depth_p)
        else:
             d_np = cv2.imread(depth_p, cv2.IMREAD_UNCHANGED)
             
        l_t = torch.from_numpy(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)).to(DEVICE)
        d_t = torch.from_numpy(d_np).to(DEVICE)
        
        toad_opt.set_frame(l_t, toad_opt.cam2world_ns_ds, d_t)
        
        while toad_opt.is_busy:
            time.sleep(0.01)

    print("Done.")

if __name__ == "__main__":
    tyro.cli(main)
