import pyzed.sl as sl
import numpy as np
from autolab_core import CameraIntrinsics, PointCloud, RgbCloud
from raftstereo.raft_stereo import *
from raftstereo.utils.utils import InputPadder
import argparse
from dataclasses import dataclass, field
from typing import List
import os
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# Configuration class for the RAFT stereo model parameters
class RAFTConfig:
    # Path to the pre-trained RAFT stereo model checkpoint
    restore_ckpt: str = str(os.path.join(DIR_PATH,'../dependencies/raftstereo/models/raftstereo-middlebury.pth'))
    # Hidden dimensions for network layers
    hidden_dims: List[int] = [128]*3
    # Correlation implementation type
    corr_implementation: str = "reg"
    # Whether to use a shared backbone for both images
    shared_backbone: bool = False
    # Correlation pyramid levels for feature matching
    corr_levels: int = 4
    # Radius of correlation lookups
    corr_radius: int = 4
    # Number of downsampling layers
    n_downsample: int = 2
    # Normalization type for context network
    context_norm: str = "batch"
    # Whether to use slow-fast GRU for iterative updates
    slow_fast_gru: bool = False
    # Enable mixed precision for faster computation and memory efficiency
    mixed_precision: bool = False
    # Number of GRU layers in the update block
    n_gru_layers: int = 3
    
# ZED stereo camera interface with RAFT stereo depth estimation
class Zed:
    """
    ZED camera wrapper for stereo image capture and depth estimation using RAFT-Stereo algorithm.
    Provides functions for camera initialization, image capture, depth estimation, and point cloud generation.
    """
    def __init__(self, flip_mode, resolution, fps, cam_id=None, recording_file=None, start_time=0.0):
        """
        Initialize the ZED camera with specified parameters.
        
        Args:
            flip_mode (bool): Whether to flip the camera image
            resolution (str): Camera resolution ('720p', '1080p', or '2k')
            fps (int): Frames per second
            cam_id (int, optional): Camera serial number for multi-camera setups
            recording_file (str, optional): Path to SVO recording file for playback
            start_time (float, optional): Start time in seconds for SVO playback
        """
        init = sl.InitParameters()
        if cam_id is not None:
            init.set_from_serial_number(cam_id)
            self.cam_id = cam_id
        self.height_ = None
        self.width_ = None
        
        # Set camera flip mode
        if flip_mode:
            init.camera_image_flip = sl.FLIP_MODE.ON
        else:
            init.camera_image_flip = sl.FLIP_MODE.OFF
            
        # Configure for SVO file playback if provided
        if recording_file is not None:
            init.set_from_svo_file(recording_file)
            
        # Configure camera resolution
        if resolution == '720p':
            init.camera_resolution = sl.RESOLUTION.HD720
            self.height_ = 720
            self.width_ = 1280
        elif resolution == '1080p':
            init.camera_resolution = sl.RESOLUTION.HD1080
            self.height_ = 1080
            self.width_ = 1920
        elif resolution == '2k':
            init.camera_resolution = sl.RESOLUTION.HD2k
            self.height_ = 1242
            self.width_ = 2208
        else:
            print("Only 720p, 1080p, and 2k supported by Zed")
            exit()
            
        # Disable native ZED depth computation (we'll use RAFT-Stereo instead)
        init.depth_mode = sl.DEPTH_MODE.NONE
        init.sdk_verbose = 1
        init.camera_fps = fps
        self.cam = sl.Camera()
        init.camera_disable_self_calib = True
        
        # Open the camera with the specified parameters
        status = self.cam.open(init)
        self.recording_file = recording_file
        self.start_time = start_time
        
        # Set SVO playback position if applicable
        if recording_file is not None:
            fps = self.cam.get_camera_information().camera_configuration.fps
            self.cam.set_svo_position(int(start_time * fps))
            
        # Check if camera opened successfully
        if status != sl.ERROR_CODE.SUCCESS:
            print("Camera Open : " + repr(status) + ". Exit program.")
            exit()
        else:
            print("Opened camera")
            
        # Calculate stereo parameters for depth estimation
        left_cx = self.get_K(cam="left")[0, 2]
        right_cx = self.get_K(cam="right")[0, 2]
        self.cx_diff = right_cx - left_cx  # Horizontal principal point difference for disparity calculation
        self.f_ = self.get_K(cam="left")[0,0]  # Focal length
        self.cx_ = left_cx  # Principal point x-coordinate
        self.cy_ = self.get_K(cam="left")[1,2]  # Principal point y-coordinate
        self.Tx_ = self.get_stereo_transform()[0,3]  # Baseline (translation between cameras)
        
        # RAFT-Stereo parameters
        self.valid_iters_ = 32  # Number of iterations for RAFT-Stereo flow estimation
        self.padder_ = InputPadder(torch.empty(1,3,self.height_,self.width_).shape, divis_by=32)
        self.model = self.create_raft()  # Initialize RAFT-Stereo model

    def create_raft(self):
        """
        Initialize and load the RAFT-Stereo model for depth estimation.
        
        Returns:
            torch.nn.Module: Loaded RAFT-Stereo model in evaluation mode
        """
        raft_args = RAFTConfig()
        model = torch.nn.DataParallel(RAFTStereo(raft_args), device_ids=[0])
        model.load_state_dict(torch.load(raft_args.restore_ckpt))

        model = model.module
        model = model.to('cuda')
        model = model.eval()
        return model
        
    def load_image_raft(self, im):
        """
        Prepare image for RAFT-Stereo model inference.
        
        Args:
            im (numpy.ndarray): RGB image array
            
        Returns:
            torch.Tensor: Processed image tensor on GPU
        """
        img = torch.from_numpy(im).permute(2,0,1).float()
        return img[None].to('cuda')
    
    def get_depth_image_and_pointcloud(self, left_img, right_img, from_frame):
        """
        Compute depth image and point cloud from stereo images using RAFT-Stereo.
        
        Args:
            left_img (numpy.ndarray): Left RGB image
            right_img (numpy.ndarray): Right RGB image
            from_frame (str): Coordinate frame name for point cloud
            
        Returns:
            tuple: (depth_image, points, rgbs) containing depth map and colored point cloud
        """
        with torch.no_grad():
            # Prepare images for RAFT model
            image1 = self.load_image_raft(left_img)
            image2 = self.load_image_raft(right_img)
            image1, image2 = self.padder_.pad(image1, image2)
            
            # Run RAFT-Stereo to compute disparity (negative of optical flow)
            _, flow_up = self.model(image1, image2, iters=self.valid_iters_, test_mode=True)
            flow_up = self.padder_.unpad(flow_up).squeeze()

            flow_up_np = -flow_up.detach().cpu().numpy().squeeze()
            
            # Convert disparity to depth using the stereo camera equation:
            # depth = (focal_length * baseline) / disparity
            depth_image = (self.f_ * self.Tx_) / abs(flow_up_np + self.cx_diff)  
            rows, cols = depth_image.shape
            y, x = np.meshgrid(range(rows), range(cols), indexing="ij")

            # Convert depth image to x,y,z point cloud using the pinhole camera model
            Z = depth_image  # Depth values
            # X = (x - cx) * Z / fx
            X = (x - self.cx_) * Z / self.f_
            # Y = (y - cy) * Z / fy
            Y = (y - self.cy_) * Z / self.f_
            points = np.stack((X,Y,Z), axis=-1)
            rgbs = left_img
            
            # Remove points with zero depth (invalid or background points)
            non_zero_indices = np.all(points != [0, 0, 0], axis=-1)
            points = points[non_zero_indices]
            rgbs = rgbs[non_zero_indices]
            
            # Format as PointCloud and RgbCloud objects for compatibility with autolab_core
            points = points.reshape(-1,3)
            rgbs = rgbs.reshape(-1,3)
            points = PointCloud(points.T, from_frame)
            rgbs = RgbCloud(rgbs.T, from_frame)
            return depth_image, points, rgbs 

    def get_frame(self, depth=True, cam="left"):
        """
        Capture a frame from the ZED camera and optionally compute depth.
        
        Args:
            depth (bool): Whether to compute depth
            cam (str): Which camera to use as reference ("left" or "right")
            
        Returns:
            tuple: (left_image, right_image, depth_map) or None if frame capture failed
        """
        res = sl.Resolution()
        res.width = self.width_
        res.height = self.height_
        if self.cam.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve stereo images
            left_rgb = sl.Mat()
            right_rgb = sl.Mat()
            self.cam.retrieve_image(left_rgb, sl.VIEW.LEFT, sl.MEM.CPU, res)
            self.cam.retrieve_image(right_rgb, sl.VIEW.RIGHT, sl.MEM.CPU, res)
            left, right = (
                torch.from_numpy(
                    np.flip(left_rgb.get_data()[..., :3], axis=2).copy()
                ).cuda(),
                torch.from_numpy(
                    np.flip(right_rgb.get_data()[..., :3], axis=2).copy()
                ).cuda(),
            )
            
            # Compute depth if requested
            if depth:
                left_torch, right_torch = left.permute(2, 0, 1), right.permute(2, 0, 1)

                # Handle different reference views (left or right camera)
                if cam == "left":
                    flow = raft_inference(left_torch, right_torch, self.model)
                else:
                    right_torch = torch.flip(right_torch, dims=[2])
                    left_torch = torch.flip(left_torch, dims=[2])
                    flow = raft_inference(right_torch, left_torch, self.model)

                # Compute depth from disparity using stereo camera equation
                fx = self.get_K()[0, 0]  # Focal length
                depth = (
                    fx * self.get_stereo_transform()[0, 3] / (flow.abs() + self.cx_diff)
                )

                if cam != "left":
                    depth = torch.flip(depth, dims=[1])
            else:
                depth = None
            return left, right, depth
        elif self.cam.grab() == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            print("End of recording file")
            return None, None, None
        else:
            raise RuntimeError("Could not grab frame")

    def get_K(self, cam="left"):
        """
        Get camera intrinsic matrix (calibration matrix K).
        
        Args:
            cam (str): Which camera to use ("left" or "right")
            
        Returns:
            numpy.ndarray: 3x3 intrinsic matrix K
        """
        calib = (
            self.cam.get_camera_information().camera_configuration.calibration_parameters
        )
        if cam == "left":
            intrinsics = calib.left_cam
        else:
            intrinsics = calib.right_cam
        K = np.array(
            [
                [intrinsics.fx, 0, intrinsics.cx],
                [0, intrinsics.fy, intrinsics.cy],
                [0, 0, 1],
            ]
        )
        return K

    def get_intr(self, cam="left"):
        """
        Get camera intrinsics in autolab_core format.
        
        Args:
            cam (str): Which camera to use ("left" or "right")
            
        Returns:
            CameraIntrinsics: Camera intrinsics object
        """
        calib = (
            self.cam.get_camera_information().camera_configuration.calibration_parameters
        )
        if cam == "left":
            intrinsics = calib.left_cam
        else:
            intrinsics = calib.right_cam
        return CameraIntrinsics(
            frame="zed",
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.cx,
            cy=intrinsics.cy,
            width=1280,
            height=720,
        )

    def get_stereo_transform(self):
        """
        Get transformation matrix from left to right camera.
        
        Returns:
            numpy.ndarray: 4x4 transformation matrix (in meters)
        """
        transform = (
            self.cam.get_camera_information().camera_configuration.calibration_parameters.stereo_transform.m
        )
        transform[:3, 3] /= 1000  # Convert from millimeters to meters
        return transform

    def start_record(self, out_path):
        """
        Start recording to an SVO file.
        
        Args:
            out_path (str): Output file path
        """
        recordingParameters = sl.RecordingParameters()
        recordingParameters.compression_mode = sl.SVO_COMPRESSION_MODE.H264
        recordingParameters.video_filename = out_path
        err = self.cam.enable_recording(recordingParameters)

    def stop_record(self):
        """Stop recording to SVO file."""
        self.cam.disable_recording()

    def get_rgb_depth(self, cam="left"):
        """
        Get RGB images and depth map.
        
        Args:
            cam (str): Which camera to use as reference ("left" or "right")
            
        Returns:
            tuple: (left_image, right_image, depth_map) as numpy arrays
        """
        left, right, depth = self.get_frame(cam=cam)
        return left.cpu().numpy(), right.cpu().numpy(), depth.cpu().numpy()
    
    def get_rgb(self, cam="left"):
        """
        Get only RGB images without computing depth.
        
        Args:
            cam (str): Which camera to use as reference ("left" or "right")
            
        Returns:
            tuple: (left_image, right_image) as numpy arrays
        """
        left, right, _ = self.get_frame(depth=False, cam=cam)
        return left.cpu().numpy(), right.cpu().numpy()

    def get_ns_intrinsics(self):
        """
        Get camera intrinsics in NeRF Studio format.
        
        Returns:
            dict: Camera intrinsics dictionary compatible with NeRF Studio
        """
        calib = (
            self.cam.get_camera_information().camera_configuration.calibration_parameters
        )
        calibration_parameters_l = calib.left_cam
        return {
            "w": self.width_,
            "h": self.height_,
            "fl_x": calibration_parameters_l.fx,
            "fl_y": calibration_parameters_l.fy,
            "cx": calibration_parameters_l.cx,
            "cy": calibration_parameters_l.cy,
            "k1": calibration_parameters_l.disto[0],
            "k2": calibration_parameters_l.disto[1],
            "p1": calibration_parameters_l.disto[3],
            "p2": calibration_parameters_l.disto[4],
            "camera_model": "OPENCV",
        }

    def get_zed_depth(self):
        """
        Get native ZED depth map (not using RAFT-Stereo).
        Note: This requires depth_mode to be enabled in InitParameters.
        
        Returns:
            numpy.ndarray: Depth map
        """
        if self.cam.grab() == sl.ERROR_CODE.SUCCESS:
            depth = sl.Mat()
            self.cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
            return depth.get_data()
        else:
            raise RuntimeError("Could not grab frame")

    def close(self):
        """Close the camera and release resources."""
        self.cam.close()
        self.cam = None
        print("Closed camera")

    def reopen(self):
        """Reopen the camera with previously specified parameters."""
        if self.cam is None:
            init = sl.InitParameters()
            # Configuration for SVO file playback
            if self.recording_file is not None:
                init.set_from_svo_file(self.recording_file)
                init.camera_image_flip = sl.FLIP_MODE.OFF
                init.depth_mode = sl.DEPTH_MODE.NONE
                init.camera_resolution = sl.RESOLUTION.HD1080
                init.sdk_verbose = 1
                init.camera_fps = 15
            # Configuration for live camera
            else:
                init.camera_resolution = sl.RESOLUTION.HD720
                init.sdk_verbose = 1
                init.camera_fps = 15
                init.camera_image_flip = sl.FLIP_MODE.OFF
                init.depth_mode = sl.DEPTH_MODE.NONE
                init.depth_minimum_distance = 100  # millimeters
            
            self.cam = sl.Camera()
            init.camera_disable_self_calib = True
            status = self.cam.open(init)
            
            # Set SVO playback position if applicable
            if self.recording_file is not None:
                fps = self.cam.get_camera_information().camera_configuration.fps
                self.cam.set_svo_position(int(self.start_time * fps))
                
            # Check if camera opened successfully
            if status != sl.ERROR_CODE.SUCCESS:
                print("Camera Open : " + repr(status) + ". Exit program.")
                exit()
            else:
                print("Opened camera")
                print(
                    "Current Exposure is set to: ",
                    self.cam.get_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE),
                )
            
            # Reinitialize RAFT model and stereo parameters
            self.model = self.create_raft()
            left_cx = self.get_K(cam="left")[0, 2]
            right_cx = self.get_K(cam="right")[0, 2]
            self.cx_diff = right_cx - left_cx