import pyrealsense2 as rs
import numpy as np
import torch
from threading import Lock
import trimesh
from autolab_core import RigidTransform
from pathlib import Path

class RealSense():
    width: int
    height: int
    raft_lock: Lock
    zed_mesh: trimesh.Trimesh
    cam_to_zed: RigidTransform

    def __init__(self, flip_mode=False, resolution='720p', fps=30, cam_id=None):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Resolve resolution
        # Note: D455 requires matching fps for depth and color
        # 640x480@30fps works for both streams
        # 1280x720 only works at 5fps for depth
        if resolution == '720p':
            w, h = 1280, 720
            fps = min(fps, 5)  # D455 depth max at 720p
        elif resolution == '1080p':
            w, h = 1920, 1080
            fps = 5
        elif resolution == '480p':
            w, h = 640, 480
            fps = min(fps, 30)
        else:
            w, h = 640, 480 # Default fallback
            
        self.width = w
        self.height = h
        self.fps = fps
        
        if cam_id:
            self.config.enable_device(str(cam_id))
            
        self.config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
        
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        
        # Dummy lock to satisfy checks in main script
        self.raft_lock = Lock()
        
        # Load visualization mesh (using ZED mesh as placeholder or a box)
        # Assuming we keep the existing mesh file or use a simple box
        try:
            zed2_path = Path(__file__).parent / Path("data/ZED2.stl")
            if zed2_path.exists():
                self.zed_mesh = trimesh.load(str(zed2_path))
            else:
                self.zed_mesh = trimesh.creation.box(extents=[0.1, 0.03, 0.03])
        except:
             self.zed_mesh = trimesh.creation.box(extents=[0.1, 0.03, 0.03])
             
        # Offset (cam_to_bsae) - placeholder identity or approx
        self.cam_to_zed = RigidTransform(
            rotation=np.eye(3),
            translation=np.array([0, 0, 0]), 
            from_frame='camera', to_frame='base'
        )
        
        # "cam" object for settings - creating a dummy helper
        class CameraSettings:
            def set_camera_settings(self, *args, **kwargs):
                print("RealSense: Set camera settings not implemented (ignored)")
        self.cam = CameraSettings()

    def get_K(self):
        intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        K = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ])
        return K

    def get_distortion(self):
        intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        coeffs = np.array(intrinsics.coeffs, dtype=np.float32)
        # OpenCV expects 5 or 8 coeffs. RealSense provides 5.
        if coeffs.shape[0] < 5:
            coeffs = np.pad(coeffs, (0, 5 - coeffs.shape[0]))
        return coeffs[:5]

    def get_frame(self, depth=True):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None, None, None
            
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data()) # usually uint16 in mm
        
        # Convert to tensors
        # Color: HxWx3 uint8
        color_tensor = torch.from_numpy(color_image).cuda()
        
        # Depth: HxW float32 in meters
        depth_tensor = torch.from_numpy(depth_image.astype(np.float32) * 0.001).cuda()
        
        # Right image is None for Mono/RealSense usually (unless we have stereo streams enabled and aligned)
        return color_tensor, None, depth_tensor

    def close(self):
        self.pipeline.stop()

    @staticmethod
    def project_depth(
        rgb: torch.Tensor,
        depth: torch.Tensor,
        K: torch.Tensor,
        depth_threshold: float = 1.0,
        subsample: int = 4,
    ):
        """Deproject RGBD image to point cloud, using provided intrinsics.
        Also threshold/subsample pointcloud for visualization speed."""

        img_wh = rgb.shape[:2][::-1]

        grid = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(img_wh[0], device="cuda"),
                    torch.arange(img_wh[1], device="cuda"),
                    indexing="xy",
                ),
                2,
            )
            + 0.5
        )

        homo_grid = torch.concat(
            [grid, torch.ones((grid.shape[0], grid.shape[1], 1), device="cuda")],
            dim=2
        ).reshape(-1, 3)
        local_dirs = torch.matmul(torch.linalg.inv(K),homo_grid.T).T
        points = (local_dirs * depth.reshape(-1,1)).float()
        points = points.reshape(-1,3)

        mask = depth.reshape(-1, 1) <= depth_threshold
        points = points.reshape(-1, 3)[mask.flatten()][::subsample].cpu().numpy()
        colors = rgb.reshape(-1, 3)[mask.flatten()][::subsample].cpu().numpy()

        return (points, colors)
