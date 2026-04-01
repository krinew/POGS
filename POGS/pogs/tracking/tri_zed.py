import pyzed.sl as sl
from typing import Optional, Tuple
import torch
import numpy as np
from threading import Lock
import plotly
from plotly import express as px
import trimesh
from pathlib import Path
# from raftstereo.raft_stereo import *
from autolab_core import RigidTransform
import torch
import torch.nn.functional as tfn
import os
import numpy as np
import pathlib
from matplotlib import pyplot as plt
torch._C._jit_set_profiling_executor(False)

def is_tensor(data):
    return type(data) == torch.Tensor

def is_tuple(data):
    return isinstance(data, tuple)

def is_list(data):
    return isinstance(data, list) or isinstance(data, torch.nn.ModuleList)

def is_dict(data):
    return isinstance(data, dict) or isinstance(data, torch.nn.ModuleDict)

def is_seq(data):
    return is_tuple(data) or is_list(data)

def iterate1(func):
    """Decorator to iterate over a list (first argument)"""
    def inner(var, *args, **kwargs):
        if is_seq(var):
            return [func(v, *args, **kwargs) for v in var]
        elif is_dict(var):
            return {key: func(val, *args, **kwargs) for key, val in var.items()}
        else:
            return func(var, *args, **kwargs)
    return inner


@iterate1
def interpolate(tensor, size, scale_factor, mode):
    if size is None and scale_factor is None:
        return tensor
    if is_tensor(size):
        size = size.shape[-2:]
    return tfn.interpolate(
        tensor, size=size, scale_factor=scale_factor,
        recompute_scale_factor=False, mode=mode,
        align_corners=None,
    )


def resize_input(
    rgb: torch.Tensor,
    intrinsics: torch.Tensor = None,
    resize: tuple = None
):
    """Resizes input data

    Args:
        rgb (torch.Tensor): input image (B,3,H,W)
        intrinsics (torch.Tensor): camera intrinsics (B,3,3)
        resize (tuple, optional): resize shape. Defaults to None.

    Returns:
        rgb: resized image (B,3,h,w)
        intrinsics: resized intrinsics (B,3,3)
    """

    # Don't resize if not requested
    if resize is None:
        if intrinsics is None:
            return rgb
        else:
            return rgb, intrinsics
    # Resize rgb
    orig_shape = [float(v) for v in rgb.shape[-2:]]
    rgb = interpolate(rgb, mode="bilinear", scale_factor=None, size=resize)
    # Return only rgb if there are no intrinsics
    if intrinsics is None:
        return rgb
    # Resize intrinsics
    shape = [float(v) for v in rgb.shape[-2:]]
    intrinsics = intrinsics.clone()
    intrinsics[:, 0] *= shape[1] / orig_shape[1]
    intrinsics[:, 1] *= shape[0] / orig_shape[0]
    # return resized input
    return rgb, intrinsics

def format_image(rgb):
    return torch.tensor(rgb.transpose(2,0,1)[None]).to(torch.float32).cuda() / 255.0
class StereoModel(torch.nn.Module):
    """Learned Stereo model.

    Takes as input two images plus intrinsics and outputs a metrically scaled depth map.

    Taken from: https://github.com/ToyotaResearchInstitute/mmt_stereo_inference
    Paper here: https://arxiv.org/pdf/2109.11644.pdf
    Authors: Krishna Shankar, Mark Tjersland, Jeremy Ma, Kevin Stone, Max Bajracharya

    Pre-trained checkpoint here: s3://tri-ml-models/efm/depth/stereo.pt

    Args:
        cfg (Config): configuration file to initialize the model
        ckpt (str, optional): checkpoint path to load a pre-trained model. Defaults to None.
        baseline (float): Camera baseline. Defaults to 0.12 (ZED baseline)
    """

    def __init__(self, ckpt: str = None):
        super().__init__()
        # Initialize model
        self.model = torch.jit.load(ckpt).cuda()
        self.model.eval()

    def inference(
        self,
        baseline: float,
        rgb_left: torch.Tensor,
        rgb_right: torch.Tensor,
        intrinsics: torch.Tensor,
        resize: tuple = None,
    ):
        """Performs inference on input data

        Args:
            rgb_left (torch.Tensor): input float32 image (B,3,H,W)
            rgb_right (torch.Tensor): input float32 image (B,3,H,W)
            intrinsics (torch.Tensor): camera intrinsics (B,3,3)
            resize (tuple, optional): resize shape. Defaults to None.

        Returns:
            depth: output depth map (B,1,H,W)
        """
        
        rgb_left, intrinsics = resize_input(
            rgb=rgb_left, intrinsics=intrinsics, resize=resize
        )
        rgb_right = resize_input(rgb=rgb_right, resize=resize)

        with torch.no_grad():
            output, _ = self.model(rgb_left, rgb_right)

        disparity_sparse = output["disparity_sparse"]
        mask = disparity_sparse != 0
        depth = torch.zeros_like(disparity_sparse)
        depth[mask] = baseline * intrinsics[0, 0, 0] / disparity_sparse[mask]
        # depth = baseline * intrinsics[0, 0, 0] / output["disparity"]
        rgb = (rgb_left.squeeze(0).permute(1,2,0).cpu().detach().numpy()*255).astype(np.uint8)
        return depth, output["disparity"], disparity_sparse,rgb


class Zed():
    width: int
    """Width of the rgb/depth images."""
    height: int
    """Height of the rgb/depth images."""
    raft_lock: Lock
    """Lock for the camera, for raft-stereo depth!"""

    zed_mesh: trimesh.Trimesh
    """Trimesh of the ZED camera."""
    cam_to_zed: RigidTransform
    """Transform from left camera to ZED camera base."""

    def __init__(self, flip_mode, resolution, fps, cam_id=None, recording_file=None, start_time=0.0):
        init = sl.InitParameters()
        if cam_id is not None:
            init.set_from_serial_number(cam_id)
            self.cam_id = cam_id
        self.width = None
        self.debug_ = False
        self.height = None
        self.init_res = None
        # Set camera flip mode
        if flip_mode:
            init.camera_image_flip = sl.FLIP_MODE.ON
        else:
            init.camera_image_flip = sl.FLIP_MODE.OFF
            
        if recording_file is not None:
            init.set_from_svo_file(recording_file)
            
        # Configure camera resolution
        if resolution == '720p':
            init.camera_resolution = sl.RESOLUTION.HD720
            self.height = 720
            self.width = 1280
            self.init_res = 1280
        elif resolution == '1080p':
            init.camera_resolution = sl.RESOLUTION.HD1080
            self.height = 1080
            self.width = 1920
            self.init_res = 1920
        elif resolution == '2k':
            init.camera_resolution = sl.RESOLUTION.HD2k
            self.height = 1242
            self.width = 2208
            self.init_res = 2208
        else:
            print("Only 720p, 1080p, and 2k supported by Zed")
            exit()
        # Disable native ZED depth computation (we'll use RAFT-Stereo instead)
        init.depth_mode = sl.DEPTH_MODE.NONE
        init.sdk_verbose = 1
        init.camera_fps = fps
        self.cam = sl.Camera()
        init.camera_disable_self_calib = True
        status = self.cam.open(init)
        if recording_file is not None:
            fps = self.cam.get_camera_information().camera_configuration.fps
            self.cam.set_svo_position(int(start_time*fps))
        if status != sl.ERROR_CODE.SUCCESS: #Ensure the camera has opened succesfully
            print("Camera Open : "+repr(status)+". Exit program.")
            exit()
        else:
            print("Opened camera")
        res = sl.Resolution()
        res.width = self.width
        res.height = self.height
        left_cx = self.get_K(cam="left")[0, 2]
        right_cx = self.get_K(cam="right")[0, 2]
        self.cx_diff = right_cx - left_cx  # /1920
        self.f_ = self.get_K(cam="left")[0,0]
        self.cx_ = left_cx
        self.cy_ = self.get_K(cam="left")[1,2]
        self.baseline_ = self.cam.get_camera_information().camera_configuration.calibration_parameters.stereo_transform.get_translation().get()[0] / 1000.0
        self.intrinsics_ = torch.tensor([[
            [self.f_,0,self.cx_],
            [0,self.f_,self.cy_],
            [0,0,1]
        ]]).to(torch.float32).cuda()
        # Create lock for raft -- gpu threading messes up CUDA memory state, with curobo...
        self.raft_lock = Lock()
        self.dir_path = pathlib.Path(__file__).parent.resolve()
        self.stereo_ckpt = os.path.join(self.dir_path,'models/stereo_20230724.pt') #We use stereo model from this paper: https://arxiv.org/abs/2109.11644. However, you can sub this in for any realtime stereo model (including the default Zed model).
        with self.raft_lock:
            self.model = StereoModel(self.stereo_ckpt)

        # left_cx = self.get_K(cam='left')[0,2]
        # right_cx = self.get_K(cam='right')[0,2]
        # self.cx_diff = (right_cx-left_cx)
        

        
        # For visualiation.
        
        zedM_path = Path(__file__).parent / Path("data/ZEDM.stl")
        zed2_path = Path(__file__).parent / Path("data/ZED2.stl")
        self.zedM_mesh = trimesh.load(str(zedM_path))
        self.zed2_mesh = trimesh.load(str(zed2_path))
        # assert isinstance(zed_mesh, trimesh.Trimesh)
        self.zed_mesh = self.zed2_mesh
        self.cam_to_zed = RigidTransform(
            rotation=RigidTransform.quaternion_from_axis_angle(
                np.array([1, 0, 0]) * (np.pi / 2)
            ),
            translation=np.array([0.06, 0.042, -0.035]),
        )

    def prime_get_frame(self):
        res = sl.Resolution()
        res.width = self.width
        res.height = self.height
        if self.cam.grab() == sl.ERROR_CODE.SUCCESS:
            left_rgb = sl.Mat()
            right_rgb = sl.Mat()
            self.cam.retrieve_image(left_rgb, sl.VIEW.LEFT)
            self.cam.retrieve_image(right_rgb, sl.VIEW.RIGHT)
            self.height = self.height - self.height % 32
            self.width = self.width - self.width % 32
            left_cropped = np.flip(left_rgb.get_data()[:self.height,:self.width,:3], axis=2).copy()
            right_cropped = np.flip(right_rgb.get_data()[:self.height,:self.width,:3], axis=2).copy()
            with self.raft_lock:
                tridepth, disparity, disparity_sparse,cropped_rgb = self.model.inference(rgb_left=format_image(left_cropped),rgb_right=format_image(right_cropped),intrinsics=self.intrinsics_,baseline=self.baseline_)
            return left_cropped,right_cropped
        else:
            print("Couldn't grab frame")
            
    def get_frame(
        self, depth=True
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        res = sl.Resolution()
        res.width = self.width
        res.height = self.height
        r = self.width/self.init_res
        if self.cam.grab() == sl.ERROR_CODE.SUCCESS:
            left_rgb = sl.Mat()
            right_rgb = sl.Mat()
            self.cam.retrieve_image(left_rgb, sl.VIEW.LEFT, sl.MEM.CPU, res)
            self.cam.retrieve_image(right_rgb, sl.VIEW.RIGHT, sl.MEM.CPU, res)
            self.height = self.height - self.height % 32
            self.width = self.width - self.width % 32
            left_cropped = np.flip(left_rgb.get_data()[:self.height,:self.width,:3], axis=2).copy()
            right_cropped = np.flip(right_rgb.get_data()[:self.height,:self.width,:3], axis=2).copy()
            
            with self.raft_lock:
                tridepth, disparity, disparity_sparse,cropped_rgb = self.model.inference(rgb_left=format_image(left_cropped),rgb_right=format_image(right_cropped),intrinsics=self.intrinsics_,baseline=self.baseline_)
            if(self.debug_):
                import pdb
                pdb.set_trace()
                plt.imshow(tridepth.detach().cpu().numpy()[0,0],cmap='jet')
                plt.savefig('/home/lifelong/prime_raft.png')
                import pdb
                pdb.set_trace()
            return torch.from_numpy(left_cropped).cuda(), torch.from_numpy(right_cropped).cuda(), tridepth[0,0]
        elif self.cam.grab() == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            print("End of recording file")
            return None,None,None
        else:
            raise RuntimeError("Could not grab frame")
    
    def get_K(self,cam='left') -> np.ndarray:
        calib = self.cam.get_camera_information().camera_configuration.calibration_parameters
        if cam=='left':
            intrinsics = calib.left_cam
        else:
            intrinsics = calib.right_cam
        r = self.width/self.init_res
        K = np.array([[intrinsics.fx*r, 0, intrinsics.cx*r], 
                      [0, intrinsics.fy*r, intrinsics.cy*r], 
                      [0, 0, 1]])
        return K

    def get_stereo_transform(self):
        transform = self.cam.get_camera_information().camera_configuration.calibration_parameters.stereo_transform.m
        transform[:3,3] /= 1000#convert to meters
        return transform

    def start_record(self, out_path):
        recordingParameters = sl.RecordingParameters()
        recordingParameters.compression_mode = sl.SVO_COMPRESSION_MODE.H264
        recordingParameters.video_filename = out_path
        err = self.cam.enable_recording(recordingParameters)

    def stop_record(self):
        self.cam.disable_recording()

    @staticmethod
    def plotly_render(frame) -> plotly.graph_objs.Figure:
        fig = px.imshow(frame)
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            yaxis_visible=False,
            yaxis_showticklabels=False,
            xaxis_visible=False,
            xaxis_showticklabels=False,
        )
        return fig

    @staticmethod
    def project_depth(
        rgb: torch.Tensor,
        depth: torch.Tensor,
        K: torch.Tensor,
        depth_threshold: float = 1.0,
        subsample: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray]:
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


import tyro
def main(name: str) -> None:
# def main() -> None:

    import torch
    from viser import ViserServer
    # zed = Zed(recording_file="exps/eyeglasses/2024-06-06_014947/traj.svo2")
     #jzed = Zed(recording_file="test.svo2")
    # import pdb; pdb.set_trace()
    zed = Zed(recording_file = "exps/scissors/2024-06-06_155342/traj.svo2")
    # zed.start_record(f"/home/chungmin/Documents/please2/toad/motion_vids/{name}.svo2")
    import os
    # os.makedirs(out_dir,exist_ok=True)
    i = 0


    #code for visualizing poincloud
    import viser
    from matplotlib import pyplot as plt
    import viser.transforms as tf
    v = ViserServer()
    gui_reset_up = v.add_gui_button(
        "Reset up direction",
        hint="Set the camera control 'up' direction to the current camera's 'up'.",
    )

    @gui_reset_up.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None
        client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
            [0.0, -1.0, 0.0]
        )
    while True:
        left,right,depth = zed.get_frame()
        left = left.cpu().numpy()
        depth = depth.cpu().numpy()
        # import matplotlib.pyplot as plt
        # plt.imshow(left)
        # plt.show()
        K = zed.get_K()
        T_world_camera = np.eye(4)

        img_wh = left.shape[:2][::-1]

        grid = (
            np.stack(np.meshgrid(np.arange(img_wh[0]), np.arange(img_wh[1])), 2) + 0.5
        )

        homo_grid = np.concatenate([grid,np.ones((grid.shape[0],grid.shape[1],1))],axis=2).reshape(-1,3)
        local_dirs = np.matmul(np.linalg.inv(K),homo_grid.T).T
        points = (local_dirs * depth.reshape(-1,1)).astype(np.float32)
        points = points.reshape(-1,3)
        v.add_point_cloud("points", points = points.reshape(-1,3), colors=left.reshape(-1,3),point_size=.001)

if __name__ == "__main__":
    tyro.cli(main)