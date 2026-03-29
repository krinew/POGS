import typing
from dataclasses import dataclass, field
from typing import Literal, Type, Optional
from pathlib import Path

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import json
# from nerfstudio.configs import base_config as cfg
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
import trimesh
import viser

from dataclasses import dataclass, field
from nerfstudio.models.base_model import ModelConfig
from pogs.data.utils.patch_embedding_dataloader import PatchEmbeddingDataloader
from pogs.pogs import POGSModelConfig
from nerfstudio.models.splatfacto import SH2RGB
from pogs.encoders.image_encoder import BaseImageEncoderConfig
from pogs.data.full_images_datamanager import FullImageDatamanagerConfig
from sklearn.neighbors import NearestNeighbors

from nerfstudio.viewer.viewer import VISER_NERFSTUDIO_SCALE_RATIO
from gsplat.cuda._torch_impl import _quat_to_rotmat
from scipy.spatial.transform import Rotation as Rot
from typing import Literal, Type, Optional
from nerfstudio.viewer.viewer_elements import *
import torch
import numpy as np 
import math
import open3d as o3d
import os
import os.path as osp
import time
import threading
import copy

def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
        ],
        dim=-1,
    )
def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5

def get_clip_patchloader(image, pipeline, image_scale):
    clip_cache_path = Path("dummy_cache2.npy")
    import time
    model_name = str(time.time())
    image = image.permute(2,0,1)[None,...]
    patchloader = PatchEmbeddingDataloader(
        cfg={
            "tile_ratio": image_scale,
            "stride_ratio": .25,
            "image_shape": list(image.shape[2:4]),
            "model_name": model_name,
        },
        device='cuda:0',
        model=pipeline.image_encoder,
        image_list=image,
        cache_path=clip_cache_path,
    )
    return patchloader

def get_grid_embeds_patch(patchloader, rn, cn, im_h, im_w, img_scale):
    "create points, which is a meshgrid of x and y coordinates, with a z coordinate of 1.0"
    r_res = im_h // rn
    c_res = im_w // cn
    points = torch.stack(torch.meshgrid(torch.arange(0, im_h,r_res), torch.arange(0,im_w,c_res)), dim=-1).cuda().long()
    points = torch.cat([torch.zeros((*points.shape[:-1],1),dtype=torch.int64,device='cuda'),points],dim=-1)
    embeds = patchloader(points.view(-1,3))
    return embeds, points

def get_2d_embeds(image: torch.Tensor, scale: float, pipeline):
    patchloader = get_clip_patchloader(image, pipeline=pipeline, image_scale=scale)
    embeds, points = get_grid_embeds_patch(patchloader, image.shape[0] * scale,image.shape[1] * scale, image.shape[0], image.shape[1], scale)
    return embeds, points


@dataclass
class POGSPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: POGSPipeline)
    """target class to instantiate"""
    datamanager: FullImageDatamanagerConfig = FullImageDatamanagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = POGSModelConfig()
    """specifies the model config"""
    network: BaseImageEncoderConfig = BaseImageEncoderConfig()
    """specifies the vision network config"""

class POGSPipeline(VanillaPipeline):
    def __init__(
        self,
        config: POGSPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
        use_clip : bool = True,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode

        self.datamanager: FullImageDatamanagerConfig = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank
        )
        seed_pts = None
        if (
            hasattr(self.datamanager, "train_dataparser_outputs")
            and "points3D_xyz" in self.datamanager.train_dataparser_outputs.metadata
        ):
            pts = self.datamanager.train_dataparser_outputs.metadata["points3D_xyz"]
            pts_rgb = self.datamanager.train_dataparser_outputs.metadata["points3D_rgb"]
            seed_pts = (pts, pts_rgb)
        # self.datamanager.to(device)


        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            grad_scaler=grad_scaler,
            image_encoder=self.datamanager.image_encoder,
            datamanager=self.datamanager,
            seed_points=seed_pts,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

        self.use_clip = use_clip
        self.plot_verbose = True
        
        self.img_count = 0

        self.viewer_control = self.model.viewer_control

        self.state_stack = []

        self.a_interaction_method = ViewerDropdown(
            "Interaction Method",
            default_value="Interactive",
            options=["Interactive", "Clustering"],
            cb_hook=self._update_interaction_method
        )

        self.click_gaussian = ViewerButton(name="Click", cb_hook=self._click_gaussian)
        self.click_location = None
        self.click_handle = None


        self.crop_to_click = ViewerButton(name="Crop to Click", cb_hook=self._crop_to_click, disabled=True)
        self.crop_group = []

        self.add_crop_to_group_list = ViewerButton(name="Add Crop to Group List", cb_hook=self._add_crop_to_group_list, disabled=True)
        self.add_crop_to_previous_group = ViewerButton(name="Add Crop to Previous Group", cb_hook=self._add_crop_to_previous_group, disabled=True)
        self.view_crop_group_list = ViewerButton(name="View Crop Group List", cb_hook=self._view_crop_group_list, disabled=True)
        
        self.load_state = ViewerButton(name="Load State", cb_hook=self._load_state, disabled=False)
        self.crop_group_list = []
        self.crop_group_tf_list = []
        self.model_keep_inds = None

        self.reset_state = ViewerButton(name="Reset State", cb_hook=self._reset_state, disabled=True)

        self.z_export_options = ViewerCheckbox(name="Export Options", default_value=False, cb_hook=self._update_export_options)
        self.z_export_options_visible_gaussians = ViewerButton(
            name="Export Visible Gaussians",
            visible=False,
            cb_hook=self._export_visible_gaussians
            )
        self.z_export_options_cluster_labels = ViewerButton(
            name="Export Cluster",
            visible=False,
            cb_hook=self._export_clusters
            )

        self.group_labels_local, self.group_masks_local, self.group_masks_global = None, None, None
        
        self.traj_dirs = []
        self.traj_dir = self.config.datamanager.data
        for root, dirs, files in os.walk(self.traj_dir):
            for dir in dirs:
                if dir.startswith("traj-"):
                    self.traj_dirs.append(dir)
        if len(self.traj_dirs) > 0:
            self.load_traj_file = ViewerDropdown("Load File", default_value=self.traj_dirs[-1], options=self.traj_dirs, cb_hook=self._load_traj_file)
            self.traj_file = Path(osp.join(self.traj_dir, self.traj_dirs[-1], "part_deltas_traj.npy"))
            if self.traj_file.exists():
                self.traj = np.load(self.traj_file, allow_pickle=True)
                self.preview_frame_slider = ViewerSlider("Preview Frame", min_value=0, max_value=self.traj.shape[0] - 1, step=1, default_value=0, cb_hook=self._preview_frame_slider)
                self.play_button = ViewerButton("Play", cb_hook=self._play_button)
                self.pause_button = ViewerButton("Pause", cb_hook=self._pause_button)
                self.framerate_number = ViewerNumber("FPS", default_value=3.0)
                self.framerate_buttons = ViewerButtonGroup("", default_value=3, options = ("3", "5", "10"), cb_hook=self._framerate_buttons)


    def _framerate_buttons(self, button: ViewerButtonGroup) -> None:
        self.framerate_number.value = float(self.framerate_buttons.value)
    
    def _play_button(self, button: ViewerButton) -> None:
        self.pause = False
        assert self.traj_file.exists(), f"Trajectory file {self.traj_file} does not exist"
        
        def play() -> None:
            # pass
            while self.pause == False:
                max_frame = int(self.traj.shape[0])
                if max_frame > 0:
                    assert self.preview_frame_slider is not None
                    self.preview_frame_slider.value = (self.preview_frame_slider.value + 1) % max_frame
                time.sleep(1.0 / self.framerate_number.value)

        threading.Thread(target=play).start()

    # Pause the trajectory when the pause button is pressed.
    def _pause_button(self, button: ViewerButton) -> None:
        self.pause = True
        
    def _preview_frame_slider(self, slider: ViewerSlider) -> None:
        assert self.traj_file.exists(), f"Trajectory file {self.traj_file} does not exist"
        frame = self.traj[self.preview_frame_slider.value]
        xyz0 = self.traj[0]
        
        for i, mask in enumerate(self.group_masks_local):
            rigid_transform_mat = frame[i].as_matrix()
            rigid_transform_mat[:3,3] = rigid_transform_mat[:3,3] - xyz0[i].translation()
            means_centered = torch.subtract(self.init_means[mask], self.init_means[mask].mean(dim=0))
            means_centered_homog = torch.cat([means_centered, torch.ones(means_centered.shape[0], 1).to(self.device)], dim=1)
            self.model.gauss_params["means"][mask] = ((torch.from_numpy(rigid_transform_mat).to(torch.float32).cuda() @ means_centered_homog.T).T)[:, :3] + self.init_means[mask].mean(dim=0)
            self.model.gauss_params["quats"][mask] = torch.Tensor(
                Rot.from_matrix(
                torch.matmul(torch.from_numpy(frame[i].as_matrix()[:3,:3]).to(torch.float32).cuda(), _quat_to_rotmat(self.init_quats[mask])).cpu()
                ).as_quat()).to(self.device)[:, [3, 0, 1, 2]]
                
        self.viewer_control.viewer._trigger_rerender()

    def _queue_state(self):
        """Save current state to stack"""
        import copy
        self.state_stack.append(copy.deepcopy({k:v.detach() for k,v in self.model.gauss_params.items()}))
        self.reset_state.set_disabled(False)


    def _reset_state(self, button: ViewerButton, pop = True):
        """Revert to previous saved state"""

        assert len(self.state_stack) > 0, "No previous state to revert to"
        if pop:
            prev_state = self.state_stack.pop()
        else:
            prev_state = self.state_stack[-1]
        for name in self.model.gauss_params.keys():
            self.model.gauss_params[name] = prev_state[name]

        self.click_location = None
        if self.click_handle is not None:
            self.click_handle.remove()
        self.click_handle = None

        self.click_gaussian.set_disabled(False)

        self.crop_to_click.set_disabled(True)
        # self.move_current_crop.set_disabled(True)
        self.crop_group = []
        if self.crop_transform_handle is not None:
            self.crop_transform_handle.remove()
            self.crop_transform_handle = None
        if len(self.state_stack) == 0:
            self.reset_state.set_disabled(True)
        self.add_crop_to_group_list.set_disabled(True)

    def _reset_crop_group_list(self, button: ViewerButton):
        """Reset the crop group list"""
        self.crop_group_list = []
        self.crop_group_tf_list = []
        self.model_keep_inds = []
        self.add_crop_to_group_list.set_disabled(True)
        self.view_crop_group_list.set_disabled(True)
    
    def _add_crop_to_group_list(self, button: ViewerButton):
        """Add the current crop to the group list"""
        self.crop_group_list.append(self.crop_group[0])
        self.crop_group_tf_list.append(copy.copy(self.crop_transform_handle))
        self.crop_transform_handle.remove()
        self._reset_state(None, pop=False)
        self.view_crop_group_list.set_disabled(False)
    
    def _add_crop_to_previous_group(self, button: ViewerButton):
        """Combine the current crop with the previous group"""
        self.crop_group_list[-1] = torch.cat([self.crop_group_list[-1], self.crop_group[0]])
        self._reset_state(None, pop=False)
        self.view_crop_group_list.set_disabled(False)

    def _view_crop_group_list(self, button: ViewerButton):
        if len(self.crop_group_list) == 0:
            return
        if len(self.state_stack) == 0:
            return

        keep_inds = []
        for inds in self.crop_group_list:
            keep_inds.extend(inds)
        keep_inds = torch.stack(keep_inds)
        prev_state = self.state_stack[-1]
        for name in self.model.gauss_params.keys():
            self.model.gauss_params[name] = prev_state[name][keep_inds]
        self.model.keep_inds = keep_inds
        self._export_clusters(None)
        self.z_export_options_cluster_labels.visible = True

    def _crop_to_click(self, button: ViewerButton):
        """Crop to click location"""
        assert self.click_location is not None, "Need to specify click location"

        self._queue_state()  # Save current state
        curr_means = self.model.gauss_params['means'].detach()
        self.model.eval()

        # The only way to reset is to reset the state using the reset button.
        self.click_gaussian.set_disabled(True)  # Disable user from changing click
        self.crop_to_click.set_disabled(True)  # Disable user from changing click

        # Get the 3D location of the click
        location = self.click_location
        location = torch.tensor(location).view(1, 3).to(self.device)

        # The list of positions to query for garfield features. The first one is the click location.
        positions = torch.cat([location, curr_means])  # N x 3

        # Create a kdtree, to get the closest gaussian to the click-point.
        points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(curr_means.cpu().numpy()))
        kdtree = o3d.geometry.KDTreeFlann(points)
        _, inds, _ = kdtree.search_knn_vector_3d(location.view(3, -1).float().detach().cpu().numpy(), 10)

        # get the closest point to the sphere, using kdtree
        sphere_inds = inds


        keep_list = []

        if self.model.cluster_labels == None:
            instances = self.model.get_grouping_at_points(positions)  # (1+N, 256)
            click_instance = instances[0]

            affinity = torch.norm(click_instance - instances, dim=1)[1:]

            # Filter out points that have affinity < 0.5 (i.e., not likely to be in the same group)
            keeps = torch.where(affinity > 0.5)[0].cpu()
            keep_points = points.select_by_index(keeps.tolist())  # indices of gaussians

            # Here, we desire the gaussian groups to be grouped tightly together spatially. 
            # We use DBSCAN to group the gaussians together, and choose the cluster that contains the click point.
            # Note that there may be spuriously high affinity between points that are spatially far apart,
            #  possibly due two different groups being considered together at an odd angle / far viewpoint.

            # If there are too many points, we downsample them first before DBSCAN.
            # Then, we assign the filtered points to the cluster of the nearest downsampled point.
            if len(keeps) > 5000:
                curr_point_min = keep_points.get_min_bound()
                curr_point_max = keep_points.get_max_bound()

                _, _, curr_points_ds_ids = keep_points.voxel_down_sample_and_trace(
                    voxel_size=0.0001,
                    min_bound=curr_point_min,
                    max_bound=curr_point_max,
                )
                curr_points_ds_ids = np.array([points[0] for points in curr_points_ds_ids])
                curr_points_ds = keep_points.select_by_index(curr_points_ds_ids)
                curr_points_ds_selected = np.zeros(len(keep_points.points), dtype=bool)
                curr_points_ds_selected[curr_points_ds_ids] = True

                _clusters = np.asarray(curr_points_ds.cluster_dbscan(eps=0.02, min_points=5))
                nn_model = NearestNeighbors(
                    n_neighbors=1, algorithm="auto", metric="euclidean"
                ).fit(np.asarray(curr_points_ds.points))

                _, indices = nn_model.kneighbors(np.asarray(keep_points.points)[~curr_points_ds_selected])

                clusters = np.zeros(len(keep_points.points), dtype=int)
                clusters[curr_points_ds_selected] = _clusters
                clusters[~curr_points_ds_selected] = _clusters[indices[:, 0]]

            else:
                clusters = np.asarray(keep_points.cluster_dbscan(eps=0.02, min_points=5))

            # Choose the cluster that contains the click point. If there is none, move to the next scale.
            cluster_inds = clusters[np.isin(keeps, sphere_inds)]
            cluster_inds = cluster_inds[cluster_inds != -1]

            cluster_ind = cluster_inds[0]

            keeps = keeps[np.where(clusters == cluster_ind)]

            keep_list.append(keeps)
            
            if len(keep_list) == 0:
                print("No gaussians within crop, aborting")
                # The only way to reset is to reset the state using the reset button.
                self.click_gaussian.set_disabled(False)
                self.crop_to_click.set_disabled(False)
                return
        else:
            # Handle case where we have precomputed cluster labels
            vote = int(torch.tensor(self.model.cluster_labels[sphere_inds].mode())[0].item()) # mode group ID from the click sphere samples
            
            keep_inds_list = torch.where(self.model.cluster_labels == vote)[0] # get all points in the same group as the click sphere samples
            keep_points_o3d = points.select_by_index(keep_inds_list.tolist()) # clustered points in o3d format for DBSCAN
            
            sphere_ind_vote = torch.where(self.model.cluster_labels[sphere_inds] == vote)[0]
            
            sphere_inds_keep = [(torch.where(keep_inds_list == torch.tensor(sphere_inds)[i])[0]).item() for i in sphere_ind_vote.tolist()]
            # Secondary clustering in cartesian space to filter outliers
            group_clusters = keep_points_o3d.cluster_dbscan(eps=0.010, min_points=1)

            inner_vote = torch.tensor(group_clusters)[sphere_inds_keep].mode()[0].item()
            keep_inds_list_inner = torch.where(torch.tensor(group_clusters) == inner_vote)[0]
            keep_list = [keep_inds_list[keep_inds_list_inner]]
            
        
        table_bounding_cube_filename = self.datamanager.get_datapath().joinpath("table_bounding_cube.json")
        with open(table_bounding_cube_filename, 'r') as json_file: 
            bounding_box_dict = json.load(json_file)
        table_z_val = bounding_box_dict['table_height'] + 0.015 #- 0.01 # Removes everything below this value to represent the table and anything below. Found 0.008 to be good value for this
        # table_z_val = -0.165 # z value of the table to filter out of our clusters
        keep_list = [keep_list[0][torch.where(curr_means[keep_list[0]][:,2] > table_z_val)[0].cpu()]] # filter out table points
        # Remove the click handle + visualization
        self.click_location = None
        self.click_handle.remove()
        self.click_handle = None
        
        self.crop_group = keep_list
        
        self.add_crop_to_group_list.set_disabled(False)
        if len(self.crop_group_list) > 0:
            self.add_crop_to_previous_group.set_disabled(False)


        keep_inds = self.crop_group[0]
        prev_state = self.state_stack[-1]
        for name in self.model.gauss_params.keys():
            self.model.gauss_params[name] = prev_state[name][keep_inds]

        """Add a transform control to the current scene, and update the model accordingly."""

        self.viewer_control.viewer._trigger_rerender()
        scene_centroid = self.model.gauss_params['means'].detach().mean(dim=0)
        
        
        ## Delete if reorienting to bbox is iffy
        points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.model.gauss_params['means'].detach().cpu().numpy()))

        obb = points.get_oriented_bounding_box()

        R = obb.R
        target_z = np.array([0, 0, 1])
        dot_products = np.abs(R.T @ target_z)
        z_index = np.argmax(dot_products)
        remaining_indices = [i for i in range(3) if i != z_index]
        new_x = R[:, remaining_indices[0]]
        new_y = R[:, remaining_indices[1]]

        # Create a new orthonormal basis with z-axis as [0, 0, 1]
        new_z = target_z
        new_x = new_x / np.linalg.norm(new_x)  
        new_y = np.cross(new_z, new_x) 
        new_y = new_y / np.linalg.norm(new_y) 

        R_new = np.column_stack((new_x, new_y, new_z))
        
        import viser.transforms as vtf

        oriented_quat = vtf.SO3.from_matrix(R_new)
        
        self.crop_transform_handle = self.viewer_control.viser_server.add_transform_controls(
            name=f"/obj_transform",
            position=(VISER_NERFSTUDIO_SCALE_RATIO*scene_centroid).cpu().numpy(),
            wxyz = oriented_quat.wxyz,
        )

        @self.crop_transform_handle.on_update
        def _(_):
            handle_position = torch.tensor(self.crop_transform_handle.position).to(self.device)
            handle_position = handle_position / VISER_NERFSTUDIO_SCALE_RATIO
            handle_rotmat = _quat_to_rotmat(torch.tensor(self.crop_transform_handle.wxyz).to(self.device).float())

            self.viewer_control.viewer._trigger_rerender()

            
    def _update_interaction_method(self, dropdown: ViewerDropdown):
        """Update the UI based on the interaction method"""
        hide_in_interactive = (not (dropdown.value == "Interactive")) # i.e., hide if in interactive mode

        self.cluster_scene.set_hidden((not hide_in_interactive))
        self.cluster_scene_scale.set_hidden((not hide_in_interactive))
        self.cluster_scene_shuffle_colors.set_hidden((not hide_in_interactive))

        self.click_gaussian.set_hidden(hide_in_interactive)
        self.crop_to_click.set_hidden(hide_in_interactive)
        self.crop_to_group_level.set_hidden(hide_in_interactive)
        # self.move_current_crop.set_hidden(hide_in_interactive)
        self.move_crop_frame.set_hidden(hide_in_interactive)

    def _click_gaussian(self, button: ViewerButton):
        """Start listening for click-based 3D point specification.
        Refer to garfield_interaction.py for more details."""
        def del_handle_on_rayclick(click: ViewerClick):
            self._on_rayclick(click)
            self.click_gaussian.set_disabled(False)
            self.crop_to_click.set_disabled(False)
            self.viewer_control.unregister_click_cb(del_handle_on_rayclick)

        self.click_gaussian.set_disabled(True)
        self.viewer_control.register_click_cb(del_handle_on_rayclick)

    def _on_rayclick(self, click: ViewerClick):
        """On click, calculate the 3D position of the click and visualize it.
        Refer to garfield_interaction.py for more details."""

        cam = self.viewer_control.get_camera(500, None, 0)
        cam2world = cam.camera_to_worlds[0, :3, :3]
        import viser.transforms as vtf

        x_pi = vtf.SO3.from_x_radians(np.pi).as_matrix().astype(np.float32)
        world2cam = (cam2world @ x_pi).inverse()
        # rotate the ray around into cam coordinates
        newdir = world2cam @ torch.tensor(click.direction).unsqueeze(-1)
        z_dir = newdir[2].item()
        # project it into coordinates with matrix
        K = cam.get_intrinsics_matrices()[0]
        coords = K @ newdir
        coords = coords / coords[2]
        pix_x, pix_y = int(coords[0]), int(coords[1])
        self.model.eval()
        outputs = self.model.get_outputs(cam.to(self.device))
        self.model.train()
        with torch.no_grad():
            depth = outputs["depth"][pix_y, pix_x].cpu().numpy()

        self.click_location = np.array(click.origin) + np.array(click.direction) * (depth / z_dir)

        sphere_mesh = trimesh.creation.icosphere(radius=0.2)
        sphere_mesh.visual.vertex_colors = (0.0, 1.0, 0.0, 1.0)  # type: ignore
        self.click_handle = self.viewer_control.viser_server.add_mesh_trimesh(
            name=f"/click",
            mesh=sphere_mesh,
            position=VISER_NERFSTUDIO_SCALE_RATIO * self.click_location,
        )
        
    def _update_export_options(self, checkbox: ViewerCheckbox):
        """Update the UI based on the export options"""
        self.z_export_options_visible_gaussians.set_hidden(not checkbox.value)
    
    def _export_clusters(self, button: ViewerButton):
        """Export the cluster information to a .npy file"""
        output_dir = f"outputs/{self.datamanager.config.dataparser.data.name}"
        filename = Path(output_dir) / f"clusters.npy"
        
        cgtf = []
        for i in range(len(self.crop_group_tf_list)): 
            tf = np.zeros(7)
            tf[:4] = self.crop_group_tf_list[i].wxyz
            tf[4:] = self.crop_group_tf_list[i].position / VISER_NERFSTUDIO_SCALE_RATIO
            cgtf.append(tf) # w x y z translation
            
        self.cgtf_stack = np.stack(cgtf)
        if self.model.cluster_labels is not None and self.model.keep_inds is not None:
            np.save(filename, np.array([self.model.cluster_labels, self.model.keep_inds, self.cgtf_stack], dtype=object))
        else:
            print("No cluster labels to export")
            
    def _load_state(self, button: ViewerButton):
        """Load the state from a .npy file"""
        
        # add to state stack
        self.state_stack.append(self.model.gauss_params)
        output_dir = f"outputs/{self.datamanager.config.dataparser.data.name}"
        filename = Path(output_dir) / f"clusters.npy"
        if filename.exists():
            data = np.load(filename, allow_pickle=True)
            self.model.cluster_labels = data[0]
            self.model.keep_inds = data[1]
            self.cgtf_stack = data[2]
        else:
            print("No state to load")
        
        # Update the crop group list
        self.crop_group_list = []
        self.crop_group_tf_list = []
        for i in range(len(self.cgtf_stack)):
            self.crop_group_list.append(self.model.keep_inds)
            self.crop_group_tf_list.append(self.cgtf_stack[i])
        
        # View the crop group list
        keep_inds = []
        for inds in self.crop_group_list:
            keep_inds.extend(inds)
        keep_inds = torch.stack(keep_inds)
        prev_state = self.state_stack[-1]
        for name in self.model.gauss_params.keys():
            self.model.gauss_params[name] = prev_state[name][keep_inds]
        self.model.keep_inds = keep_inds
        
        
    
    def _export_visible_gaussians(self, button: ViewerButton):
        """Export the visible gaussians to a .ply file"""
        output_dir = f"outputs/{self.datamanager.config.dataparser.data.name}"
        segmented_filename = Path(output_dir) / f"prime_seg_gaussians.ply"
        full_filename = Path(output_dir) / f"prime_full_gaussians.ply"




        model=self.model

        with torch.no_grad():
            positions = model.means.cpu().numpy()
            if model.config.sh_degree > 0:
                shs_0 = model.shs_0.contiguous().cpu().numpy()
            colors = model.colors.cpu().numpy()
            normalized_colors = (colors - np.min(colors)) / (np.max(colors) - np.min(colors))

            
            positions = positions.astype('float64')
            normalized_colors = normalized_colors.astype('float64')

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(positions)
            pcd.colors = o3d.utility.Vector3dVector(normalized_colors)
            o3d.io.write_point_cloud(str(segmented_filename),pcd)
            
        prev_state = self.state_stack[-1]
        with torch.no_grad():
            positions = prev_state["means"].cpu().numpy()
            features_dc = prev_state["features_dc"].cpu().numpy()
            if(self.model.config.sh_degree > 0):
                colors = SH2RGB(features_dc)
            else:
                torch.sigmoid(features_dc)
            normalized_colors = (colors - np.min(colors)) / (np.max(colors) - np.min(colors))

            
            positions = positions.astype('float64')
            normalized_colors = normalized_colors.astype('float64')

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(positions)
            pcd.colors = o3d.utility.Vector3dVector(normalized_colors)
            o3d.io.write_point_cloud(str(full_filename),pcd)
        
    def _load_traj_file(self, dropdown: ViewerDropdown) -> None:
        """Load a trajectory file"""
        self.traj_file = Path(osp.join(self.traj_dir, self.load_traj_file.value, "part_deltas_traj.npy"))
        if self.traj_file.exists():
            self.traj = np.load(self.traj_file, allow_pickle=True)
            self.preview_frame_slider.remove()
            self.preview_frame_slider = ViewerSlider("Preview Frame", min_value=0, max_value=self.traj.shape[0] - 1, step=1, default_value=0, cb_hook=self._preview_frame_slider)
            self.preview_frame_slider.install(self.viewer_control.viser_server)