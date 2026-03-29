import torch
import viser.transforms as vtf
import time
import numpy as np
import trimesh
from typing import List, Tuple
import moviepy as mpy
from copy import deepcopy

from nerfstudio.utils.eval_utils import eval_setup
from pathlib import Path
from threading import Lock
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.viewer.viewer import Viewer
from nerfstudio.configs.base_config import ViewerConfig
from pogs.pogs_pipeline import POGSPipeline
from nerfstudio.utils import writer
from nerfstudio.models.splatfacto import SH2RGB

from pogs.tracking.rigid_group_optimizer import RigidGroupOptimizer, RigidGroupOptimizerConfig
from pogs.tracking.toad_object import ToadObject
from pogs.tracking.observation import PosedObservation, Frame
from pogs.encoders.openclip_encoder import OpenCLIPNetwork
import open3d as o3d
from pogs.tracking.observation import Future

class Optimizer:
    """Wrapper around 1) RigidGroupOptimizer and 2) GraspableToadObject.
    Operates in camera frame, not world frame."""

    init_cam_pose: torch.Tensor
    """Initial camera pose, in OpenCV format.
    This is aligned with the camera pose provided in __init__,
    and is in world coordinates/scale."""
    viewer_ns: Viewer
    """Viewer for nerfstudio visualization (not the same as robot visualization)."""

    num_groups: int
    """Number of object parts."""

    toad_object: ToadObject
    """Meshes + grasps for object parts."""

    optimizer: RigidGroupOptimizer
    """Optimizer for part poses."""
    
    optimizer_config: RigidGroupOptimizerConfig = RigidGroupOptimizerConfig()
    """Configuration for the rigid group optimizer."""
    
    MATCH_RESOLUTION: int = 500
    """Camera resolution for RigidGroupOptimizer."""

    initialized: bool = False
    """Whether the object pose has been initialized. This is set to `False` at `ToadOptimizer` initialization."""
    
    render_features: bool = False
    """Whether features are rendered in the gsplat eval mode"""

    def __init__(
        self,
        config_path: Path,  # path to the nerfstudio config file
        K: np.ndarray,  # camera intrinsics
        width: int,  # camera width
        height: int,  # camera height
        init_cam_pose: torch.Tensor,  # initial camera pose in OpenCV format
    ):
        self.config_path = config_path

        clusters = config_path.parent.parent.parent.joinpath("clusters.npy") # For preloading the cluster info for pre-clustered objects instead of clustering interactively
        print("clusters file", clusters)
        if not clusters.exists():
            print(f"clusters.npy file does not exist. \nProceed with interactive clustering.")
            self.cluster_from_file = None
        else:
            self.cluster_from_file = np.load(clusters, allow_pickle=True)
        
        # Load the POGSPipeline.
        train_config, self.pipeline, _, _ = eval_setup(config_path)
        assert isinstance(self.pipeline, POGSPipeline)
        train_config.logging.local_writer.enable = False

        assert self.pipeline.datamanager.train_dataset is not None
        dataset_scale = self.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale

        writer.setup_local_writer(train_config.logging, max_iter=train_config.max_num_iterations)
        self.viewer_ns = Viewer(
            ViewerConfig(
                default_composite_depth=False,
                num_rays_per_chunk=-1
            ),
            config_path.parent,
            self.pipeline.datamanager.get_datapath(),
            self.pipeline,
            train_lock=Lock()
        )
        assert self.viewer_ns.train_lock is not None

        self.keep_inds = None
        self.group_labels, self.group_masks, self.group_masks_global = self._setup_crops_and_groups()
        
        self.max_relevancy_label = None
        self.max_relevancy_text = None
        
        self.is_servoing = False
        
        self.follow_max_relevancy_label = None
        self.follow_max_relevancy_text = None

        self.place_max_relevancy_label = None
        self.place_max_relevancy_text = None

        self.num_groups = len(self.group_masks)
        
        assert init_cam_pose.shape == (1, 3, 4)
        self.init_cam_pose = deepcopy(init_cam_pose)

        # For nerfstudio, feed the camera as:
        #  - opengl format
        #  - in nerfstudio scale
        #  - as `Cameras` object
        #  - with `MATCH_RESOLUTION` resolution.
        init_cam_ns = torch.cat([
            init_cam_pose[0], torch.tensor([0, 0, 0, 1], dtype=torch.float32).reshape(1, 4)
        ], dim=0) @ (torch.from_numpy(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])).float())
        init_cam_ns = init_cam_ns[None, :3, :]
        init_cam_ns[:, :3, 3] = init_cam_ns[:, :3, 3] * dataset_scale  # convert to meters
        assert init_cam_ns.shape == (1, 3, 4)

        cam2world_ns = Cameras(
            camera_to_worlds=init_cam_ns,
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            width=width,
            height=height,
        )
        self.cam2world_ns = deepcopy(cam2world_ns)

        print("Ratio: " + str(self.MATCH_RESOLUTION / min(width, height)))
        cam2world_ns.rescale_output_resolution(
            self.MATCH_RESOLUTION / min(width, height)
        )
        
        # Set up the optimizer.
        self.cam2world_ns_ds = cam2world_ns
        self.dataset_scale = dataset_scale

        self.orig_means = self.pipeline.model.gauss_params["means"].detach().clone()
        self.orig_quats = self.pipeline.model.gauss_params["quats"].detach().clone()
        self.orig_scales = self.pipeline.model.gauss_params["scales"].detach().clone()
        self.orig_opacities = self.pipeline.model.gauss_params["opacities"].detach().clone()    
        self.orig_features_dc = self.pipeline.model.gauss_params["features_dc"].detach().clone()
        self.orig_features_rest = self.pipeline.model.gauss_params["features_rest"].detach().clone()
        
        self.optimizer = RigidGroupOptimizer(
            self.optimizer_config,
            self.pipeline.model,
            group_masks=self.group_masks,
            group_labels=self.group_labels,
            dataset_scale=dataset_scale,
            render_lock=self.viewer_ns.train_lock,
        )

        # Initialize the object -- remember that ToadObject works in world scale,
        # since grasps + etc are in world scale.
        start = time.time()
        self.toad_object = ToadObject.from_points_and_clusters(
            self.optimizer.init_means.detach().cpu().numpy(),
            self.optimizer.group_labels.detach().cpu().numpy(),
            scene_scale=self.optimizer.dataset_scale,
        )
        print(f"Time taken for init (object): {time.time() - start:.2f} s")

        self.initialized = False
    
    def background_snapshot(self) -> torch.Tensor:
        """Get a snapshot of the background."""
        cam = self.cam2world_ns
        # binary and all tensors in list self.group_masks_global
        clustered_objects_global_mask = (torch.sum(torch.stack(self.group_masks_global), axis=0) > 0)
        all_means = self.pipeline.__dict__['state_stack'][0]['means'][~clustered_objects_global_mask].to('cuda')
        all_quats = self.pipeline.__dict__['state_stack'][0]['quats'][~clustered_objects_global_mask].to('cuda')
        all_scales = self.pipeline.__dict__['state_stack'][0]['scales'][~clustered_objects_global_mask].to('cuda')
        all_opacities = self.pipeline.__dict__['state_stack'][0]['opacities'][~clustered_objects_global_mask].to('cuda')
        all_features_dc = self.pipeline.__dict__['state_stack'][0]['features_dc'][~clustered_objects_global_mask].to('cuda')
        all_features_rest = self.pipeline.__dict__['state_stack'][0]['features_rest'][~clustered_objects_global_mask].to('cuda')
        
        self.pipeline.model.gauss_params["means"] = all_means
        self.pipeline.model.gauss_params["quats"] = all_quats
        self.pipeline.model.gauss_params["scales"] = all_scales
        self.pipeline.model.gauss_params["opacities"] = all_opacities
        self.pipeline.model.gauss_params["features_dc"] = all_features_dc
        self.pipeline.model.gauss_params["features_rest"] = all_features_rest
        
        
        outputs = self.pipeline.model.get_outputs(cam.to('cuda'), tracking=False, BLOCK_WIDTH=16, rgb_only = True)
                
        background = outputs["rgb"].squeeze().detach().cpu().numpy()
        self.pipeline.model.gauss_params["means"] = self.orig_means.clone()
        self.pipeline.model.gauss_params["quats"] = self.orig_quats.clone()
        self.pipeline.model.gauss_params["scales"] = self.orig_scales.clone()
        self.pipeline.model.gauss_params["opacities"] = self.orig_opacities.clone()
        self.pipeline.model.gauss_params["features_dc"] = self.orig_features_dc.clone()
        self.pipeline.model.gauss_params["features_rest"] = self.orig_features_rest.clone()
        return background
            
    def reset_optimizer(self) -> None:
        """Re-generate self.optimizer."""
        self.optimizer.reset_transforms()
        del self.optimizer
        self.pipeline.model.gauss_params["means"] = self.orig_means.clone()
        self.pipeline.model.gauss_params["quats"] = self.orig_quats.clone()
        self.optimizer = RigidGroupOptimizer(
            self.optimizer_config,
            self.pipeline.model,
            group_masks=self.group_masks,
            group_labels=self.group_labels,
            dataset_scale=self.dataset_scale,
            render_lock=self.viewer_ns.train_lock,
        )
    def _cluster_from_file(self):
        self.keep_inds = self.cluster_from_file[1]
        self.pipeline.model.keep_inds = self.cluster_from_file[1]
        self.pipeline.model.cluster_labels = self.cluster_from_file[0]
        self.tfs = self.cluster_from_file[2] # (n,7) quat-pos
        self.pipeline.cgtf_stack = self.cluster_from_file[2]
        self.pipeline.model.cgtf_stack = self.cluster_from_file[2]
        keep_inds_mask = torch.zeros_like(self.pipeline.model.cluster_labels)
        keep_inds_mask[self.keep_inds] = 1
        keep_inds_mask = keep_inds_mask.to(torch.bool)
        
        cluster_labels = self.pipeline.model.cluster_labels[self.keep_inds].to(torch.int32)
        cluster_labels_global = self.pipeline.model.cluster_labels.to(torch.int32)
        self.pipeline._queue_state()
        prev_state = self.pipeline.state_stack[-1]
        for name in self.pipeline.model.gauss_params.keys():
            self.pipeline.model.gauss_params[name] = prev_state[name][self.keep_inds]
        
        return cluster_labels, keep_inds_mask, cluster_labels_global
    
    def _cluster_interactively(self):
        _ = input("Model populated (interactively crop and press enter to continue)")
        self.keep_inds = self.pipeline.model.keep_inds
        
        keep_inds_mask = torch.zeros_like(self.pipeline.model.cluster_labels)
        keep_inds_mask[self.keep_inds] = 1
        keep_inds_mask = keep_inds_mask.to(torch.bool)
        
        cluster_labels = self.pipeline.model.cluster_labels[self.keep_inds].to(torch.int32)
        cluster_labels_global = self.pipeline.model.cluster_labels.to(torch.int32)
        
        self.tfs = self.pipeline.cgtf_stack # (n,7) quat-pos
        self.pipeline.model.cgtf_stack = self.pipeline.cgtf_stack
        return cluster_labels, keep_inds_mask, cluster_labels_global
    
    def _setup_crops_and_groups(self) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Set up the crops and groups for the optimizer, interactively."""
        if self.cluster_from_file is not None: # load cached cluster file, otherwise, interactively cluster
            try:
                if getattr(self.pipeline.model, "best_scales") is None:
                    raise TypeError
                cluster_labels, keep_inds_mask, cluster_labels_global = self._cluster_from_file()
                print("Model populated. Clustered from cache file")
            except TypeError:
                print("Model not populated yet. Please wait...")
                # Wait for the user to set up the crops and groups.
                while getattr(self.pipeline.model, "best_scales") is None:
                    time.sleep(0.1)
                cluster_labels, keep_inds_mask, cluster_labels_global = self._cluster_from_file()
                print("Model populated. Clustered from cache file")
        else:
            try:
                if getattr(self.pipeline.model, "best_scales") is None:
                    raise TypeError
                cluster_labels, keep_inds_mask, cluster_labels_global = self._cluster_interactively()
                print("Clustered interactively")
            except TypeError:
                print("Model not populated yet. Please wait...")
                # Wait for the user to set up the crops and groups.
                while getattr(self.pipeline.model, "best_scales") is None:
                    time.sleep(0.1)
                cluster_labels, keep_inds_mask, cluster_labels_global = self._cluster_interactively()
                print("Clustered interactively")

        self.pipeline.model.mapping, cluster_labels_keep = torch.unique(cluster_labels, return_inverse=True)
        group_masks = [(cid == cluster_labels_keep).cuda() for cid in range(cluster_labels_keep.max().item() + 1)]
        
        group_masks_global = [((cid == cluster_labels_global) & keep_inds_mask).cuda() for cid in self.pipeline.model.mapping]
        self.pipeline.model.render_features = self.render_features
        return cluster_labels_keep.int().cuda(), group_masks, group_masks_global

    def set_frame(self, rgb, ns_camera, depth) -> None:
        """Set the first frame for the optimizer -- doesn't optimize the poses yet."""
        target_frame_rgb = (rgb/255)
        
        frame = Frame(rgb=target_frame_rgb, camera=ns_camera, dino_fn=self.pipeline.datamanager.dino_dataloader.get_pca_feats, metric_depth_img=depth)
        
        self.optimizer.set_frame(frame)
        
    def set_observation(self, rgb, ns_camera, depth) -> None:
        """Set the frame for the optimizer -- doesn't optimize the poses yet."""
        target_frame_rgb = (rgb/255)
        
        frame = PosedObservation(rgb=target_frame_rgb, camera=ns_camera, dino_fn=self.pipeline.datamanager.dino_dataloader.get_pca_feats, metric_depth_img=depth)
        if hasattr(self.optimizer, 'frame'):
            frame_dict = self.optimizer.frame.__dict__
            for attr in list(frame_dict.keys()):
                if isinstance(frame_dict[attr], torch.Tensor):
                    frame_dict[attr] = frame_dict[attr].detach().cpu()
                    del frame_dict[attr]
            if hasattr(self.optimizer.frame, '_roi_frames'):
                for roiframe in self.optimizer.frame._roi_frames:
                    roiframe_dict = roiframe.__dict__
                    for attr in list(roiframe_dict.keys()):
                        if isinstance(roiframe_dict[attr], torch.Tensor):
                            roiframe_dict[attr] = roiframe_dict[attr].detach().cpu()
                        if isinstance(roiframe_dict[attr], Future):
                            del roiframe_dict[attr]
                    del roiframe
                del self.optimizer.frame._roi_frames

            del self.optimizer.frame
        
        self.optimizer.set_observation(frame)

    def init_obj_pose(self):
        """Initialize the object pose, and render the object pose optimization process.
        Also updates `initialized` to `True`."""
        # retval only matters for visualization
        start = time.time()
        renders = self.optimizer.initialize_obj_pose(render=True,n_seeds=7)
        print(f"Time taken for init (pose opt): {time.time() - start:.2f} s")

        start = time.time()
        for idx, render in enumerate(renders):
            if len(render)>1:
                render = [r.detach().cpu().numpy()*255 for r in render]
                # save video as test_camopt.mp4
                out_clip = mpy.ImageSequenceClip(render, fps=30)  
                out_clip.write_videofile(f"test_camopt{idx}.mp4")
        print(f"Time taken for init (video): {time.time() - start:.2f} s")
        
        # Assert there are no nans in part_deltas
        assert not torch.isnan(self.optimizer.part_deltas).any().item()
        if torch.isnan(self.optimizer.part_deltas).any().item():
            exit()
        self.initialized = True

    def step_opt(self,niter):
        """Run the optimizer for `niter` iterations."""
        assert self.initialized, "Please initialize the object pose first."
        outputs = self.optimizer.step(niter=niter)
        return outputs

    def get_pointcloud(self) -> trimesh.PointCloud:
        """Get the pointcloud of the object parts in camera frame."""

        with torch.no_grad():
            self.optimizer.apply_to_model(self.optimizer.part_deltas, self.optimizer.centroids, self.optimizer.group_labels)
        points = self.optimizer.dig_model.means.clone().detach()
        colors = SH2RGB(self.optimizer.dig_model.colors.clone().detach())
        points = points / self.optimizer.dataset_scale
        pc = trimesh.PointCloud(points.cpu().numpy(), colors=colors.cpu().numpy())  # pointcloud in world frame

        cam2world = torch.cat([
            self.init_cam_pose.squeeze(),
            torch.Tensor([[0, 0, 0, 1]]).to(self.init_cam_pose.device)
        ], dim=0)
        pc.vertices = trimesh.transform_points(
            pc.vertices,
            cam2world.inverse().cpu().numpy()
        )  # pointcloud in camera frame.
        return pc

    def get_parts2cam(self,keyframe=None) -> List[vtf.SE3]:
        """Get the parts' poses in camera frame. Wrapper for `RigidGroupOptimizer.get_poses_relative_to_camera`."""
        # Note: `get_poses_relative_to_camera` has dataset_scale scaling built in.
        parts2cam = self.optimizer.get_poses_relative_to_camera(self.init_cam_pose.squeeze().cuda(),keyframe=keyframe)

        # Convert to vtf.SE3.
        parts2cam_vtf = [
            vtf.SE3.from_rotation_and_translation(
                rotation=vtf.SO3.from_matrix(pose[:3,:3].cpu().numpy()),
                translation=pose[:3,3].cpu().numpy()
            ) for pose in parts2cam
        ]
        return parts2cam_vtf
    
    def get_parts2world(self,keyframe=None) -> List[vtf.SE3]:
        """Get the parts' poses in world frame. Wrapper for `RigidGroupOptimizer.get_poses_relative_to_camera`."""
        # Note: `get_poses_relative_to_camera` has dataset_scale scaling built in.
        parts2world = self.optimizer.get_part_poses(keyframe=keyframe)

        # Convert to vtf.SE3.
        parts2world_vtf = [
            vtf.SE3.from_rotation_and_translation(
                rotation=vtf.SO3.from_matrix(pose[:3,:3].cpu().numpy()),
                translation=pose[:3,3].cpu().numpy()
            ) for pose in parts2world
        ]
        return parts2world_vtf

    def get_hands(self, keyframe: int) -> List[trimesh.Trimesh]:
        """Get hands in camera frame."""
        self.optimizer.apply_keyframe(keyframe)
        hands = [deepcopy(_) for _ in self.optimizer.hand_frames[keyframe]]
        T_obj_world = self.optimizer.get_registered_o2w().cpu().numpy()
        T_world_cam = np.linalg.inv(np.concatenate([
            self.init_cam_pose.squeeze().cpu().numpy(),
            np.array([[0, 0, 0, 1]])
        ], axis=0))
        T_obj_cam = T_world_cam @ T_obj_world

        # hands are in object frame! We want it in camera frame.
        for h in hands:
            h.apply_transform(T_obj_cam)
            h.vertices = h.vertices / self.optimizer.dataset_scale

        return hands
    
    def get_clip_relevancy(self, clip_encoder: OpenCLIPNetwork) -> int:
        n_phrases = len(clip_encoder.positives)
        n_phrases_maxs = [None for _ in range(n_phrases)]
        n_phrases_sims = [None for _ in range(n_phrases)]
        scales_list = torch.linspace(0.0, 0.5, 30).to(self.optimizer.pogs_model.device)

        all_probs = []
        
        init_means = self.optimizer.init_means # (N, 3)
        distances, indicies = self.optimizer.pogs_model.k_nearest_sklearn(init_means, 3, True)
        distances = torch.from_numpy(distances).to(self.optimizer.pogs_model.device)
        indicies = torch.from_numpy(indicies).view(-1)
        weights = torch.sigmoid(self.optimizer.init_opacities[indicies].view(-1, 4))
        weights = torch.nn.Softmax(dim=-1)(weights)
        points = init_means[indicies]
        
        hash_encoding = self.optimizer.pogs_model.gaussian_field.get_hash(points) # (N, 3) -> (N, 96)
        hash_encoding = hash_encoding.view(-1, 4, hash_encoding.shape[1])
        hash_encoding = (hash_encoding * weights.unsqueeze(-1))
        hash_encoding = hash_encoding.sum(dim=1)
        
        for i, scale in enumerate(scales_list):
            clip_feats = self.optimizer.pogs_model.gaussian_field.get_clip_outputs_from_feature(hash_encoding, 
                scale.to(self.optimizer.pogs_model.device) *  
                torch.ones(self.optimizer.pogs_model.num_points, 1, device=self.optimizer.pogs_model.device)) # (N, 96) -> (N, 512)

            for j in range(n_phrases):
                probs = clip_encoder.get_relevancy(clip_feats / (clip_feats.norm(dim=-1, keepdim=True)+1e-6), 0).view(self.optimizer.pogs_model.num_points, -1)
                
                pos_prob = probs[..., 0:1]
                all_probs.append((pos_prob.max(), scale))
                if n_phrases_maxs[j] is None or pos_prob.max() > n_phrases_sims[j].max():
                    n_phrases_maxs[j] = scale
                    n_phrases_sims[j] = pos_prob
        relevancy = n_phrases_sims[0]
        
        return relevancy
    
    def state_to_ply(self, obj_id: int = None):
        """Write translated gaussian means to a ply file for grasping subprocess."""
        global_filename = self.config_path.parent.joinpath("global.ply")
                
        # Global state
        prev_state = self.pipeline.state_stack[-1]
        positions = prev_state["means"].detach().cpu().numpy().copy() # [N, 3]
        features_dc = prev_state["features_dc"].detach().cpu().numpy()

        if(self.pipeline.model.config.sh_degree > 0):
            colors = SH2RGB(features_dc)
        else:
            colors = torch.sigmoid(features_dc)
        normalized_colors = (colors - np.min(colors)) / (np.max(colors) - np.min(colors))

        positions = positions.astype('float64')
        normalized_colors = normalized_colors.astype('float64')
        # Homogenize positions for transformation
        positions = np.hstack((positions, np.ones((positions.shape[0], 1))))
        
        deltas = self.optimizer.part_deltas.detach().clone().cpu().numpy() # [num_groups, 7] for x,y,z,qw,qx,qy,qz
        for idx, delta in enumerate(deltas):
            mask = self.group_masks_global[idx].cpu().numpy()
            initial_part2world = self.optimizer.get_initial_part2world(idx).detach().cpu().numpy()

            positions[mask] = positions[mask] - np.hstack([np.array(initial_part2world[:3,3]), 0]) # world frame to object frame (initial rotation is identity)
                        
            quatxyzw = np.concatenate([delta[4:], delta[3:4]])
            delta_transform = vtf.SE3.from_rotation_and_translation(
                rotation=vtf.SO3.from_quaternion_xyzw(quatxyzw),
                translation=delta[:3]
            ).as_matrix()
            
            positions[mask] = (delta_transform @ (positions[mask].T)).T # apply delta transform in object frame
            positions[mask] = positions[mask] + np.hstack([np.array(initial_part2world[:3,3]), 0]) # object frame back to world frame
            
        # Un-homogenize positions
        positions = positions[:, :3]    

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions)
        pcd.colors = o3d.utility.Vector3dVector(normalized_colors)
        o3d.io.write_point_cloud(str(global_filename), pcd)
        
        # Local state
        if obj_id is not None:
            local_filename = self.config_path.parent.joinpath("local.ply")
            local_positions = positions[self.group_masks_global[obj_id].cpu().numpy()]
            local_colors = normalized_colors[self.group_masks_global[obj_id].cpu().numpy()]
            local_positions = local_positions.astype('float64')
            local_colors = local_colors.astype('float64')
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(local_positions)
            pcd.colors = o3d.utility.Vector3dVector(local_colors)
            o3d.io.write_point_cloud(str(local_filename), pcd)