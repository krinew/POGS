from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Type, Union

import numpy as np
import torch

try:
    from gsplat.rendering import rasterization
    from gsplat.strategy import DefaultStrategy
except ImportError:
    print("Please install gsplat>=1.0.0")
from nerfstudio.models.splatfacto import RGB2SH
from torch.nn import Parameter
import torch.nn.functional as F
from torchvision.transforms.functional import resize

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras

# need following import for background color override
from nerfstudio.model_components import renderers
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.rich_utils import CONSOLE

# sms imports
from nerfstudio.models.splatfacto import SplatfactoModelConfig, SplatfactoModel
from nerfstudio.viewer.viewer_elements import ViewerButton, ViewerSlider, ViewerControl
from pogs.fields.gaussian_field import GaussianField
from pogs.encoders.image_encoder import BaseImageEncoder
from pogs.field_components.gaussian_fieldheadnames import GaussianFieldHeadNames
from nerfstudio.model_components import losses
from pogs.model_components.losses import DepthLossType, mse_depth_loss, depth_ranking_loss, pearson_correlation_depth_loss
from nerfstudio.utils.colormaps import apply_colormap
import viser.transforms as vtf
from cuml.cluster.hdbscan import HDBSCAN
import cv2
import open3d as o3d
import time
from collections import OrderedDict

from pogs.data.utils.dino_dataloader import get_img_resolution, MAX_DINO_SIZE
from sklearn.neighbors import NearestNeighbors

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
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )

def generate_random_colors(N=5000) -> torch.Tensor:
    """Generate random colors for visualization"""
    hs = np.random.uniform(0, 1, size=(N, 1))
    ss = np.random.uniform(0.6, 0.61, size=(N, 1))
    vs = np.random.uniform(0.84, 0.95, size=(N, 1))
    hsv = np.concatenate([hs, ss, vs], axis=-1)
    # convert to rgb
    rgb = cv2.cvtColor((hsv * 255).astype(np.uint8)[None, ...], cv2.COLOR_HSV2RGB)
    return torch.Tensor(rgb.squeeze() / 255.0)

# @torch.compile()
def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat

@dataclass
class POGSModelConfig(SplatfactoModelConfig):

    _target: Type = field(default_factory=lambda: POGSModel)
    split_screen_size: float = 0.025
    """if a gaussian is more than this percent of screen space, split it"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    clip_loss_weight: float = 0.5
    """weight of clip loss"""
    output_depth_during_training: bool = True
    """If True, output depth during training. Otherwise, only output depth during evaluation."""
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="SO3xR3"))
    """Config of the camera optimizer to use"""
    gaussian_dim:int = 64
    """Dimension the gaussians actually store as features"""
    dim: int = 64
    """Output dimension of the feature rendering"""
    dino_rescale_factor: int = 5
    """How much to upscale rendered dino for supervision"""
    min_mask_screensize: float = 0.003
    """Minimum screen size of masks to use for supervision"""
    depth_loss_mult: float = 0.05
    """Lambda of the depth loss."""
    depth_loss_type: DepthLossType = DepthLossType.PEARSON_LOSS
    """Depth loss type."""
    num_downscales: int = 0
    use_scale_regularization: bool = True
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 3.0
    """threshold of ratio of gaussian max to min scale before applying regularization"""
    # max_gs_num: int = 30_000
    # """Maximum number of GSs. Default to 1_000_000."""

class POGSModel(SplatfactoModel):

    config: POGSModelConfig

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        if "seed_points" in kwargs:
            self.seed_pts = kwargs["seed_points"]
        else:
            self.seed_pts = None
        super().__init__(*args, **kwargs)
        self.loaded_ckpt = False
        self.localized_query = None
        self.best_scales = None
    
    def populate_modules(self):
        super().populate_modules()
        self.gauss_params['dino_feats'] = torch.nn.Parameter(torch.randn((self.num_points, self.config.gaussian_dim)))
        torch.inverse(torch.ones((1, 1), device="cuda:0"))# https://github.com/pytorch/pytorch/issues/90613
        
        self.gaussian_field = GaussianField()
        self.datamanager = self.kwargs["datamanager"]
        self.image_encoder: BaseImageEncoder = self.kwargs["image_encoder"]

        self.nn = torch.nn.Sequential(
            torch.nn.Linear(self.config.gaussian_dim, 64, bias = False),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64, bias = False),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64, bias = False),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.config.dim, bias = False)
        )
        
        # POGS init
        self.steps_since_add = 0
        
        self.viewer_control = ViewerControl()
        self.viser_scale_ratio = 0.1
        self.colormap = generate_random_colors()
        self.cluster_scene = ViewerButton(name="Cluster Scene", cb_hook=self._cluster_scene, disabled=False)
        self.toggle_rgb_cluster = ViewerButton(name="Toggle RGB/Cluster", cb_hook=self._togglergbcluster, disabled=False)
        self.cluster_scene_shuffle_colors = ViewerButton(name="Reshuffle Cluster Colors", cb_hook=self._reshuffle_cluster_colors, disabled=False)
        self.cluster_labels = None
        self.keep_inds = None
        self.cgtf_stack = None
        self.render_features = True
        self.mapping = None # maps tracked object_id to cluster label
        self.rgb1_cluster0 = True
        self.temp_opacities = None
        self.frame_on_word = ViewerButton("Best Guess", cb_hook=self.localize_query_cb)
        self.relevancy_thresh = ViewerSlider("Relevancy Thresh", 0.0, 0, 1.0, 0.01)
        self.cluster_eps = ViewerSlider("Cluster Eps", 0.012, 0.005, 0.1, 0.005)
        

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        super().load_state_dict(dict, **kwargs)
        # here we need to do some hacky stuff....
        # Convert gauss_params from ParameterDict to a simple OrderedDict of Tensors
        # This is critical for allowing backprop through the gauss_params
        newdict = OrderedDict()
        for k, v in self.gauss_params.items():
            newdict[k] = torch.Tensor(v)
        del self.gauss_params
        self.gauss_params = newdict

    def k_nearest_sklearn(self, x: torch.Tensor, k: int, include_self: bool = False):
        """
        Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        if include_self:
            return distances.astype(np.float32), indices
        else:
            return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        gpg = {
            name: [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        }
        gpg["lerf"] = list(self.gaussian_field.parameters())
        gpg['dino_feats'] = [self.gauss_params['dino_feats']]

        return gpg

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        gps['nn_projection'] = list(self.nn.parameters())
        self.camera_optimizer.get_param_groups(param_groups=gps)
        return gps

    @staticmethod
    def get_empty_outputs(width: int, height: int, background: torch.Tensor) -> Dict[str, Union[torch.Tensor, List]]:
        rgb = background.repeat(height, width, 1)
        depth = background.new_ones(*rgb.shape[:2], 1) * 10
        accumulation = background.new_zeros(*rgb.shape[:2], 1)
        return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}

    def get_outputs(self, camera: Cameras, tracking=False, obj_id=None, invert = False, BLOCK_WIDTH=16, rgb_only = False) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        outputs = {}
        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        background = self._get_background_color()
        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), self.background_color
                )
        else:
            crop_ids = None
        
        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore
        
        if obj_id is not None:
            if invert:
                crop_ids = torch.where(self.cluster_labels[self.keep_inds] != self.mapping[obj_id].item())[0]
            else:
                crop_ids = torch.where(self.cluster_labels[self.keep_inds] == self.mapping[obj_id].item())[0]
            
        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
            dino_crop = self.gauss_params['dino_feats'][crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats
            dino_crop = self.gauss_params['dino_feats']

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        if not self.rgb1_cluster0 and self.cluster_labels is not None:
            labels = self.cluster_labels.numpy()
            features_dc = self.gauss_params['features_dc'].detach().clone()
            features_rest = self.gauss_params['features_rest'].detach().clone()
            for c_id in range(0, labels.max().astype(int) + 1):
                # set the colors of the gaussians accordingly using colormap from matplotlib
                cluster_mask = np.where(labels == c_id)
                features_dc[cluster_mask] = RGB2SH(self.colormap[c_id, :3].to(self.gauss_params['features_dc']))
                features_rest[cluster_mask] = torch.zeros_like(features_rest[cluster_mask])
            if crop_ids is not None:
                colors_crop = torch.cat((features_dc[:, None, :][crop_ids], features_rest[crop_ids]), dim=1)
            else:
                colors_crop = torch.cat((features_dc[:, None, :], features_rest), dim=1)

        for i in range(len(self.image_encoder.positives)):
            if self.image_encoder.positives[i] == '':
                self.temp_opacities = None
        
        if self.cluster_labels is not None and self.temp_opacities is not None:
            if crop_ids is not None:
                opacities_crop = self.temp_opacities[crop_ids]
            else:
                opacities_crop = self.temp_opacities

        K = camera.get_intrinsics_matrices().cuda()
        K[:, :2, :] *= camera_scale_fac
        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            sh_degree_to_use = None
            
        render, alpha, info = rasterization(
            means=means_crop,
            quats=quats_crop,
            scales=torch.exp(scales_crop), # rasterization does normalization internally
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=self.strategy.absgrad if isinstance(self.strategy, DefaultStrategy) else False,
            rasterize_mode=self.config.rasterize_mode,
        )
        
        if self.training:
            self.info = info
            self.strategy.step_pre_backward(
                self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
            )
        if self.training and info["means2d"].requires_grad:
            info["means2d"].retain_grad()

        alpha = alpha[:, ...]
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)
        outputs["rgb"] = rgb.squeeze(0)
        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None
        outputs["depth"] = depth_im
        outputs["accumulation"] = alpha.squeeze(0)
        outputs["background"] = background
        
        if rgb_only:
            return outputs
        
        if self.datamanager.use_clip or self.loaded_ckpt and not tracking:
            if (self.step - self.datamanager.lerf_step > 0):
                
                ########################
                # CLIP Relevancy Field #
                ########################
                reset_interval = self.config.reset_alpha_every * self.config.refine_every
                field_output = None
                if self.training and self.step>self.config.warmup_length and (self.step % reset_interval > self.num_train_data + self.config.refine_every  or self.step < (self.config.reset_alpha_every * self.config.refine_every)):
                    clip_hash_encoding = self.gaussian_field.get_hash(self.means)

                    # Might not be handling this optimally
                    if 'image_downscale_factor' in self.datamanager.train_dataset._dataparser_outputs.metadata.keys():
                        rgb_downscale = self.datamanager.train_dataset._dataparser_outputs.metadata['image_downscale_factor']
                        downscale_factor = self.datamanager.config.clip_downscale_factor / rgb_downscale
                    else:
                        downscale_factor = self.datamanager.config.clip_downscale_factor
                    camera.rescale_output_resolution(1 / downscale_factor)

                    clip_W, clip_H = camera.width.item(), camera.height.item()
                    clipK = camera.get_intrinsics_matrices().cuda()
                    
                    field_output, alpha, info = rasterization(
                        means=means_crop,
                        quats=quats_crop,
                        scales=torch.exp(scales_crop),
                        opacities=torch.sigmoid(opacities_crop).squeeze(-1),
                        colors=clip_hash_encoding,
                        viewmats=viewmat, # [1, 4, 4]
                        Ks=clipK,  # [1, 3, 3]
                        width=clip_W,
                        height=clip_H,
                        tile_size=BLOCK_WIDTH,
                        packed=False,
                        near_plane=0.01,
                        far_plane=1e10,
                        render_mode="RGB",
                        sparse_grad=False,
                        absgrad=True,
                        rasterize_mode=self.config.rasterize_mode,
                    )

                    # rescale the camera back to original dimensions
                    camera.rescale_output_resolution(downscale_factor)
                    
                    self.random_pixels = self.datamanager.random_pixels.to(self.device)

                    clip_scale = self.datamanager.curr_scale * torch.ones((self.random_pixels.shape[0],1),device=self.device)
                    clip_scale = clip_scale * clip_H * (depth_im.view(-1, 1)[::downscale_factor**2][self.random_pixels] / camera.fy.item())

                    field_output = self.gaussian_field.get_outputs_from_feature(field_output.view(clip_H*clip_W, -1), clip_scale, self.random_pixels)

                    clip_output = field_output[GaussianFieldHeadNames.CLIP].to(dtype=torch.float32)

                    outputs["clip"] = clip_output
                    outputs["clip_scale"] = clip_scale

                    outputs["instance"] = field_output[GaussianFieldHeadNames.INSTANCE].to(dtype=torch.float32)

                if camera.metadata is not None:
                    if "clip_downscale_factor" not in camera.metadata and not rgb_only:
                        # N x B x 1; N
                        max_across, self.best_scales, instances_out = self.get_max_across(means_crop, quats_crop, scales_crop, opacities_crop, viewmat, K, H, W, preset_scales=None)

                        if not torch.isnan(instances_out).any():
                            outputs["group_feats"] = instances_out
                        else:
                            print("instance loss may be nan")

                        if not hasattr(self, "image_encoder"): # Await Load
                            if not hasattr(self.image_encoder, "positives"):
                                time.sleep(0.2)
                                
                        for i in range(len(self.image_encoder.positives)):
                            max_across[i][max_across[i] < self.relevancy_thresh.value] = 0
                            outputs[f"relevancy_{i}"] = max_across[i].view(H, W, -1)

        # DINO stuff
        if (self.step - self.datamanager.dino_step > 0) or tracking:
            p_size = 14 

            downscale = (self.config.dino_rescale_factor*MAX_DINO_SIZE/max(H,W))/p_size
            
            if camera.metadata is not None:
                downscale = 1.0 if ("clip_downscale_factor" not in camera.metadata and not tracking) else downscale
            elif tracking:
                downscale = 1.0
            h,w = get_img_resolution(H, W, p = p_size)
            dino_K = K.clone()
            dino_K[:, :2, :] *= downscale
            dino_h,dino_w = self.config.dino_rescale_factor*(h//p_size),self.config.dino_rescale_factor*(w//p_size)
            if camera.metadata is not None:
                if "clip_downscale_factor" not in camera.metadata:
                    dino_h,dino_w = H,W
            elif tracking:
                dino_h,dino_w = H,W
            dino_feats, dino_alpha, _ = rasterization(
                means=means_crop.detach() if self.training else means_crop,
                quats=quats_crop.detach(),
                scales=torch.exp(scales_crop).detach(),
                opacities=torch.sigmoid(opacities_crop).squeeze(-1).detach(),
                colors=dino_crop,
                viewmats=viewmat,  # [1, 4, 4]
                Ks=dino_K,  # [1, 3, 3]
                width=dino_w,
                height=dino_h,
                packed=False,
                near_plane=0.01,
                far_plane=1e10,
                render_mode="RGB",
                sparse_grad=False,
                absgrad=False,
                rasterize_mode=self.config.rasterize_mode,
            )
            feat_shape = dino_feats.shape
            if torch.isnan(dino_alpha).any() or torch.isinf(dino_alpha).any():
                print("NaNs or Infs detected in dino_alpha")
            if torch.isnan(dino_feats).any() or torch.isinf(dino_feats).any():
                print("NaNs or Infs detected in dino_feats")
            dino_alpha_clamped = torch.clamp(dino_alpha, min=1e-6)

            dino_feats = torch.where(dino_alpha > 0, dino_feats / dino_alpha_clamped.detach(), torch.zeros(self.config.gaussian_dim, device=self.device))
            nn_inputs = dino_feats.view(-1,self.config.gaussian_dim)
            dino_feats = self.nn(nn_inputs).view(*feat_shape[:-1],-1)
            if not self.training:
                dino_feats[dino_alpha.squeeze(-1) < 0.8] = 0
            outputs['dino'] = dino_feats.squeeze(0)
        return outputs
    
    def _get_downscale_factor(self):
        return 1 # no downscaling for Lang Embed
    
    def reshape_termination_depth(self, termination_depth, output_depth_shape):
        termination_depth = F.interpolate(termination_depth.permute(2, 0, 1).unsqueeze(0), size=(output_depth_shape[0], output_depth_shape[1]), mode='bilinear', align_corners=False)
        # Remove the extra dimensions added by unsqueeze and permute
        termination_depth = termination_depth.squeeze(0).permute(1, 2, 0)
        return termination_depth

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        
        if self.config.depth_loss_type not in (DepthLossType.NONE,) and self.training and "depth_image" in batch:
            output_depth_shape = outputs["depth"].shape[:2]
            if (
                losses.FORCE_PSEUDODEPTH_LOSS
                and self.config.depth_loss_type not in losses.PSEUDODEPTH_COMPATIBLE_LOSSES
            ):
                raise ValueError(
                    f"Forcing pseudodepth loss, but depth loss type ({self.config.depth_loss_type}) must be one of {losses.PSEUDODEPTH_COMPATIBLE_LOSSES}"
                )
            if self.config.depth_loss_type in (DepthLossType.MSE,):
                metrics_dict["depth_loss"] = torch.Tensor([0.0]).to(self.device)
                termination_depth = batch["depth_image"].to(self.device)
                termination_depth = self.reshape_termination_depth(termination_depth, output_depth_shape)

                metrics_dict["depth_loss"] = mse_depth_loss(
                    termination_depth, outputs["depth"])
            
            elif self.config.depth_loss_type in (DepthLossType.PEARSON_LOSS,):
                metrics_dict["depth_loss"] = torch.Tensor([0.0]).to(self.device)
                termination_depth = batch["depth_image"].to(self.device)
                termination_depth = self.reshape_termination_depth(termination_depth, output_depth_shape)
                
                metrics_dict["depth_loss"] = pearson_correlation_depth_loss(
                    termination_depth, outputs["depth"])
            
            elif self.config.depth_loss_type in (DepthLossType.SPARSENERF_RANKING,):
                metrics_dict["depth_ranking"] = depth_ranking_loss(
                    outputs["depth"], batch["depth_image"].to(self.device)
                )
            else:
                raise NotImplementedError(f"Unknown depth loss type {self.config.depth_loss_type}")
            
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        margin = 1.0
        if self.training and 'clip' in outputs and 'clip' in batch: 
            unreduced_clip = self.config.clip_loss_weight * torch.nn.functional.huber_loss(
                outputs["clip"], batch["clip"].to(self.device).to(torch.float32), delta=1.25, reduction="none"
            )
            loss_dict["clip_loss"] = unreduced_clip.sum(dim=-1).nanmean()

        if self.training and 'instance' in outputs and 'instance_masks' in batch: 
            mask = batch["instance_masks"]
            if len(mask) > 2:
                instance_loss = torch.tensor(0.0, device=self.device)

                idx = torch.randperm(len(mask)-1)
                
                total_ray_count = outputs["instance"].shape[0]

                count = 0
                
                # Contrastive loss between mask features
                for i in range(len(mask)-2):
                    if ((mask[idx[i]].sum()/total_ray_count <= self.config.min_mask_screensize) 
                        or 
                        (mask[idx[i+1]].sum()/total_ray_count <= self.config.min_mask_screensize)):
                        continue
                    instance_loss += (
                        F.relu(margin - torch.norm(outputs["instance"][mask[idx[i]]].mean(dim=0) - outputs["instance"][mask[idx[i+1]]].mean(dim=0), p=2, dim=-1))).nansum()
                    count += 1
                    
                # Encourage features within a mask to be close to each other
                for i in range(len(mask)-1):
                    if (mask[i].sum()/total_ray_count <= self.config.min_mask_screensize):
                        continue

                    else:
                        instance_loss += F.relu(torch.norm(outputs["instance"][mask[idx[i]]] - outputs["instance"][mask[idx[i]]].mean(dim=0).repeat(mask[idx[i]].sum(),1), p=2, dim=-1)).nanmean()
                        count += 1

                # Push the negative mask to ones normed vector
                instance_loss += 0.1 * F.relu(torch.norm(outputs["instance"][mask[-1]] - (torch.ones(128, device=self.device)/torch.ones(128, device=self.device).norm()).repeat(mask[-1].sum(),1), p=2, dim=-1)).nanmean()
                count += 1
                        
                loss = instance_loss / count
                if loss != 0:
                    loss_dict["instance_loss"] = loss

        if 'dino' in outputs and 'dino' in batch:
            gt = batch['dino']
            gt = resize(gt.permute(2,0,1), (outputs['dino'].shape[0], outputs['dino'].shape[1])).permute(1,2,0)
            loss_dict['dino_loss'] = torch.nn.functional.mse_loss(outputs['dino'],gt)
            if not hasattr(self,'nearest_ids') or self.num_points != self.nearest_ids.shape[0]:
                from cuml.neighbors import NearestNeighbors
                model = NearestNeighbors(n_neighbors=3)
                means = self.means.detach().cpu().numpy()
                model.fit(means)
                _, self.nearest_ids = model.kneighbors(means)
            # encourage the nearest neighbors to have similar dino feats
            if self.step > (self.datamanager.dino_step+1000):
                loss_dict['dino_nn_loss'] = .01*self.gauss_params['dino_feats'][self.nearest_ids].var(dim=1).sum()
        
        if self.config.depth_loss_type not in (DepthLossType.NONE,) and 'depth' in outputs and 'depth_image' in batch:
            assert metrics_dict is not None and ("depth_loss" in metrics_dict or "depth_ranking" in metrics_dict)
            if "depth_ranking" in metrics_dict:
                loss_dict["depth_ranking"] = (
                    self.config.depth_loss_mult
                    * np.interp(self.step, [0, 2000], [0, 0.2])
                    * metrics_dict["depth_ranking"]
                )
            
            if "depth_loss" in metrics_dict:
                loss_dict["depth_loss"] = self.config.depth_loss_mult * metrics_dict["depth_loss"]
        return loss_dict


    def localize_query_cb(self,element):
        if len(self.image_encoder.positives) == 0:
            return
        with torch.no_grad():

            # Do K nearest neighbors for each point and then avg the clip hash for each point based on the KNN
            means_freeze = self.means.data.detach().clone()
            distances, indicies = self.k_nearest_sklearn(means_freeze, 3, True)
            distances = torch.from_numpy(distances).to(self.device)
            indicies = torch.from_numpy(indicies).view(-1)
            weights = torch.sigmoid(self.opacities[indicies].view(-1, 4))
            weights = torch.nn.Softmax(dim=-1)(weights)
            points = means_freeze[indicies]
            clip_hash_encoding = self.gaussian_field.get_hash(points)
            clip_hash_encoding = clip_hash_encoding.view(-1, 4, clip_hash_encoding.shape[1])
            clip_hash_encoding = (clip_hash_encoding * weights.unsqueeze(-1))
            clip_hash_encoding = clip_hash_encoding.sum(dim=1)
            clip_feats = self.gaussian_field.get_outputs_from_feature(clip_hash_encoding, self.best_scales[0].to(self.device) * torch.ones(self.num_points, 1, device=self.device))[GaussianFieldHeadNames.CLIP].to(dtype=torch.float32)
            relevancy = self.image_encoder.get_relevancy(clip_feats / (clip_feats.norm(dim=-1, keepdim=True)+1e-6), 0).view(self.num_points, -1)
            
            labels = self.cluster_labels.numpy()
            avg_relevancy_per_cluster = []
            for c_id in range(0, labels.max().astype(int) + 1):
                # set the colors of the gaussians accordingly using colormap from matplotlib
                cluster_mask = np.where(labels == c_id)

                avg_relevancy_per_cluster.append(relevancy[cluster_mask][..., 0].mean().item())

            cluster_argmax = np.array(avg_relevancy_per_cluster).argmax()

            cluster_mask = np.where(labels != cluster_argmax)

            self.temp_opacities = self.opacities.data.clone()
            self.temp_opacities[cluster_mask[0]] = self.opacities.data.min()/2

            self.viewer_control.viewer._trigger_rerender()  # trigger viewer rerender

    def crop_to_word_cb(self,element):
        with torch.no_grad():
            # Do K nearest neighbors for each point and then avg the clip hash for each point based on the KNN
            distances, indicies = self.k_nearest_sklearn(self.means.data, 3, True)
            distances = torch.from_numpy(distances).to(self.device)
            indicies = torch.from_numpy(indicies).to(self.device).view(-1)
            weights = torch.sigmoid(self.opacities[indicies].view(-1, 4))
            weights = torch.nn.Softmax(dim=-1)(weights)
            points = self.means[indicies]
            clip_hash_encoding = self.gaussian_field.get_hash(points)
            clip_hash_encoding = clip_hash_encoding.view(-1, 4, clip_hash_encoding.shape[1])
            clip_hash_encoding = (clip_hash_encoding * weights.unsqueeze(-1))
            clip_hash_encoding = clip_hash_encoding.sum(dim=1)
            clip_feats = self.gaussian_field.get_outputs_from_feature(clip_hash_encoding, self.best_scales[0].to(self.device) * torch.ones(self.num_points, 1, device=self.device))[GaussianFieldHeadNames.CLIP].to(dtype=torch.float32)
            relevancy = self.image_encoder.get_relevancy(clip_feats / (clip_feats.norm(dim=-1, keepdim=True)+1e-6), 0).view(self.num_points, -1)
            color = apply_colormap(relevancy[..., 0:1])
            self.crop_ids = (relevancy[..., 0] / relevancy[..., 0].max() > self.relevancy_thresh.value)

            self.viewer_control.viser_server.add_point_cloud(
                "Relevancy", 
                self.means.numpy(force=True)[self.crop_ids.cpu()] * 10, 
                color.numpy(force=True)[self.crop_ids.cpu()], 
                0.01
                )

            # Add a slider to debug the relevancy values
                        
            #Define all crop viewer elements
            self._crop_center_init = self.means[self.crop_ids].mean(dim=0).cpu().numpy()
            self.original_means = self.means.data.clone()

            self._crop_handle = self.viewer_control.viser_server.add_transform_controls(
                "Crop Points", 
                depth_test=False, 
                line_width=4.0)
            world_center = tuple(p / self.viser_scale_ratio for p in self._crop_center_init)
            self._crop_handle.position = world_center

            @self._crop_handle.on_update
            def _update_crop_handle(han):
                if self._crop_center_init is None:
                    return
                new_center = np.array(self._crop_handle.position) * self.viser_scale_ratio
                delta = new_center - self._crop_center_init
                displacement = torch.zeros_like(self.means)
                displacement[self.crop_ids] = torch.from_numpy(delta).to(self.device).to(self.means.dtype)
                
                curr_to_world = torch.from_numpy(vtf.SE3(np.concatenate((self._crop_handle.wxyz, self._crop_handle.position * self.viser_scale_ratio))).as_matrix()).to(self.device).to(self.means.dtype)
                transform = torch.from_numpy(vtf.SE3(np.concatenate((self._crop_handle.wxyz, (self._crop_handle.position * self.viser_scale_ratio) - self._crop_center_init))).as_matrix()).to(self.device).to(self.means.dtype)

                transformed_points = self.original_means.clone()
                homogeneous_points = torch.cat((transformed_points[self.crop_ids], torch.ones(transformed_points[self.crop_ids].shape[0], 1, device=self.device, dtype=self.means.dtype)), dim=1)
                transformed_homogeneous = curr_to_world @ transform @ torch.inverse(curr_to_world) @ homogeneous_points.transpose(0,1)
                transformed_homogeneous = transformed_homogeneous.transpose(0,1)
                transformed_points[self.crop_ids] = transformed_homogeneous[:, :3] / transformed_homogeneous[:, 3:4]
                self.means.data = transformed_points

            self.viewer_control.viser_server.add_point_cloud(
                "Centroid", 
                np.array([(self._crop_center_init / self.viser_scale_ratio)]), 
                np.array([0,0,0]), 
                0.1
                )

    def reset_crop_cb(self, element):
        self.crop_ids = None
        self.means.data = self.original_means
        self._crop_center_init = None
        self._crop_handle.visible = False

    def _reshuffle_cluster_colors(self, button: ViewerButton):
        """Reshuffle the cluster colors, if clusters defined using `_cluster_scene`."""
        if self.cluster_labels is None:
            return
        self.cluster_scene_shuffle_colors.set_disabled(True)  # Disable user from reshuffling colors
        self.colormap = generate_random_colors()
        colormap = self.colormap

        labels = self.cluster_labels

        features_dc = self.gauss_params['features_dc'].detach()
        features_rest = self.gauss_params['features_rest'].detach()
        for c_id in range(0, labels.max().int().item() + 1):
            # set the colors of the gaussians accordingly using colormap from matplotlib
            cluster_mask = np.where(labels == c_id)
            features_dc[cluster_mask] = RGB2SH(colormap[c_id, :3].to(self.gauss_params['features_dc']))
            features_rest[cluster_mask] = 0

        self.gauss_params['features_dc'] = torch.nn.Parameter(self.gauss_params['features_dc'])
        self.gauss_params['features_rest'] = torch.nn.Parameter(self.gauss_params['features_rest'])
        self.cluster_scene_shuffle_colors.set_disabled(False)

    def _cluster_scene(self, button: ViewerButton):
        """Cluster the scene, and assign gaussian colors based on the clusters.
        Also populates self.crop_group_list with the clusters group indices."""

        self.cluster_scene.set_disabled(True)  # Disable user from clustering, while clustering

        labels = self.cluster(self.cluster_eps.value)

        opacities = self.gauss_params['opacities'].detach()
        opacities[labels < 0] = -100  # hide unclustered gaussians
        self.gauss_params['opacities'] = torch.nn.Parameter(opacities.float())

        self.cluster_labels = torch.Tensor(labels)

        self.rgb1_cluster0 = not self.rgb1_cluster0
        self.cluster_scene.set_disabled(False)
        
        self.viewer_control.viewer._trigger_rerender()  # trigger viewer rerender
    
    def cluster(self, eps = 0.1):
        positions = self.means.data.clone().detach()
        distances, indicies = self.k_nearest_sklearn(positions, 3, True)
        distances = torch.from_numpy(distances).to(self.device)
        indicies = torch.from_numpy(indicies).view(-1)
        weights = torch.sigmoid(self.opacities[indicies].view(-1, 4))
        weights = torch.nn.Softmax(dim=-1)(weights)
        points = positions[indicies]
        hash_encoding = self.gaussian_field.get_hash(points)
        hash_encoding = hash_encoding.view(-1, 4, hash_encoding.shape[1])
        hash_encoding = (hash_encoding * weights.unsqueeze(-1))
        hash_encoding = hash_encoding.sum(dim=1)
        group_feats = self.gaussian_field.get_instance_outputs_from_feature(hash_encoding).to(dtype=torch.float32).cpu().detach().numpy()
        positions = positions.cpu().numpy()

        start = time.time()

        # Cluster the gaussians using HDBSCAN.
        # We will first cluster the downsampled gaussians, then 
        # assign the full gaussians to the spatially closest downsampled gaussian.

        vec_o3d = o3d.utility.Vector3dVector(positions)
        pc_o3d = o3d.geometry.PointCloud(vec_o3d)
        min_bound = np.clip(pc_o3d.get_min_bound(), -1, 1)
        max_bound = np.clip(pc_o3d.get_max_bound(), -1, 1)
        pc, _, ids = pc_o3d.voxel_down_sample_and_trace(
            0.0001, min_bound, max_bound
        )
        if len(ids) > 1e6:
            print(f"Too many points ({len(ids)}) to cluster... aborting.")
            print( "Consider using interactive select to reduce points before clustering.")
            print( "Are you sure you want to cluster? Press y to continue, else return.")
            # wait for input to continue, if yes then continue, else return
            if input() != "y":
                self.cluster_scene.set_disabled(False)
                return

        id_vec = np.array([points[0] for points in ids])  # indices of gaussians kept after downsampling
        group_feats_downsampled = group_feats[id_vec]
        positions_downsampled = np.array(pc.points)

        print(f"Clustering {group_feats_downsampled.shape[0]} gaussians... ", end="", flush=True)

        # Run cuml-based HDBSCAN
        clusterer = HDBSCAN(
            cluster_selection_epsilon=eps,
            min_samples=50,
            min_cluster_size=300,
            allow_single_cluster=False,
        ).fit(group_feats_downsampled)

        non_clustered = np.ones(positions.shape[0], dtype=bool)
        non_clustered[id_vec] = False
        labels = clusterer.labels_.copy()
        clusterer.labels_ = -np.ones(positions.shape[0], dtype=np.int32)
        clusterer.labels_[id_vec] = labels

        # Assign the full gaussians to the spatially closest downsampled gaussian, with scipy NearestNeighbors.
        positions_np = positions[non_clustered]
        if positions_np.shape[0] > 0:  # i.e., if there were points removed during downsampling
            k = 1
            nn_model = NearestNeighbors(
                n_neighbors=k, algorithm="auto", metric="euclidean"
            ).fit(positions_downsampled)
            _, indices = nn_model.kneighbors(positions_np)
            clusterer.labels_[non_clustered] = labels[indices[:, 0]]

        labels = clusterer.labels_
        print(f"done. Took {time.time()-start} seconds. Found {labels.max() + 1} clusters.")

        noise_mask = labels == -1
        if noise_mask.sum() != 0 and (labels>=0).sum() > 0:
            # if there is noise, but not all of it is noise, relabel the noise
            valid_mask = labels >=0
            valid_positions = positions[valid_mask]
            k = 1
            nn_model = NearestNeighbors(
                n_neighbors=k, algorithm="auto", metric="euclidean"
            ).fit(valid_positions)
            noise_positions = positions[noise_mask]
            _, indices = nn_model.kneighbors(noise_positions)
            # for now just pick the closest cluster
            noise_relabels = labels[valid_mask][indices[:, 0]]
            labels[noise_mask] = noise_relabels
            clusterer.labels_ = labels

        labels = clusterer.labels_
        return labels
    
    def _togglergbcluster(self, button: ViewerButton):
        self.rgb1_cluster0 = not self.rgb1_cluster0
        self.viewer_control.viewer._trigger_rerender()  # trigger viewer rerender
    
    @torch.no_grad()
    def get_grouping_at_points(self, positions):
        """Get the grouping features at a set of points."""
        # Apply distortion, calculate hash values, then normalize

        x = self.gaussian_field.get_hash(positions)
        x = x / x.norm(dim=-1, keepdim=True)

        return self.gaussian_field.get_instance_outputs_from_feature(x)

    def get_max_across(self, means_crop, quats_crop, scales_crop, opacities_crop, viewmat, K, H, W, preset_scales=None):
        n_phrases = len(self.image_encoder.positives)
        n_phrases_maxs = [None for _ in range(n_phrases)]
        n_phrases_sims = [None for _ in range(n_phrases)]
        scales_list = torch.linspace(0.0, 0.5, 30).to(self.device)
        all_probs = []
        BLOCK_WIDTH = 16

        with torch.no_grad():
            hash_encoding = self.gaussian_field.get_hash(self.means)

            field_output, alpha, info = rasterization(
                        means=means_crop,
                        quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
                        scales=torch.exp(scales_crop),
                        opacities=torch.sigmoid(opacities_crop).squeeze(-1),
                        colors=hash_encoding,
                        viewmats=viewmat, # [1, 4, 4]
                        Ks=K,  # [1, 3, 3]
                        width=W,
                        height=H,
                        tile_size=BLOCK_WIDTH,
                        packed=False,
                        near_plane=0.01,
                        far_plane=1e10,
                        render_mode="RGB",
                        sparse_grad=False,
                        absgrad=True,
                        rasterize_mode=self.config.rasterize_mode,
            )

        for i, scale in enumerate(scales_list):
            with torch.no_grad():
                out = self.gaussian_field.get_outputs_from_feature(field_output.view(H*W, -1), scale * torch.ones(H*W, 1, device=self.device))
                instances_output_im = out[GaussianFieldHeadNames.INSTANCE].to(dtype=torch.float32).view(H, W, -1)
                clip_output_im = out[GaussianFieldHeadNames.CLIP].to(dtype=torch.float32).view(H, W, -1)

            for j in range(n_phrases):
                if preset_scales is None or j == i:

                    probs = self.image_encoder.get_relevancy(clip_output_im.view(-1, self.image_encoder.embedding_dim), j)
                    pos_prob = probs[..., 0:1]
                    all_probs.append((pos_prob.max(), scale))
                    if n_phrases_maxs[j] is None or pos_prob.max() > n_phrases_sims[j].max():
                        n_phrases_maxs[j] = scale
                        n_phrases_sims[j] = pos_prob

        return torch.stack(n_phrases_sims), torch.Tensor(n_phrases_maxs), instances_output_im