import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from typing import List, Optional
import kornia
from pogs.pogs import POGSModel
from contextlib import nullcontext
from nerfstudio.engine.schedulers import (
    ExponentialDecayScheduler,
    ExponentialDecaySchedulerConfig,
)
import warp as wp
from pogs.tracking.atap_loss import ATAPLoss
from pogs.tracking.utils import *
import viser.transforms as vtf
import trimesh
from typing import Tuple
from pogs.tracking.utils2 import *
from pogs.tracking.observation import PosedObservation, Frame
import wandb
from dataclasses import dataclass
from nerfstudio.cameras.cameras import Cameras
import copy

@dataclass
class RigidGroupOptimizerConfig:
    use_depth: bool = False
    rank_loss_mult: float = 0.1
    rank_loss_erode: int = 5
    depth_loss_mult = 3.7
    depth_ignore_threshold: float = 0.26  # in meters
    use_atap: bool = False
    pose_lr: float = 0.004
    pose_lr_final: float = 0.0008
    rot_lr_scaler: float = 3.0
    mask_hands: bool = False
    do_obj_optim: bool = False
    blur_kernel_size: int = 5
    blur2_kernel_size: int = 45
    clip_grad: float = 0.8
    use_roi = True
    use_mask_loss = False
    roi_inflate_proportion: float = 0.25
    roi_inflate: float = 75
    
class RigidGroupOptimizer:
    """From: part, To: object. in current world frame. Part frame is centered at part centroid, and object frame is centered at object centroid."""

    def __init__(
        self,
        config: RigidGroupOptimizerConfig,
        pogs_model: POGSModel,
        group_masks: List[torch.Tensor],
        group_labels: torch.Tensor,
        dataset_scale: float,
        render_lock = nullcontext(),
        use_wandb = False,
    ):
        """
        This one takes in a list of gaussian ID masks to optimize local poses for
        Each rigid group can be optimized independently, with no skeletal constraints
        """
        torch.autograd.set_detect_anomaly(True)
        self.config = config
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(project="POGS", save_code=False)
        self.dataset_scale = dataset_scale
        self.tape = None
        self.is_initialized = False
        self.hand_lefts = [] #list of bools for each hand frame
        self.pogs_model = pogs_model

        self.group_labels = group_labels
        self.group_masks = group_masks
        
        # store a 7-vec of trans, rotation for each group (x,y,z,qw,qx,qy,qz)
        self.part_deltas = torch.zeros(
            len(group_masks), 7, dtype=torch.float32, device="cuda"
        )
        self.part_deltas[:, 3:] = torch.tensor(
            [1, 0, 0, 0], dtype=torch.float32, device="cuda"
        )
        self.part_deltas = torch.nn.Parameter(self.part_deltas)
        self.part_deltas.requires_grad_(True)
        k = self.config.blur_kernel_size
        s = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
        self.blur = kornia.filters.GaussianBlur2d((k, k), (s, s))
        k = self.config.blur2_kernel_size
        s = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
        self.blur2 = kornia.filters.GaussianBlur2d((k, k), (s, s))
        self.part_optimizer = torch.optim.Adam([self.part_deltas], lr=self.config.pose_lr)
        self.keyframes = []
        #hand_frames stores a list of hand vertices and faces for each keyframe, stored in the OBJECT COORDINATE FRAME
        self.hand_frames = []
        # lock to prevent blocking the render thread if provided
        self.render_lock = render_lock
        if self.config.use_atap and len(group_masks) > 1:
            self.atap = ATAPLoss(pogs_model, group_masks, group_labels, self.dataset_scale)
        else:
            self.config.use_atap = False
        self.init_means = self.pogs_model.gauss_params["means"].detach().clone()
        self.init_quats = self.pogs_model.gauss_params["quats"].detach().clone()
        self.init_opacities = self.pogs_model.gauss_params["opacities"].detach().clone()
        
        # Save the initial initial part to object transforms
        self.init_p2w = torch.empty(len(self.group_masks), 4, 4, dtype=torch.float32, device="cuda")
        self.init_p2w_7vec = torch.zeros(len(self.group_masks), 7, dtype=torch.float32, device="cuda")
        self.p2manual_tf = torch.empty(len(self.group_masks), 4, 4, dtype=torch.float32, device="cuda")
        self.p2manual_tf_SE3 = []
        self.init_p2w_7vec[:,3] = 1.0
        
        cg2w = self.pogs_model.cgtf_stack # (n,7) wxyz - xyz
        
        for i,g in enumerate(self.group_masks):
            gp_centroid = self.init_means[g].mean(dim=0)
            self.init_p2w_7vec[i,:3] = gp_centroid
            self.init_p2w[i,:,:] = torch.from_numpy(vtf.SE3.from_rotation_and_translation(
                vtf.SO3.identity(), (gp_centroid).cpu().numpy()
            ).as_matrix()).float().cuda()
            
            # find p2cg transform for ith cg2w
            se3 = vtf.SE3.from_rotation_and_translation(
                vtf.SO3(cg2w[i,:4]), cg2w[i,4:] - gp_centroid.cpu().numpy()
            )

            self.p2manual_tf_SE3.append(se3) # n times SE3 objects
            self.p2manual_tf[i,:,:] = torch.from_numpy(se3.as_matrix()).float().cuda() # (n, 4, 4)
        
    def initialize_obj_pose(self, niter=100, n_seeds=6, render=False):
        renders1 = []
        renders2 = []
        assert not self.is_initialized, "Can only initialize once"

        def try_opt(start_pose_adj, niter, use_depth, use_mask=False, rndr = False, use_roi = False):
            "tries to optimize for the initial pose, returns loss and pose + GS render if requested"
            self.reset_transforms()
            whole_pose_adj = start_pose_adj.detach().clone()
            if start_pose_adj.shape[0] != len(self.group_masks):
                whole_pose_adj.repeat(len(self.group_masks), 1)
            else:
                assert start_pose_adj.shape[0] == len(self.group_masks), start_pose_adj.shape
            whole_pose_adj = torch.nn.Parameter(whole_pose_adj)
            optimizer = torch.optim.Adam([whole_pose_adj], lr=0.005)
            for i in range(niter):
                tape = wp.Tape()
                optimizer.zero_grad()
                with tape:
                    loss, outputs = self.get_optim_loss(self.frame, whole_pose_adj, True, use_depth, False, False, False, use_mask=use_mask, use_roi=use_roi)
                loss.backward()
                tape.backward()
                optimizer.step()
                if rndr:
                    with torch.no_grad():
                        if isinstance(self.frame, PosedObservation):
                            frame = self.frame.frame
                            outputs = self.pogs_model.get_outputs(frame.camera, tracking=True)
                            renders2.append(outputs["rgb"].detach())
                        else:
                            frame = self.frame
                            outputs = self.pogs_model.get_outputs(frame.camera, tracking=True)
                            renders1.append(outputs["rgb"].detach())
            self.is_initialized = True
            return loss, whole_pose_adj.data.detach()

        best_loss = float("inf")

        whole_pose_adj = torch.zeros(len(self.group_masks), 7, dtype=torch.float32, device="cuda")
        # x y z qw qx qy qz
        z_rot = 0.0
        quat = torch.from_numpy(vtf.SO3.from_z_radians(z_rot).wxyz).cuda()
        whole_pose_adj[:, :3] = torch.zeros(3, dtype=torch.float32, device="cuda")
        whole_pose_adj[:, 3:] = quat
        loss, final_poses = try_opt(whole_pose_adj, niter, use_depth = False, use_mask = False, rndr = render)

        if loss is not None and loss < best_loss:
            best_loss = loss
            # best_outputs = outputs
            best_poses = final_poses
        self.set_observation(PosedObservation(rgb=self.frame.rgb, camera=self.frame.camera, dino_fn=self.frame._dino_fn, metric_depth_img=self.frame.depth))
        _, best_poses = try_opt(best_poses, 70, use_depth=True, rndr=render, use_mask=self.config.use_mask_loss, use_roi=True)# do a few optimization steps with depth
        with self.render_lock:
            self.apply_to_model(
                best_poses,
                self.group_labels,
            )
            
        self.part_deltas = best_poses
        self.part_deltas = torch.nn.Parameter(self.part_deltas)
        self.part_deltas.requires_grad_(True)
        self.part_optimizer = torch.optim.Adam([self.part_deltas], lr=self.config.pose_lr)

        self.prev_part_deltas = best_poses

        del loss
        del self.frame
        torch.cuda.empty_cache()
        return renders1, renders2
    
    @property
    def objreg2objinit(self):
        return torch_posevec_to_mat(self.obj_delta).squeeze()
    
    def get_poses_relative_to_camera(self, c2w: torch.Tensor, keyframe: Optional[int] = None):
        """
        Returns the current group2cam transform as defined by the specified camera pose in world coords
        c2w: 3x4 tensor of camera to world transform

        Coordinate origin of the object aligns with world axes and centered at centroid

        returns:
        Nx4x4 tensor of obj2camera transform for each of the N groups, in the same ordering as the cluster labels
        """
        with torch.no_grad():
            assert c2w.shape == (3, 4)
            c2w = torch.cat(
                [
                    c2w,
                    torch.tensor([0, 0, 0, 1], dtype=torch.float32, device="cuda").view(
                        1, 4
                    ),
                ],
                dim=0,
            )
            obj2cam_physical_batch = torch.empty(
                len(self.group_masks), 4, 4, dtype=torch.float32, device="cuda"
            )
            for i in range(len(self.group_masks)):
                obj2world_physical = self.get_part2world_transform(i)
                obj2world_physical[:3,3] /= self.dataset_scale
                obj2cam_physical_batch[i, :, :] = c2w.inverse().matmul(obj2world_physical)

        return obj2cam_physical_batch
    
    def get_part_poses(self, keyframe: Optional[int] = None):
        """
        Returns the current group2world transform 

        Coordinate origin of the object aligns with world axes and centered at centroid
        returns:
        Nx4x4 tensor of obj2world transform for each of the N groups, in the same ordering as the cluster labels
        """
        with torch.no_grad():
            obj2cam_physical_batch = torch.empty(
                len(self.group_masks), 4, 4, dtype=torch.float32, device="cuda"
            )
            for i in range(len(self.group_masks)):
                obj2world_physical = self.get_part2world_transform(i)
                obj2world_physical[:3,3] /= self.dataset_scale
                obj2cam_physical_batch[i, :, :] = obj2world_physical
        return obj2cam_physical_batch
    
    def get_partdelta_transform(self,i):
        """
        returns the transform from part_i to parti_i init at keyframe index given
        """
        return torch_posevec_to_mat(self.part_deltas[i].unsqueeze(0)).squeeze()
    

    def get_part2world_transform(self,i):
        """
        returns the transform from part_i to world
        """
        R_delta = torch.from_numpy(vtf.SO3(self.part_deltas[i, 3:].cpu().numpy()).as_matrix()).float().cuda()
        # we premultiply by rotation matrix to line up the 
        initial_part2world = self.get_initial_part2world(i)

        part2world = initial_part2world.clone()
        part2world[:3,:3] = R_delta[:3,:3].matmul(part2world[:3,:3]) # rotate around world frame
        part2world[:3,3] += self.part_deltas[i,:3] # translate in world frame
        return part2world
    
    def get_initial_part2world(self,i):
        return self.init_p2w[i]
    

    def get_optim_loss(self, frame: Frame, part_deltas, use_dino, use_depth, use_rgb, use_atap, use_hand_mask, use_mask, use_roi = False):
        """
        Returns a backpropable loss for the given frame
        """
        feats_dict = {
            "real_rgb": [],
            "real_dino": [],
            "real_depth": [],
            "real_mask": [],
            "rendered_rgb": [],
            "rendered_dino": [],
            "rendered_depth": [],
            "accumulation": [],
            "valids": [],
                }
        with self.render_lock:
            self.pogs_model.eval()
            self.apply_to_model(
                part_deltas, self.group_labels
            )
            if not use_roi:
                outputs = self.pogs_model.get_outputs(frame.camera, tracking=True, BLOCK_WIDTH=8, rgb_only=False)
                feats_dict["real_rgb"]=frame.rgb
                feats_dict["real_dino"]=frame.dino_feats.to(frame.camera.device)
                feats_dict["real_depth"]=frame.depth
                feats_dict["rendered_rgb"]=outputs['rgb']
                feats_dict["rendered_dino"]=self.blur(outputs['dino'].permute(2,0,1)[None]).squeeze().permute(1,2,0)
                feats_dict["rendered_depth"]=outputs['depth']

                if not outputs["accumulation"].any():
                    return None
            else:
                for i in reversed(range(len(self.group_masks))):
                    camera = frame.roi_frames[i].camera
                    if not use_dino: render_rgb_only = True 
                    else: render_rgb_only = False
                    outputs = self.pogs_model.get_outputs(camera, tracking=True, obj_id=i, BLOCK_WIDTH=8, rgb_only=render_rgb_only)

                    out_mask = (outputs['accumulation'] > 0.85)
                                            
                    valids = (out_mask.squeeze(-1) & (~frame.roi_frames[i].depth.isnan().squeeze(-1)))
                    
                    feats_dict["real_rgb"].append(frame.roi_frames[i].rgb)
                    if use_depth:
                        real_depth = frame.roi_frames[i].depth
                        valid_depths = kornia.morphology.erosion(valids.unsqueeze(0).unsqueeze(0).to(float),torch.ones(5,5, device=valids.device)).to(bool).squeeze(0).squeeze(0)
                        depths = real_depth[valid_depths]
                        if len(depths) > 0:
                            depths_median = torch.median(depths)
                            # reject outlier depths
                            reject = (real_depth > depths_median * 1.3).squeeze(-1)
                            valid_depths[reject] = 0
                            
                        masked_depth = real_depth * valid_depths.unsqueeze(-1)
                        mask_zeros = torch.where(masked_depth == 0, 0, 1)
                        masked_depth_rendered = outputs['depth'] * valid_depths.unsqueeze(-1) * mask_zeros
                        valids = valid_depths.unsqueeze(-1) * mask_zeros
                        valids = kornia.morphology.erosion(valids.squeeze(-1).unsqueeze(0).unsqueeze(0).to(float),torch.ones(9,9, device=valids.device)).to(bool).squeeze(0).permute(1,2,0).squeeze(-1)
                        masked_depth = masked_depth * valids.unsqueeze(-1)
                        masked_depth_rendered = masked_depth_rendered * valids.unsqueeze(-1)
                        feats_dict['valids'].append(valids.unsqueeze(-1))
                        feats_dict["real_depth"].append(masked_depth)
                        feats_dict["rendered_depth"].append(masked_depth_rendered)

                    else:
                        feats_dict['valids'].append(kornia.morphology.erosion(valids.unsqueeze(0).unsqueeze(0).to(float),torch.ones(9,9, device=valids.device)).to(bool).squeeze(0).permute(1,2,0))
                    
                    if use_mask:
                        feats_dict["real_mask"].append((frame.roi_frames[i].mask.to(torch.float32)).unsqueeze(-1))
                        
                    if use_dino:
                        dino_feats = frame.roi_frames[i].dino_feats.to(camera.device)
                        feats_dict["real_dino"].append(dino_feats)
                    feats_dict["rendered_rgb"].append(outputs['rgb'])
                    if use_dino:
                        feats_dict["rendered_dino"].append(self.blur(outputs['dino'].permute(2,0,1)[None]).squeeze().permute(1,2,0))
                        
                    accum = outputs['accumulation']
                    feats_dict["accumulation"].append(accum.to(torch.float32))                    

                for key in feats_dict.keys():
                    if len(feats_dict[key]) > 0:
                        for i in range(len(self.group_masks)):
                                feats_dict[key][i] = feats_dict[key][i].contiguous().view(-1, feats_dict[key][i].shape[-1])
                        feats_dict[key] = torch.cat(feats_dict[key])
        if use_dino:   
            loss = (feats_dict["real_dino"] - feats_dict["rendered_dino"]).norm(dim=-1).nanmean()
        if use_rgb:
            rgb_loss = (feats_dict["real_rgb"] - feats_dict["rendered_rgb"]).abs().mean()
            loss = rgb_loss
            if self.use_wandb:
                wandb.log({"rgb_loss": rgb_loss.item()})

        if self.use_wandb:
            wandb.log({"DINO mse_loss": loss.mean().item()})
        if use_depth:
            physical_depth = feats_dict["rendered_depth"] / self.dataset_scale
            valids = feats_dict['valids']

            physical_depth_clamped = torch.clamp(physical_depth, min=1e-8, max=1.0)[valids]
            real_depth_clamped = torch.clamp(feats_dict["real_depth"], min=1e-8, max=1.0)[valids]
            pix_loss = (physical_depth_clamped - real_depth_clamped) ** 2
            pix_loss = pix_loss[(pix_loss < self.config.depth_ignore_threshold**2)]
            if self.use_wandb:
                wandb.log({"depth_loss": pix_loss.mean().item()})
            if torch.isnan(pix_loss.mean()).any():
                pass
            else:
                loss = loss + self.config.depth_loss_mult * pix_loss.mean()
        if use_mask and "real_mask" in feats_dict:
            mask_bce_loss = F.binary_cross_entropy(feats_dict["accumulation"], feats_dict["real_mask"])
            if self.use_wandb:
                wandb.log({"mask_bce_loss": mask_bce_loss.mean().item()})
            loss = loss + 0.6 * mask_bce_loss
        
        if use_atap:
            weights = torch.ones(len(self.group_masks), len(self.group_masks),dtype=torch.float32,device='cuda')
            atap_loss = self.atap(weights)
            if self.use_wandb:
                wandb.log({"atap_loss": atap_loss.item()})
            loss = loss + atap_loss

        return loss, outputs
        
    def step(self, niter=1, use_depth=False, use_rgb=False):
        part_scheduler = ExponentialDecayScheduler(
            ExponentialDecaySchedulerConfig(
                lr_final=self.config.pose_lr_final, max_steps=niter
            )
        ).get_scheduler(self.part_optimizer, self.config.pose_lr)
        self.prev_part_deltas = copy.deepcopy(self.part_deltas.detach())
        for i in range(niter):
            # renormalize rotation representation
            with torch.no_grad():
                self.part_deltas[:, 3:] = self.part_deltas[:, 3:] / self.part_deltas[:, 3:].norm(dim=1, keepdim=True)
            tape = wp.Tape()
            self.part_optimizer.zero_grad()

            # Compute loss
            with tape:
                if self.config.use_roi:
                    frame = self.frame
                else:
                    frame = self.frame.frame

                use_dino = True

                loss, outputs = self.get_optim_loss(frame, self.part_deltas, use_dino,
                        use_depth, use_rgb, self.config.use_atap, self.config.mask_hands, self.config.use_mask_loss, self.config.use_roi)
            if loss is not None:
                loss.backward()
                #tape backward needs to be after loss backward since loss backward propagates gradients to the outputs of warp kernels
                tape.backward()

            self.part_optimizer.step()
            part_scheduler.step()
            
            if self.use_wandb:
                part_grad_norms = self.part_deltas.grad.norm(dim=1)
                for i in range(len(self.group_masks)):
                    wandb.log({f"part_delta{i} grad_norm": part_grad_norms[i].item()})
                wandb.log({"loss": loss.item()})
        # reset lr
        self.part_optimizer.param_groups[0]["lr"] = self.config.pose_lr
        with torch.no_grad():
            with self.render_lock:
                self.pogs_model.eval()
                self.apply_to_model(
                        self.part_deltas, self.group_labels
                    )
                if self.config.use_roi:
                    outputs = self.pogs_model.get_outputs(self.frame.frame.camera, tracking=True, rgb_only=True)
        out_dict = {k:i.detach() for k,i in outputs.items()}
        del loss
        torch.cuda.empty_cache()
        return out_dict

    def apply_to_model(self, part_deltas, group_labels):
        """
        Takes the current part_deltas and applies them to each of the group masks
        """
        self.reset_transforms()
        new_quats = torch.empty_like(
            self.pogs_model.gauss_params["quats"], requires_grad=False
        )
        new_means = torch.empty_like(
            self.pogs_model.gauss_params["means"], requires_grad=True
        )
        wp.launch(
            kernel=apply_to_model,
            dim=self.pogs_model.num_points,
            inputs = [
                wp.from_torch(self.init_p2w_7vec),
                wp.from_torch(part_deltas),
                wp.from_torch(group_labels),
                wp.from_torch(self.pogs_model.gauss_params["means"], dtype=wp.vec3),
                wp.from_torch(self.pogs_model.gauss_params["quats"]),
            ],
            outputs=[wp.from_torch(new_means, dtype=wp.vec3), wp.from_torch(new_quats)],
        )
        self.pogs_model.gauss_params["quats"] = new_quats
        self.pogs_model.gauss_params["means"] = new_means


    @torch.no_grad()
    def register_keyframe(self, lhands: List[trimesh.Trimesh], rhands: List[trimesh.Trimesh]):
        """
        Saves the current pose_deltas as a keyframe
        """
        # hand vertices are given in world coordinates
        w2o = self.get_registered_o2w().inverse().cpu().numpy()
        all_hands = lhands + rhands
        is_lefts = [True]*len(lhands) + [False]*len(rhands)
        if len(all_hands)>0:
            all_hands = [hand.apply_transform(w2o) for hand in all_hands]
            self.hand_frames.append(all_hands)
            self.hand_lefts.append(is_lefts)
        else:
            self.hand_frames.append([])
            self.hand_lefts.append([])

        partdeltas = torch.empty(len(self.group_masks), 4, 4, dtype=torch.float32, device="cuda")
        for i in range(len(self.group_masks)):
            partdeltas[i] = self.get_partdelta_transform(i)
        self.keyframes.append(partdeltas)
        
    @torch.no_grad()
    def apply_keyframe(self, i):
        """
        Applies the ith keyframe to the pose_deltas
        """
        deltas_to_apply = torch.empty(len(self.group_masks), 7, dtype=torch.float32, device="cuda")
        for j in range(len(self.group_masks)):
            delta = self.keyframes[i][j]
            deltas_to_apply[j,:3] = delta[:3,3]
            deltas_to_apply[j,3:] = torch.from_numpy(vtf.SO3.from_matrix(delta[:3,:3].cpu().numpy()).wxyz).cuda()
        self.apply_to_model(deltas_to_apply, self.group_labels)
    
    def save_trajectory(self, path: Path):
        """
        Saves the trajectory to a file
        """
        torch.save({
            "keyframes": self.keyframes,
            "hand_frames": self.hand_frames,
            "hand_lefts": self.hand_lefts
        }, path)

    def load_trajectory(self, path: Path):
        """
        Loads the trajectory from a file. Sets keyframes and hand_frames.
        """
        data = torch.load(path)
        self.keyframes = [d.cuda() for d in data["keyframes"]]
        self.hand_frames = data['hand_frames']
        self.hand_lefts = data['hand_lefts']

    def reset_transforms(self):
        with torch.no_grad():
            self.pogs_model.gauss_params["means"] = self.init_means.detach().clone()
            self.pogs_model.gauss_params["quats"] = self.init_quats.detach().clone()
    
    def render_mask(self, cam: Cameras, obj_id: int):
        """
        Render the mask of an object given a certain camera pose and object index
        """
        with torch.no_grad():
            outputs = self.pogs_model.get_outputs(cam,tracking=True, obj_id=obj_id, BLOCK_WIDTH=8, rgb_only=True)
            object_mask = outputs["accumulation"] > 0.9
            if ~object_mask.any():
                raise RuntimeError("Object left ROI")
            return object_mask
        
    def calculate_roi(self, obj_id: int, cam: Cameras = None):
        """
        Calculate the ROI for the object given a certain camera pose and object index
        """
        with torch.no_grad():
            if cam is not None:
                object_mask = self.render_mask(cam, obj_id)
            else: 
                object_mask = self.frame._obj_masks[obj_id].squeeze(0)

            valids = torch.where(object_mask)
            valid_xs = valids[1]/object_mask.shape[1]
            valid_ys = valids[0]/object_mask.shape[0] # normalize to 0-1

            if cam is not None:
                inflate_amnt = (
                    max((self.config.roi_inflate_proportion*(valid_xs.max() - valid_xs.min()).item()), (self.config.roi_inflate/cam.width.item())),
                    max((self.config.roi_inflate_proportion*(valid_ys.max() - valid_ys.min()).item()), (self.config.roi_inflate/cam.height.item()))
                ) # x, y
            else:
                inflate_amnt = ((self.config.roi_inflate_proportion*(valid_xs.max() - valid_xs.min()).item()),
                                (self.config.roi_inflate_proportion*(valid_ys.max() - valid_ys.min()).item()))
            xmin, xmax, ymin, ymax = max(0,valid_xs.min().item() - inflate_amnt[0]), min(1,valid_xs.max().item() + inflate_amnt[0]),\
                                max(0,valid_ys.min().item() - inflate_amnt[1]), min(1,valid_ys.max().item() + inflate_amnt[1])
            return xmin, xmax, ymin, ymax
        
    def set_frame(self, frame: Frame):
        """
        Sets the rgb_frame to optimize the pose for
        rgb_frame: HxWxC tensor image
        """
        self.frame = frame
        
    def set_observation(self, frame: PosedObservation):
        """
        Sets the rgb_frame to optimize the pose for
        rgb_frame: HxWxC tensor image
        """
        
        assert self.is_initialized, "Must initialize first with the first frame"
        if self.config.use_roi:
            for obj_id in range(len(self.group_masks)):
                xmin, xmax, ymin, ymax = self.calculate_roi(obj_id, cam = frame.frame.camera)
                frame.add_roi(xmin, xmax, ymin, ymax)
        self.frame = frame
        
