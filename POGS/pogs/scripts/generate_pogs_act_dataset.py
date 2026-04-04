"""Generate timestep-wise dynamic POGS datasets from raw RLBench demos.

This script replays recorded RLBench episodes from the raw dataset split,
runs POGS tracking at every timestep, and saves a per-episode `.pkl` file with:

- full-scene tracked point clouds per timestep
- gripper state trajectory
- ACT-style action trajectory (next-state pose/open/collision)

Output format per episode:
{
    "tracked_coords": (T, N, 3),
    "tracked_colors": (T, N, 3),
    "tracked_dino": (T, N, 64),
    "tracked_detic": (T, N, 64),
    "tracked_clusters": (T, N, 1),
    "joint_positions": (T, 7),
    "gripper_open": (T, 1),
    "action": (T, 9),
    "variation_id": int,
    "task_goal": (512,),
}
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_SCRIPT_DIR, "../../")))
sys.path.insert(0, os.path.normpath(os.path.join(_SCRIPT_DIR, "../../../PointCloudMatters")))

from pogs.tracking.optim import Optimizer


def dbg(msg: str) -> None:
    """Lightweight debug logger for dataset generation."""
    print(f"[DEBUG][generate_pogs_act_dataset] {msg}", flush=True)


_DBG_ONCE_FLAGS: set[str] = set()


def dbg_once(key: str, msg: str) -> None:
    """Print a debug line only once per process for noisy diagnostics."""
    if key not in _DBG_ONCE_FLAGS:
        _DBG_ONCE_FLAGS.add(key)
        dbg(msg)


def task_file_to_task_class(task_file: str):
    import importlib

    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module(f"rlbench.tasks.{name}")
    mod = importlib.reload(mod)
    return getattr(mod, class_name)


def sample_to_fixed_size_semantics(
    coords: np.ndarray,
    colors: np.ndarray,
    dino: np.ndarray,
    detic: np.ndarray,
    clusters: np.ndarray,
    num_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample points with semantic features to a fixed size using farthest point sampling (FPS)."""
    dbg(
        "sample_to_fixed_size_semantics: "
        f"coords={coords.shape}, colors={colors.shape}, dino={dino.shape}, "
        f"detic={detic.shape}, clusters={clusters.shape}, target={num_points}"
    )
    if coords.shape[0] == 0:
        raise RuntimeError("Tracked point cloud is empty.")

    # Inherit FPS logic from Pointnet_Pointnet2_pytorch models
    import sys
    import os
    import torch
    _PN2_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "../../../Pointnet_Pointnet2_pytorch/models"))
    if _PN2_ROOT not in sys.path:
        sys.path.insert(0, _PN2_ROOT)
    from pointnet2_utils import farthest_point_sample

    if coords.shape[0] >= num_points:
        dbg(f"Applying FPS to reduce from {coords.shape[0]} to {num_points} points")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        coords_t = torch.from_numpy(coords).unsqueeze(0).to(device)  # [1, N, 3]
        with torch.no_grad():
            idx_t = farthest_point_sample(coords_t, num_points)
        idx = idx_t.squeeze(0).cpu().numpy()
    else:
        dbg(f"Point count {coords.shape[0]} < {num_points}; sampling with replacement")
        idx = np.random.choice(coords.shape[0], num_points, replace=True)

    out = (
        coords[idx].astype(np.float32),
        colors[idx].astype(np.float32),
        dino[idx].astype(np.float32),
        detic[idx].astype(np.float32),
        clusters[idx].astype(np.float32)
    )
    dbg(
        "sample_to_fixed_size_semantics out: "
        f"coords={out[0].shape}, colors={out[1].shape}, dino={out[2].shape}, detic={out[3].shape}, clusters={out[4].shape}"
    )
    return out


def extract_full_scene_pointcloud(
    optimizer: Optimizer,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract full-scene tracked Gaussians directly from memory along with semantic features."""
    with torch.no_grad():
        dbg("extract_full_scene_pointcloud: start")
        dbg(
            f"part_deltas shape={tuple(optimizer.optimizer.part_deltas.shape)} "
            f"group_labels shape={tuple(optimizer.group_labels.shape)}"
        )
        uniq_group_labels = torch.unique(optimizer.group_labels)
        dbg_once(
            "group_labels_unique",
            f"group_labels unique values={uniq_group_labels.detach().cpu().tolist()} "
            f"(single value [0] is expected when tracking exactly one crop-group object)",
        )

        # Apply latest rigid tracking deltas to the currently active tracked subset.
        optimizer.optimizer.apply_to_model(optimizer.optimizer.part_deltas, optimizer.group_labels)

        # Full static scene (all gaussians) lives in state_stack.
        prev_state = optimizer.pipeline.state_stack[-1]
        means = prev_state["means"].detach().cpu().float().clone()

        # Tracked subset positions are updated in current model gauss_params.
        tracked_means = optimizer.pipeline.model.gauss_params["means"].detach().cpu().float()

        # Insert tracked subset back into the full scene buffer.
        keep_inds = optimizer.keep_inds.cpu() if optimizer.keep_inds is not None else optimizer.pipeline.model.keep_inds.cpu()
        dbg(
            f"full_means={tuple(means.shape)}, tracked_means={tuple(tracked_means.shape)}, "
            f"keep_inds={tuple(keep_inds.shape)}"
        )
        if tracked_means.shape[0] != keep_inds.shape[0]:
            raise ValueError(
                "Tracked means size mismatch: "
                f"tracked={tracked_means.shape[0]} vs keep_inds={keep_inds.shape[0]}"
            )
        means[keep_inds] = tracked_means

        features_dc = prev_state["features_dc"].detach().cpu().float()

        opacities = prev_state["opacities"].detach().cpu().float()

        # semantics extraction from POGS gaussians
        dino_feats = prev_state.get("dino_feats")
        if dino_feats is not None:
            dino_feats = dino_feats.detach().cpu().float()

        detic_feats = prev_state.get("detic_feats")
        if detic_feats is not None:
            detic_feats = detic_feats.detach().cpu().float()

        cluster_labels = None
        if hasattr(optimizer.pipeline.model, "cluster_labels") and optimizer.pipeline.model.cluster_labels is not None:
            cluster_labels = optimizer.pipeline.model.cluster_labels.detach().cpu().float()
            dbg(f"raw cluster_labels shape={tuple(cluster_labels.shape)} dtype={cluster_labels.dtype}")
            if cluster_labels.ndim == 1:
                cluster_labels = cluster_labels.unsqueeze(-1)
            else:
                cluster_labels = cluster_labels.reshape(cluster_labels.shape[0], -1)[:, :1]

            if cluster_labels.shape[0] != means.shape[0]:
                dbg(
                    "ERROR: cluster_labels and means mismatch: "
                    f"cluster_labels={cluster_labels.shape[0]}, means={means.shape[0]}"
                )
                raise ValueError(
                    f"Dataset generation failed: cluster label count ({cluster_labels.shape[0]}) "
                    f"does not match gaussian count ({means.shape[0]}). Please re-export clusters.npy from the viewer."
                )
            unique_clusters = torch.unique(cluster_labels)
            dbg(
                f"cluster_labels normalized shape={tuple(cluster_labels.shape)} "
                f"unique_count={unique_clusters.numel()}"
            )
            dbg_once(
                "cluster_labels_preview",
                f"cluster_labels preview unique values (first up to 16): "
                f"{unique_clusters[:16].detach().cpu().tolist()}"
            )
        else:
            dbg("No cluster_labels found on model; defaulting to zeros")
            cluster_labels = torch.zeros((means.shape[0], 1), dtype=torch.float32)

        # Opacity filter mechanism
        opacity_vals = torch.sigmoid(opacities).squeeze(-1)
        keep = opacity_vals > 0.05
        dbg(f"opacity keep count={int(keep.sum().item())}/{keep.shape[0]}")
        if keep.any():
            means = means[keep]
            features_dc = features_dc[keep]
            if dino_feats is not None:
                dino_feats = dino_feats[keep]
            if detic_feats is not None:
                detic_feats = detic_feats[keep]

            cluster_labels = cluster_labels[keep]

        # Convert SH -> RGB
        C0 = 0.28209479177387814
        colors = torch.clamp(features_dc * C0 + 0.5, 0.0, 1.0)

        coords_np = means.numpy()
        colors_np = colors.numpy()
        dino_np = dino_feats.numpy() if dino_feats is not None else np.zeros((coords_np.shape[0], 64), dtype=np.float32)
        detic_np = detic_feats.numpy() if detic_feats is not None else np.zeros((coords_np.shape[0], 64), dtype=np.float32)
        cluster_np = cluster_labels.numpy()
        dbg(
            "extract arrays before sampling: "
            f"coords={coords_np.shape}, colors={colors_np.shape}, "
            f"dino={dino_np.shape}, detic={detic_np.shape}, clusters={cluster_np.shape}"
        )

    return sample_to_fixed_size_semantics(coords_np, colors_np, dino_np, detic_np, cluster_np, max_points)


def build_action_from_obs(obs) -> np.ndarray:
    """Build one ACT action vector from an RLBench observation.

    Action layout here is 9D (xyz + quat + gripper + collision), matching
    the same preprocessing path as PointCloudMatters RLBench ACT datasets.
    """
    return np.concatenate(
        [
            np.asarray(obs.gripper_pose, dtype=np.float32),
            np.asarray([obs.gripper_open], dtype=np.float32),
            np.asarray([obs.ignore_collisions], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def get_episode_ids(episodes_root: Path) -> list[int]:
    episode_ids = []
    for d in sorted(episodes_root.glob("episode*")):
        if d.is_dir():
            try:
                episode_ids.append(int(d.name.replace("episode", "")))
            except ValueError:
                continue
    return episode_ids


def obs_to_tensors(obs, camera: str, device: str, match_size: tuple[int, int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    import cv2
    rgb = getattr(obs, f"{camera}_rgb").astype(np.float32)
    depth = getattr(obs, f"{camera}_depth").astype(np.float32)
    dbg(
        f"obs_to_tensors in: camera={camera}, rgb={rgb.shape}/{rgb.dtype}, "
        f"depth={depth.shape}/{depth.dtype}, match_size={match_size}"
    )
    
    if match_size is not None and (rgb.shape[1] != match_size[0] or rgb.shape[0] != match_size[1]):
        rgb = cv2.resize(rgb, match_size, interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, match_size, interpolation=cv2.INTER_NEAREST)
        dbg(f"obs_to_tensors resized to rgb={rgb.shape}, depth={depth.shape}")
        
    return (
        torch.from_numpy(rgb).to(device),
        torch.from_numpy(depth).to(device),
    )


def save_episode(
    out_path: Path,
    tracked_coords: list[np.ndarray],
    tracked_colors: list[np.ndarray],
    tracked_dino: list[np.ndarray],
    tracked_detic: list[np.ndarray],
    tracked_clusters: list[np.ndarray],
    joint_positions: list[np.ndarray],
    gripper_open: list[np.ndarray],
    actions: list[np.ndarray],
    variation_id: int,
) -> None:
    payload = {
        "tracked_coords": np.stack(tracked_coords, axis=0).astype(np.float32),
        "tracked_colors": np.stack(tracked_colors, axis=0).astype(np.float32),
        "tracked_dino": np.stack(tracked_dino, axis=0).astype(np.float32),
        "tracked_detic": np.stack(tracked_detic, axis=0).astype(np.float32),
        "tracked_clusters": np.stack(tracked_clusters, axis=0).astype(np.float32),
        "joint_positions": np.stack(joint_positions, axis=0).astype(np.float32),
        "gripper_open": np.stack(gripper_open, axis=0).astype(np.float32),
        "action": np.stack(actions, axis=0).astype(np.float32),
        "variation_id": int(variation_id),
        # Keep this key for compatibility with existing ACT dataloaders.
        "task_goal": np.zeros(512, dtype=np.float32),
    }
    with out_path.open("wb") as f:
        pickle.dump(payload, f)


def process_episode(
    optimizer: Optimizer,
    demo,
    out_path: Path,
    camera: str,
    max_points: int,
    first_niters: int,
    niters: int,
    device: str,
    variation_id: int,
    encoder,
) -> None:
    t0 = time.time()
    dbg(
        f"process_episode start: out_path={out_path}, camera={camera}, "
        f"max_points={max_points}, first_niters={first_niters}, niters={niters}, variation_id={variation_id}"
    )
    observations = list(demo._observations)
    dbg(f"observations count={len(observations)}")
    if len(observations) < 2:
        raise RuntimeError("Episode must contain at least 2 timesteps.")

    optimizer.reset_optimizer()
    dbg("optimizer.reset_optimizer done")

    cam_w = int(optimizer.cam2world_ns_ds.width[0].item())
    cam_h = int(optimizer.cam2world_ns_ds.height[0].item())
    match_size = (cam_w, cam_h)
    dbg(f"optimizer camera ds size: width={cam_w}, height={cam_h}, match_size={match_size}")

    first_rgb, first_depth = obs_to_tensors(observations[0], camera, device, match_size)
    optimizer.set_frame(first_rgb, optimizer.cam2world_ns_ds, first_depth)
    dbg(f"set_frame complete: first_rgb={tuple(first_rgb.shape)}, first_depth={tuple(first_depth.shape)}")
    optimizer.init_obj_pose()
    dbg("optimizer.init_obj_pose complete")

    embeds = []
    joint_positions = []
    gripper_open = []

    raw_frames = []

    for t, obs in enumerate(observations):
        step_t0 = time.time()
        dbg(f"step {t + 1}/{len(observations)}: gripper_open={obs.gripper_open}")
        rgb_t, depth_t = obs_to_tensors(obs, camera, device, match_size)
        raw_frames.append((rgb_t.cpu().numpy() * 255).astype(np.uint8))

        optimizer.set_observation(rgb_t, optimizer.cam2world_ns_ds, depth_t)
        niter = first_niters if t == 0 else niters
        dbg(f"calling step_opt with niter={niter}")
        optimizer.step_opt(niter=niter)
        dbg("step_opt complete")


        coords_t, colors_t, dino_t, detic_t, cluster_t = extract_full_scene_pointcloud(optimizer, max_points)
        dbg(
            f"extracted pointcloud: coords={coords_t.shape}, colors={colors_t.shape}, "
            f"dino={dino_t.shape}, detic={detic_t.shape}, clusters={cluster_t.shape}"
        )

        # Preprocess features according to PointCloudMatters PCM
        # Normalize color to [-1, 1]
        colors_t_255 = colors_t * 255.0 if colors_t.max() <= 1.0 else colors_t
        colors_t_norm = colors_t_255 / 127.5 - 1.0

        # Stack features: [color, coords, dino, detic, cluster]
        features_t = np.concatenate([colors_t_norm, coords_t, dino_t, detic_t, cluster_t], axis=-1)
        dbg(f"features_t shape={features_t.shape}, min={features_t.min():.4f}, max={features_t.max():.4f}")

        input_dict = {
            "coord": torch.from_numpy(coords_t).to(device),
            "feat": torch.from_numpy(features_t).to(device),
            "offset": torch.tensor([coords_t.shape[0]], dtype=torch.int32, device=device)
        }

        with torch.no_grad():
            feat_out = encoder(input_dict)  # (max_points, 1024)
            # Take the global feature which is intrinsically identically repeated for all N points
            global_embed = feat_out[0].detach().cpu().numpy().astype(np.float32)
            embeds.append(global_embed)
            dbg(f"encoder output shape={tuple(feat_out.shape)}; global_embed shape={global_embed.shape}")

        joint_positions.append(np.asarray(obs.gripper_pose, dtype=np.float32))
        gripper_open.append(np.asarray([obs.gripper_open], dtype=np.float32))
        dbg(f"step {t + 1} complete in {time.time() - step_t0:.2f}s")

    actions = []
    for t in range(len(observations)):
        next_obs = observations[t + 1] if (t + 1) < len(observations) else observations[t]
        actions.append(build_action_from_obs(next_obs))
    dbg(f"built actions count={len(actions)}; action_dim={actions[0].shape[0] if actions else 'n/a'}")

    # import moviepy.editor as mpy
    # print(f"Generating raw_camera_view.mp4 with {len(raw_frames)} frames.")
    # out_clip_raw = mpy.ImageSequenceClip(raw_frames, fps=10)
    # out_clip_raw.write_videofile("raw_camera_view.mp4")


    payload = {
        "obs_embeds": np.stack(embeds, axis=0).astype(np.float32),
        "joint_positions": np.stack(joint_positions, axis=0).astype(np.float32),
        "gripper_open": np.stack(gripper_open, axis=0).astype(np.float32),
        "action": np.stack(actions, axis=0).astype(np.float32),
        "variation_id": int(variation_id),
        "task_goal": np.zeros(512, dtype=np.float32),
    }

    # Save atomically to prevent partial-file corruption if Ctrl+C is pressed
    tmp_path = out_path.with_suffix(".pkl.tmp")
    dbg(f"writing payload to temp file: {tmp_path}")
    with tmp_path.open("wb") as f:
        pickle.dump(payload, f)
    tmp_path.rename(out_path)
    dbg(
        f"episode saved: {out_path} | embeds={payload['obs_embeds'].shape}, "
        f"actions={payload['action'].shape}, file_size={out_path.stat().st_size} bytes"
    )
    dbg(f"process_episode done in {time.time() - t0:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dynamic POGS ACT episodes")
    parser.add_argument("--task", required=True, help="RLBench task, e.g. open_drawer")
    parser.add_argument(
        "--raw-root",
        required=True,
        help="Raw RLBench split root containing task folders (train/val)",
    )
    parser.add_argument("--pogs-config", required=True, help="Path to trained POGS config.yml")
    parser.add_argument("--out-dir", required=True, help="Directory to write episode*.pkl")
    parser.add_argument("--camera", default="front", help="RLBench camera stream to track")
    parser.add_argument("--max-points", type=int, default=8192)
    parser.add_argument("--first-niters", type=int, default=15)
    parser.add_argument("--niters", type=int, default=5)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=-1, help="-1 means all")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pointnet2-ckpt", type=str, required=False, default=None, help="Path to pre-trained PointNet++ checkpoint (optional)")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Auto-resume dataset generation by skipping existing .pkl files")
    args = parser.parse_args()
    dbg(f"args={vars(args)}")

    import sys
    sys.path.append("/home/pi0/POGS-ACT-implementation/POGS/PointCloudMatters")
    from src.models.components.pcd_encoder.pointnet2_encoder import PointNet2Encoder

    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig

    raw_root = Path(args.raw_root)
    episodes_root = raw_root / args.task / "all_variations" / "episodes"
    dbg(f"raw_root={raw_root}")
    dbg(f"episodes_root={episodes_root}")
    if not episodes_root.exists():
        raise FileNotFoundError(f"Episodes root not found: {episodes_root}")

    episode_ids = [eid for eid in get_episode_ids(episodes_root) if eid >= args.start_episode]
    if args.max_episodes > 0:
        episode_ids = episode_ids[: args.max_episodes]
    dbg(f"selected episode_ids={episode_ids}")
    if not episode_ids:
        raise RuntimeError("No episodes selected. Check --raw-root, --task, and episode range args.")

    obs_config = ObservationConfig()
    obs_config.set_all(False)
    getattr(obs_config, f"{args.camera}_camera").set_all(True)
    obs_config.gripper_pose = True
    obs_config.gripper_open = True

    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(collision_checking=False),
            gripper_action_mode=Discrete(),
        ),
        dataset_root=str(raw_root),
        obs_config=obs_config,
        headless=args.headless,
    )
    dbg("Launching RLBench environment")
    env.launch()
    dbg("RLBench environment launched")

    task_cls = task_file_to_task_class(args.task)
    task = env.get_task(task_cls)
    dbg(f"task loaded: {task_cls}")

    first_ep = episode_ids[0]
    variation_path = episodes_root / f"episode{first_ep}" / "variation_number.pkl"
    with variation_path.open("rb") as f:
        variation_id = pickle.load(f)
    dbg(f"first episode={first_ep}, variation_id={variation_id}")

    task.set_variation(-1)
    first_demo = task.get_demos(
        1,
        random_selection=False,
        live_demos=False,
        from_episode_number=first_ep,
    )[0]
    task.set_variation(variation_id)

    # Just read the first observation directly from the loaded demo sequence
    first_obs = first_demo._observations[0]
    dbg(f"first observation loaded for episode {first_ep}")

    K = np.asarray(first_obs.misc[f"{args.camera}_camera_intrinsics"], dtype=np.float32)
    extrinsics = np.asarray(first_obs.misc[f"{args.camera}_camera_extrinsics"], dtype=np.float32)
    print(f"Extracted camera intrinsics K (abs focal lengths):\n{K}")
    print(f"Raw RLBench extrinsics (PyRep c2w):\n{extrinsics}")
    dbg(f"K shape={K.shape}, extrinsics shape={extrinsics.shape}")

    # RLBench camera_extrinsics IS already camera-to-world (c2w) in PyRep convention.
    # The POGS Optimizer.__init__ applies R_x_pi internally (PyRep→OpenGL conversion).
    # We just pass the raw extrinsics directly.
    if extrinsics.shape == (4, 4):
        init_cam_pose = torch.from_numpy(extrinsics[:3, :]).float().unsqueeze(0)
    elif extrinsics.shape == (3, 4):
        init_cam_pose = torch.from_numpy(extrinsics).float().unsqueeze(0)
    else:
        raise RuntimeError(f"Unexpected extrinsics shape: {extrinsics.shape}")
    print(f"init_cam_pose (raw c2w, passed to Optimizer):\n{init_cam_pose}")
    dbg(f"init_cam_pose shape={tuple(init_cam_pose.shape)}")

    height, width = getattr(first_obs, f"{args.camera}_rgb").shape[:2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dbg(f"first frame size: width={width}, height={height}, device={device}")

    # dataparser_transform is identity for this model (trained with auto_scale_poses=false)
    # so we skip that multiplication.


    # Instantiate encoder (135 channels = colors(3)+coords(3)+dino(128)+cluster(1))
    encoder = PointNet2Encoder(
        in_channels=135,
        pretrained_path=args.pointnet2_ckpt,
        freeze=False,
    ).to(device)
    encoder.eval()
    dbg("PointNet2Encoder initialized and set to eval mode")

    optimizer = Optimizer(
        Path(args.pogs_config),
        K,
        width,
        height,
        init_cam_pose,
    )
    dbg(
        f"Optimizer initialized: keep_inds_len={optimizer.keep_inds.shape[0] if optimizer.keep_inds is not None else 'None'}, "
        f"group_labels={tuple(optimizer.group_labels.shape)}, num_groups={optimizer.num_groups}"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dbg(f"out_dir={out_dir}")

    print(f"Generating {len(episode_ids)} episodes for task={args.task}")
    try:
        for episode_id in tqdm(episode_ids):
            episode_t0 = time.time()
            out_path = out_dir / f"episode{episode_id}.pkl"
            dbg(f"Starting episode {episode_id} -> {out_path}")

            if args.resume and out_path.exists():
                print(f"[Resume] Skipping episode {episode_id} as {out_path.name} already exists.")
                continue

            variation_path = episodes_root / f"episode{episode_id}" / "variation_number.pkl"
            with variation_path.open("rb") as f:
                variation_id = pickle.load(f)
            dbg(f"episode {episode_id} variation_id={variation_id}")

            task.set_variation(-1)
            demo = task.get_demos(
                1,
                random_selection=False,
                live_demos=False,
                from_episode_number=episode_id,
            )[0]
            task.set_variation(variation_id)
            dbg(f"demo loaded for episode {episode_id}; observation_count={len(demo._observations)}")

            # Removed try-except to preserve full traceback during debugging.
            process_episode(
                optimizer=optimizer,
                demo=demo,
                out_path=out_path,
                camera=args.camera,
                max_points=args.max_points,
                first_niters=args.first_niters,
                niters=args.niters,
                device=device,
                variation_id=int(variation_id),
                encoder=encoder,
            )
            dbg(f"episode {episode_id} finished in {time.time() - episode_t0:.2f}s")
    finally:
        dbg("Shutting down RLBench environment")
        env.shutdown()

    print(f"Done. Saved episodes to: {out_dir}")


if __name__ == "__main__":
    main()
