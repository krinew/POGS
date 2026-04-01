"""Precompute PointNet++ embeddings for dynamic POGS ACT episodes.

Reads episode `.pkl` files that contain timestep-wise tracked point clouds,
encodes each timestep with PointNet++, and saves compact embedding episodes
for ACT training.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.transformpcd import (  # noqa: E402
    CollectPCD,
    ComposePCD,
    GridSamplePCD,
    NormalizeColorPCD,
    ToTensorPCD,
)
from src.models.components.pcd_encoder.pointnet2_encoder import PointNet2Encoder  # noqa: E402


def build_transform(grid_size: float) -> ComposePCD:
    return ComposePCD(
        [
            GridSamplePCD(
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=False,
                return_displacement=False,
                keys=("coord", "color"),
            ),
            NormalizeColorPCD(),
            ToTensorPCD(),
            CollectPCD(keys=["coord"], feat_keys=["color", "coord"]),
        ]
    )


def encode_timestep(
    encoder: PointNet2Encoder,
    transform: ComposePCD,
    coords: np.ndarray,
    colors: np.ndarray,
    device: str,
) -> np.ndarray:
    colors_255 = colors * 255.0 if colors.max() <= 1.0 else colors
    pcd = transform({"coord": coords, "color": colors_255})

    pcd = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in pcd.items()}
    with torch.no_grad():
        feats = encoder(pcd)  # (N, 1024), global feature repeated per point
    return feats[0].detach().cpu().numpy().astype(np.float32)


def process_episode(
    in_path: Path,
    out_path: Path,
    encoder: PointNet2Encoder,
    transform: ComposePCD,
    device: str,
    keep_pointcloud: bool,
) -> None:
    with in_path.open("rb") as f:
        episode = pickle.load(f)

    coords_seq = episode["tracked_coords"]
    colors_seq = episode["tracked_colors"]

    embeds = []
    for t in range(coords_seq.shape[0]):
        embeds.append(
            encode_timestep(
                encoder=encoder,
                transform=transform,
                coords=coords_seq[t],
                colors=colors_seq[t],
                device=device,
            )
        )

    out = {
        "obs_embeds": np.stack(embeds, axis=0).astype(np.float32),
        "joint_positions": np.asarray(episode["joint_positions"], dtype=np.float32),
        "gripper_open": np.asarray(episode["gripper_open"], dtype=np.float32),
        "action": np.asarray(episode["action"], dtype=np.float32),
        "variation_id": int(episode.get("variation_id", 0)),
        "task_goal": np.asarray(episode.get("task_goal", np.zeros(512)), dtype=np.float32),
    }

    if keep_pointcloud:
        out["tracked_coords"] = np.asarray(coords_seq, dtype=np.float32)
        out["tracked_colors"] = np.asarray(colors_seq, dtype=np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(out, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute PointNet++ embeddings for POGS ACT")
    parser.add_argument("--input-dir", required=True, help="Directory with episode*.pkl dynamic POGS episodes")
    parser.add_argument("--out-dir", required=True, help="Output directory for embedding episodes")
    parser.add_argument("--pointnet2-ckpt", required=True, help="Path to pretrained PointNet++ checkpoint")
    parser.add_argument("--grid-size", type=float, default=0.005)
    parser.add_argument("--max-episodes", type=int, default=-1, help="-1 means all episodes")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--keep-pointcloud", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    episode_files = sorted(input_dir.glob("episode*.pkl"))
    if args.max_episodes > 0:
        episode_files = episode_files[: args.max_episodes]

    if not episode_files:
        raise RuntimeError(f"No episode files found in {input_dir}")

    device = args.device
    transform = build_transform(grid_size=args.grid_size)

    encoder = PointNet2Encoder(
        in_channels=6,
        pretrained_path=args.pointnet2_ckpt,
        freeze=True,
    ).to(device)
    encoder.eval()

    for ep_path in tqdm(episode_files, desc="Encoding episodes"):
        out_path = out_dir / ep_path.name
        process_episode(
            in_path=ep_path,
            out_path=out_path,
            encoder=encoder,
            transform=transform,
            device=device,
            keep_pointcloud=args.keep_pointcloud,
        )

    print(f"Done. Saved embedding episodes to: {out_dir}")


if __name__ == "__main__":
    main()
