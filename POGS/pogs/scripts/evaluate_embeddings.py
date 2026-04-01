"""
Embedding Quality Evaluation for PointNet++ Encoder.

Evaluates the quality of PointNet++ embeddings by:
  1. Encoding a set of point clouds → 1024-dim vectors
  2. Training a lightweight decoder to reconstruct point clouds from embeddings
  3. Computing Chamfer distance between original and reconstructed PCs
  4. Visualizing original vs reconstructed side-by-side with Open3D

This provides a proxy task to measure how much geometric information is
preserved in the pretrained PointNet++ features, _without_ needing any
downstream policy training.

Usage:
    # Evaluate on PLY files from POGS exports:
    python evaluate_embeddings.py \\
        --pcd-dir   /path/to/ply/files \\
        --checkpoint ../../Pointnet_Pointnet2_pytorch/log/part_seg/pointnet2_part_seg_msg/checkpoints/best_model.pth \\
        --epochs    100 \\
        --visualize

    # Or test on a single POGS checkpoint:
    python evaluate_embeddings.py \\
        --pogs-checkpoint /path/to/step-XXXXX.ckpt \\
        --checkpoint ...
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 1. POGS root (contains 'pogs' module)
sys.path.insert(0, os.path.normpath(os.path.join(_SCRIPT_DIR, "../..")))
# 2. PointCloudMatters root (contains 'src' module)
sys.path.insert(0, os.path.normpath(os.path.join(_SCRIPT_DIR, "../../../PointCloudMatters")))
# 3. PointOps lib (required by ACT components)
sys.path.insert(0, os.path.normpath(os.path.join(_SCRIPT_DIR, "../../../PointCloudMatters/libs/pointops")))
# 4. PointNet++ models root
sys.path.insert(0, os.path.normpath(os.path.join(_SCRIPT_DIR, "../../../Pointnet_Pointnet2_pytorch/models")))

from pogs.gs_to_pointcloud import extract_pointcloud_from_gaussians, pogs_to_pcm_input  # noqa: E402
from src.models.components.pcd_encoder.pointnet2_encoder import PointNet2Encoder        # noqa: E402


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class PointCloudDecoder(nn.Module):
    """Simple MLP decoder: 1024-dim embedding → (N, 3) point cloud.

    Args:
        latent_dim: input embedding dimension (1024 for PointNet++)
        num_points: number of output points to reconstruct
    """

    def __init__(self, latent_dim: int = 1024, num_points: int = 1024):
        super().__init__()
        self.num_points = num_points
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, num_points * 3),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim) embedding
        Returns:
            points: (B, num_points, 3)
        """
        out = self.net(z)
        return out.view(z.shape[0], self.num_points, 3)


# ---------------------------------------------------------------------------
# Chamfer Distance
# ---------------------------------------------------------------------------

def chamfer_distance(pc1: torch.Tensor, pc2: torch.Tensor) -> torch.Tensor:
    """Bidirectional Chamfer Distance (mean, not sum).

    Args:
        pc1: (B, N, 3)
        pc2: (B, M, 3)
    Returns:
        scalar mean Chamfer distance over batch
    """
    # B x N x M pairwise distances
    diff = pc1.unsqueeze(2) - pc2.unsqueeze(1)    # (B, N, M, 3)
    dist = (diff ** 2).sum(-1)                     # (B, N, M)

    cd_1 = dist.min(dim=2).values.mean(dim=1)      # (B,) – each pt in p1 → nearest in p2
    cd_2 = dist.min(dim=1).values.mean(dim=1)      # (B,) – each pt in p2 → nearest in p1
    return (cd_1 + cd_2).mean()


# ---------------------------------------------------------------------------
# Dataset from PLY files / POGS checkpoints
# ---------------------------------------------------------------------------

def load_ply_as_dense_tensor(ply_path: str, n_points: int = 1024, device: str = "cpu"):
    """Load a PLY file and return (C, N) tensor."""
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    # FPS to n_points
    if pts.shape[0] > n_points:
        from pogs.gs_to_pointcloud import _fps_numpy
        idx = _fps_numpy(pts, n_points)
        pts = pts[idx]
    elif pts.shape[0] < n_points:
        idx = np.random.choice(pts.shape[0], n_points, replace=True)
        pts = pts[idx]
    # Normalise to unit sphere
    centre = pts.mean(0, keepdims=True)
    pts -= centre
    scale  = np.max(np.sqrt((pts ** 2).sum(-1)))
    pts /= (scale + 1e-8)
    return torch.from_numpy(pts).float().to(device)  # (N, 3)


def build_dataset_from_plys(ply_dir: str, n_points: int, device: str):
    ply_files = [os.path.join(ply_dir, f) for f in os.listdir(ply_dir) if f.endswith(".ply")]
    if not ply_files:
        raise RuntimeError(f"No .ply files found in {ply_dir}")
    dataset = [load_ply_as_dense_tensor(f, n_points, device) for f in ply_files]
    print(f"Loaded {len(dataset)} point clouds from {ply_dir}")
    return dataset  # list of (N, 3) tensors


def build_dataset_from_pogs(pogs_ckpt: str, n_points: int, device: str):
    """Extract a single point cloud from a POGS checkpoint."""
    ckpt = torch.load(pogs_ckpt, map_location=device)
    state = ckpt.get("pipeline", ckpt)
    PREFIX = "_model.gauss_params."
    gauss_params = {k[len(PREFIX):]: v.to(device) for k, v in state.items() if k.startswith(PREFIX)}
    coords, _ = extract_pointcloud_from_gaussians(gauss_params, max_points=n_points)
    # Normalise
    coords = coords.astype(np.float32)
    coords -= coords.mean(0, keepdims=True)
    scale = np.max(np.sqrt((coords ** 2).sum(-1)))
    coords /= (scale + 1e-8)
    pc = torch.from_numpy(coords).float().to(device)
    return [pc]   # single-element dataset; we'll use with augmentation


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def encode_pc_batch(encoder: PointNet2Encoder, pcs: torch.Tensor, device: str) -> torch.Tensor:
    """Encode a batch of dense point clouds through the PointNet2 encoder.

    Args:
        pcs: (B, N, 3)  – normalised point clouds
    Returns:
        z: (B, 1024)
    """
    B, N, _ = pcs.shape
    # Build PCM-style dict
    flat_coord = pcs.reshape(B * N, 3)
    # Use xyz as features (no colour available when loading from PLY)
    flat_feat  = flat_coord  # (B*N, 3) — coord only; pad to 6 channels
    flat_feat  = torch.cat([flat_feat, flat_feat], dim=-1)  # (B*N, 6) repeat xyz as colour
    offsets    = torch.arange(1, B + 1, device=device, dtype=torch.int32) * N

    pcd_dict = {"coord": flat_coord, "feat": flat_feat, "offset": offsets}

    # Forward through encoder
    raw = encoder(pcd_dict)       # (B*N, 1024) global feat replicated
    # Take first point of each cloud (they're all the same global feature)
    z = raw.view(B, N, 1024)[:, 0, :]  # (B, 1024)
    return z


def train_decoder(encoder, decoder, dataset, args, device):
    optimizer = optim.Adam(decoder.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    n_points  = args.num_points
    losses    = []

    for epoch in range(args.epochs):
        # Sample batch from dataset (with random sub-sampling / jitter as augmentation)
        indices = np.random.choice(len(dataset), min(args.batch_size, len(dataset)), replace=True)
        batch   = torch.stack([dataset[i] for i in indices]).to(device)  # (B, N, 3)

        # Augment: random jitter
        batch = batch + 0.002 * torch.randn_like(batch)

        # Subsample to n_points if needed
        if batch.shape[1] > n_points:
            idx   = torch.randperm(batch.shape[1])[:n_points]
            batch = batch[:, idx, :]

        with torch.no_grad():
            z = encode_pc_batch(encoder, batch, device)       # (B, 1024)

        recon = decoder(z)                                    # (B, n_points, 3)
        loss  = chamfer_distance(recon, batch)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:4d}/{args.epochs}  Chamfer loss: {loss.item():.6f}")

    return losses


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_comparison(original: np.ndarray, reconstructed: np.ndarray, title: str = ""):
    """Show original (blue) and reconstructed (red) point clouds side by side."""
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D not available — skipping visualization")
        return

    pcd_orig = o3d.geometry.PointCloud()
    pcd_orig.points = o3d.utility.Vector3dVector(original)
    pcd_orig.paint_uniform_color([0.2, 0.4, 0.9])   # blue = original

    pcd_recon = o3d.geometry.PointCloud()
    # Offset reconstructed cloud to the right for side-by-side comparison
    offset = np.array([original[:, 0].max() - reconstructed[:, 0].min() + 0.1, 0, 0])
    pcd_recon.points = o3d.utility.Vector3dVector(reconstructed + offset)
    pcd_recon.paint_uniform_color([0.9, 0.3, 0.2])  # red = reconstructed

    print(f"\n[Visualization] {title}")
    print("  Blue  = original | Red = reconstructed")
    print("  Close the window to continue …")
    o3d.visualization.draw_geometries([pcd_orig, pcd_recon], window_name=title)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PointNet++ Embedding Quality Evaluation")

    # Input sources (mutually exclusive)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pcd-dir",          help="Directory of .ply files")
    source.add_argument("--pogs-checkpoint",  help="Path to POGS .ckpt file")

    parser.add_argument(
        "--checkpoint",
        default=os.path.join(
            _SCRIPT_DIR,
            "../../Pointnet_Pointnet2_pytorch/log/part_seg/pointnet2_part_seg_msg/checkpoints/best_model.pth",
        ),
        help="Path to pretrained PointNet++ checkpoint",
    )
    parser.add_argument("--num-points",  type=int,   default=1024)
    parser.add_argument("--epochs",      type=int,   default=200)
    parser.add_argument("--batch-size",  type=int,   default=8)
    parser.add_argument("--visualize",   action="store_true", help="Show Open3D visualization")
    parser.add_argument("--save-decoder",default=None,        help="Path to save trained decoder .pth")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[EvalEmb] Device: {device}")

    # 1. Build dataset
    if args.pcd_dir:
        dataset = build_dataset_from_plys(args.pcd_dir, args.num_points, device)
    else:
        dataset = build_dataset_from_pogs(args.pogs_checkpoint, args.num_points, device)

    # 2. Load PointNet++ encoder
    print(f"[EvalEmb] Loading encoder from {args.checkpoint} …")
    encoder = PointNet2Encoder(
        in_channels=6,
        pretrained_path=args.checkpoint,
        freeze=True,       # keep encoder fixed; only train decoder
    ).to(device)
    encoder.eval()
    print(f"[EvalEmb] Encoder loaded. Output dim = {encoder.num_channels}")

    # 3. Build decoder
    decoder = PointCloudDecoder(latent_dim=encoder.num_channels, num_points=args.num_points).to(device)
    print(f"[EvalEmb] Decoder: {sum(p.numel() for p in decoder.parameters()):,} parameters")

    # 4. Train decoder
    print(f"\n[EvalEmb] Training decoder for {args.epochs} epochs …")
    losses = train_decoder(encoder, decoder, dataset, args, device)

    print(f"\n[EvalEmb] Final Chamfer loss = {losses[-1]:.6f}")
    print(f"          Best  Chamfer loss = {min(losses):.6f}")

    # 5. Optionally save decoder
    if args.save_decoder:
        torch.save(decoder.state_dict(), args.save_decoder)
        print(f"[EvalEmb] Decoder saved to {args.save_decoder}")

    # 6. Visualize
    if args.visualize:
        decoder.eval()
        for i, pc_gt in enumerate(dataset[:3]):
            pc_gt_b = pc_gt.unsqueeze(0)                     # (1, N, 3)
            with torch.no_grad():
                z     = encode_pc_batch(encoder, pc_gt_b, device)
                recon = decoder(z).squeeze(0)                # (N, 3)

            cd = chamfer_distance(recon.unsqueeze(0), pc_gt_b).item()
            print(f"\n  Cloud {i+1}  Chamfer = {cd:.6f}")

            orig_np  = pc_gt.cpu().numpy()
            recon_np = recon.cpu().numpy()
            visualize_comparison(orig_np, recon_np, title=f"Cloud {i+1}  |  Chamfer={cd:.4f}")


if __name__ == "__main__":
    main()
