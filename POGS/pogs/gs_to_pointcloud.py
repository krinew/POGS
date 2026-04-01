"""
Gaussian → Point Cloud conversion utilities for POGS.

Extracts point clouds from trained POGS 3D Gaussian Splatting models and
formats them for consumption by the PointCloudMatters (PCM) pipeline.

Usage (standalone):
    from pogs.gs_to_pointcloud import extract_pointcloud_from_gaussians, format_for_pcm

    coords, colors = extract_pointcloud_from_gaussians(pipeline.model.gauss_params)
    pcd_dict = format_for_pcm(coords, colors)

The approach mirrors `_export_visible_gaussians` in pogs_pipeline.py but
returns tensors ready for the PCM data pipeline rather than writing to disk.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Optional, Tuple, Dict


# ---------------------------------------------------------------------------
# Spherical Harmonics helpers (mirrors pogs_pipeline.SH2RGB)
# ---------------------------------------------------------------------------

def _sh2rgb(sh: torch.Tensor) -> torch.Tensor:
    """Convert 0th-order SH coefficient to RGB [0, 1]."""
    C0 = 0.28209479177387814
    return torch.clamp(sh * C0 + 0.5, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_pointcloud_from_gaussians(
    gauss_params: Dict,
    opacity_threshold: float = 0.0,
    max_points: Optional[int] = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a coloured point cloud from POGS Gaussian parameters.

    Args:
        gauss_params: dict-like containing at least:
            - ``means``       : (N, 3)  float  – Gaussian centers
            - ``features_dc`` : (N, 3)  float  – DC SH coefficients (→ RGB)
            - ``opacities``   : (N, 1)  float  – pre-activation opacities
        opacity_threshold: Minimum *sigmoid* opacity to include a Gaussian.
            Use 0.0 to include all Gaussians (recommended if you've already
            clustered/cropped the scene).
        max_points: Downsample to this many points using Farthest Point
            Sampling.  None = no downsampling.

    Returns:
        coords: (M, 3)  float64  – XYZ positions
        colors: (M, 3)  float64  – RGB values in [0, 1]
    """
    with torch.no_grad():
        means = _to_tensor(gauss_params["means"])                # (N, 3)
        features_dc = _to_tensor(gauss_params["features_dc"])   # (N, 3)
        opacities = _to_tensor(gauss_params["opacities"])        # (N, 1)

        # Opacity filter
        opacity_vals = torch.sigmoid(opacities).squeeze(-1)      # (N,)
        if opacity_threshold > 0.0:
            keep = opacity_vals > opacity_threshold
            means = means[keep]
            features_dc = features_dc[keep]

        # Convert SH → RGB
        colors = _sh2rgb(features_dc)                            # (N, 3)

        # FPS downsampling
        if max_points is not None and means.shape[0] > max_points:
            idx = _fps_numpy(means.cpu().numpy(), max_points)
            means = means[idx]
            colors = colors[idx]

    coords = means.cpu().numpy().astype(np.float64)
    colors = colors.cpu().numpy().astype(np.float64)
    return coords, colors


def _to_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float()
    return torch.tensor(np.array(x), dtype=torch.float32)


def _fps_numpy(points: np.ndarray, n_sample: int) -> np.ndarray:
    """Farthest Point Sampling (numpy implementation).

    Args:
        points: (N, 3) float
        n_sample: number of points to sample

    Returns:
        indices: (n_sample,) int
    """
    N = points.shape[0]
    n_sample = min(n_sample, N)

    selected = np.zeros(n_sample, dtype=np.int64)
    distances = np.full(N, np.inf)

    # Start from random point
    idx = np.random.randint(0, N)
    for i in range(n_sample):
        selected[i] = idx
        centroid = points[idx]
        d = np.sum((points - centroid) ** 2, axis=1)
        distances = np.minimum(distances, d)
        idx = np.argmax(distances)

    return selected


# ---------------------------------------------------------------------------
# PCM format adapter
# ---------------------------------------------------------------------------

def format_for_pcm(
    coords: np.ndarray,
    colors: np.ndarray,
    device: str = "cuda",
    grid_size: float = 0.005,
) -> Dict[str, torch.Tensor]:
    """Convert raw point cloud arrays to a PCM-compatible batch dict.

    Applies PCM's standard preprocessing:
        1. Grid (voxel) downsampling
        2. Color normalisation (divide by 255 if > 1 — coordinates are in [0,1])
        3. Tensor conversion

    Args:
        coords: (N, 3)  XYZ positions (any scale)
        colors: (N, 3)  RGB values in [0, 1]
        device: target device string
        grid_size: voxel grid size for downsampling (meters)

    Returns:
        dict with keys:
            ``coord``      : (M, 3) float32
            ``feat``       : (M, 6) float32  – [R, G, B, X, Y, Z] (PCM feat format)
            ``offset``     : (1,)   int32    – [M]   (single cloud batch)
    """
    import open3d as o3d

    # Grid downsampling via Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
    pcd_down = pcd.voxel_down_sample(voxel_size=grid_size)

    pts = np.asarray(pcd_down.points, dtype=np.float32)
    rgb = np.asarray(pcd_down.colors, dtype=np.float32)

    # Feature = [color_norm, xyz_norm]  (mirrors PCM's CollectPCD)
    # Normalise color to [-1, 1] (PCM NormalizeColorPCD divides by 127.5 - 1)
    rgb_norm = (rgb - 0.5) * 2.0   # [0,1] → [-1, 1]
    feat = np.concatenate([rgb_norm, pts], axis=-1)  # (M, 6)

    coord_t  = torch.from_numpy(pts).to(device)
    feat_t   = torch.from_numpy(feat).to(device)
    offset_t = torch.tensor([pts.shape[0]], dtype=torch.int32, device=device)

    return {"coord": coord_t, "feat": feat_t, "offset": offset_t}


# ---------------------------------------------------------------------------
# High-level convenience function
# ---------------------------------------------------------------------------

def pogs_to_pcm_input(
    gauss_params: Dict,
    max_points: int = 8192,
    opacity_threshold: float = 0.0,
    grid_size: float = 0.005,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """End-to-end: POGS gauss_params → PCM batch dict.

    Args:
        gauss_params: POGS model gauss_params dict
        max_points: FPS downsample target (applied before grid sampling)
        opacity_threshold: opacity cutoff for Gaussians
        grid_size: voxel grid size (m)
        device: target device

    Returns:
        PCM-compatible dict {coord, feat, offset}
    """
    coords, colors = extract_pointcloud_from_gaussians(
        gauss_params,
        opacity_threshold=opacity_threshold,
        max_points=max_points,
    )
    return format_for_pcm(coords, colors, device=device, grid_size=grid_size)
