"""
PointNet++ encoder for point cloud processing.

Uses PCM's pointops library for FPS, ball_query, grouping, and interpolation.
Matches the same interface as PointNet (input_dict → per-point features).

Architecture follows the standard PointNet++ SSG design:
  - 3 Set Abstraction (SA) layers for hierarchical downsampling + local feature learning
  - 3 Feature Propagation (FP) layers for upsampling back to original resolution
  - Outputs per-point features (N, num_channels) matching what ACTPCD expects

Reference:
  - Qi et al., "PointNet++: Deep Hierarchical Feature Learning on Point Sets" (NeurIPS 2017)
  - yanx27/Pointnet_Pointnet2_pytorch
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import pointops
from pointops.functions.utils import offset2batch


class SharedMLP(nn.Module):
    """Shared MLP applied to grouped point features."""

    def __init__(self, in_channels, mlp_channels, bn=True):
        super().__init__()
        layers = []
        for out_channels in mlp_channels:
            layers.append(nn.Linear(in_channels, out_channels, bias=not bn))
            if bn:
                layers.append(nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.mlp = nn.Sequential(*layers)
        self.out_channels = mlp_channels[-1]

    def forward(self, x):
        """
        Args:
            x: (M * nsample, C) or (M, C) features
        Returns:
            (M * nsample, out_channels) or (M, out_channels) features
        """
        return self.mlp(x)


class SetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction layer using pointops.

    Performs: FPS → Ball Query → Grouping → SharedMLP → MaxPool
    """

    def __init__(self, npoint, radius, nsample, in_channels, mlp_channels, group_all=False):
        """
        Args:
            npoint: number of points to sample (None if group_all)
            radius: ball query radius
            nsample: max samples in ball query
            in_channels: input feature channels (excluding xyz)
            mlp_channels: list of output channels for shared MLP
            group_all: if True, group all points (global feature)
        """
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        # MLP processes (relative xyz + features) → grouped features
        self.mlp = SharedMLP(3 + in_channels, mlp_channels)

    def forward(self, xyz, features, offset):
        """
        Args:
            xyz: (N, 3) point coordinates
            features: (N, C) point features, or None
            offset: (B,) cumulative point counts per batch

        Returns:
            new_xyz: (M, 3) sampled point coordinates
            new_features: (M, D) aggregated features
            new_offset: (B,) new cumulative point counts
        """
        B = offset.shape[0]

        if self.group_all:
            # Group ALL points into a single group per batch
            new_xyz, new_features, new_offset = self._group_all(xyz, features, offset)
        else:
            # FPS to select npoint centroids
            new_offset = self._compute_new_offset(offset, B)
            fps_idx = pointops.farthest_point_sampling(xyz, offset, new_offset)
            new_xyz = xyz[fps_idx.long()]  # (M, 3)

            # Ball query to find neighbors
            idx, _ = pointops.ball_query(
                self.nsample, self.radius, 0.0,
                xyz, offset, new_xyz, new_offset
            )  # (M, nsample)

            M = new_xyz.shape[0]

            # Group features: gather neighbor coords + features
            grouped_xyz = xyz[idx.view(-1).long()].view(M, self.nsample, 3)
            grouped_xyz = grouped_xyz - new_xyz.unsqueeze(1)  # relative coords (M, nsample, 3)

            if features is not None:
                grouped_features = features[idx.view(-1).long()].view(M, self.nsample, -1)
                grouped_features = torch.cat([grouped_xyz, grouped_features], dim=-1)  # (M, nsample, 3+C)
            else:
                grouped_features = grouped_xyz  # (M, nsample, 3)

            # Apply MLP to each neighbor
            MN = M * self.nsample
            grouped_features = grouped_features.reshape(MN, -1)  # (M*nsample, 3+C)
            grouped_features = self.mlp(grouped_features)  # (M*nsample, D)
            grouped_features = grouped_features.view(M, self.nsample, -1)  # (M, nsample, D)

            # Max pool over neighbors
            new_features = grouped_features.max(dim=1)[0]  # (M, D)

        return new_xyz, new_features, new_offset

    def _compute_new_offset(self, offset, B):
        """Compute new offset after FPS sampling."""
        new_offset_list = []
        count = 0
        prev = 0
        for i in range(B):
            n_i = (offset[i] - prev).item()
            m_i = min(self.npoint, n_i)
            count += m_i
            new_offset_list.append(count)
            prev = offset[i].item()
        return torch.tensor(new_offset_list, dtype=torch.int32, device=offset.device)

    def _group_all(self, xyz, features, offset):
        """Group all points per batch into a single group."""
        B = offset.shape[0]
        batch = offset2batch(offset)

        new_xyz_list = []
        new_features_list = []

        for b in range(B):
            mask = (batch == b)
            xyz_b = xyz[mask]  # (N_b, 3)
            centroid = xyz_b.mean(dim=0, keepdim=True)  # (1, 3)
            new_xyz_list.append(centroid)

            relative_xyz = xyz_b - centroid  # (N_b, 3)

            if features is not None:
                feat_b = features[mask]  # (N_b, C)
                grouped = torch.cat([relative_xyz, feat_b], dim=-1)  # (N_b, 3+C)
            else:
                grouped = relative_xyz  # (N_b, 3)

            # Apply MLP
            grouped = self.mlp(grouped)  # (N_b, D)

            # Max pool
            pooled = grouped.max(dim=0, keepdim=True)[0]  # (1, D)
            new_features_list.append(pooled)

        new_xyz = torch.cat(new_xyz_list, dim=0)  # (B, 3)
        new_features = torch.cat(new_features_list, dim=0)  # (B, D)
        new_offset = torch.arange(1, B + 1, dtype=torch.int32, device=offset.device)

        return new_xyz, new_features, new_offset


class FeaturePropagation(nn.Module):
    """
    PointNet++ Feature Propagation layer.

    Upsamples features from coarser to finer level using distance-weighted
    interpolation + skip connection + MLP.
    """

    def __init__(self, in_channels, mlp_channels):
        """
        Args:
            in_channels: channels from interpolated features + skip features
            mlp_channels: list of output channels for MLP
        """
        super().__init__()
        self.mlp = SharedMLP(in_channels, mlp_channels)

    def forward(self, xyz_coarse, xyz_fine, features_coarse, features_fine, offset_coarse, offset_fine):
        """
        Args:
            xyz_coarse: (M, 3) coarser level coordinates
            xyz_fine: (N, 3) finer level coordinates
            features_coarse: (M, C1) coarser level features
            features_fine: (N, C2) finer level features (skip connection), or None
            offset_coarse: (B,) coarser offset
            offset_fine: (B,) finer offset

        Returns:
            new_features: (N, D) upsampled features
        """
        # Interpolate coarse features to fine level using 3-NN distance-weighted interpolation
        interpolated = pointops.interpolation(
            xyz_coarse, xyz_fine, features_coarse,
            offset_coarse, offset_fine, k=3
        )  # (N, C1)

        # Skip connection
        if features_fine is not None:
            interpolated = torch.cat([interpolated, features_fine], dim=-1)  # (N, C1+C2)

        # MLP
        new_features = self.mlp(interpolated)  # (N, D)

        return new_features


class PointNet2(nn.Module):
    """
    PointNet++ SSG encoder compatible with PCM's ACTPCD backbone interface.

    Architecture:
        SA1: npoint=512, radius=0.2, nsample=32, MLP=[64, 64, 128]
        SA2: npoint=128, radius=0.4, nsample=64, MLP=[128, 128, 256]
        SA3: global aggregation, MLP=[256, 512, 1024]
        FP3: MLP=[256, 256]
        FP2: MLP=[256, 128]
        FP1: MLP=[128, 128, 512]

    Input: input_dict with keys 'coord', 'feat', 'offset', 'grid_coord'
    Output: per-point features (N, 512)
    """

    def __init__(
        self,
        in_channels=6,
        num_classes=0,
        pretrained_path=None,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.embedding_table = None

        # --- Set Abstraction layers ---
        # in_channels is the feature dim (e.g. 6 for color+coord, 70 for color+coord+dino)
        # The 'feat' from CollectPCD is concat of [color, coord], so feat_channels = in_channels
        # SA layers receive features WITHOUT xyz (xyz is handled separately in grouping)
        self.sa1 = SetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channels=in_channels,
            mlp_channels=[64, 64, 128]
        )
        self.sa2 = SetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channels=128,
            mlp_channels=[128, 128, 256]
        )
        self.sa3 = SetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channels=256,
            mlp_channels=[256, 512, 1024],
            group_all=True
        )

        # --- Feature Propagation layers ---
        # FP3: interpolate from SA3 (1024-dim) to SA2 level, skip with SA2 features (256-dim)
        self.fp3 = FeaturePropagation(
            in_channels=1024 + 256,
            mlp_channels=[256, 256]
        )
        # FP2: interpolate from FP3 (256-dim) to SA1 level, skip with SA1 features (128-dim)
        self.fp2 = FeaturePropagation(
            in_channels=256 + 128,
            mlp_channels=[256, 128]
        )
        # FP1: interpolate from FP2 (128-dim) to original level, skip with input features
        self.fp1 = FeaturePropagation(
            in_channels=128 + in_channels,
            mlp_channels=[128, 128, 512]
        )

        # Optional classification head
        self.final = (
            nn.Linear(512, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.num_channels = num_classes if num_classes > 0 else 512

        # Load pretrained weights if specified
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, path):
        """Load pretrained weights, skipping classification head if shapes differ."""
        import os
        if not os.path.exists(path):
            print(f"[PointNet2] Pretrained path {path} not found, skipping weight loading.")
            return

        state_dict = torch.load(path, map_location="cpu")
        # Handle nested state dicts (e.g., {"model_state_dict": {...}})
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Filter out incompatible keys
        model_dict = self.state_dict()
        compatible = {}
        for k, v in state_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                compatible[k] = v
            else:
                print(f"[PointNet2] Skipping incompatible key: {k}")

        model_dict.update(compatible)
        self.load_state_dict(model_dict, strict=False)
        print(f"[PointNet2] Loaded {len(compatible)}/{len(state_dict)} pretrained parameters from {path}")

    def forward(self, input_dict):
        """
        Args:
            input_dict: dict with keys:
                - 'coord': (N, 3) point coordinates
                - 'feat': (N, C) point features (C = in_channels)
                - 'offset': (B,) cumulative point counts
                - 'grid_coord': (N, 3) quantized grid coordinates (unused by PointNet++)

        Returns:
            features: (N, num_channels) per-point features
        """
        xyz = input_dict["coord"]       # (N, 3)
        features = input_dict["feat"]   # (N, C)
        offset = input_dict["offset"]   # (B,)

        # Save for skip connections
        l0_xyz = xyz
        l0_features = features
        l0_offset = offset

        # --- Encoder (Set Abstraction) ---
        l1_xyz, l1_features, l1_offset = self.sa1(l0_xyz, l0_features, l0_offset)
        l2_xyz, l2_features, l2_offset = self.sa2(l1_xyz, l1_features, l1_offset)
        l3_xyz, l3_features, l3_offset = self.sa3(l2_xyz, l2_features, l2_offset)

        # --- Decoder (Feature Propagation) ---
        l2_features = self.fp3(l3_xyz, l2_xyz, l3_features, l2_features, l3_offset, l2_offset)
        l1_features = self.fp2(l2_xyz, l1_xyz, l2_features, l1_features, l2_offset, l1_offset)
        l0_features = self.fp1(l1_xyz, l0_xyz, l1_features, l0_features, l1_offset, l0_offset)

        # Optional classification head
        out = self.final(l0_features)

        return out
