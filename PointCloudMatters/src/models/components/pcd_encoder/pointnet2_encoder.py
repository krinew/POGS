"""
PointNet++ MSG Encoder wrapper for PointCloudMatters (PCM).

Uses the encoder backbone (SA layers only, no task-specific heads) from
PointNet++ part-segmentation MSG architecture.  Output is a single
1024-dim global feature vector per cloud, compatible with ACTPCD pipeline.

Usage:
    encoder = PointNet2Encoder(in_channels=6,
                               pretrained_path="path/to/best_model.pth")

The PointNet++ code is imported from Pointnet_Pointnet2_pytorch which lives
alongside PointCloudMatters in the workspace:
    /home/pi0/POGS-ACT-implementation/POGS/Pointnet_Pointnet2_pytorch/
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Locate and register the PointNet++ source tree on sys.path
# ---------------------------------------------------------------------------
_POINTNET2_ROOT = os.path.join(
    os.path.dirname(__file__),
    # PointCloudMatters/src/models/components/pcd_encoder  ->  repo root ->
    # Pointnet_Pointnet2_pytorch
    "../../../../../../Pointnet_Pointnet2_pytorch/models",
)
_POINTNET2_ROOT = os.path.normpath(_POINTNET2_ROOT)
if _POINTNET2_ROOT not in sys.path:
    sys.path.insert(0, _POINTNET2_ROOT)

from pointnet2_utils import PointNetSetAbstraction, PointNetSetAbstractionMsg  # noqa: E402


# ---------------------------------------------------------------------------
# Pure encoder backbone (SA layers only, no propagation / classification head)
# ---------------------------------------------------------------------------

class PointNet2EncoderBackbone(nn.Module):
    """PointNet++ MSG encoder backbone.

    Produces:
      - l3_xyz  : (B, 3, 1)  – single global centroid
      - l3_points: (B, 1024, 1) – 1024-dim feature for the whole cloud

    Architecture mirrors the part-seg MSG encoder before the feature-
    propagation layers.
    """

    def __init__(self, in_channels: int = 3):
        """
        Args:
            in_channels: Additional feature channels beyond XYZ.
                         3 = XYZ only, 6 = XYZ + RGB, etc.
        """
        super().__init__()
        # Number of extra channels (features beyond xyz)
        additional_channel = in_channels - 3  # e.g. 3 for RGB

        # SA1: 512 centroids, multi-scale radii
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.1, 0.2, 0.4],
            nsample_list=[32, 64, 128],
            in_channel=3 + additional_channel,
            mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
        )
        # SA2: 128 centroids
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=128,
            radius_list=[0.4, 0.8],
            nsample_list=[64, 128],
            in_channel=128 + 128 + 64,
            mlp_list=[[128, 128, 256], [128, 196, 256]],
        )
        # SA3: global pooling → 1024-dim bottleneck
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=512 + 3,
            mlp=[256, 512, 1024],
            group_all=True,
        )

    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor] = None):
        """
        Args:
            xyz   : (B, 3, N)  – point coordinates
            points: (B, C, N)  – per-point features (None → use xyz)
        Returns:
            l3_xyz   : (B, 3, 1)
            l3_points: (B, 1024, 1)
        """
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        return l3_xyz, l3_points


# ---------------------------------------------------------------------------
# PCM-compatible wrapper
# ---------------------------------------------------------------------------

class PointNet2Encoder(nn.Module):
    """PCM-compatible PointNet++ MSG encoder.

    Interface matches PCM's existing encoders (PointNet / SpUNet):
      - Constructor: ``in_channels``, optional ``pretrained_path``
      - Forward: ``input_dict`` → flat feature tensor ``(total_points, C)``
                 where C = num_channels = 1024
      - Attribute: ``num_channels``

    The forward method accepts PCM's sparse-tensor dict format:
        {
          "coord"  : (total_pts, 3)   float32  – world-space XYZ
          "feat"   : (total_pts, C)   float32  – per-point features
          "offset" : (B,)             int32    – cumulative point counts
        }
    and internally reformats to the dense (B, C, N) tensors expected by
    PointNet++.  Variable-length batches are zero-padded to max length.
    """

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 0,  # kept for API compat; unused
        pretrained_path: Optional[str] = None,
        freeze: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_channels = 1024  # global bottleneck dim

        self.backbone = PointNet2EncoderBackbone(in_channels=in_channels)

        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _load_pretrained(self, path: str):
        """Load pretrained PointNet++ weights.

        Supports checkpoints from:
          - part_seg MSG  (pointnet2_part_seg_msg best_model.pth)
          - cls MSG       (pointnet2_cls_msg       best_model.pth)
        Only the SA-layer weights are loaded; task-specific heads / FP layers
        are silently skipped.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"PointNet++ pretrained checkpoint not found: {path}"
            )
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)


        # State dicts may be stored directly or nested under 'model_state_dict'
        if isinstance(checkpoint, dict):
            state = checkpoint.get(
                "model_state_dict", checkpoint.get("state_dict", checkpoint)
            )
        else:
            state = checkpoint

        # Strip module. prefix from DDP checkpoints
        state = {k.replace("module.", ""): v for k, v in state.items()}

        # Only keep keys that belong to sa1/sa2/sa3  (skip fp, conv, cls …)
        backbone_state = {
            k: v
            for k, v in state.items()
            if k.startswith("sa1.") or k.startswith("sa2.") or k.startswith("sa3.")
        }

        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
        loaded = len(backbone_state) - len(missing)
        print(
            f"[PointNet2Encoder] Loaded {loaded}/{len(backbone_state)} weights "
            f"from {os.path.basename(path)}.  "
            f"Missing={len(missing)}, Unexpected={len(unexpected)}"
        )

    # ------------------------------------------------------------------
    # Sparse-tensor dict → dense tensors helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_sparse_dict(input_dict: dict):
        coord = input_dict["coord"]    # either (total, 3) or (B, N, 3)
        feat  = input_dict.get("feat", None)
        offset = input_dict["offset"]  # (B,) cumulative

        # If DataLoader stacked into dense batch tensor (B, N, 3)
        if coord.dim() == 3:
            B, N, _ = coord.shape
            clouds = []
            for b in range(B):
                xyz_i = coord[b].permute(1, 0)  # (3, N)
                if feat is not None:
                    f_i = feat[b].permute(1, 0) if feat.dim() == 3 else feat[b*N:(b+1)*N].permute(1, 0)
                    clouds.append(torch.cat([xyz_i, f_i], dim=0))
                else:
                    clouds.append(xyz_i)
            return clouds

        # Sparse flat format (total_pts, 3)
        prev = 0
        clouds = []
        for end in offset.tolist():
            if isinstance(end, list):
                end = end[0]
            end = int(end)
            xyz_i = coord[prev:end].permute(1, 0)  # (3, Ni)
            if feat is not None:
                f_i = feat[prev:end].permute(1, 0)
                clouds.append(torch.cat([xyz_i, f_i], dim=0))
            else:
                clouds.append(xyz_i)
            prev = end
        return clouds
    @staticmethod
    def _pad_to_dense(clouds: list, device):
        """Zero-pad variable-length clouds to a single dense batch.

        Returns:
            xyz   : (B, 3, N_max)
            feat  : (B, C, N_max) or None
        """
        C_total = clouds[0].shape[0]  # 3 + feature_dim
        N_max = max(c.shape[1] for c in clouds)
        B = len(clouds)

        dense = torch.zeros(B, C_total, N_max, device=device, dtype=torch.float32)
        for i, c in enumerate(clouds):
            dense[i, :, : c.shape[1]] = c

        xyz  = dense[:, :3, :]      # (B, 3, N_max)
        feat = dense[:, 3:, :] if C_total > 3 else None
        return xyz, feat

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_dict: dict) -> torch.Tensor:
        device = input_dict["coord"].device
        clouds = self._unpack_sparse_dict(input_dict)

        xyz, feat = self._pad_to_dense(clouds, device)   # (B,3,N), (B,C,N)

        # PointNet++ forward
        _, l3_points = self.backbone(xyz, feat)  # l3_points: (B, 1024, 1)
        global_feat = l3_points.squeeze(-1)      # (B, 1024)

        # Return (total_pts, 1024) — replicate per cloud
        # Handle both dense (B,N,3) and sparse (total,3) coord formats
        coord = input_dict["coord"]
        if coord.dim() == 3:
            # Dense: each cloud has same N points
            B, N, _ = coord.shape
            parts = [global_feat[b:b+1].expand(N, -1) for b in range(B)]
        else:
            # Sparse: use offset
            offset = input_dict["offset"]
            if isinstance(offset, list):
                offset = torch.tensor([o[0] if isinstance(o, list) else int(o) for o in offset])
            parts = []
            prev = 0
            for b, end_val in enumerate(offset.tolist()):
                end = int(end_val)
                n_i = end - prev
                parts.append(global_feat[b:b+1].expand(n_i, -1))
                prev = end

        return torch.cat(parts, dim=0)  # (total_pts, 1024)
