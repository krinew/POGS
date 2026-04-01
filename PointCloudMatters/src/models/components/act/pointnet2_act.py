"""ACT policy variants for PointNet++ features.

This module supports two observation modes:
1. Online point cloud encoding (PointNet2Encoder inside the policy).
2. Precomputed PointNet++ embeddings (no backbone in the policy).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from src.models.components.act.act import ACTRLBench


class ACTRLBenchPointNet2(ACTRLBench):
    """ACT policy that uses PointNet++ for point cloud perception.

    Unlike ACTPCD / ACTRLBenchPCD which use a sparse-conv backbone with
    per-point features + FPS token extraction, this class:
      1. Feeds the point cloud to PointNet2Encoder → (B, 1024) global vector.
      2. Projects to hidden_dim.
      3. Treats that as a *single* observation token fed to the transformer.

    Args:
        backbone: PointNet2Encoder instance (or any encoder that accepts a
                  PCM-style dict and returns (total_pts, num_channels)).
        transformer: ACT Transformer (shared with parent).
        encoder: ACT VAE encoder (shared with parent).
        hidden_dim: transformer hidden dimension (default 512).
        num_queries: action chunk size.
        num_cameras: ignored (set to 0 internally, PCD replaces images).
        action_dim: action space dimension.
        qpos_dim: proprioception vector dimension.
        env_state_dim: extra state dimension (usually 0).
        latent_dim: VAE latent dimension.
        action_loss: loss function for actions.
        klloss: KL divergence loss.
        kl_weight: weight for KL term.
        goal_cond_dim: CLIP goal conditioning dimension (0 = no goal).
        freeze_backbone: freeze PointNet++ weights.
        ignore_vae: skip VAE (inference only).
        rot_type: rotation representation ("6d").
        collision: predict collision flag.
        position_loss_weight: weight for position loss.
    """

    def __init__(
        self,
        backbone,
        transformer,
        encoder,
        hidden_dim: int = 512,
        num_queries: int = 100,
        num_cameras: int = 0,   # not used, kept for config compat
        action_dim: int = 11,
        qpos_dim: int = 11,
        env_state_dim: int = 0,
        latent_dim: int = 32,
        action_loss=None,
        klloss=None,
        kl_weight: float = 10.0,
        goal_cond_dim: int = 512,
        obs_feature_pos_embedding=None,  # not used; kept for config compat
        freeze_backbone: bool = False,
        ignore_vae: bool = False,
        rot_type: str = "6d",
        collision: bool = False,
        position_loss_weight: float = 10.0,
        obs_embed_dim: int = 1024,
        **kwargs
    ):
        super().__init__(
            backbone=backbone,
            transformer=transformer,
            encoder=encoder,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_cameras=0,          # we handle obs ourselves
            action_dim=action_dim,
            qpos_dim=qpos_dim,
            env_state_dim=env_state_dim,
            latent_dim=latent_dim,
            action_loss=action_loss,
            klloss=klloss,
            kl_weight=kl_weight,
            goal_cond_dim=goal_cond_dim,
            obs_feature_pos_embedding=None,
            freeze_backbone=freeze_backbone,
            ignore_vae=ignore_vae,
            rot_type=rot_type,
            collision=collision,
            position_loss_weight=position_loss_weight,
            
        )

        # Remove the Conv2d input_proj built by ACT.build_encoder
        # (it is for 2-D feature maps; we use PointNet++ global vectors)
        self.input_proj = None

        # Project feature vector (PointNet2 output or precomputed embedding) -> hidden_dim.
        if backbone is not None:
            feature_dim = backbone.num_channels
        else:
            feature_dim = obs_embed_dim

        self.pcd_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Positional embedding for the single PCD observation token
        # Shape: (1, hidden_dim) – learned
        self.pcd_pos_embed = nn.Embedding(1, hidden_dim)

    # ------------------------------------------------------------------
    # Override build_encoder to avoid creating the Conv2d input_proj
    # ------------------------------------------------------------------

    def build_encoder(self):
        """Build encoder components, skipping the image-backbone Conv2d."""
        import torch
        from src.models.components.act.utils import get_sinusoid_encoding_table

        # Robot state projection
        self.input_proj_robot_state = nn.Linear(self.qpos_dim, self.hidden_dim)

        # VAE components
        self.cls_embed = nn.Embedding(1, self.hidden_dim)
        self.encoder_action_proj = nn.Linear(self.action_dim, self.hidden_dim)
        self.encoder_joint_proj = nn.Linear(self.qpos_dim, self.hidden_dim)
        self.latent_proj = nn.Linear(self.hidden_dim, self.latent_dim * 2)

        self.register_buffer(
            "pos_table",
            get_sinusoid_encoding_table(1 + 1 + self.num_queries, self.hidden_dim),
        )

        if self.goal_cond_dim > 0:
            self.proj_goal_cond_emb = nn.Linear(self.goal_cond_dim, self.hidden_dim)

    # ------------------------------------------------------------------
    # Override build_decoder to adjust additional_pos_embed size
    # ------------------------------------------------------------------

    def build_decoder(self):
        """Build decoder components."""
        self.action_head = nn.Linear(self.hidden_dim, self.action_dim)
        self.is_pad_head = nn.Linear(self.hidden_dim, 1)
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
        self.latent_out_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        # 2 = latent + proprio  (+ 1 optional goal_cond)
        self.additional_pos_embed = nn.Embedding(
            2 + int(self.goal_cond_dim > 0), self.hidden_dim
        )

    # ------------------------------------------------------------------
    # Core: encode point cloud and return (src, pos) for transformer
    # ------------------------------------------------------------------

    def _encode_pcd(self, pcd_dict: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode point cloud to a single transformer token.

        Args:
            pcd_dict: PCM-style dict {coord, feat, offset, ...}

        Returns:
            src: (hidden_dim, B, 1) – observation token (seq-first)
            pos: (hidden_dim, B, 1) – positional embedding (seq-first)
        """
        if self.backbone is None:
            raise RuntimeError("backbone is None; provide 'obs_embeds' in data_dict instead of 'pcds'.")

        # backbone: produces (total_pts, C) with global feat repeated per point
        # Fix DataLoader collation: list of tensors → single tensor
        if isinstance(pcd_dict.get("offset"), list):
            pcd_dict["offset"] = torch.cat([o.view(-1) for o in pcd_dict["offset"]])
        if isinstance(pcd_dict.get("coord"), list):
            pcd_dict["coord"] = torch.cat(pcd_dict["coord"], dim=0)
        if isinstance(pcd_dict.get("feat"), list):
            pcd_dict["feat"] = torch.cat(pcd_dict["feat"], dim=0)
        raw_feat = self.backbone(pcd_dict)  # (total_pts, 1024)

        # Extract per-sample global vector from the first point of each cloud
        offset = pcd_dict["offset"]
        if isinstance(offset, list):
            offset = torch.tensor([o[0] if isinstance(o, list) else int(o) 
                                   for o in offset])
        elif isinstance(offset, torch.Tensor) and offset.dim() > 1:
            offset = offset[:, 0]

        global_feats = []
        prev = 0
        for end_val in offset.tolist():
            end = int(end_val)
            global_feats.append(raw_feat[prev])
            prev = end
        global_feat = torch.stack(global_feats, dim=0)  # (B, 1024)

        obs_token = self.pcd_proj(global_feat)    # (B, hidden_dim)
        # Expand to (1, B, hidden_dim) for transformer (seq-len = 1)
        src = obs_token.unsqueeze(0)              # (1, B, hidden_dim)

        # Positional embedding
        pos_embed = self.pcd_pos_embed.weight     # (1, hidden_dim)
        pos = pos_embed.unsqueeze(1).expand(1, obs_token.shape[0], -1)  # (1,B,H)

        # Transformer expects (B, C, H, W)-style src for the Transformer module
        # which then rearranges internally.  We feed it as (B, hidden_dim, 1, 1).
        src = rearrange(src, "s b c -> b c 1 s")   # (B, hidden_dim, 1, 1)
        pos = rearrange(pos, "s b c -> b c 1 s")   # (B, hidden_dim, 1, 1)

        return src, pos

    def _encode_obs_embed(self, obs_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode precomputed embeddings to a single transformer token."""
        if obs_embeds.dim() != 2:
            raise RuntimeError(f"Expected obs_embeds shape (B, D), got {tuple(obs_embeds.shape)}")

        obs_token = self.pcd_proj(obs_embeds)       # (B, hidden_dim)
        src = obs_token.unsqueeze(0)                # (1, B, hidden_dim)

        pos_embed = self.pcd_pos_embed.weight
        pos = pos_embed.unsqueeze(1).expand(1, obs_token.shape[0], -1)

        src = rearrange(src, "s b c -> b c 1 s")   # (B, hidden_dim, 1, 1)
        pos = rearrange(pos, "s b c -> b c 1 s")   # (B, hidden_dim, 1, 1)
        return src, pos

    # ------------------------------------------------------------------
    # Override forward_obs_embed from ACT base class
    # ------------------------------------------------------------------

    def forward_obs_embed(self, data_dict: dict) -> dict:
        """Produce observation embeddings from PCD input or precomputed embeddings."""
        qpos = data_dict["qpos"]
        latent_input = data_dict["latent_input"]

        if "obs_embeds" in data_dict:
            src, pos = self._encode_obs_embed(data_dict["obs_embeds"])
        else:
            pcd_dict = data_dict["pcds"]
            if isinstance(pcd_dict, list):
                pcd_dict = pcd_dict[0]
            src, pos = self._encode_pcd(pcd_dict)

        latent_input = latent_input.unsqueeze(0)  # (1, B, H)
        proprio_input = self.input_proj_robot_state(qpos).unsqueeze(0)  # (1,B,H)

        if self.goal_cond_dim > 0:
            goal = data_dict.get("goal_cond", None)
            if goal is not None:
                if goal.dim() > 2:
                    goal = goal.reshape(goal.shape[0], -1)
                goal_emb = self.proj_goal_cond_emb(goal).unsqueeze(0)
                proprio_input = torch.cat([proprio_input, goal_emb], dim=0)

        data_dict["src"] = src
        data_dict["pos"] = pos
        data_dict["latent_input"] = latent_input
        data_dict["proprio_input"] = proprio_input

        return data_dict
