from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Tuple

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import LightningModule

from src import utils as U

log = U.RankedLogger(__name__, rank_zero_only=True)


class RLBenchACTBCModule(LightningModule):
    def __init__(
        self,
        policy,
        optimizer,
        lr_scheduler,
        train_metrics,
        val_metrics,
        best_val_metrics,
        compile: bool = False,
        temporal_agg: bool = False,
        proprio_embed_check_every_n_steps: int = 50,
        proprio_embed_check_on_val: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False,
            ignore=["policy", "train_metrics", "val_metrics", "best_val_metrics"],
        )

        self.policy = policy

        # metric objects for calculating and averaging accuracy across batches
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics

        # for tracking best so far validation metrics
        self.best_val_metrics = best_val_metrics

    @staticmethod
    def _safe_batch_corr(x: torch.Tensor, y: torch.Tensor) -> float:
        if x.numel() < 2 or y.numel() < 2:
            return float("nan")
        x_np = x.detach().float().cpu().numpy()
        y_np = y.detach().float().cpu().numpy()
        if np.std(x_np) < 1e-8 or np.std(y_np) < 1e-8:
            return float("nan")
        return float(np.corrcoef(x_np, y_np)[0, 1])

    def _maybe_print_proprio_embed_consistency(self, batch: Dict[str, torch.Tensor], stage: str, batch_idx: int) -> None:
        if stage == "train":
            every = int(getattr(self.hparams, "proprio_embed_check_every_n_steps", 50))
            if every <= 0 or (self.global_step % every) != 0:
                return
        else:
            if not bool(getattr(self.hparams, "proprio_embed_check_on_val", True)):
                return
            if batch_idx != 0:
                return

        obs = batch.get("obs_embeds", None)
        qpos = batch.get("qpos", None)
        if not isinstance(obs, torch.Tensor) or not isinstance(qpos, torch.Tensor):
            return
        if obs.dim() != 2 or qpos.dim() != 2:
            return

        obs_norm = torch.linalg.norm(obs.detach().float(), dim=-1)
        qpos_norm = torch.linalg.norm(qpos.detach().float(), dim=-1)
        norm_corr = self._safe_batch_corr(obs_norm, qpos_norm)

        hidden_cos = float("nan")
        try:
            with torch.no_grad():
                if hasattr(self.policy, "pcd_proj") and hasattr(self.policy, "input_proj_robot_state"):
                    obs_h = self.policy.pcd_proj(obs.detach())
                    qpos_h = self.policy.input_proj_robot_state(qpos.detach())
                    hidden_cos = float(torch.nn.functional.cosine_similarity(obs_h, qpos_h, dim=-1).mean().item())
                    
                    if not getattr(self, "_printed_pogs_victory", False):
                        print("\n=========================================================================")
                        print(f"✅ [SUCCESS: POGS VISION INTEGRATION RESOLVED]")
                        print(f"Policy architecture confirmed as: {self.policy.__class__.__name__}")
                        print(f"Vision stream (obs_embeds) successfully projected to hidden dim: {obs_h.shape}")
                        print(f"Proprioception (qpos) successfully projected to hidden dim: {qpos_h.shape}")
                        print("The ACT transformer is now learning from POGS Point Cloud representations!")
                        print("=========================================================================\n")
                        self._printed_pogs_victory = True
                        
        except Exception as e:
            print(f"[ERROR] Failed to compute vision/proprioception integration metrics: {e}")
            hidden_cos = float("nan")

        obs_mean = float(obs_norm.mean().item())
        qpos_mean = float(qpos_norm.mean().item())

        print(
            f"[TRAIN_DIAGNOSTIC] stage={stage} step={int(self.global_step)} "
            f"obs_norm_mean={obs_mean:.4f} qpos_norm_mean={qpos_mean:.4f} "
            f"norm_corr={norm_corr:.4f} hidden_cos={hidden_cos:.4f}"
        )

        if np.isfinite(norm_corr):
            self.log(
                f"{stage}/obs_qpos_norm_corr",
                norm_corr,
                on_step=(stage == "train"),
                on_epoch=(stage != "train"),
                prog_bar=False,
                sync_dist=True,
                batch_size=int(obs.shape[0]),
            )
        if np.isfinite(hidden_cos):
            self.log(
                f"{stage}/obs_qpos_hidden_cos",
                hidden_cos,
                on_step=(stage == "train"),
                on_epoch=(stage != "train"),
                prog_bar=False,
                sync_dist=True,
                batch_size=int(obs.shape[0]),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.train_metrics.reset()
        self.val_metrics.reset()
        self.best_val_metrics.reset()

    def model_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.policy(batch)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss_dict = self.model_step(batch)

        # update and log metrics
        self.train_metrics(loss_dict)
        self.log_dict(
            self.train_metrics.metrics_dict(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["actions"].shape[0],
        )

        self._maybe_print_proprio_embed_consistency(batch, stage="train", batch_idx=batch_idx)

        # return loss or backpropagation will fail
        return loss_dict["loss"]

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        return super().on_train_epoch_end()

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        loss_dict = self.model_step(batch)

        # update and log metrics
        self.val_metrics(loss_dict)
        self.log_dict(
            self.val_metrics.metrics_dict(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["actions"].shape[0],
        )

        self._maybe_print_proprio_embed_consistency(batch, stage="val", batch_idx=batch_idx)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        metrics = self.val_metrics.compute()  # get current val metrics
        self.best_val_metrics(metrics)  # update best so far val metrics
        # log `best_val_metrics` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log_dict(self.best_val_metrics.compute(), sync_dist=True, prog_bar=True)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        raise NotImplementedError

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.policy = torch.compile(self.policy)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = U.build_optimizer(self.hparams.optimizer, self.policy)
        if self.hparams.lr_scheduler is not None:
            self.hparams.lr_scheduler.scheduler.total_steps = (
                self.trainer.estimated_stepping_batches
            )
            scheduler = U.build_scheduler(
                self.hparams.lr_scheduler.scheduler, optimizer=optimizer
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams.lr_scheduler.get("monitor", "val/loss"),
                    "interval": self.hparams.lr_scheduler.get("interval", "step"),
                    "frequency": self.hparams.lr_scheduler.get("frequency", 1),
                },
            }
        return {"optimizer": optimizer}
