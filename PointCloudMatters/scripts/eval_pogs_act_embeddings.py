#!/usr/bin/env python3
"""Evaluate quality of precomputed POGS-ACT episode embeddings.

This script includes both basic diagnostics and research-backed metrics.

Research-inspired metrics included:
- Alignment/Uniformity on hypersphere pairs
  (Wang & Isola, ICML 2020)
- Effective rank / spectral spread metrics
  (used in RankMe-style representation diagnostics)
- Linear probe of action-delta predictability from embeddings
  (standard in representation-learning evaluations)

Expected per-episode keys (minimum):
- obs_embeds: (T, D)
Optional but used when present:
- action: (T, A)
- joint_positions: (T, J)
- gripper_open: (T, 1)
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a_std = a.std()
    b_std = b.std()
    if a_std < 1e-12 or b_std < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _cosine_sim_rows(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    x0n = np.linalg.norm(x0, axis=1) + 1e-12
    x1n = np.linalg.norm(x1, axis=1) + 1e-12
    return np.sum(x0 * x1, axis=1) / (x0n * x1n)


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _alignment_sq(z0: np.ndarray, z1: np.ndarray) -> float:
    if z0.size == 0 or z1.size == 0:
        return float("nan")
    d = z0 - z1
    return float(np.sum(d * d, axis=1).mean())


def _uniformity_sampled(z: np.ndarray, max_pairs: int, rng: np.random.Generator) -> float:
    n = z.shape[0]
    if n < 2:
        return float("nan")
    max_pairs = max(1, int(max_pairs))
    i = rng.integers(0, n, size=max_pairs)
    j = rng.integers(0, n, size=max_pairs)
    valid = i != j
    if not np.any(valid):
        return float("nan")
    i = i[valid]
    j = j[valid]
    dist_sq = np.sum((z[i] - z[j]) ** 2, axis=1)
    return float(np.log(np.mean(np.exp(-2.0 * dist_sq)) + 1e-12))


def _effective_rank(x: np.ndarray) -> float:
    if x.shape[0] < 2:
        return float("nan")
    x0 = x - x.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(x0, full_matrices=False)
    power = s * s
    p = power / (power.sum() + 1e-12)
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))


def _participation_ratio(x: np.ndarray) -> float:
    if x.shape[0] < 2:
        return float("nan")
    x0 = x - x.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(x0, full_matrices=False)
    power = s * s
    num = power.sum() ** 2
    den = np.sum(power * power) + 1e-12
    return float(num / den)


def _ridge_multitarget_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    reg: float,
) -> np.ndarray:
    x_train = x_train.astype(np.float64)
    y_train = y_train.astype(np.float64)
    x_test = x_test.astype(np.float64)

    x_mean = x_train.mean(axis=0, keepdims=True)
    y_mean = y_train.mean(axis=0, keepdims=True)
    x0 = x_train - x_mean
    y0 = y_train - y_mean

    n, d = x0.shape
    reg = float(max(reg, 1e-9))

    if n >= d:
        a = x0.T @ x0 + reg * np.eye(d, dtype=np.float64)
        b = x0.T @ y0
        w = np.linalg.solve(a, b)
    else:
        k = x0 @ x0.T + reg * np.eye(n, dtype=np.float64)
        alpha = np.linalg.solve(k, y0)
        w = x0.T @ alpha

    y_pred = (x_test - x_mean) @ w + y_mean
    return y_pred.astype(np.float32)


def _r2_multitarget(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    y_mean = y_true.mean(axis=0, keepdims=True)
    ss_tot = np.sum((y_true - y_mean) ** 2, axis=0)
    # Ignore near-constant targets where R2 is numerically unstable.
    valid = ss_tot > 1e-6
    if not np.any(valid):
        return float("nan")
    r2 = 1.0 - (ss_res[valid] / (ss_tot[valid] + 1e-12))
    return float(np.mean(r2))


def _temporal_linear_probe(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    reg: float,
) -> tuple[float, float]:
    n = min(x.shape[0], y.shape[0])
    if n < 10:
        return float("nan"), float("nan")

    x = x[:n]
    y = y[:n]

    split = int(round(n * train_ratio))
    split = max(2, min(split, n - 2))

    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    y_pred = _ridge_multitarget_predict(x_train, y_train, x_test, reg=reg)
    r2 = _r2_multitarget(y_test, y_pred)
    mae = float(np.mean(np.abs(y_test - y_pred)))
    return r2, mae


def _find_episode_files(embed_root: Path, task: str | None) -> list[Path]:
    if task is not None:
        task_dir = embed_root / task
        if not task_dir.exists():
            raise FileNotFoundError(f"Task folder not found: {task_dir}")
        return sorted(task_dir.glob("episode*.pkl"))

    direct = sorted(embed_root.glob("episode*.pkl"))
    if direct:
        return direct

    files: list[Path] = []
    for task_dir in sorted(embed_root.iterdir()):
        if task_dir.is_dir():
            files.extend(sorted(task_dir.glob("episode*.pkl")))
    return files


def evaluate(
    embed_root: Path,
    task: str | None,
    max_episodes: int,
    uniformity_pairs: int,
    probe_reg: float,
    probe_train_ratio: float,
) -> dict[str, Any]:
    files = _find_episode_files(embed_root, task)
    if not files:
        raise RuntimeError(f"No episode*.pkl files found under {embed_root}")

    if max_episodes > 0:
        files = files[:max_episodes]

    rng = np.random.default_rng(0)

    per_episode: list[dict[str, Any]] = []
    all_embeds: list[np.ndarray] = []

    dims_seen: set[int] = set()
    min_t = math.inf
    max_t = 0
    total_t = 0

    required_keys = ["obs_embeds"]
    optional_keys = ["action", "joint_positions", "gripper_open", "variation_id", "task_goal"]
    schema_missing_counts: dict[str, int] = {k: 0 for k in required_keys + optional_keys}

    for fp in files:
        with fp.open("rb") as f:
            data = pickle.load(f)

        miss_req = [k for k in required_keys if k not in data]
        for k in miss_req:
            schema_missing_counts[k] += 1
        for k in optional_keys:
            if k not in data:
                schema_missing_counts[k] += 1

        if miss_req:
            per_episode.append(
                {
                    "file": str(fp),
                    "valid": False,
                    "error": f"missing required keys: {miss_req}",
                }
            )
            continue

        obs_embeds = np.asarray(data["obs_embeds"], dtype=np.float32)
        if obs_embeds.ndim != 2:
            per_episode.append(
                {
                    "file": str(fp),
                    "valid": False,
                    "error": f"obs_embeds should be 2D (T,D), got shape {obs_embeds.shape}",
                }
            )
            continue

        t_steps, emb_dim = obs_embeds.shape
        dims_seen.add(int(emb_dim))
        min_t = min(min_t, t_steps)
        max_t = max(max_t, t_steps)
        total_t += t_steps

        all_embeds.append(obs_embeds)

        if t_steps >= 2:
            z0 = obs_embeds[:-1]
            z1 = obs_embeds[1:]
            d_embed = z1 - z0
            cos_consec = _cosine_sim_rows(z0, z1)
            l2_consec = np.linalg.norm(d_embed, axis=1)

            z_all_n = _l2_normalize_rows(obs_embeds)
            z0_n = z_all_n[:-1]
            z1_n = z_all_n[1:]
            alignment_sq = _alignment_sq(z0_n, z1_n)
            uniformity = _uniformity_sampled(z_all_n, max_pairs=uniformity_pairs, rng=rng)
        else:
            d_embed = np.asarray([], dtype=np.float32)
            cos_consec = np.asarray([], dtype=np.float32)
            l2_consec = np.asarray([], dtype=np.float32)
            alignment_sq = float("nan")
            uniformity = float("nan")

        eff_rank = _effective_rank(obs_embeds)
        pr = _participation_ratio(obs_embeds)
        rank_norm = float(min(obs_embeds.shape[0], obs_embeds.shape[1]))
        eff_rank_ratio = float(eff_rank / rank_norm) if rank_norm > 0 else float("nan")
        pr_ratio = float(pr / rank_norm) if rank_norm > 0 else float("nan")

        action = data.get("action", None)
        corr = float("nan")
        probe_r2 = float("nan")
        probe_mae = float("nan")
        if action is not None:
            action = np.asarray(action, dtype=np.float32)
            if action.ndim == 2 and action.shape[0] >= 2 and d_embed.size:
                d_action = action[1:] - action[:-1]
                n_da = min(d_embed.shape[0], d_action.shape[0])
                corr = _safe_corr(
                    np.linalg.norm(d_embed[:n_da], axis=1),
                    np.linalg.norm(d_action[:n_da], axis=1),
                )

                x_probe = obs_embeds[:n_da]
                y_probe = d_action[:n_da]
                probe_r2, probe_mae = _temporal_linear_probe(
                    x_probe,
                    y_probe,
                    train_ratio=probe_train_ratio,
                    reg=probe_reg,
                )

        per_episode.append(
            {
                "file": str(fp),
                "valid": True,
                "timesteps": int(t_steps),
                "embed_dim": int(emb_dim),
                "embed_norm_mean": float(np.linalg.norm(obs_embeds, axis=1).mean()),
                "embed_norm_std": float(np.linalg.norm(obs_embeds, axis=1).std()),
                "consec_cos_mean": float(cos_consec.mean()) if cos_consec.size else float("nan"),
                "consec_cos_min": float(cos_consec.min()) if cos_consec.size else float("nan"),
                "consec_l2_mean": float(l2_consec.mean()) if l2_consec.size else float("nan"),
                "embed_action_delta_corr": corr,
                "alignment_sq": alignment_sq,
                "uniformity": uniformity,
                "effective_rank": eff_rank,
                "effective_rank_ratio": eff_rank_ratio,
                "participation_ratio": pr,
                "participation_ratio_norm": pr_ratio,
                "linear_probe_action_r2": probe_r2,
                "linear_probe_action_mae": probe_mae,
            }
        )

    valid_eps = [e for e in per_episode if e.get("valid", False)]
    if not valid_eps:
        raise RuntimeError("No valid episodes with required schema were found.")

    embed_cat = np.concatenate(all_embeds, axis=0)
    embed_norms = np.linalg.norm(embed_cat, axis=1)

    dim_std = embed_cat.std(axis=0)
    low_var_dims = int((dim_std < 1e-4).sum())

    consec_cos_all = np.asarray(
        [e["consec_cos_mean"] for e in valid_eps if not np.isnan(e["consec_cos_mean"])],
        dtype=np.float32,
    )
    corr_all = np.asarray(
        [e["embed_action_delta_corr"] for e in valid_eps if not np.isnan(e["embed_action_delta_corr"])],
        dtype=np.float32,
    )
    alignment_all = np.asarray(
        [e["alignment_sq"] for e in valid_eps if not np.isnan(e["alignment_sq"])],
        dtype=np.float32,
    )
    uniformity_all = np.asarray(
        [e["uniformity"] for e in valid_eps if not np.isnan(e["uniformity"])],
        dtype=np.float32,
    )
    eff_rank_ratio_all = np.asarray(
        [e["effective_rank_ratio"] for e in valid_eps if not np.isnan(e["effective_rank_ratio"])],
        dtype=np.float32,
    )
    pr_ratio_all = np.asarray(
        [e["participation_ratio_norm"] for e in valid_eps if not np.isnan(e["participation_ratio_norm"])],
        dtype=np.float32,
    )
    probe_r2_all = np.asarray(
        [e["linear_probe_action_r2"] for e in valid_eps if not np.isnan(e["linear_probe_action_r2"])],
        dtype=np.float32,
    )
    probe_mae_all = np.asarray(
        [e["linear_probe_action_mae"] for e in valid_eps if not np.isnan(e["linear_probe_action_mae"])],
        dtype=np.float32,
    )

    summary: dict[str, Any] = {
        "embed_root": str(embed_root),
        "task": task,
        "episodes_scanned": len(files),
        "episodes_valid": len(valid_eps),
        "timesteps_total": int(total_t),
        "timesteps_min": int(min_t),
        "timesteps_max": int(max_t),
        "embed_dims_seen": sorted(list(dims_seen)),
        "schema_missing_counts": schema_missing_counts,
        "embedding_stats": {
            "mean": float(embed_cat.mean()),
            "std": float(embed_cat.std()),
            "abs_mean": float(np.abs(embed_cat).mean()),
            "norm_mean": float(embed_norms.mean()),
            "norm_std": float(embed_norms.std()),
            "norm_p05": float(np.percentile(embed_norms, 5)),
            "norm_p50": float(np.percentile(embed_norms, 50)),
            "norm_p95": float(np.percentile(embed_norms, 95)),
        },
        "temporal_stats": {
            "episode_consec_cos_mean": float(np.nanmean(consec_cos_all)) if consec_cos_all.size else float("nan"),
            "episode_consec_cos_std": float(np.nanstd(consec_cos_all)) if consec_cos_all.size else float("nan"),
            "highly_static_episode_ratio_cos_gt_0_999": float((consec_cos_all > 0.999).mean()) if consec_cos_all.size else float("nan"),
        },
        "alignment_stats": {
            "embed_action_delta_corr_mean": float(np.nanmean(corr_all)) if corr_all.size else float("nan"),
            "embed_action_delta_corr_std": float(np.nanstd(corr_all)) if corr_all.size else float("nan"),
            "episodes_with_corr": int(corr_all.size),
        },
        "research_metrics": {
            "references": [
                "Alignment/Uniformity: Wang & Isola (ICML 2020)",
                "Rank diagnostics (effective rank): RankMe-style spectral metrics",
                "Linear probe: standard representation-learning diagnostic",
            ],
            "settings": {
                "uniformity_pairs": int(uniformity_pairs),
                "probe_reg": float(probe_reg),
                "probe_train_ratio": float(probe_train_ratio),
            },
            "alignment_sq_mean": float(np.nanmean(alignment_all)) if alignment_all.size else float("nan"),
            "alignment_sq_std": float(np.nanstd(alignment_all)) if alignment_all.size else float("nan"),
            "uniformity_mean": float(np.nanmean(uniformity_all)) if uniformity_all.size else float("nan"),
            "uniformity_std": float(np.nanstd(uniformity_all)) if uniformity_all.size else float("nan"),
            "effective_rank_ratio_mean": float(np.nanmean(eff_rank_ratio_all)) if eff_rank_ratio_all.size else float("nan"),
            "effective_rank_ratio_std": float(np.nanstd(eff_rank_ratio_all)) if eff_rank_ratio_all.size else float("nan"),
            "participation_ratio_norm_mean": float(np.nanmean(pr_ratio_all)) if pr_ratio_all.size else float("nan"),
            "participation_ratio_norm_std": float(np.nanstd(pr_ratio_all)) if pr_ratio_all.size else float("nan"),
            "linear_probe_action_r2_mean": float(np.nanmean(probe_r2_all)) if probe_r2_all.size else float("nan"),
            "linear_probe_action_r2_std": float(np.nanstd(probe_r2_all)) if probe_r2_all.size else float("nan"),
            "linear_probe_action_mae_mean": float(np.nanmean(probe_mae_all)) if probe_mae_all.size else float("nan"),
            "linear_probe_action_mae_std": float(np.nanstd(probe_mae_all)) if probe_mae_all.size else float("nan"),
            "episodes_with_linear_probe": int(probe_r2_all.size),
        },
        "collapse_checks": {
            "low_variance_dims_lt_1e_4": low_var_dims,
            "low_variance_ratio": float(low_var_dims / embed_cat.shape[1]),
        },
        "per_episode": per_episode,
    }

    return summary


def print_report(summary: dict[str, Any], show_per_episode: bool) -> None:
    print("=== Embedding Evaluation Summary ===")
    print(f"embed_root               : {summary['embed_root']}")
    print(f"task                     : {summary['task']}")
    print(f"episodes scanned/valid   : {summary['episodes_scanned']} / {summary['episodes_valid']}")
    print(f"timesteps total|min|max  : {summary['timesteps_total']} | {summary['timesteps_min']} | {summary['timesteps_max']}")
    print(f"embed dims seen          : {summary['embed_dims_seen']}")

    es = summary["embedding_stats"]
    print("--- embedding stats ---")
    print(f"mean/std                 : {es['mean']:.6f} / {es['std']:.6f}")
    print(f"abs_mean                 : {es['abs_mean']:.6f}")
    print(f"norm mean/std            : {es['norm_mean']:.6f} / {es['norm_std']:.6f}")
    print(f"norm p05/p50/p95         : {es['norm_p05']:.6f} / {es['norm_p50']:.6f} / {es['norm_p95']:.6f}")

    ts = summary["temporal_stats"]
    print("--- temporal stats ---")
    print(f"consecutive cos mean/std : {ts['episode_consec_cos_mean']:.6f} / {ts['episode_consec_cos_std']:.6f}")
    print(f"static ratio (cos>0.999) : {ts['highly_static_episode_ratio_cos_gt_0_999']:.6f}")

    al = summary["alignment_stats"]
    print("--- action alignment ---")
    print(f"corr mean/std            : {al['embed_action_delta_corr_mean']:.6f} / {al['embed_action_delta_corr_std']:.6f}")
    print(f"episodes with corr       : {al['episodes_with_corr']}")

    rm = summary["research_metrics"]
    print("--- research metrics ---")
    print(f"alignment_sq mean/std    : {rm['alignment_sq_mean']:.6f} / {rm['alignment_sq_std']:.6f}")
    print(f"uniformity mean/std      : {rm['uniformity_mean']:.6f} / {rm['uniformity_std']:.6f}")
    print(f"eff_rank_ratio mean/std  : {rm['effective_rank_ratio_mean']:.6f} / {rm['effective_rank_ratio_std']:.6f}")
    print(f"pr_ratio mean/std        : {rm['participation_ratio_norm_mean']:.6f} / {rm['participation_ratio_norm_std']:.6f}")
    print(f"probe R2 mean/std        : {rm['linear_probe_action_r2_mean']:.6f} / {rm['linear_probe_action_r2_std']:.6f}")
    print(f"probe MAE mean/std       : {rm['linear_probe_action_mae_mean']:.6f} / {rm['linear_probe_action_mae_std']:.6f}")
    print(f"episodes with probe      : {rm['episodes_with_linear_probe']}")

    cc = summary["collapse_checks"]
    print("--- collapse checks ---")
    print(f"low-variance dims        : {cc['low_variance_dims_lt_1e_4']}")
    print(f"low-variance ratio       : {cc['low_variance_ratio']:.6f}")

    miss = summary["schema_missing_counts"]
    print("--- schema missing counts ---")
    for k in sorted(miss.keys()):
        print(f"{k:24s}: {miss[k]}")

    if show_per_episode:
        print("--- per-episode ---")
        for ep in summary["per_episode"]:
            if not ep.get("valid", False):
                print(f"INVALID | {ep['file']} | {ep['error']}")
                continue
            print(
                "OK | "
                f"{Path(ep['file']).name} | "
                f"T={ep['timesteps']} D={ep['embed_dim']} | "
                f"norm={ep['embed_norm_mean']:.4f}+/-{ep['embed_norm_std']:.4f} | "
                f"cos={ep['consec_cos_mean']:.4f} | "
                f"align={ep['alignment_sq']:.4f} uni={ep['uniformity']:.4f} | "
                f"rank={ep['effective_rank_ratio']:.4f} pr={ep['participation_ratio_norm']:.4f} | "
                f"probe_r2={ep['linear_probe_action_r2']:.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate POGS-ACT episode embeddings")
    parser.add_argument("--embed-root", type=Path, required=True, help="Root containing task folders or episode*.pkl files")
    parser.add_argument("--task", type=str, default=None, help="Optional task folder name, e.g. open_drawer")
    parser.add_argument("--max-episodes", type=int, default=-1, help="Limit number of episodes for quick checks")
    parser.add_argument("--uniformity-pairs", type=int, default=50000, help="Random pair count used for uniformity metric")
    parser.add_argument("--probe-reg", type=float, default=1e-2, help="Ridge regularization for temporal linear probe")
    parser.add_argument("--probe-train-ratio", type=float, default=0.8, help="Train split ratio for temporal linear probe")
    parser.add_argument("--show-per-episode", action="store_true", help="Print per-episode metrics")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional path to save full JSON report")
    args = parser.parse_args()

    summary = evaluate(
        embed_root=args.embed_root,
        task=args.task,
        max_episodes=args.max_episodes,
        uniformity_pairs=args.uniformity_pairs,
        probe_reg=args.probe_reg,
        probe_train_ratio=args.probe_train_ratio,
    )
    print_report(summary, args.show_per_episode)

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved report: {args.out_json}")


if __name__ == "__main__":
    main()
