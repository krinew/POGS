"""Summarize per-group tracking diagnostics from a random rollout JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _finite_mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _finite_max(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(max(vals)) if vals else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect per-group rollout diagnostics.")
    parser.add_argument("rollout_json", type=Path)
    parser.add_argument("--step", type=int, default=0)
    args = parser.parse_args()

    obj = json.loads(args.rollout_json.read_text(encoding="utf-8"))
    rows = obj.get("rows", [])
    if not rows:
        raise SystemExit(f"No rows found in {args.rollout_json}")

    step_rows = [r for r in rows if int(r.get("step", -1)) == int(args.step)]
    row = step_rows[0] if step_rows else rows[min(max(args.step, 0), len(rows) - 1)]

    print(f"rollout: {args.rollout_json}")
    print(f"step: {row.get('step')} / rows={len(rows)}")
    print(
        "global: "
        f"total={row.get('opt_total_loss_last')} "
        f"depth={row.get('opt_depth_loss_last')} "
        f"mask={row.get('opt_mask_bce_loss_last')} "
        f"dino={row.get('opt_dino_loss_last')} "
        f"ignored_groups={row.get('seg_ignored_groups')}"
    )

    render_stats = {int(s["group"]): s for s in row.get("seg_render_group_stats", [])}
    assignments = {int(a["group"]): a for a in row.get("seg_assignments", [])}
    per_group = row.get("per_group_loss_metrics", {}) or {}
    per_pose = row.get("per_group_pose_step_metrics", {}) or {}
    groups = sorted(set(render_stats) | set(assignments) | {int(k.split("_")[1]) for k in per_group if k.startswith("group_")})

    header = (
        "group  area  visible  target  ratio  iou    depth_err(assign)  depth_err(render)  "
        "depth_loss  mask_bce  dino_all  dino_valid  ignored reason"
    )
    print("\n" + header)
    print("-" * len(header))
    for g in groups:
        s = render_stats.get(g, {})
        a = assignments.get(g, {})
        prefix = f"group_{g:02d}_"
        def metric(name: str):
            return per_group.get(prefix + name + "_last", per_group.get(prefix + name, float("nan")))
        def pose_metric(name: str):
            return per_pose.get(prefix + name, float("nan"))

        print(
            f"g{g:02d}   "
            f"{float(s.get('render_area', a.get('render_area', math.nan))):5.0f}  "
            f"{float(a.get('visible_area', math.nan)):7.0f}  "
            f"{float(a.get('target_area', math.nan)):6.0f}  "
            f"{float(a.get('target_render_ratio', math.nan)):5.3f}  "
            f"{float(a.get('iou', math.nan)):5.3f}  "
            f"{float(a.get('median_depth_err_m', math.nan)):16.4f}  "
            f"{float(s.get('median_abs_depth_err', math.nan)):17.4f}  "
            f"{float(metric('depth_loss')):10.6f}  "
            f"{float(metric('mask_bce_loss')):8.5f}  "
            f"{float(metric('dino_loss_all')):8.4f}  "
            f"{float(metric('dino_loss_valid')):10.4f}  "
            f"{a.get('ignored')} {a.get('ignore_reason', '')}  "
            f"step_trans={float(pose_metric('step_trans_delta_m')):.4f}m"
        )

    by_group: dict[int, dict[str, list[float]]] = {}
    for r in rows:
        for s in r.get("seg_render_group_stats", []):
            g = int(s["group"])
            by_group.setdefault(g, {"area": [], "render_depth_err": []})
            by_group[g]["area"].append(float(s.get("render_area", math.nan)))
            by_group[g]["render_depth_err"].append(float(s.get("median_abs_depth_err", math.nan)))

    print("\nAcross rollout:")
    for g in sorted(by_group):
        stats = by_group[g]
        print(
            f"g{g:02d}: mean_area={_finite_mean(stats['area']):.1f} "
            f"mean_render_depth_err={_finite_mean(stats['render_depth_err']):.4f} "
            f"max_render_depth_err={_finite_max(stats['render_depth_err']):.4f}"
        )


if __name__ == "__main__":
    main()
