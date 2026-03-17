#!/usr/bin/env python3
"""
Animated comparison for two NeuralFoil runs.

- Left: overlaid objective curves where objective = -reward (lower is better), log y-scale.
- Right: current shape image from each run (stacked).
"""

import os
import csv as csv_mod
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 200,
}


def load_csv(results_dir):
    csv_path = os.path.join(results_dir, "results.csv")
    rows = []
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        for i, row in enumerate(reader):
            it = int(row["iteration"])
            reward = float(row["reward"])
            # Prefer explicit design id if present, otherwise synthesize for PSO.
            design = row.get("design", "")
            if not design:
                particle = row.get("particle", "")
                if particle != "":
                    design = f"iter_{it:04d}_p{int(float(particle)):03d}"
                else:
                    design = f"design_{i}"
            rows.append({"iteration": it, "reward": reward, "design": design})
    return rows


def cap_rows(rows, max_iter):
    if max_iter is None:
        return rows
    return [r for r in rows if r["iteration"] <= max_iter]


def compute_plot_x(rows):
    """Fractional x: iteration + within-iteration index/total."""
    iters = np.array([r["iteration"] for r in rows], dtype=int)
    if len(iters) == 0:
        return np.array([], dtype=float)
    counts = {}
    for v in iters:
        counts[v] = counts.get(v, 0) + 1
    seen = {}
    x = np.zeros(len(iters), dtype=float)
    for i, v in enumerate(iters):
        idx = seen.get(v, 0)
        seen[v] = idx + 1
        denom = counts[v]
        x[i] = float(v) if denom <= 1 else float(v) + idx / float(denom)
    return x


def objective_from_rewards(rows):
    r = np.array([row["reward"] for row in rows], dtype=float)
    obj = np.maximum(-r, 1e-9)
    best = np.minimum.accumulate(obj) if len(obj) else obj
    return obj, best


def get_shape_image(results_dir, design_ref):
    p = os.path.join(results_dir, design_ref, "save", "shape.png")
    if os.path.exists(p):
        return Image.open(p)
    return None


def make_comparison(dir_a, dir_b, label_a=None, label_b=None, max_iter=80, step=1, output_dir=None):
    rows_a = cap_rows(load_csv(dir_a), max_iter)
    rows_b = cap_rows(load_csv(dir_b), max_iter)
    label_a = label_a or os.path.basename(dir_a)
    label_b = label_b or os.path.basename(dir_b)

    n = max(len(rows_a), len(rows_b))
    if n == 0:
        print("No rows to render.")
        return

    x_a = compute_plot_x(rows_a)
    x_b = compute_plot_x(rows_b)
    obj_a, best_a = objective_from_rewards(rows_a)
    obj_b, best_b = objective_from_rewards(rows_b)

    all_obj = np.concatenate([obj_a, obj_b]) if len(obj_a) and len(obj_b) else (obj_a if len(obj_a) else obj_b)
    p5, p95 = np.percentile(all_obj, 5), np.percentile(all_obj, 95)
    y_lo = max(min(np.min(all_obj) * 0.8, p5 * 0.8), 1e-9)
    y_hi = max(np.max(all_obj) * 1.2, p95 * 1.2)
    x_min = -0.1
    x_max = max(float(np.max(x_a)) if len(x_a) else 0.0, float(np.max(x_b)) if len(x_b) else 0.0) + 0.1

    output_dir = output_dir or os.path.dirname(dir_a)
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update(STYLE)
    frame_indices = list(range(0, n, max(1, step)))
    if frame_indices[-1] != n - 1:
        frame_indices.append(n - 1)

    frames = []
    print(f"[comparison] Rendering {len(frame_indices)} frames ...")
    for fi, up_to in enumerate(frame_indices):
        k_a = min(up_to + 1, len(rows_a))
        k_b = min(up_to + 1, len(rows_b))

        fig = plt.figure(figsize=(14, 7), facecolor="white")
        gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.06)
        gs_right = gs[1].subgridspec(2, 1, hspace=0.12)

        # Left: one overlaid objective plot
        ax = fig.add_subplot(gs[0])
        if k_a > 0:
            ax.scatter(x_a[:k_a], np.clip(obj_a[:k_a], y_lo, y_hi), s=12, alpha=0.5, color="#1f77b4",
                       edgecolors="k", linewidths=0.1, zorder=2)
            ax.plot(x_a[:k_a], np.clip(best_a[:k_a], y_lo, y_hi), color="#1f77b4", lw=1.5, zorder=3,
                    label=f"{label_a} best")
            ax.scatter([x_a[k_a - 1]], [np.clip(obj_a[k_a - 1], y_lo, y_hi)], s=80,
                       facecolors="none", edgecolors="#1f77b4", linewidths=1.8, zorder=5,
                       label=f"{label_a} current")
        if k_b > 0:
            ax.scatter(x_b[:k_b], np.clip(obj_b[:k_b], y_lo, y_hi), s=12, alpha=0.5, color="#d62728",
                       edgecolors="k", linewidths=0.1, zorder=2)
            ax.plot(x_b[:k_b], np.clip(best_b[:k_b], y_lo, y_hi), color="#d62728", lw=1.5, zorder=3,
                    label=f"{label_b} best")
            ax.scatter([x_b[k_b - 1]], [np.clip(obj_b[k_b - 1], y_lo, y_hi)], s=80,
                       facecolors="none", edgecolors="#d62728", linewidths=1.8, zorder=5,
                       label=f"{label_b} current")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_lo, y_hi)
        ax.set_yscale("log")
        ax.set_xlabel("LLM calls")
        ax.set_ylabel("Objective (-reward, lower is better)")
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95, ncol=2)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        b_a = best_a[k_a - 1] if k_a > 0 else np.nan
        b_b = best_b[k_b - 1] if k_b > 0 else np.nan
        ax.set_title(f"Up to frame {up_to}: best={b_a:.4f} vs {b_b:.4f}", fontweight="medium", pad=8)

        # Right top: current design from A
        ax_img_a = fig.add_subplot(gs_right[0])
        ax_img_a.axis("off")
        if k_a > 0:
            dref_a = rows_a[k_a - 1]["design"]
            img_a = get_shape_image(dir_a, dref_a)
            if img_a is not None:
                ax_img_a.imshow(img_a)
                ax_img_a.set_title(f"{label_a} ({dref_a})", fontsize=9, fontweight="medium", pad=4)
            else:
                ax_img_a.text(0.5, 0.5, "(no shape.png)", ha="center", va="center",
                              fontsize=10, transform=ax_img_a.transAxes, color="#888")

        # Right bottom: current design from B
        ax_img_b = fig.add_subplot(gs_right[1])
        ax_img_b.axis("off")
        if k_b > 0:
            dref_b = rows_b[k_b - 1]["design"]
            img_b = get_shape_image(dir_b, dref_b)
            if img_b is not None:
                ax_img_b.imshow(img_b)
                ax_img_b.set_title(f"{label_b} ({dref_b})", fontsize=9, fontweight="medium", pad=4)
            else:
                ax_img_b.text(0.5, 0.5, "(no shape.png)", ha="center", va="center",
                              fontsize=10, transform=ax_img_b.transAxes, color="#888")

        fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.08)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)

        if (fi + 1) % 20 == 0:
            print(f"  {fi + 1}/{len(frame_indices)} frames")

    for _ in range(8):
        frames.append(frames[-1])
    out = os.path.join(output_dir, "comparison_evolution.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=150, loop=0, optimize=True)
    print(f"  -> {out}  ({len(frames)} frames, {os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_a")
    parser.add_argument("dir_b")
    parser.add_argument("--label-a", default=None)
    parser.add_argument("--label-b", default=None)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    make_comparison(
        args.dir_a,
        args.dir_b,
        label_a=args.label_a,
        label_b=args.label_b,
        max_iter=args.max_iter,
        step=args.step,
        output_dir=args.output_dir,
    )
