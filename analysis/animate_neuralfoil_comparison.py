#!/usr/bin/env python3
"""
Animated comparison for any number of NeuralFoil runs.

- Left: overlaid objective curves. Maximisation rewards (e.g. ld_ratio) plotted directly on linear scale;
         minimisation rewards (e.g. -Cd) plotted as -reward on log scale.
- Right: best shape image from each run (stacked vertically).

Usage:
    python animate_neuralfoil_comparison.py dir_a dir_b [dir_c ...] \\
        --labels "islands" "PSO" "v2_batch" \\
        --max-iter 200 --step 2
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

# Colours assigned per run in order.
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]


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
            # Read running-best column written by each framework:
            #   islands/v2/v2_batch write "best_reward"; GA/PSO writes "gbest_reward".
            br = row.get("gbest_reward") or row.get("best_reward")
            best_reward = float(br) if br else None
            rows.append({"iteration": it, "reward": reward, "design": design,
                         "best_reward": best_reward})
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
        x[i] = float(v) if counts[v] <= 1 else float(v) + idx / float(counts[v])
    return x


def objective_from_rewards(rows):
    r = np.array([row["reward"] for row in rows], dtype=float)
    # Detect maximisation (positive rewards like ld_ratio) vs minimisation (negative like -Cd).
    valid = r[r > -9.0]  # exclude -10.0 failure sentinel
    maximizing = len(valid) > 0 and np.median(valid) > 0
    if maximizing:
        obj = r.copy()
    else:
        obj = np.maximum(-r, 1e-9)
    # Use explicit running-best from CSV when available (required for PSO where
    # np.maximum.accumulate over all particles gives wrong per-iteration best).
    if rows[0]["best_reward"] is not None:
        br = np.array([row["best_reward"] for row in rows], dtype=float)
        best = br.copy() if maximizing else np.maximum(-br, 1e-9)
    else:
        best = np.maximum.accumulate(obj) if maximizing else np.minimum.accumulate(obj)
    return obj, best, maximizing


def get_shape_image(results_dir, design_ref):
    p = os.path.join(results_dir, design_ref, "save", "shape.png")
    if os.path.exists(p):
        return Image.open(p)
    return None


def make_comparison(dirs, labels=None, max_iter=80, step=1, output_dir=None):
    assert len(dirs) >= 2, "Pass at least 2 result directories."
    labels = labels or [os.path.basename(d) for d in dirs]
    colors = (COLORS * ((len(dirs) // len(COLORS)) + 1))[:len(dirs)]

    all_rows  = [cap_rows(load_csv(d), max_iter) for d in dirs]

    if not any(all_rows):
        print("No rows to render.")
        return

    all_x     = [compute_plot_x(rows) for rows in all_rows]
    all_objs  = []
    all_bests = []
    maximizing = False
    for rows in all_rows:
        if rows:
            obj, best, mx = objective_from_rewards(rows)
            maximizing = maximizing or mx
        else:
            obj, best = np.array([]), np.array([])
        all_objs.append(obj)
        all_bests.append(best)

    combined = np.concatenate([o for o in all_objs if len(o)])
    p5, p95 = np.percentile(combined, 5), np.percentile(combined, 95)
    if maximizing:
        y_lo = min(np.min(combined) * 0.9, p5 * 0.9)
        y_hi = np.max(combined) * 1.1
    else:
        y_lo = max(min(np.min(combined) * 0.8, p5 * 0.8), 1e-9)
        y_hi = max(np.max(combined) * 1.2, p95 * 1.2)
    x_min = -0.1
    x_max = max((float(np.max(x)) if len(x) else 0.0) for x in all_x) + 0.1

    output_dir = output_dir or os.path.dirname(dirs[0])
    os.makedirs(output_dir, exist_ok=True)

    # Animate by iteration so that multi-particle frameworks (PSO: 30 rows/iter)
    # produce the same number of frames as single-evaluation frameworks.
    all_iters = sorted(set(
        r["iteration"] for rows in all_rows for r in rows
    ))
    frame_iters = all_iters[::max(1, step)]
    if frame_iters and frame_iters[-1] != all_iters[-1]:
        frame_iters.append(all_iters[-1])

    n_runs = len(dirs)
    plt.rcParams.update(STYLE)
    frames = []
    print(f"[comparison] Rendering {len(frame_iters)} frames ({n_runs} runs) ...")

    for fi, iter_k in enumerate(frame_iters):
        k_list = [sum(1 for r in rows if r["iteration"] <= iter_k) for rows in all_rows]

        fig = plt.figure(figsize=(14, max(5, 3 * n_runs)), facecolor="white")
        gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.06)
        gs_right = gs[1].subgridspec(n_runs, 1, hspace=0.12)

        # Left: overlaid objective plot.
        ax = fig.add_subplot(gs[0])
        best_vals = []
        for ri in range(n_runs):
            rows, x, obj, best = all_rows[ri], all_x[ri], all_objs[ri], all_bests[ri]
            color, label = colors[ri], labels[ri]
            k = k_list[ri]
            if k == 0:
                best_vals.append(np.nan)
                continue
            ax.scatter(x[:k], np.clip(obj[:k], y_lo, y_hi), s=12, alpha=0.5,
                       color=color, edgecolors="k", linewidths=0.1, zorder=2)
            ax.plot(x[:k], np.clip(best[:k], y_lo, y_hi), color=color, lw=1.5,
                    zorder=3, label=f"{label} best")
            ax.scatter([x[k - 1]], [np.clip(obj[k - 1], y_lo, y_hi)], s=80,
                       facecolors="none", edgecolors=color, linewidths=1.8,
                       zorder=5, label=f"{label} current")
            best_vals.append(float(best[k - 1]))

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_lo, y_hi)
        if not maximizing:
            ax.set_yscale("log")
        ax.set_xlabel("Evaluations")
        ax.set_ylabel("Reward (higher is better)" if maximizing else "Objective (-reward, lower is better)")
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95, ncol=2)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        best_str = "  vs  ".join(
            f"{l}={v:.4f}" for l, v in zip(labels, best_vals) if not np.isnan(v)
        )
        best_label = "best" if maximizing else "best obj"
        ax.set_title(f"Iteration {iter_k} — {best_label}: {best_str}", fontweight="medium", pad=8)

        # Right: one image panel per run, stacked vertically.
        for ri in range(n_runs):
            ax_img = fig.add_subplot(gs_right[ri])
            ax_img.axis("off")
            k = k_list[ri]
            if k > 0:
                best_row = max(all_rows[ri][:k], key=lambda r: r["reward"])
                img = get_shape_image(dirs[ri], best_row["design"])
                if img is not None:
                    ax_img.imshow(img)
                    ax_img.set_title(f"{labels[ri]} best ({best_row['design']})",
                                     fontsize=9, fontweight="medium", pad=4,
                                     color=colors[ri])
                else:
                    ax_img.text(0.5, 0.5, "(no shape.png)", ha="center", va="center",
                                fontsize=10, transform=ax_img.transAxes, color="#888")

        fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.08)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)

        if (fi + 1) % 20 == 0:
            print(f"  {fi + 1}/{len(frame_iters)} frames")

    for _ in range(8):
        frames.append(frames[-1])
    out = os.path.join(output_dir, "comparison_evolution.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=150, loop=0, optimize=True)
    print(f"  -> {out}  ({len(frames)} frames, {os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", help="Result directories (2 or more)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Display labels, one per directory")
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.dirs):
        parser.error(f"--labels count ({len(args.labels)}) must match dirs count ({len(args.dirs)})")

    make_comparison(
        args.dirs,
        labels=args.labels,
        max_iter=args.max_iter,
        step=args.step,
        output_dir=args.output_dir,
    )
