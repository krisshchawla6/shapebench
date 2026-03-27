#!/usr/bin/env python3
"""
Animated comparison for any number of NeuralFoil runs — x-axis in cumulative
model evaluations (one unit = one NeuralFoil call) rather than iteration number.

Each framework maps differently:
  islands / v2 / v2_batch : 1 eval per iteration  (1 LLM call / eval)
  v3_dynamic_optimizer    : ~30 evals per iteration (1 LLM call / ~30 evals)
  GA / PSO                : 30 evals per iteration (0 LLM calls)

LLM usage is auto-detected from CSV columns and shown in the legend.

Optional dashed horizontal reference line from an adjoint/IPOPT result.

Usage:
    python animate_neuralfoil_comparison_evals.py dir_a dir_b [dir_c ...] \\
        --labels "islands" "v3" "PSO" \\
        --max-iter 600 --step 2 \\
        --adjoint environments/NeuralFoil/results/adjoint_run \\
        --adjoint-label "Adjoint (IPOPT)"
"""

import json
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

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]


def load_csv(results_dir):
    csv_path = os.path.join(results_dir, "results.csv")
    rows = []
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        for i, row in enumerate(reader):
            it = int(row["iteration"])
            reward = float(row["reward"])
            design = row.get("design", "")
            if not design:
                particle = row.get("particle", "")
                if particle != "":
                    design = f"iter_{it:04d}_p{int(float(particle)):03d}"
                else:
                    design = f"design_{i}"
            br = row.get("gbest_reward") or row.get("best_reward")
            best_reward = float(br) if br else None
            rows.append({"iteration": it, "reward": reward, "design": design,
                         "best_reward": best_reward, "_cols": set(row.keys())})
    return rows


def cap_rows(rows, max_iter):
    if max_iter is None:
        return rows
    return [r for r in rows if r["iteration"] <= max_iter]


def detect_llm_annotation(rows):
    """Infer LLM-calls-per-eval from CSV columns.

    - PSO/GA:     has 'particle' column   → 0 LLM calls
    - v3:         has 'sample_type' col   → 1 LLM call per iteration (batch of N evals)
    - islands/v2: neither                 → 1 LLM call per eval
    """
    if not rows:
        return ""
    cols = rows[0]["_cols"]
    if "particle" in cols:
        return "0 LLM calls"
    elif "sample_type" in cols:
        n_iters = len(set(r["iteration"] for r in rows))
        batch = round(len(rows) / n_iters) if n_iters else 1
        return f"1 LLM / {batch} evals"
    else:
        return "1 LLM / eval"


def objective_from_rewards(rows):
    r = np.array([row["reward"] for row in rows], dtype=float)
    valid = r[r > -9.0]
    maximizing = len(valid) > 0 and np.median(valid) > 0
    if maximizing:
        obj = r.copy()
    else:
        obj = np.maximum(-r, 1e-9)
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


def load_adjoint_reference(adjoint_dir):
    path = os.path.join(adjoint_dir, "save", "results.json")
    if not os.path.exists(path):
        print(f"[adjoint] results.json not found at {path}")
        return None, None
    with open(path) as f:
        d = json.load(f)
    weighted_cd = d.get("weighted_cd") or d.get("weighted_CD_mean")
    feasible = d.get("feasible", None)
    if weighted_cd is None:
        print(f"[adjoint] weighted_cd not found in {path}")
        return None, None
    return float(weighted_cd), feasible


def make_comparison(dirs, labels=None, max_iter=80, step=1, output_dir=None,
                    adjoint_dir=None, adjoint_label="Adjoint (IPOPT)"):
    assert len(dirs) >= 2, "Pass at least 2 result directories."
    labels = labels or [os.path.basename(d) for d in dirs]
    colors = (COLORS * ((len(dirs) // len(COLORS)) + 1))[:len(dirs)]

    all_rows = [cap_rows(load_csv(d), max_iter) for d in dirs]

    if not any(all_rows):
        print("No rows to render.")
        return

    # x-axis: cumulative model evaluation index (row position in each CSV)
    all_x = [np.arange(len(rows), dtype=float) for rows in all_rows]

    # Auto-detect LLM usage and annotate labels for legend
    annotated_labels = []
    for rows, label in zip(all_rows, labels):
        ann = detect_llm_annotation(rows)
        annotated_labels.append(f"{label} ({ann})" if ann else label)

    all_objs = []
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

    adjoint_ref = None
    if adjoint_dir is not None:
        weighted_cd, adj_feasible = load_adjoint_reference(adjoint_dir)
        if weighted_cd is not None:
            adjoint_ref = -weighted_cd if maximizing else weighted_cd
            feas_str = " (feasible)" if adj_feasible else " (infeasible)"
            print(f"[adjoint] reference={adjoint_ref:.6f}{feas_str}")

    combined = np.concatenate([o for o in all_objs if len(o)])
    p5, p95 = np.percentile(combined, 5), np.percentile(combined, 95)
    if maximizing:
        y_lo = min(np.min(combined) * 0.9, p5 * 0.9)
        y_hi = np.max(combined) * 1.1
        if adjoint_ref is not None:
            y_hi = max(y_hi, adjoint_ref * 1.05)
    else:
        y_lo = max(min(np.min(combined) * 0.8, p5 * 0.8), 1e-9)
        y_hi = max(np.max(combined) * 1.2, p95 * 1.2)
        if adjoint_ref is not None:
            y_lo = min(y_lo, adjoint_ref * 0.95)

    x_min = -5
    x_max = max((float(np.max(x)) if len(x) else 0.0) for x in all_x) * 1.02

    output_dir = output_dir or os.path.dirname(dirs[0])
    os.makedirs(output_dir, exist_ok=True)

    # Per-run frame sequences: resample each run's iteration list to the same
    # number of frames so all runs start and finish at the same frame,
    # each advancing at a pace proportional to its own iteration count.
    per_run_iters = [
        sorted(set(r["iteration"] for r in rows)) for rows in all_rows
    ]
    # Total frames driven by the run with the most unique iterations.
    total_frames = max(len(iters) for iters in per_run_iters) // max(1, step)
    total_frames = max(total_frames, 2)

    # For each run resample its iteration list to total_frames points, always
    # including the last iteration so the final state is shown.
    per_run_frame_iters = []
    for iters in per_run_iters:
        indices = np.linspace(0, len(iters) - 1, total_frames, dtype=int)
        per_run_frame_iters.append([iters[i] for i in indices])

    n_runs = len(dirs)
    plt.rcParams.update(STYLE)
    frames = []
    print(f"[comparison] Rendering {total_frames} frames ({n_runs} runs, per-run steps) ...")

    for fi in range(total_frames):
        iter_k_list = [per_run_frame_iters[ri][fi] for ri in range(n_runs)]
        # k = number of rows (evals) up to and including each run's current iter
        k_list = [sum(1 for r in rows if r["iteration"] <= iter_k_list[ri])
                  for ri, rows in enumerate(all_rows)]

        fig = plt.figure(figsize=(14, max(5, 3 * n_runs)), facecolor="white")
        gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.06)
        gs_right = gs[1].subgridspec(n_runs, 1, hspace=0.12)

        ax = fig.add_subplot(gs[0])
        best_vals = []
        for ri in range(n_runs):
            rows, x, obj, best = all_rows[ri], all_x[ri], all_objs[ri], all_bests[ri]
            color = colors[ri]
            alabel = annotated_labels[ri]
            k = k_list[ri]
            if k == 0:
                best_vals.append(np.nan)
                continue
            ax.scatter(x[:k], np.clip(obj[:k], y_lo, y_hi), s=12, alpha=0.5,
                       color=color, edgecolors="k", linewidths=0.1, zorder=2)
            ax.plot(x[:k], np.clip(best[:k], y_lo, y_hi), color=color, lw=1.5,
                    zorder=3, label=f"{alabel} best")
            ax.scatter([x[k - 1]], [np.clip(obj[k - 1], y_lo, y_hi)], s=80,
                       facecolors="none", edgecolors=color, linewidths=1.8,
                       zorder=5, label=f"{labels[ri]} current")
            best_vals.append(float(best[k - 1]))

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_lo, y_hi)
        if not maximizing:
            ax.set_yscale("log")
        ax.set_xlabel("Model (Airfoil Design) Evaluations")
        ax.set_ylabel("Reward (higher is better)" if maximizing
                      else "Objective (−reward, lower is better)")

        if adjoint_ref is not None:
            ax.axhline(adjoint_ref, color="#333333", lw=1.2, ls="--", zorder=4,
                       label=f"{adjoint_label} ({adjoint_ref:.4f})")

        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95, ncol=2)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

        evals_str = "  |  ".join(
            f"{labels[ri]}={k_list[ri]} evals" for ri in range(n_runs)
        )
        best_str = "  vs  ".join(
            f"{labels[ri]}={v:.4f}" for ri, v in enumerate(best_vals) if not np.isnan(v)
        )
        best_label = "best" if maximizing else "best obj"
        iters_str = "  |  ".join(
            f"{labels[ri]} iter {iter_k_list[ri]}" for ri in range(n_runs)
        )
        ax.set_title(
            f"[{iters_str}]\n{best_label}: {best_str}",
            fontweight="medium", pad=8, fontsize=9,
        )

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

        fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.08)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)

        if (fi + 1) % 20 == 0:
            print(f"  {fi + 1}/{total_frames} frames")

    for _ in range(8):
        frames.append(frames[-1])
    out = os.path.join(output_dir, "comparison_evolution_evals.gif")
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
    parser.add_argument("--adjoint", default=None, metavar="DIR",
                        help="Optional adjoint result directory; draws a dashed reference line")
    parser.add_argument("--adjoint-label", default="Adjoint (IPOPT)",
                        help="Label for the adjoint reference line")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.dirs):
        parser.error(f"--labels count ({len(args.labels)}) must match dirs count ({len(args.dirs)})")

    make_comparison(
        args.dirs,
        labels=args.labels,
        max_iter=args.max_iter,
        step=args.step,
        output_dir=args.output_dir,
        adjoint_dir=args.adjoint,
        adjoint_label=args.adjoint_label,
    )
