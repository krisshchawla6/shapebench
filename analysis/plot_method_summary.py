#!/usr/bin/env python3
"""
Plot a method summary across multiple runs (seeds) of the same method:
  1. Mean best trajectory (objective vs. model evaluations) with min/max band.
  2. Airfoil outline of the best design found across all runs.
  3. Saves trajectory data to a CSV for future combined-method plots.

The saved CSV has columns:
    eval, mean_best, min_best, max_best
so it can be loaded alongside CSVs from other methods in a combined plot.

Usage:
    python analysis/plot_method_summary.py \\
        environments/NeuralFoil/results/run_v3_attempt_2/ \\
        environments/NeuralFoil/results/run_v3_attempt_3/ \\
        environments/NeuralFoil/results/run_v3_attempt_4/ \\
        --method-label "v3 (LLM)" \\
        --output-dir environments/NeuralFoil/results/summary_v3/ \\
        --adjoint environments/NeuralFoil/results/adjoint_run/ \\
        --max-evals 6000
"""

import argparse
import csv as csv_mod
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import aerosandbox as asb


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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(results_dir, max_evals=None):
    """Load results.csv; return list of dicts with eval index and best_reward."""
    csv_path = os.path.join(results_dir, "results.csv")
    rows = []
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        for i, row in enumerate(reader):
            if max_evals is not None and i >= max_evals:
                break
            it = int(row.get("iteration") or row.get("call") or i)
            reward = float(row["reward"])
            design = row.get("design", "")
            if not design:
                particle = row.get("particle", "")
                restart  = row.get("restart", "")
                call_col = row.get("call", "")
                if particle != "":
                    design = f"iter_{it:04d}_p{int(float(particle)):03d}"
                elif call_col != "" and restart != "":
                    design = f"call_{int(float(call_col)):05d}_r{int(float(restart))}"
                else:
                    design = f"design_{i}"
            br = row.get("gbest_reward") or row.get("best_reward")
            best_reward = float(br) if br else None
            fp = row.get("fitness_penalty", "")
            ft = row.get("fitness_total", "")
            rows.append({
                "eval": i,
                "iteration": it,
                "reward": reward,
                "design": design,
                "best_reward": best_reward,
                "fitness_penalty": float(fp) if fp != "" else None,
                "fitness_total":   float(ft) if ft != "" else None,
            })
    return rows


def best_trajectory(rows):
    """Return (eval_indices, best_so_far) arrays.

    Uses best_reward column if available (already the running best);
    otherwise recomputes it with a running max.
    """
    evals = np.array([r["eval"] for r in rows], dtype=float)
    if rows[0]["best_reward"] is not None:
        best = np.array([r["best_reward"] for r in rows], dtype=float)
    else:
        rewards = np.array([r["reward"] for r in rows], dtype=float)
        best = np.maximum.accumulate(rewards)
    return evals, best


def best_fitness_total_trajectory(rows):
    """Return (eval_indices, best_fitness_total_so_far) arrays using fitness_total column.

    fitness_total = fitness_objective + fitness_penalty; running max gives the
    best combined objective+penalty seen so far.  Returns (None, None) if absent.
    """
    if rows[0]["fitness_total"] is None:
        return None, None
    evals = np.array([r["eval"] for r in rows], dtype=float)
    # Treat rows with fitness_total==0 and reward==-10 as total evaluation
    # failures (n_solved=0, n_unreachable=0): the aero solver returned nothing,
    # so both fitness_objective and fitness_penalty are 0 by default, giving a
    # spuriously high fitness_total=0 that would permanently dominate the
    # running max over all real designs (which have fitness_total < 0).
    totals = np.array(
        [(-np.inf if (r["fitness_total"] == 0.0 and r["reward"] == -10.0) else r["fitness_total"])
         if r["fitness_total"] is not None else -np.inf
         for r in rows],
        dtype=float,
    )
    best = np.maximum.accumulate(totals)
    return evals, best


def is_minimizing(rows):
    """Return True if the objective is to minimise (reward is negative, e.g. weighted CD)."""
    rewards = np.array([r["reward"] for r in rows if r["reward"] > -9.0], dtype=float)
    if len(rewards) == 0:
        return True
    return np.median(rewards) < 0


# ---------------------------------------------------------------------------
# Adjoint reference
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Airfoil plotting helper
# ---------------------------------------------------------------------------

def plot_best_airfoil(ax, dirs, all_rows, minimizing, output_dir=None):
    """Find the best design across all runs and plot its airfoil outline.

    Reward is always higher=better by framework convention, regardless of
    whether the underlying objective is minimised or maximised.
    """
    best_val = -np.inf
    best_dir = None
    best_design = None

    for d, rows in zip(dirs, all_rows):
        if not rows:
            continue
        # Best design = highest reward across all evaluations in this run
        best_row = max(rows, key=lambda r: r["reward"])
        if best_row["reward"] > best_val:
            best_val = best_row["reward"]
            best_dir = d
            best_design = best_row["design"]

    if best_dir is None or best_design is None:
        ax.text(0.5, 0.5, "(no best design found)", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888")
        ax.axis("off")
        return

    # Always plot from Kulfan params for a clean sharp TE.
    # Pre-rendered shape.png images have a blunt TE artifact from coordinate
    # discretisation; plotting directly lets us explicitly close at (1, 0).
    # PSO: params in design.json at the design dir root.
    # v3:  params in save/results.json under the "design" key.
    design_json = os.path.join(best_dir, best_design, "design.json")
    results_json = os.path.join(best_dir, best_design, "save", "results.json")

    if os.path.exists(design_json):
        with open(design_json) as f:
            params = json.load(f)
    elif os.path.exists(results_json):
        with open(results_json) as f:
            params = json.load(f)["design"]
    else:
        # Fall back to any pre-saved best_*.json in the output directory whose
        # filename contains the design name (e.g. best_flash2_5_attempt22_iter182_o13.json).
        params = None
        if output_dir is not None:
            for fname in sorted(os.listdir(output_dir)):
                if fname.endswith(".json"):
                    with open(os.path.join(output_dir, fname)) as f:
                        d = json.load(f)
                    if "design" in d:
                        params = d["design"]
                        break
        if params is None:
            ax.text(0.5, 0.5, "(design params not found)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#888")
            ax.axis("off")
            return

    airfoil = asb.KulfanAirfoil(
        upper_weights=np.array(params["upper_weights"], dtype=float),
        lower_weights=np.array(params["lower_weights"], dtype=float),
        leading_edge_weight=float(params["leading_edge_weight"]),
        TE_thickness=0.0,
    )
    coords = np.array(airfoil.coordinates)
    # Force closure at the sharp trailing edge (1, 0)
    coords = np.vstack([coords, [[1.0, 0.0]]])
    ax.plot(coords[:, 0], coords[:, 1], "b-", linewidth=1.5)
    ax.fill(coords[:, 0], coords[:, 1], alpha=0.15, color="steelblue")
    ax.set_aspect("equal")
    ax.set_xlabel("x/c", fontsize=9)
    ax.set_ylabel("y/c", fontsize=9)

    # Read aero results for title annotations
    title_lines = [f"Best: {best_design}  (reward={best_val:.5f})"]
    aero_path = os.path.join(best_dir, best_design, "save", "results.json")
    if os.path.exists(aero_path):
        with open(aero_path) as f:
            aero = json.load(f)
        wcd = aero.get("weighted_CD_mean")
        feasible = aero.get("feasible")
        alphas = aero.get("alphas")
        if wcd is not None:
            feas_str = "feasible" if feasible else "infeasible"
            title_lines.append(f"weighted CD = {wcd:.5f}  ({feas_str})")
        if alphas:
            title_lines.append(
                f"α = {alphas[0]:.2f}° … {alphas[-1]:.2f}°  ({len(alphas)} CL targets)"
            )
    ax.set_title("\n".join(title_lines), fontsize=8.5, fontweight="medium", pad=4)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_summary(dirs, method_label="method", output_dir=None,
                 adjoint_dir=None, adjoint_label="Adjoint (IPOPT)",
                 max_evals=None, n_eval_points=500, color="#1f77b4",
                 output_name="summary", failed_dirs=None):

    output_dir = output_dir or os.path.join(os.path.dirname(dirs[0]), "summary")
    os.makedirs(output_dir, exist_ok=True)

    all_rows = [load_csv(d, max_evals) for d in dirs]
    all_rows = [r for r in all_rows if r]  # drop empty

    if not all_rows:
        print("No data found.")
        return

    minimizing = is_minimizing(all_rows[0])

    # Build a common eval grid up to the longest run. Shorter runs (including
    # failed/partial runs) are forward-filled at their final best value by
    # np.interp, which clamps to the last data point beyond the run's end.
    max_len = max(len(rows) for rows in all_rows)
    eval_grid = np.linspace(0, max_len - 1, n_eval_points)

    # Interpolate each run's best trajectory onto the common grid.
    interp_bests = []
    for rows in all_rows:
        evals, best = best_trajectory(rows)
        # Convert to objective (positive = better for plotting)
        obj_best = -best if minimizing else best
        interp = np.interp(eval_grid, evals, obj_best)
        interp_bests.append(interp)

    interp_bests = np.array(interp_bests)  # shape: (n_runs, n_eval_points)
    run_lengths  = np.array([len(rows) for rows in all_rows])

    # Compute statistics using only runs with real data at each eval point.
    # Runs that have ended are excluded (not forward-filled) from the stats.
    n_points    = len(eval_grid)
    mean_best   = np.full(n_points, np.nan)
    median_best = np.full(n_points, np.nan)
    min_best    = np.full(n_points, np.nan)
    max_best    = np.full(n_points, np.nan)
    n_active    = np.zeros(n_points, dtype=int)

    for i, ev in enumerate(eval_grid):
        active = interp_bests[run_lengths - 1 >= ev, i]
        if len(active) > 0:
            mean_best[i]   = np.mean(active)
            median_best[i] = np.median(active)
            min_best[i]    = np.min(active)
            max_best[i]    = np.max(active)
            n_active[i]    = len(active)

    # Identify failed runs by explicit --failed-dirs list (matched by abspath).
    # Only these runs get the dashed-line / × marker treatment.
    failed_set = set(os.path.abspath(d) for d in (failed_dirs or []))
    is_failed  = [os.path.abspath(d) in failed_set for d in dirs]

    failed_lengths = [int(run_lengths[i]) for i, f in enumerate(is_failed) if f]
    short_ends     = sorted([l - 1 for l in failed_lengths])
    n_short        = len(short_ends)
    n_complete     = len(all_rows) - n_short

    # Split index: first grid point where a failed run ends
    if short_ends:
        split_idx = int(np.searchsorted(eval_grid, short_ends[0]))
        split_idx = max(1, min(split_idx, n_points - 1))
    else:
        split_idx = n_points

    # Adjoint reference
    adjoint_ref = None
    if adjoint_dir is not None:
        weighted_cd, adj_feasible = load_adjoint_reference(adjoint_dir)
        if weighted_cd is not None:
            adjoint_ref = weighted_cd if minimizing else -weighted_cd
            feas_str = " (feasible)" if adj_feasible else " (infeasible)"
            print(f"[adjoint] reference={adjoint_ref:.6f}{feas_str}")

    # -----------------------------------------------------------------------
    # Save trajectory CSV (only real data — NaN where all runs ended)
    # -----------------------------------------------------------------------
    csv_out = os.path.join(output_dir, f"{output_name}_trajectory.csv")
    with open(csv_out, "w", newline="") as f:
        writer = csv_mod.writer(f)
        writer.writerow(["eval", "n_active", "mean_best", "median_best", "min_best", "max_best"])
        for i, ev in enumerate(eval_grid):
            def fmt(v):
                return f"{v:.8f}" if not np.isnan(v) else ""
            writer.writerow([
                f"{ev:.1f}", n_active[i],
                fmt(mean_best[i]), fmt(median_best[i]),
                fmt(min_best[i]),  fmt(max_best[i]),
            ])
    print(f"[summary] Trajectory CSV -> {csv_out}")

    # -----------------------------------------------------------------------
    # Save fitness_total CSV (objective space = -fitness_total, apples-to-apples
    # with BO/L-BFGS-B reward).  Only written if fitness_total column is present.
    # -----------------------------------------------------------------------
    ft_trajs = []
    for rows in all_rows:
        et, bt = best_fitness_total_trajectory(rows)
        if et is not None:
            # Convert to objective space (positive, lower is better)
            ft_trajs.append(np.interp(eval_grid, et, -bt))

    if ft_trajs:
        ft_arr      = np.array(ft_trajs)   # (n_runs, n_points)
        ft_mean     = np.full(n_points, np.nan)
        ft_median   = np.full(n_points, np.nan)
        ft_min      = np.full(n_points, np.nan)
        ft_max      = np.full(n_points, np.nan)
        ft_n_active = np.zeros(n_points, dtype=int)
        for i, ev in enumerate(eval_grid):
            active = ft_arr[run_lengths - 1 >= ev, i]
            if len(active) > 0:
                ft_mean[i]     = np.mean(active)
                ft_median[i]   = np.median(active)
                ft_min[i]      = np.min(active)
                ft_max[i]      = np.max(active)
                ft_n_active[i] = len(active)

        ft_csv_out = os.path.join(output_dir, f"{output_name}_trajectory_fitness_total.csv")
        with open(ft_csv_out, "w", newline="") as f:
            writer = csv_mod.writer(f)
            writer.writerow(["eval", "n_active", "mean_best", "median_best", "min_best", "max_best"])
            for i, ev in enumerate(eval_grid):
                def fmt(v):
                    return f"{v:.8f}" if not np.isnan(v) else ""
                writer.writerow([
                    f"{ev:.1f}", ft_n_active[i],
                    fmt(ft_mean[i]), fmt(ft_median[i]),
                    fmt(ft_min[i]),  fmt(ft_max[i]),
                ])
        print(f"[summary] Fitness-total CSV -> {ft_csv_out}")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor="white",
                             gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.08})
    ax = axes[0]

    ylabel = ("Objective (−reward, lower is better)" if minimizing
              else "Reward (higher is better)")

    # Clip to 1e-9 to keep log scale valid; mask NaN grid points
    valid   = ~np.isnan(median_best)
    ev_v    = eval_grid[valid]
    med_v   = np.maximum(median_best[valid], 1e-9)
    min_v   = np.maximum(min_best[valid],    1e-9)
    max_v   = np.maximum(max_best[valid],    1e-9)

    # Y-limits
    all_vals = np.concatenate([med_v, min_v, max_v])
    if minimizing:
        y_lo = max(np.min(all_vals) * 0.8, 1e-9)
        y_hi = np.max(all_vals) * 1.2
        if adjoint_ref is not None:
            y_lo = min(y_lo, adjoint_ref * 0.95)
    else:
        y_lo = np.min(all_vals) * 0.9
        y_hi = np.max(all_vals) * 1.1
        if adjoint_ref is not None:
            y_hi = max(y_hi, adjoint_ref * 1.05)

    # Solid region — all runs active
    ax.fill_between(ev_v[ev_v <= eval_grid[split_idx - 1]],
                    min_v[ev_v <= eval_grid[split_idx - 1]],
                    max_v[ev_v <= eval_grid[split_idx - 1]],
                    color=color, alpha=0.2)
    ax.plot(ev_v[ev_v <= eval_grid[split_idx - 1]],
            med_v[ev_v <= eval_grid[split_idx - 1]],
            color=color, lw=2.0)

    # Dashed region — some runs have ended (fewer active)
    if split_idx < n_points:
        mask_dash = ev_v >= eval_grid[split_idx - 1]
        ax.fill_between(ev_v[mask_dash], min_v[mask_dash], max_v[mask_dash],
                        color=color, alpha=0.10)
        ax.plot(ev_v[mask_dash], med_v[mask_dash],
                color=color, lw=2.0, linestyle="--")

    # × markers at each short run's end point on the median line
    for end_eval in short_ends:
        med_at_end = float(np.interp(end_eval, ev_v, med_v))
        ax.plot(end_eval, med_at_end, marker="x", color="#333333",
                markersize=8, markeredgewidth=1.5, zorder=5)

    # Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_entries = [
        Patch(facecolor=color, alpha=0.2,
              label=f"Min–max range ({n_complete} complete runs)"),
        Line2D([0], [0], color=color, lw=2.0,
               label=f"Median best ({len(all_rows)} runs)"),
    ]
    if n_short > 0:
        legend_entries += [
            Line2D([0], [0], color=color, lw=2.0, linestyle="--",
                   label=f"Fewer active runs ({n_short} stopped early)"),
            Line2D([0], [0], marker="x", color="#333333", lw=0,
                   markersize=8, markeredgewidth=1.5, label="Run stopped"),
        ]
    if adjoint_ref is not None:
        ax.axhline(adjoint_ref, color="#333333", lw=1.2, ls="--", zorder=4)
        legend_entries.append(
            Line2D([0], [0], color="#333333", lw=1.2, linestyle="--",
                   label=f"{adjoint_label} (reward = {-adjoint_ref:.4f})")
        )
    ax.legend(handles=legend_entries, loc="upper right", framealpha=0.95)

    if minimizing:
        ax.set_yscale("log")
    ax.set_xlim(-5, eval_grid[-1] * 1.02)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Model (Airfoil Design) Evaluations")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{method_label}  —  {len(all_rows)} runs,  "
                 f"up to {int(eval_grid[-1])} evals each",
                 fontweight="medium", pad=8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    # Airfoil panel
    plot_best_airfoil(axes[1], dirs, all_rows, minimizing, output_dir=output_dir)

    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.12)
    plot_out = os.path.join(output_dir, f"{output_name}.png")
    fig.savefig(plot_out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[summary] Plot -> {plot_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dirs", nargs="+",
                        help="Result directories — one per run/seed of the same method")
    parser.add_argument("--method-label", default="method",
                        help="Label for this method in plots and filenames")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save summary.png and trajectory.csv")
    parser.add_argument("--adjoint", default=None, metavar="DIR",
                        help="Adjoint result directory for reference line")
    parser.add_argument("--adjoint-label", default="Adjoint (IPOPT)")
    parser.add_argument("--max-evals", type=int, default=None,
                        help="Cap evaluation count per run")
    parser.add_argument("--n-eval-points", type=int, default=500,
                        help="Number of points on the common eval grid (default: 500)")
    parser.add_argument("--color", default="#1f77b4",
                        help="Line/fill colour for this method (hex, default: blue)")
    parser.add_argument("--output-name", default="summary",
                        help="Base filename for outputs (default: summary → summary.png, summary_trajectory.csv)")
    parser.add_argument("--failed-dirs", nargs="*", default=[],
                        help="Subset of dirs that are failed/crashed runs (get dashed line + × marker)")
    args = parser.parse_args()

    make_summary(
        dirs=args.dirs,
        method_label=args.method_label,
        output_dir=args.output_dir,
        adjoint_dir=args.adjoint,
        adjoint_label=args.adjoint_label,
        max_evals=args.max_evals,
        n_eval_points=args.n_eval_points,
        color=args.color,
        output_name=args.output_name,
        failed_dirs=args.failed_dirs,
    )
