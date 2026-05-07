"""
generate_convergence_plots.py

For every (env, reward, method) in /scratch/ShapeEvolve/best_rewards/, discover all
seed/attempt run directories, read their best-so-far trajectories, and produce
convergence plots matching the standardised LAM500 / SuperWing analysis format:

  • Serif font, tick direction "in"
  • Log x-axis, geomspace grid
  • Median + min/max band per method
  • Two-legend system (method colours + style key)
  • No top/right spines, facecolor="white"

Output: /scratch/ShapeEvolve/best_rewards_plots/{env}/{reward}.png
"""

import csv as csv_mod
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

BEST_REWARDS = "/scratch/ShapeEvolve/best_rewards"
OUT_BASE     = "/scratch/ShapeEvolve/best_rewards_plots"

SKIP_ENTRIES = {
    "analysis", "reward.py", ".DS_Store", "__pycache__",
    "collect_best.py", "collect_best_bn.py", "collect_best_ceras.py",
    "collect_best_drivaer.py", "collect_best_ds.py", "collect_best_mi.py",
    "collect_best_neuralfoil.py", "collect_best_neuralfoil_benchmark700.py",
    "collect_best_vn.py", "analyze_rankings.py", "analyze_standard_stats.py",
}

# ── Style ─────────────────────────────────────────────────────────────────────
STYLE = {
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset":  "cm",
    "font.size":         9,
    "axes.labelsize":    11,
    "axes.titlesize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   8.5,
    "axes.linewidth":    0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "figure.dpi":        150,
}

# ── Method metadata — colour + linestyle ────────────────────────────────────
# Canonical key → (display label, hex colour, linestyle)
FW_META = {
    "lbfgsb":               dict(label="L-BFGS-B",             color="#e377c2", ls="-"),
    "cmaes":                dict(label="CMA-ES",                color="#d62728", ls="-"),
    "GA":                   dict(label="PSO / GA",              color="#1f77b4", ls="-"),
    "GA_parallel":          dict(label="PSO (parallel)",        color="#17becf", ls="-"),
    "GAp":                  dict(label="PSO / GA",              color="#1f77b4", ls="-"),
    "BO_torch":             dict(label="Bayesian Opt.",         color="#ff7f0e", ls="-"),
    "BO":                   dict(label="Bayesian Opt.",         color="#ff7f0e", ls="-"),
    "v3_dynamic_optimizer": dict(label="v3 (LLM+local)",       color="#2ca02c", ls="--"),
    "v3_flash2_5":          dict(label="v3 flash2.5",          color="#bcbd22", ls="--"),
    "v3":                   dict(label="v3 (LLM+local)",       color="#2ca02c", ls="--"),
    "openevolve_adapter":   dict(label="OpenEvolve",           color="#9467bd", ls="--"),
    "openevolve":           dict(label="OpenEvolve",           color="#9467bd", ls="--"),
    "shinka_adapter":       dict(label="Shinka",               color="#8c564b", ls="--"),
    "shinka":               dict(label="Shinka",               color="#8c564b", ls="--"),
}
_DEFAULT_FW = dict(label=None, color="#7f7f7f", ls="-")

# Longest-prefix-first list for method-name canonicalisation
_METHOD_PREFIXES = [
    "lbfgsb", "cmaes",
    "GA_parallel", "GA", "GAp",
    "BO_torch", "BO",
    "v3_dynamic_optimizer", "v3_flash2_5", "v3",
    "openevolve_adapter", "openevolve",
    "shinka_adapter", "shinka",
]


def canonical_method(name: str) -> str:
    for p in _METHOD_PREFIXES:
        if name == p or (name.startswith(p) and name[len(p):len(p)+1] in ("", "_", "-")):
            return p
    return name


# ── Trajectory loading ────────────────────────────────────────────────────────

def _running_max(arr: np.ndarray) -> np.ndarray:
    """
    Forward-fill running maximum, ignoring NaN/fail sentinels.
    Leading NaN values (before the first valid evaluation) are filled with
    the first valid value so that interp_to_grid and np.maximum.accumulate
    remain well-defined — matches the SuperWing analysis script convention.
    """
    out = np.full(len(arr), np.nan)
    cur = np.nan
    for i, v in enumerate(arr):
        if not np.isnan(v):
            cur = v if np.isnan(cur) else max(cur, v)
        out[i] = cur
    # Forward-fill leading NaNs with the first valid value
    first = next((v for v in out if not np.isnan(v)), None)
    if first is not None:
        out = np.where(np.isnan(out), first, out)
    return out


def load_csv_traj(csv_path: str):
    """
    Return 1-D running-max array from results.csv (one entry per row).
    GA uses 'gbest_reward'; all others use 'best_reward'; fallback to 'reward'.
    Returns None if file is missing / empty / all failed.
    """
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if df.empty:
        return None

    # Choose best-reward column
    if "gbest_reward" in df.columns and "best_reward" not in df.columns:
        col = "gbest_reward"
    elif "best_reward" in df.columns:
        col = "best_reward"
    elif "reward" in df.columns:
        col = "reward"
    else:
        return None

    arr = df[col].values.astype(float)
    arr[~np.isfinite(arr)] = np.nan  # only remove inf/nan

    best = _running_max(arr)
    if np.all(np.isnan(best)):
        return None
    return best


def load_oe_traj(run_dir: str):
    """
    Return 1-D running-max array from oe_run/evolution_trace.jsonl.
    Returns None if file missing or all evaluations failed.
    """
    trace = os.path.join(run_dir, "oe_run", "evolution_trace.jsonl")
    if not os.path.exists(trace):
        return None
    rows = []
    with open(trace) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sc = obj.get("child_metrics", {}).get("combined_score")
            rows.append(sc)
    if not rows:
        return None
    arr = np.array([np.nan if (s is None or not np.isfinite(float(s) if s is not None else np.nan))
                    else float(s) for s in rows])
    best = _running_max(arr)
    if np.all(np.isnan(best)):
        return None
    return best


def load_seed_traj(run_dir: str):
    """Auto-detect format and return running-max trajectory array (or None)."""
    if not run_dir or not os.path.isdir(run_dir):
        return None
    # OpenEvolve JSONL (preferred for OE runs — more granular than sparse CSV checkpoints)
    oe = load_oe_traj(run_dir)
    if oe is not None:
        return oe
    # Standard CSV
    return load_csv_traj(os.path.join(run_dir, "results.csv"))


# ── Source / seed discovery (unchanged from previous version) ─────────────────

def _climb_to_seed(path: str):
    p = path.rstrip("/")
    for _ in range(10):
        p = os.path.dirname(p)
        if not p or p == "/":
            break
        base = os.path.basename(p)
        if re.match(r'^seed\d+$', base) or re.match(r'^.+_s\d+$', base):
            return p
    return None


def get_source_dir(entry_dir: str):
    fp = os.path.join(entry_dir, "best_result.json")
    if os.path.exists(fp):
        try:
            with open(fp) as f:
                data = json.load(f)
        except Exception:
            data = {}
        source = data.get("source", {})
        if isinstance(source, dict):
            for key in ("seed_dir", "run_dir"):
                d = source.get(key)
                if d and os.path.isdir(d):
                    return d
            d = source.get("sample_dir")
            if d and os.path.isdir(d):
                # If sample_dir itself is a run dir (has results.csv / oe_run),
                # return it directly (e.g. Superwing).
                if (os.path.exists(os.path.join(d, "results.csv"))
                        or os.path.isdir(os.path.join(d, "oe_run"))):
                    return d
                climbed = _climb_to_seed(d)
                if climbed and os.path.isdir(climbed):
                    return climbed
                parent = os.path.dirname(d)
                if os.path.isdir(parent):
                    return parent
        d = data.get("source_dir")
        if d and os.path.isdir(d):
            return d

    fp = os.path.join(entry_dir, "source_info.json")
    if os.path.exists(fp):
        try:
            with open(fp) as f:
                data = json.load(f)
        except Exception:
            data = {}
        d = data.get("results_csv")
        if d:
            parent = os.path.dirname(d)
            if os.path.isdir(parent):
                return parent
        d = data.get("design_dir")
        if d:
            climbed = _climb_to_seed(d)
            if climbed and os.path.isdir(climbed):
                return climbed
        d = data.get("sample_dir")
        if d:
            climbed = _climb_to_seed(d)
            if climbed and os.path.isdir(climbed):
                return climbed
    return None


def find_seed_dirs(seed_dir: str):
    seed_dir = seed_dir.rstrip("/")
    parent   = os.path.dirname(seed_dir)
    name     = os.path.basename(seed_dir)
    if not parent or not name or not os.path.isdir(seed_dir):
        return [seed_dir] if os.path.isdir(seed_dir) else []

    def _glob(pat):
        return sorted(p for p in glob.glob(pat) if os.path.isdir(p))

    m = re.match(r'^(.*?seed)\d+$', name)
    if m:
        return _glob(os.path.join(parent, m.group(1) + '*')) or [seed_dir]

    m = re.search(r'(.*seed)\d+', name)
    if m:
        return _glob(os.path.join(parent, m.group(1) + '*')) or [seed_dir]

    m = re.search(r'attempt_\d+', name)
    if m:
        prefix = name[:m.start()]
        return _glob(os.path.join(parent, prefix + 'attempt_*')) or [seed_dir]

    m = re.match(r'^(.*_s)\d+$', name)
    if m:
        return _glob(os.path.join(parent, m.group(1) + '*')) or [seed_dir]

    return [seed_dir]


def get_all_seed_dirs(method_dir: str):
    seed_subdirs = sorted(
        s for s in glob.glob(os.path.join(method_dir, "seed*"))
        if os.path.isdir(s)
    )
    if seed_subdirs:
        run_dirs, seen = [], set()
        for sd in seed_subdirs:
            src = get_source_dir(sd)
            if src and src not in seen:
                run_dirs.append(src)
                seen.add(src)
        if run_dirs:
            return run_dirs
        return seed_subdirs

    src = get_source_dir(method_dir)
    if not src:
        return []
    return find_seed_dirs(src)


# ── Band computation (median + min/max) ────────────────────────────────────────

def interp_to_grid(curve: np.ndarray, x_out: np.ndarray) -> np.ndarray:
    """Step-interpolate curve (index 1..n) onto x_out; extend flat past run end."""
    n = len(curve)
    if n == 0:
        return np.full(len(x_out), np.nan)
    idx = np.clip(
        np.searchsorted(np.arange(1, n + 1), x_out, side="right") - 1, 0, n - 1
    )
    out = curve[idx].copy()
    out[x_out < 1] = np.nan
    return out


def compute_band(arrays, x_grid):
    """Monotone median + min/max band across seed arrays on x_grid."""
    mat = np.vstack([interp_to_grid(a, x_grid) for a in arrays])
    # Make each row monotone (running max across columns, in case of flat extend)
    mat = np.maximum.accumulate(mat, axis=1)
    n_valid = np.sum(~np.isnan(mat), axis=0)
    mask = n_valid >= max(1, len(arrays) // 2)
    med = np.where(mask, np.maximum.accumulate(np.nanmedian(mat, axis=0)), np.nan)
    lo  = np.where(mask, np.nanmin(mat, axis=0),  np.nan)
    hi  = np.where(mask, np.nanmax(mat, axis=0),  np.nan)
    return med, lo, hi


def plot_band(ax, arrays, x_grid, color, label, ls="-"):
    med, lo, hi = compute_band(arrays, x_grid)
    valid = ~np.isnan(med)
    if not valid.any():
        return False
    ax.fill_between(x_grid[valid], lo[valid], hi[valid], color=color, alpha=0.18)
    ax.plot(x_grid[valid], med[valid], color=color, lw=1.8, label=label, ls=ls)
    return True


# ── Per-reward plot ───────────────────────────────────────────────────────────

def plot_reward(env: str, reward: str, method_trajs: dict, out_path: str) -> bool:
    """
    method_trajs: {method_dir_name: [array, ...]}
    Returns True if figure was saved.
    """
    # Filter to methods that have at least one valid trajectory
    valid_methods = {
        mn: arrs for mn, arrs in method_trajs.items()
        if any(a is not None and len(a) > 0 for a in arrs)
    }
    if not valid_methods:
        return False

    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(8.5, 5), facecolor="white")

    all_flat = [a for arrs in valid_methods.values() for a in arrs
                if a is not None and len(a) > 0]
    x_max = max(len(a) for a in all_flat)
    x_grid = np.unique(np.concatenate([
        np.geomspace(1, x_max, 1200).astype(int), [x_max]
    ]))

    plotted_methods = []
    fb_idx = 0

    for method_name in sorted(valid_methods.keys()):
        arrs = [a for a in valid_methods[method_name]
                if a is not None and len(a) > 0]
        if not arrs:
            continue

        canon = canonical_method(method_name)
        m = FW_META.get(canon, _DEFAULT_FW.copy())
        label = m["label"] or method_name
        color = m["color"]
        ls    = m["ls"]
        # Assign a tab20 fallback for unmapped method names
        if color == "#7f7f7f":
            _fallback = plt.cm.tab20.colors
            color = _fallback[fb_idx % len(_fallback)]
            fb_idx += 1

        # Clip x_grid to this method's own longest run
        fw_xmax = max(len(a) for a in arrs)
        xg = x_grid[x_grid <= fw_xmax]
        if len(xg) == 0:
            continue

        did_plot = plot_band(ax, arrs, xg, color, label, ls=ls)
        if did_plot:
            plotted_methods.append((canon, method_name, label, color, ls))

    if not plotted_methods:
        plt.close()
        return False

    ax.set_xscale("log")
    ax.set_xlim(1, x_max)
    ax.set_xlabel("Function evaluations (per run)")
    ax.set_ylabel("Best reward (running max)")
    ax.set_title(f"{env}  —  {reward}", pad=6)
    ax.grid(True, which="both", alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # Method legend (dedup by label — SST has multiple runs per method)
    seen_labels = set()
    method_handles = []
    for canon, mn, label, color, ls in plotted_methods:
        if label in seen_labels:
            continue
        seen_labels.add(label)
        method_handles.append(
            Line2D([0], [0], color=color, ls=ls, lw=1.8, label=label)
        )

    style_handles = [
        Patch(facecolor="grey", alpha=0.25, label="Min–max range"),
        Line2D([0], [0], color="grey", lw=1.8, label="Median best"),
    ]

    leg1 = ax.legend(handles=method_handles, loc="lower right",
                     framealpha=0.95, title="Method", fontsize=8.5)
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc="lower left",
              framealpha=0.95, title="Style key", fontsize=8.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    envs = sorted(
        e for e in os.listdir(BEST_REWARDS)
        if os.path.isdir(os.path.join(BEST_REWARDS, e)) and e not in SKIP_ENTRIES
    )

    total_saved = 0
    total_skip  = 0

    for env in envs:
        env_dir = os.path.join(BEST_REWARDS, env)
        rewards = sorted(
            r for r in os.listdir(env_dir)
            if os.path.isdir(os.path.join(env_dir, r)) and r not in SKIP_ENTRIES
        )
        print(f"\n{'='*60}", flush=True)
        print(f"  {env}  ({len(rewards)} rewards)", flush=True)
        print(f"{'='*60}", flush=True)

        for reward in rewards:
            reward_dir  = os.path.join(env_dir, reward)
            method_dirs = sorted(
                m for m in os.listdir(reward_dir)
                if os.path.isdir(os.path.join(reward_dir, m))
                and m not in SKIP_ENTRIES
                and not re.match(r'^seed\d+$', m)
            )

            method_trajs = {}
            for mname in method_dirs:
                mdir      = os.path.join(reward_dir, mname)
                seed_dirs = get_all_seed_dirs(mdir)
                trajs     = [load_seed_traj(sd) for sd in seed_dirs]
                n_valid   = sum(1 for t in trajs if t is not None)
                print(f"  {reward}/{mname}: {len(seed_dirs)} runs, {n_valid} w/ data",
                      flush=True)
                method_trajs[mname] = trajs

            out_path = os.path.join(OUT_BASE, env, f"{reward}.png")
            saved    = plot_reward(env, reward, method_trajs, out_path)
            if saved:
                total_saved += 1
                print(f"  -> saved  {env}/{reward}.png", flush=True)
            else:
                total_skip += 1
                print(f"  -> SKIP   {reward} (no valid data)", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"Done.  {total_saved} plots saved, {total_skip} skipped.", flush=True)
    print(f"Output: {OUT_BASE}/", flush=True)


if __name__ == "__main__":
    main()
