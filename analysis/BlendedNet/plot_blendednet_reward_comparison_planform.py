"""Planform overlay comparing best designs between shapebench_5 (min-CD) and
shapebench_5_max_LD (max-L/D) for each method.

One subplot per method (2×2 grid).  Each subplot shows the two outlines
overlaid at the same scale so shape differences are immediately visible.
All subplots share the same axis limits for cross-method comparison.

Usage:
    cd /scratch/ShapeEvolve
    source /home/jack/venv_torch210/bin/activate
    python analysis/BlendedNet/plot_blendednet_reward_comparison_planform.py

Outputs:
    environments/BlendedNet/results/analysis_plots_reward_comparison/planform_reward_comparison.png/.pdf
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO_DIR, "environments", "BlendedNet", "results")
OUT_DIR = os.path.join(RESULTS_DIR, "analysis_plots_reward_comparison")

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 150,
}

# Consistent method colors (match planform_max_LD plots)
COLORS = {
    "Bayesian Opt.":    "#ff7f0e",
    "PSO (20p × 200i)": "#1f77b4",
    "CMA-ES":           "#d62728",
    "ShapeEvolve":      "#2ca02c",
}


# ── Geometry ──────────────────────────────────────────────────────────────────

C1 = 1000.0

def full_span_polygon(params):
    B1 = params["B1"]; B2 = params["B2"]; B3 = params["B3"]
    C2 = params["C2"]; C3 = params["C3"]; C4 = params["C4"]
    S1 = np.radians(params["S1"])
    S2 = np.radians(params["S2"])
    S3 = np.radians(params["S3"])

    y = np.array([0.0, B1, B1 + B2, B1 + B2 + B3])
    le = np.array([
        0.0,
        B1 * np.tan(S1),
        B1 * np.tan(S1) + B2 * np.tan(S2),
        B1 * np.tan(S1) + B2 * np.tan(S2) + B3 * np.tan(S3),
    ])
    chord = np.array([C1, C2, C3, C4])
    te = le + chord

    xs_le = y;          xs_te = y[::-1]
    ys_le = le;         ys_te = te[::-1]
    xp_le = -y[::-1];  xp_te = -y
    yp_le = le[::-1];  yp_te = te

    x = np.concatenate([xp_te, xp_le, xs_le, xs_te, [xp_te[0]]])
    y_out = np.concatenate([yp_te, yp_le, ys_le, ys_te, [yp_te[0]]])
    return x, y_out


# ── Data loading ──────────────────────────────────────────────────────────────

def _design_dir(run_dir, row, method_key):
    if method_key == "ga":
        return os.path.join(run_dir, f"iter_{int(row['iteration']):04d}_p{int(row['particle']):03d}")
    else:
        return os.path.join(run_dir, str(row["design"]))


def load_best(dirs, reward_col, method_key):
    """Returns (params, reward_val, results_dict) for the single best design."""
    best_r, best_dir, best_row = -np.inf, None, None
    for d in dirs:
        csv = os.path.join(d, "results.csv")
        if not os.path.exists(csv):
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if reward_col not in df.columns:
            continue
        idx = int(df[reward_col].idxmax())
        row = df.iloc[idx]
        r = float(row[reward_col])
        if r > best_r:
            best_r, best_dir, best_row = r, d, row
    if best_dir is None:
        return None, None, None
    ddir = _design_dir(best_dir, best_row, method_key)
    rj = os.path.join(ddir, "save", "results.json")
    if not os.path.exists(rj):
        return None, None, None
    with open(rj) as f:
        res = json.load(f)
    return res["design"], best_r, res


def _cross_metrics(res):
    """Return (mean_CD, mean_LD) computed from results dict, regardless of reward type."""
    ops = res.get("operating_points", [])
    if "mean_CD" in res:
        mean_cd = res["mean_CD"]
    elif ops:
        mean_cd = float(np.mean([op["CD_approx"] for op in ops]))
    else:
        mean_cd = None
    if "mean_LD" in res:
        mean_ld = res["mean_LD"]
    elif ops:
        cds = [op["CD_approx"] for op in ops]
        cls = [op["CL_approx"] for op in ops]
        mean_ld = float(np.mean([cl / cd for cl, cd in zip(cls, cds)]))
    else:
        mean_ld = None
    return mean_cd, mean_ld


# ── Warm-start loaders ────────────────────────────────────────────────────────

def _load_ws_best_cmaes():
    dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "run_cmaes_shapebench_5_max_LD_warmstart_cornerA_seed*_n500")))
    best_r, best_p = -np.inf, None
    for d in dirs:
        csv = os.path.join(d, "results.csv")
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv)
        if "reward" not in df.columns:
            continue
        run_best = df["reward"].max()
        if run_best > best_r:
            row = df.loc[df["reward"].idxmax()]
            rj = os.path.join(d, str(row["design"]), "save", "results.json")
            if os.path.exists(rj):
                with open(rj) as f:
                    res = json.load(f)
                best_r, best_p = run_best, res
    return best_p["design"] if best_p else None, best_r, best_p


def _load_ws_best_ga():
    dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "run_GA_parallel_shapebench_5_max_LD_warmstart_cornerA_seed*_20p_100i")))
    best_r, best_p = -np.inf, None
    for d in dirs:
        csv = os.path.join(d, "results.csv")
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv)
        if "gbest_reward" not in df.columns:
            continue
        run_best = df["gbest_reward"].max()
        if run_best > best_r:
            matches = df[df["reward"] >= run_best - 1e-4]
            if matches.empty:
                continue
            row = matches.iloc[0]
            it, pt = int(row["iteration"]), int(row["particle"])
            rj = os.path.join(d, f"iter_{it:04d}_p{pt:03d}", "save", "results.json")
            if os.path.exists(rj):
                with open(rj) as f:
                    res = json.load(f)
                best_r, best_p = run_best, res
    return best_p["design"] if best_p else None, best_r, best_p


def _load_ws_best_v3():
    dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "run_v3_flash2_5_shapebench_5_max_LD_warmstart_cornerA_attempt_*_n2000")))
    best_r, best_p_params = -np.inf, None
    for d in dirs:
        db_path = os.path.join(d, "database.json")
        if not os.path.exists(db_path):
            continue
        with open(db_path) as f:
            db = json.load(f)
        entries = db if isinstance(db, list) else list(db.values())
        for e in entries:
            r = e.get("reward", -np.inf)
            if r > best_r:
                path = e.get("path", "")
                if os.path.exists(path):
                    with open(path) as f2:
                        params = json.load(f2)
                    if "B1" in params:
                        best_r, best_p_params = r, params
    if best_p_params is None:
        return None, best_r, None
    # wrap in same results dict shape as other methods
    fake_res = {"design": best_p_params, "mean_LD": best_r, "reward": best_r, "operating_points": []}
    return best_p_params, best_r, fake_res


def load_warmstart_best_ld(name):
    """Returns (params, reward, results_dict) for warm-start max-LD; None for BO."""
    if name == "Bayesian Opt.":
        return None, None, None
    elif name == "PSO (20p × 200i)":
        return _load_ws_best_ga()
    elif name == "CMA-ES":
        return _load_ws_best_cmaes()
    elif name == "ShapeEvolve":
        return _load_ws_best_v3()
    return None, None, None


METHODS = {
    "Bayesian Opt.": {
        "cd_dirs": lambda: (sorted(glob.glob(os.path.join(RESULTS_DIR, "run_BO_torch_shapebench_5_seed*_n500"))) +
                            sorted(glob.glob(os.path.join(RESULTS_DIR, "run_BO_torch_shapebench_5_seed*_n1000")))),
        "ld_dirs": lambda: sorted(glob.glob(os.path.join(RESULTS_DIR, "run_BO_torch_shapebench_5_max_LD_seed*_n1000"))),
        "mkey": "bo",
    },
    "PSO (20p × 200i)": {
        "cd_dirs": lambda: (
            sorted(glob.glob(os.path.join(RESULTS_DIR, "run_GA_shapebench_5_seed*_20p_200i"))) +
            sorted(glob.glob(os.path.join(RESULTS_DIR, "run_GA_shapebench_5_attempt*_20p_200i")))
        ),
        "ld_dirs": lambda: sorted(glob.glob(os.path.join(RESULTS_DIR, "run_GA_parallel_shapebench_5_max_LD_seed*_20p_200i"))),
        "mkey": "ga",
    },
    "CMA-ES": {
        "cd_dirs": lambda: sorted(glob.glob(os.path.join(RESULTS_DIR, "run_cmaes_shapebench_5_seed*_n500to1000"))),
        "ld_dirs": lambda: sorted(glob.glob(os.path.join(RESULTS_DIR, "run_cmaes_shapebench_5_max_LD_seed*_n1000"))),
        "mkey": "bo",
    },
    "ShapeEvolve": {
        "cd_dirs": lambda: sorted(glob.glob(os.path.join(RESULTS_DIR, "run_v3_flash2_5_shapebench_5_attempt_*_n2000"))),
        "ld_dirs": lambda: sorted(glob.glob(os.path.join(RESULTS_DIR, "run_v3_flash2_5_shapebench_5_max_LD_attempt_*_n2000"))),
        "mkey": "v3",
    },
}

METHOD_ORDER = ["Bayesian Opt.", "PSO (20p × 200i)", "CMA-ES", "ShapeEvolve"]


# ── Annotation helper ─────────────────────────────────────────────────────────

def _ann(params, res, header=""):
    hs = params["B1"] + params["B2"] + params["B3"]
    mean_cd, mean_ld = _cross_metrics(res)
    cd_str = rf"$\overline{{C_D}}$ = {mean_cd:.5f}" if mean_cd is not None else r"$\overline{C_D}$ = n/a"
    ld_str = rf"$\overline{{L/D}}$ = {mean_ld:.2f}"  if mean_ld is not None else r"$\overline{L/D}$ = n/a"
    prefix = f"{header}\n" if header else ""
    return (
        f"{prefix}"
        f"half-span={hs:.0f} mm\n"
        f"S1={params['S1']:.0f}°  C2={params['C2']:.0f}  C4={params['C4']:.0f}\n"
        f"B2={params['B2']:.0f}\n"
        f"{cd_str}\n"
        f"{ld_str}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    plt.rcParams.update(STYLE)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load all designs
    designs = {}
    for name in METHOD_ORDER:
        cfg = METHODS[name]
        p_cd, r_cd, res_cd = load_best(cfg["cd_dirs"](), "reward", cfg["mkey"])
        p_ld, r_ld, res_ld = load_best(cfg["ld_dirs"](), "reward", cfg["mkey"])
        designs[name] = (p_cd, r_cd, res_cd, p_ld, r_ld, res_ld)
        if p_cd and p_ld:
            cd_cd, ld_cd = _cross_metrics(res_cd)
            cd_ld, ld_ld = _cross_metrics(res_ld)
            hs_cd = p_cd["B1"]+p_cd["B2"]+p_cd["B3"]
            hs_ld = p_ld["B1"]+p_ld["B2"]+p_ld["B3"]
            print(f"{name}:")
            print(f"  min-CD best → CD={cd_cd:.5f}  L/D={ld_cd:.2f}  hs={hs_cd:.0f}mm")
            print(f"  max-LD best → CD={cd_ld:.5f}  L/D={ld_ld:.2f}  hs={hs_ld:.0f}mm")

    # Compute shared axis limits from all designs
    all_x, all_y = [], []
    for p_cd, _, _res_cd, p_ld, _, _res_ld in designs.values():
        for p in [p_cd, p_ld]:
            if p is None:
                continue
            x, y = full_span_polygon(p)
            all_x.extend(x); all_y.extend(y)
    pad_x = 80; pad_y = 60
    xlim = (min(all_x) - pad_x, max(all_x) + pad_x)
    ylim = (max(all_y) + pad_y, min(all_y) - pad_y)  # inverted (LE at top)

    # Load warm-start max-LD designs
    ws_designs = {}
    for name in METHOD_ORDER:
        p_ws, r_ws, res_ws = load_warmstart_best_ld(name)
        ws_designs[name] = (p_ws, r_ws, res_ws)

    # 2 rows × 4 cols
    # Row 0: min-CD (blue solid) + max-LD random-start (red dashed) — no warm-start
    # Row 1: min-CD (blue solid) + max-LD warm-start (green dashed)
    fig, axes = plt.subplots(2, 4, figsize=(22, 16), facecolor="white",
                             sharex=True, sharey=True)

    ROW_LABELS = [
        r"min-$\overline{C_D}$ + max-$\overline{L/D}$" + "\n(no warm-start)",
        r"min-$\overline{C_D}$ + max-$\overline{L/D}$" + "\n(with warm-start, Corner A)",
    ]

    for col, name in enumerate(METHOD_ORDER):
        p_cd, r_cd, res_cd, p_ld, r_ld, res_ld = designs[name]
        p_ws, r_ws, res_ws = ws_designs[name]
        c = COLORS[name]

        for row in range(2):
            ax = axes[row, col]

            # min-CD: solid filled outline in method color
            if p_cd is not None:
                x, y = full_span_polygon(p_cd)
                ax.fill(x, y, color=c, alpha=0.20)
                ax.plot(x, y, color=c, lw=2.0,
                        label=r"min-$\overline{C_D}$" if col == 0 else "_")

            # max-LD: dashed outline only (no fill) in same method color
            if row == 0:
                p_ld_show, res_ld_show = p_ld, res_ld
                ld_label = r"max-$\overline{L/D}$ (random-start)" if col == 0 else "_"
            else:
                # fall back to random-start for BO (no warm-start run)
                p_ld_show  = p_ws  if p_ws  is not None else p_ld
                res_ld_show = res_ws if res_ws is not None else res_ld
                ld_label = (r"max-$\overline{L/D}$ (warm-start)" if (p_ws is not None and col == 0)
                            else (r"max-$\overline{L/D}$ (random, BO)" if (p_ws is None and col == 0)
                            else "_"))

            if p_ld_show is not None:
                x, y = full_span_polygon(p_ld_show)
                ax.plot(x, y, color=c, lw=2.0, ls="--", label=ld_label)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal")
            ax.axvline(0, color="grey", lw=0.5, ls="--", alpha=0.4)
            ax.grid(True, alpha=0.18)
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)
            if row == 1:
                ax.set_xlabel("Span (mm)")
            if col == 0:
                ax.set_ylabel("Chord (mm, LE → TE)")
            if row == 0:
                ax.set_title(name, fontweight="medium", pad=6)

            # annotation box — max-LD design (dashed outline)
            ann_header = (r"max-$\overline{L/D}$  [warm-start]" if (row == 1 and p_ws is not None)
                          else r"max-$\overline{L/D}$  [random-start]")
            if p_ld_show is not None and res_ld_show is not None:
                ax.text(0.97, 0.97, _ann(p_ld_show, res_ld_show, header=ann_header),
                        transform=ax.transAxes,
                        fontsize=9, va="top", ha="right", color=c,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor=c, alpha=0.85, linewidth=0.8))

            if col == 0:
                ax.legend(fontsize=10, loc="lower left", framealpha=0.9,
                          handlelength=1.5, borderpad=0.5)

    # Row descriptor text
    for row, label in enumerate(ROW_LABELS):
        axes[row, 0].text(-0.22, 0.5, label,
                          transform=axes[row, 0].transAxes,
                          fontsize=11, fontweight="bold", va="center", ha="right",
                          rotation=90,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                                    edgecolor="grey", alpha=0.85, linewidth=0.8))

    fig.suptitle(
        r"BlendedNet (BWB) — Best design planform: min-$\overline{C_D}$ vs. max-$\overline{L/D}$ per method"
        "\nTop: overlaid without warm-start.  "
        r"Bottom: max-$\overline{L/D}$ replaced by warm-start (Corner A).  "
        "Same scale across all panels.",
        fontsize=13, fontweight="medium",
    )
    fig.tight_layout(rect=[0.05, 0, 1, 0.96])

    out_png = os.path.join(OUT_DIR, "BlendedNet_planform_reward_comparison.png")
    out_pdf = os.path.join(OUT_DIR, "BlendedNet_planform_reward_comparison.pdf")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
