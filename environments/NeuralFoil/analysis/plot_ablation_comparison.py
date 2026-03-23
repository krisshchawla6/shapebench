#!/usr/bin/env python3
"""Plot best-so-far absolute reward against optimization iteration."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path("/scratch/ShapeEvolve/environments/NeuralFoil/results")
ABLATION_DIR = ROOT / "ablations_reward_fixed"
PSO_DIR = ROOT / "GA" / "particle_swarm_fixed_reward"
ORACLE_DIR = ROOT / "v3_oracle_80_30_i10_r2"
OUT_PNG = Path("/scratch/ShapeEvolve/environments/NeuralFoil/analysis/ablation_comparison_academic.png")
OUT_PDF = Path("/scratch/ShapeEvolve/environments/NeuralFoil/analysis/ablation_comparison_academic.pdf")


SERIES_STYLE = {
    "v3_oracle_80_30_i10_r2": dict(label="SE_v3", color="#d55e00", lw=2.8, ls="-", zorder=10),
    "geo_b30_f0003": dict(label="SE_geometric_30", color="#0072b2", lw=2.2, ls="-", zorder=9),
    "geo_b20_f003": dict(label="SE_geometric_20", color="#009e73", lw=2.0, ls="-", zorder=8),
    "PSO_particle_swarm": dict(label="PSO", color="#cc79a7", lw=2.0, ls="-", zorder=7),
    "dynamic_sampling_50_30_v2": dict(label="SE_dynamic", color="#56b4e9", lw=1.8, ls="-", zorder=6),
    "adaptive_b20_default": dict(label="SE_adaptive", color="#666666", lw=1.2, ls="--", zorder=2),
}


def _load_llm_curve(run_dir: Path) -> list[tuple[int, float]]:
    rows = list(csv.DictReader(run_dir.joinpath("results.csv").open(newline="", encoding="utf-8")))
    # Keep the last row per design name to remove retry-related duplicates.
    last_idx: dict[str, int] = {}
    for i, row in enumerate(rows):
        last_idx[row["design"]] = i
    deduped = [rows[i] for i in sorted(last_idx.values())]
    return _to_best_curve_by_iteration(deduped, reward_key="reward")


def _load_pso_curve(run_dir: Path) -> list[tuple[int, float]]:
    rows = list(csv.DictReader(run_dir.joinpath("results.csv").open(newline="", encoding="utf-8")))
    return _to_best_curve_by_iteration(rows, reward_key="reward")


def _to_best_curve_by_iteration(rows: list[dict[str, str]], reward_key: str) -> list[tuple[int, float]]:
    per_iter_best: dict[int, float] = {}
    for row in rows:
        iteration = int(row["iteration"])
        reward = float(row[reward_key])
        per_iter_best[iteration] = max(per_iter_best.get(iteration, float("-inf")), reward)

    best_reward = float("-inf")
    curve: list[tuple[int, float]] = []
    for iteration in sorted(per_iter_best):
        best_reward = max(best_reward, per_iter_best[iteration])
        curve.append((iteration + 1, abs(best_reward)))
    return curve


def _load_all_curves() -> dict[str, list[tuple[int, float]]]:
    curves = {
        run_dir.name: _load_llm_curve(run_dir)
        for run_dir in sorted(ABLATION_DIR.iterdir())
        if run_dir.is_dir() and run_dir.joinpath("results.csv").exists()
    }
    curves["v3_oracle_80_30_i10_r2"] = _load_llm_curve(ORACLE_DIR)
    curves["PSO_particle_swarm"] = _load_pso_curve(PSO_DIR)
    return curves


def _format_x(x: float, _: float) -> str:
    return str(int(x))


def _format_y(y: float, _: float) -> str:
    if y < 1:
        return f"{y:.2f}"
    if abs(y - round(y)) < 1e-9:
        return str(int(round(y)))
    return f"{y:.1f}"


def plot() -> None:
    curves = _load_all_curves()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
        }
    )

    fig, ax = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)

    for key, style in SERIES_STYLE.items():
        curve = curves.get(key)
        if not curve:
            continue
        xs = [p[0] for p in curve]
        ys = [p[1] for p in curve]
        ax.plot(
            xs,
            ys,
            label=style["label"],
            color=style["color"],
            linewidth=style["lw"],
            linestyle=style["ls"],
            zorder=style["zorder"],
        )
        ax.scatter(xs[-1], ys[-1], s=18, color=style["color"], zorder=style["zorder"] + 0.1)

    best_line = min(curves["geo_b30_f0003"], key=lambda p: p[1])[1]
    ax.axhline(best_line, color="0.35", linewidth=0.9, linestyle=":", zorder=1)
    max_iter = max(xs[-1] for xs in ([ [p[0] for p in c] for c in curves.values() ]))
    ax.text(max_iter - 1, best_line * 1.015, f"best observed = {best_line:.4f}", ha="right", va="bottom", color="0.35")

    ax.set_yscale("log")
    ax.set_xlim(1, max_iter + 1)
    ax.set_ylim(0.08, 12)
    ax.set_xlabel("Optimization iteration")
    ax.set_ylabel(r"Best-so-far $|$reward$|$")
    ax.set_title("NeuralFoil Optimization Ablations")
    ax.grid(True, which="major", color="0.88", linewidth=0.7)
    ax.grid(True, which="minor", color="0.94", linewidth=0.5)
    ax.xaxis.set_major_formatter(FuncFormatter(_format_x))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_y))

    legend = ax.legend(loc="upper right", frameon=True, framealpha=0.95, ncol=2)
    legend.get_frame().set_edgecolor("0.85")
    legend.get_frame().set_linewidth(0.8)

    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PDF)
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    plot()
