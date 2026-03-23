#!/usr/bin/env python3
"""Create a vertical 3x2 GIF of best-so-far airfoil evolution."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path

import aerosandbox as asb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    run_dir: Path
    color: str
    mode: str  # "llm" or "pso"


ROOT = Path("/scratch/ShapeEvolve/environments/NeuralFoil/results")
OUT_GIF = Path("/scratch/ShapeEvolve/environments/NeuralFoil/analysis/ablation_methods_3x2.gif")

METHODS = [
    Method("v3", "SE_v3", ROOT / "v3_oracle_80_30_i10_r2", "#d55e00", "llm"),
    Method("geo30", "SE_geometric_30", ROOT / "ablations_reward_fixed" / "geo_b30_f0003", "#0072b2", "llm"),
    Method("geo20", "SE_geometric_20", ROOT / "ablations_reward_fixed" / "geo_b20_f003", "#009e73", "llm"),
    Method("pso", "PSO", ROOT / "GA" / "particle_swarm_fixed_reward", "#cc79a7", "pso"),
    Method("dyn", "SE_dynamic", ROOT / "ablations_reward_fixed" / "dynamic_sampling_50_30_v2", "#56b4e9", "llm"),
    Method("adapt", "SE_adaptive", ROOT / "ablations_reward_fixed" / "adaptive_b20_default", "#666666", "llm"),
]


def _llm_design_path(run_dir: Path, design_id: str) -> Path:
    return run_dir / f"{design_id}.json"


def _pso_design_path(run_dir: Path, iteration: int, particle: int) -> Path:
    design_id = f"iter_{iteration:04d}_p{particle:03d}"
    return run_dir / design_id / "design.json"


def _load_airfoil(path: Path) -> asb.KulfanAirfoil:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return asb.KulfanAirfoil(
        name=payload.get("name", path.stem),
        upper_weights=payload["upper_weights"],
        lower_weights=payload["lower_weights"],
        leading_edge_weight=payload["leading_edge_weight"],
        TE_thickness=payload["TE_thickness"],
    )


def _dedup_llm_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    last_idx: dict[str, int] = {}
    for i, row in enumerate(rows):
        last_idx[row["design"]] = i
    return [rows[i] for i in sorted(last_idx.values())]


def _best_path_by_iteration(method: Method) -> list[tuple[int, float, Path]]:
    rows = list(csv.DictReader(method.run_dir.joinpath("results.csv").open(newline="", encoding="utf-8")))
    if method.mode == "llm":
        rows = _dedup_llm_rows(rows)

    rows_by_iter: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        rows_by_iter.setdefault(int(row["iteration"]), []).append(row)

    best_reward = float("-inf")
    best_path: Path | None = None
    out: list[tuple[int, float, Path]] = []
    max_iter = max(rows_by_iter)

    for iteration in range(max_iter + 1):
        iter_rows = rows_by_iter.get(iteration, [])
        if iter_rows:
            best_row = max(iter_rows, key=lambda r: float(r["reward"]))
            reward = float(best_row["reward"])
            if reward > best_reward:
                best_reward = reward
                if method.mode == "llm":
                    best_path = _llm_design_path(method.run_dir, best_row["design"])
                else:
                    best_path = _pso_design_path(
                        method.run_dir,
                        int(best_row["iteration"]),
                        int(best_row["particle"]),
                    )
        if best_path is not None:
            out.append((iteration + 1, abs(best_reward), best_path))
    return out


def _series_lookup(series: list[tuple[int, float, Path]], iteration_1_based: int) -> tuple[float, Path]:
    current = series[0]
    for item in series:
        if item[0] <= iteration_1_based:
            current = item
        else:
            break
    return current[1], current[2]


def _render_frame(series_map: dict[str, list[tuple[int, float, Path]]], frame_iter: int, max_iter: int) -> Image.Image:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 10,
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(8.2, 11.8), constrained_layout=True)
    fig.patch.set_facecolor("white")

    for ax, method in zip(axes.flat, METHODS):
        reward_abs, path = _series_lookup(series_map[method.key], frame_iter)
        af = _load_airfoil(path)
        x = af.x()
        y = af.y()

        ax.plot(x, y, color=method.color, linewidth=2.0)
        ax.fill(x, y, color=method.color, alpha=0.12)
        ax.axhline(0.0, color="0.85", linewidth=0.6, zorder=0)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.18, 0.22)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([-0.1, 0.0, 0.1, 0.2])
        ax.grid(True, color="0.93", linewidth=0.6)
        ax.set_title(method.label, color=method.color, fontsize=12, pad=8)
        ax.text(
            0.02,
            0.96,
            f"iter {min(frame_iter, series_map[method.key][-1][0]):d}\n|reward| = {reward_abs:.4f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="0.85", alpha=0.95),
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("$x/c$")
    for ax in axes[:, 0]:
        ax.set_ylabel("$y/c$")

    fig.suptitle(
        f"Best-so-far airfoil evolution by method  •  iteration {frame_iter}/{max_iter}",
        fontsize=14,
        y=1.01,
    )

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE)


def main() -> None:
    series_map = {method.key: _best_path_by_iteration(method) for method in METHODS}
    max_iter = max(series[-1][0] for series in series_map.values())
    frames = [_render_frame(series_map, frame_iter=i, max_iter=max_iter) for i in range(1, max_iter + 1)]
    frames[0].save(
        OUT_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=140,
        loop=0,
        optimize=False,
    )
    print(f"Saved {OUT_GIF}")


if __name__ == "__main__":
    main()
