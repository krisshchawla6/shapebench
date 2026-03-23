"""Academic-style summary plot for LHS adjoint sweep."""

import csv
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

matplotlib.use("Agg")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "adjoint")
CSV_PATH = os.path.join(RESULTS_DIR, "lhs_sweep", "results.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "analysis", "lhs_sweep_analysis.png")

BENCHMARKS = [0.078521, 0.078694, 0.078561]


def _load_rows():
    rows = []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            try:
                int(row["n_iters"]) if row["n_iters"] else None
            except ValueError:
                continue
            rows.append(row)
    return rows


def _style(ax):
    ax.grid(True, color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_linewidth(0.9)
    ax.tick_params(width=0.9, labelsize=9)


rows = _load_rows()
feas = [r for r in rows if r["feasible"] == "True"]
infeas = [r for r in rows if r["feasible"] != "True"]

cd_feas = np.array([float(r["final_weighted_cd"]) for r in feas], dtype=float)
cd_infeas = np.array([float(r["final_weighted_cd"]) for r in infeas], dtype=float)
it_feas = np.array([int(r["n_iters"]) for r in feas if r["n_iters"]], dtype=int)
it_infeas = np.array([int(r["n_iters"]) for r in infeas if r["n_iters"]], dtype=int)

if cd_feas.size == 0:
    raise RuntimeError("No feasible rows to plot.")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
    }
)

fig, axs = plt.subplots(2, 2, figsize=(10.8, 7.6))
fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.10, wspace=0.28, hspace=0.28)

# A
ax = axs[0, 0]
bins = np.linspace(cd_feas.min() - 3e-4, cd_feas.max() + 3e-4, 32)
ax.hist(cd_feas, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.4)
for v in BENCHMARKS:
    ax.axvline(v, color="#DD8452", linestyle="--", linewidth=1.0)
ax.axvline(cd_feas.min(), color="black", linewidth=1.1)
ax.set_title("A")
ax.set_xlabel("Final weighted CD")
ax.set_ylabel("Count")
_style(ax)

# B
ax = axs[0, 1]
ax.scatter(it_infeas, cd_infeas, s=12, color="#C44E52", alpha=0.45, linewidths=0)
ax.scatter(it_feas, cd_feas, s=14, color="#4C72B0", alpha=0.75, linewidths=0)
for v in BENCHMARKS:
    ax.axhline(v, color="#DD8452", linestyle="--", linewidth=1.0)
ax.set_title("B")
ax.set_xlabel("IPOPT iterations")
ax.set_ylabel("Final weighted CD")
_style(ax)

# C
ax = axs[1, 0]
x = np.arange(2)
y = np.array([len(feas), len(infeas)], dtype=float)
ax.bar(x, y, color=["#4C72B0", "#C44E52"], width=0.65)
ax.set_xticks(x, ["Feasible", "Infeasible"])
ax.set_title("C")
ax.set_ylabel("Count")
_style(ax)

# D
ax = axs[1, 1]
ax.hist(it_infeas, bins=30, color="#C44E52", alpha=0.55, density=True)
ax.hist(it_feas, bins=30, color="#4C72B0", alpha=0.65, density=True)
ax.axvline(it_feas.mean(), color="#4C72B0", linestyle="--", linewidth=1.1)
ax.axvline(it_infeas.mean(), color="#C44E52", linestyle="--", linewidth=1.1)
ax.set_title("D")
ax.set_xlabel("IPOPT iterations")
ax.set_ylabel("Density")
_style(ax)

legend_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C72B0", markersize=6, label="Feasible"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#C44E52", markersize=6, label="Infeasible"),
    Line2D([0], [0], color="#DD8452", linestyle="--", linewidth=1.0, label="Seeded adjoint"),
    Line2D([0], [0], color="black", linewidth=1.1, label="LHS best"),
]
fig.legend(handles=legend_handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.985))

n_total = len(rows)
n_feas = len(feas)
fig.suptitle(f"LHS adjoint sweep (n={n_total}, feasible={n_feas}, best={cd_feas.min():.5f})", y=0.995)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {OUT_PATH}")
