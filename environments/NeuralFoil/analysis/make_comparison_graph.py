#!/usr/bin/env python3
"""Generate the NeuralFoil-vs-XFoil comparison graph from the AeroSandbox tutorial."""

import argparse
import csv
import json
from pathlib import Path

import aerosandbox as asb
import aerosandbox.numpy as np
import aerosandbox.tools.pretty_plots as p
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def run_optimization() -> tuple[asb.KulfanAirfoil, asb.KulfanAirfoil, float]:
    cl_targets = np.array([0.8, 1.0, 1.2, 1.4, 1.5, 1.6])
    cl_weights = np.array([5, 6, 7, 8, 9, 10])
    re_targets = 500e3 * (cl_targets / 1.25) ** -0.5
    mach = 0.03

    initial_guess_airfoil = asb.KulfanAirfoil("naca0012")
    initial_guess_airfoil.name = "Initial Guess (NACA0012)"

    opti = asb.Opti()
    optimized_airfoil = asb.KulfanAirfoil(
        name="Optimized",
        lower_weights=opti.variable(
            init_guess=initial_guess_airfoil.lower_weights,
            lower_bound=-0.5,
            upper_bound=0.25,
        ),
        upper_weights=opti.variable(
            init_guess=initial_guess_airfoil.upper_weights,
            lower_bound=-0.25,
            upper_bound=0.5,
        ),
        leading_edge_weight=opti.variable(
            init_guess=initial_guess_airfoil.leading_edge_weight,
            lower_bound=-1,
            upper_bound=1,
        ),
        TE_thickness=0,
    )

    alpha = opti.variable(
        init_guess=np.degrees(cl_targets / (2 * np.pi)),
        lower_bound=-5,
        upper_bound=18,
    )

    aero = optimized_airfoil.get_aero_from_neuralfoil(
        alpha=alpha,
        Re=re_targets,
        mach=mach,
    )

    opti.subject_to(
        [
            aero["analysis_confidence"] > 0.90,
            aero["CL"] == cl_targets,
            np.diff(alpha) > 0,
            aero["CM"] >= -0.133,
            optimized_airfoil.local_thickness(x_over_c=0.33) >= 0.128,
            optimized_airfoil.local_thickness(x_over_c=0.90) >= 0.014,
            optimized_airfoil.TE_angle() >= 6.03,
            optimized_airfoil.lower_weights[0] < -0.05,
            optimized_airfoil.upper_weights[0] > 0.05,
            optimized_airfoil.local_thickness() > 0,
        ]
    )

    def wiggliness(af: asb.KulfanAirfoil):
        return sum(
            np.sum(np.diff(np.diff(array)) ** 2)
            for array in [af.lower_weights, af.upper_weights]
        )

    opti.subject_to(wiggliness(optimized_airfoil) < 2 * wiggliness(initial_guess_airfoil))
    opti.minimize(np.mean(aero["CD"] * cl_weights))

    sol = opti.solve(
        behavior_on_failure="return_last",
        options={"ipopt.mu_strategy": "monotone", "ipopt.start_with_resto": "yes"},
    )

    return initial_guess_airfoil, sol(optimized_airfoil), mach


def load_best_reward_airfoil(results_dir: Path) -> tuple[str, float, asb.KulfanAirfoil]:
    csv_path = results_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing results CSV: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")

    best_row = max(rows, key=lambda r: float(r["reward"]))
    design_id = best_row["design"]
    best_reward = float(best_row["reward"])
    json_path = results_dir / design_id / "save" / "results.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing design JSON for best reward: {json_path}")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    design = payload["design"]
    best_airfoil = asb.KulfanAirfoil(
        name=f"Run Best ({design_id})",
        upper_weights=np.array(design["upper_weights"]),
        lower_weights=np.array(design["lower_weights"]),
        leading_edge_weight=float(design["leading_edge_weight"]),
        TE_thickness=float(design["TE_thickness"]),
    )
    return design_id, best_reward, best_airfoil


def load_best_reward_airfoil_pso(results_dir: Path) -> tuple[str, float, asb.KulfanAirfoil]:
    csv_path = results_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing results CSV: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")

    best_row = max(rows, key=lambda r: float(r["reward"]))
    best_reward = float(best_row["reward"])
    iter_idx = int(best_row["iteration"])
    particle_idx = int(best_row["particle"])
    design_id = f"iter_{iter_idx:04d}_p{particle_idx:03d}"
    json_path = results_dir / design_id / "save" / "results.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing PSO design JSON for best reward: {json_path}")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    design = payload["design"]
    best_airfoil = asb.KulfanAirfoil(
        name=f"PSO Best ({design_id})",
        upper_weights=np.array(design["upper_weights"]),
        lower_weights=np.array(design["lower_weights"]),
        leading_edge_weight=float(design["leading_edge_weight"]),
        TE_thickness=float(design["TE_thickness"]),
    )
    return design_id, best_reward, best_airfoil


def make_figure(
    initial_guess_airfoil: asb.KulfanAirfoil,
    optimized_airfoil: asb.KulfanAirfoil,
    run_best_design_id: str,
    run_best_reward: float,
    run_best_airfoil: asb.KulfanAirfoil,
    pso_best_design_id: str,
    pso_best_reward: float,
    pso_best_airfoil: asb.KulfanAirfoil,
    mach: float,
    xfoil_airfoil_path: Path,
    output_path: Path,
    polar_solver: str,
):
    re_plot = 500e3
    fig, ax = plt.subplots(2, 1, figsize=(7, 8))

    airfoils_and_colors = {
        "Initial Guess": (initial_guess_airfoil, "dimgray"),
        "NeuralFoil-Optimized": (optimized_airfoil, "blue"),
        "XFoil-Optimized": (
            asb.Airfoil(coordinates=str(xfoil_airfoil_path)).to_kulfan_airfoil(),
            "darkgreen",
        ),
        "Expert-Designed (DAE-11)": (asb.Airfoil("dae11"), "red"),
        "ShapeEvolve": (run_best_airfoil, "purple"),
        "PSO": (pso_best_airfoil, "black"),
    }

    for i, (name, (af, color)) in enumerate(airfoils_and_colors.items()):
        color = p.adjust_lightness(color, 1)
        is_overlay = name in {"ShapeEvolve", "PSO"}
        overlay_line = (0, (4, 2)) if name == "ShapeEvolve" else (0, (1, 2))
        ax[0].fill(
            af.x(),
            af.y(),
            facecolor=(*color, 0.0 if is_overlay else 0.09),
            edgecolor=(*color, 0.95 if is_overlay else 0.6),
            linewidth=2.0 if is_overlay else 1.0,
            label=name,
            linestyle=overlay_line if is_overlay else (3 * i, (7, 2)),
            zorder=6 if is_overlay else (4 if "NeuralFoil" in name else 3),
        )

        alpha_sweep = np.linspace(0, 15, 41)
        if polar_solver == "xfoil":
            af_for_xfoil = af.repanel(n_points_per_side=100)
            aero = asb.XFoil(
                airfoil=af_for_xfoil,
                Re=re_plot,
                mach=mach,
                timeout=30,
                xfoil_repanel=False,
            ).alpha(alpha_sweep, start_at=5)
            if len(aero["CD"]) == 0 or len(aero["CL"]) == 0:
                raise RuntimeError(
                    "XFoil returned empty polar data. Re-run with --polar-solver neuralfoil."
                )
        elif polar_solver == "neuralfoil":
            aero = af.get_aero_from_neuralfoil(
                alpha=alpha_sweep,
                Re=re_plot,
                mach=mach,
            )
        else:
            raise ValueError(f"Unsupported polar solver: {polar_solver}")

        ax[1].plot(
            aero["CD"],
            aero["CL"],
            color=color,
            alpha=0.7,
            label=name,
            zorder=4 if "NeuralFoil" in name else 3,
        )

    shape_legend = ax[0].legend(
        fontsize=11, loc="lower right", ncol=max(1, len(airfoils_and_colors) // 2)
    )
    ax[0].add_artist(shape_legend)

    reward_legend_handles = [
        Line2D([], [], linestyle="none", label="Rewards"),
        Line2D(
            [],
            [],
            linestyle="none",
            label="NeuralFoil: -0.0793",
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            label=f"ShapeEvolve: {run_best_reward:.4f}",
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            label=f"PSO: {pso_best_reward:.4f}",
        ),
    ]
    ax[0].legend(
        handles=reward_legend_handles,
        loc="upper right",
        frameon=True,
        handlelength=0,
        handletextpad=0.2,
        borderpad=0.5,
        fontsize=9,
    )
    ax[0].set_title("Airfoil Shapes")
    ax[0].set_xlabel("$x/c$")
    ax[0].set_ylabel("$y/c$")
    ax[0].axis("equal")

    ax[1].legend(fontsize=11, loc="lower right", ncol=max(1, len(airfoils_and_colors) // 2))
    ax[1].set_title("Aerodynamic Polars (analyzed with XFoil, $\\mathrm{Re}=500\\mathrm{k}$)")
    ax[1].set_xlabel("Drag Coefficient $C_D$")
    ax[1].set_ylabel("Lift\nCoefficient\n$C_L$")
    ax[1].set_xlim(0, 0.04)
    ax[1].set_ylim(0, 1.8)

    plot_title = (
        "Comparison of NeuralFoil-Optimized-, XFoil-Optimized-,\n"
        "ShapeEvolve-, PSO-, and Expert-Designed-Airfoils"
    )
    p.show_plot(
        plot_title,
        legend=False,
        show=False,
    )
    if fig._suptitle is not None:
        fig._suptitle.set_fontsize(12)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xfoil-airfoil",
        type=Path,
        default=here / "assets" / "drela_opt6_90_dof.dat",
        help="Path to Drela's optimized airfoil coordinates.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "comparison_graph.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--run-results-dir",
        type=Path,
        default=Path("/scratch/ShapeEvolve/environments/NeuralFoil/results/ablations_reward_fixed/geo_b30_f0003"),
        help="Run directory that contains results.csv and design subfolders.",
    )
    parser.add_argument(
        "--pso-results-dir",
        type=Path,
        default=Path("/scratch/ShapeEvolve/environments/NeuralFoil/results/GA/particle_swarm_fixed_reward"),
        help="PSO run directory that contains results.csv and iter_XXXX_pXXX design subfolders.",
    )
    parser.add_argument(
        "--polar-solver",
        choices=["xfoil", "neuralfoil"],
        default="neuralfoil",
        help="Method used to compute polars for plotting.",
    )
    args = parser.parse_args()

    initial_guess_airfoil, optimized_airfoil, mach = run_optimization()
    run_best_design_id, run_best_reward, run_best_airfoil = load_best_reward_airfoil(
        args.run_results_dir
    )
    pso_best_design_id, pso_best_reward, pso_best_airfoil = load_best_reward_airfoil_pso(
        args.pso_results_dir
    )
    make_figure(
        initial_guess_airfoil=initial_guess_airfoil,
        optimized_airfoil=optimized_airfoil,
        run_best_design_id=run_best_design_id,
        run_best_reward=run_best_reward,
        run_best_airfoil=run_best_airfoil,
        pso_best_design_id=pso_best_design_id,
        pso_best_reward=pso_best_reward,
        pso_best_airfoil=pso_best_airfoil,
        mach=mach,
        xfoil_airfoil_path=args.xfoil_airfoil,
        output_path=args.output,
        polar_solver=args.polar_solver,
    )
    print(f"Saved comparison graph to: {args.output}")


if __name__ == "__main__":
    main()
