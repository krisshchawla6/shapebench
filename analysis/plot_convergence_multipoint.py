#!/usr/bin/env python3
"""
Convergence plot for NeuralFoil multi-point HPA benchmark.

Metric: best penalized objective seen so far vs. cumulative model evaluations per run.
  objective = weighted_CD + lambda * sum(violations)
  At convergence, violations -> 0 so objective -> weighted_CD for the best feasible design.

Dashed horizontal = Adjoint (IPOPT + CasADi autodiff), warm-started from NACA 0012,
converged in ~47 evaluations (weighted_CD = 0.07851).

Output: environments/NeuralFoil/results/combined_method_comparison_reward_exact_notebook/
        NeuralFoil_multipoint_objective_vs_iterations.pdf/.png
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from plot_combined_methods import plot_combined

BASE    = "environments/NeuralFoil/results"
OUT_DIR = f"{BASE}/combined_method_comparison_reward_exact_notebook"

plot_combined(
    labels=[
        "L-BFGS-B",
        "Bayesian Opt. (exact GP)",
        r"PSO (120p $\times$ 500i)",
        "ShapeEvolve",
    ],
    csv_paths=[
        (f"{BASE}/comparison_output_reward_exact_notebook_lbfgsb_nr3/"
         "comparison_evolution_evals_lbfgsb_nr3_plus_nr10_trajectory_fitness_total.csv"),
        (f"{BASE}/comparison_output_reward_exact_notebook_BO_torch_exact_GP/"
         "comparison_evolution_evals_BO_torch_trajectory_fitness_total.csv"),
        (f"{BASE}/comparison_output_reward_exact_notebook_multiple_PSO_120_particles_uptoiter_500/"
         "comparison_evolution_evals_COMBINED_trajectory_fitness_total.csv"),
        (f"{BASE}/comparison_output_reward_exact_notebook_different_v3_attempts_INCLUDING_AWS/"
         "flash2_5_summary_trajectory_fitness_total.csv"),
    ],
    colors=["#e377c2", "#ff7f0e", "#1f77b4", "#2ca02c"],
    adjoint_dir=f"{BASE}/adjoint_run_fwbounds_naca0012",
    adjoint_label="Adjoint (IPOPT)",
    x_max=60000,
    y_min=0.065,
    y_max=200,
    title=(
        r"Multi-point Drag Minimization"
        r" ($M_\infty$=0.03, Re=$4.42$–$6.25\times10^5$, 6 $C_L$ targets)"
    ),
    output_path=f"{OUT_DIR}/NeuralFoil_multipoint_objective_vs_iterations.png",
)
