"""
Run HPA multipoint IPOPT adjoint with framework-matching bounds, warm-started
from NACA0012 (same starting point as the existing adjoint_run result).

Bounds match environments/NeuralFoil/design_actions.py (same as GA/lbfgsb):
  upper_weights:       [-0.30, 0.60]
  lower_weights:       [-0.30, 0.30]
  leading_edge_weight: [-0.50, 0.50]
  TE_thickness:        [0.000, 0.010]

Results saved to:
  environments/NeuralFoil/results/adjoint_run_fwbounds_naca0012/
"""

import os
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, REPO_DIR)

NEURALFOIL_SRC = os.path.join(REPO_DIR, "environments", "NeuralFoil", "neuralfoil_src")
if os.path.isdir(NEURALFOIL_SRC) and NEURALFOIL_SRC not in sys.path:
    sys.path.insert(0, NEURALFOIL_SRC)

from environments.NeuralFoil.adjoint.optimize_multipoint_framework_bounds import (
    run_adjoint_multipoint_framework_bounds,
)

OUTPUT_DIR = os.path.join(REPO_DIR, "environments", "NeuralFoil", "results",
                           "adjoint_run_fwbounds_naca0012")


def main():
    print("=" * 60)
    print("HPA multipoint adjoint: framework bounds, warm-start NACA0012")
    print("Bounds: upper[-0.30,0.60], lower[-0.30,0.30], LE[-0.50,0.50], TE[0,0.01]")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 60)

    result = run_adjoint_multipoint_framework_bounds(
        initial_kulfan=None,   # NACA0012
        model_size="large",
        output_dir=OUTPUT_DIR,
        name="adjoint_multipoint_fwbounds_naca0012",
    )

    print(f"\nResult:")
    print(f"  weighted_cd  = {result['weighted_cd']:.6f}")
    print(f"  feasible     = {result['feasible']}")
    print(f"  solver_ok    = {result['solver_success']}")
    print(f"  n_iters      = {result['n_iters']}")
    print(f"  CMs          = {[f'{v:.5f}' for v in result['CMs']]}")
    print(f"  confs        = {[f'{v:.4f}' for v in result['analysis_confidences']]}")
    print(f"  CDs          = {[f'{v:.6f}' for v in result['CDs']]}")
    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
