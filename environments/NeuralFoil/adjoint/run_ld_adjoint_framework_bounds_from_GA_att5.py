"""
Run IPOPT adjoint with framework-matching bounds, warm-started from GA att5.

Bounds match environments/NeuralFoil/design_actions.py (same as GA/lbfgsb):
  upper_weights: [-0.30, 0.60], lower_weights: [-0.30, 0.30]
  leading_edge_weight: [-0.50, 0.50], TE_thickness: [0.000, 0.010]

CM bound: -0.130 (compensates for ~0.003 NeuralFoil->XFOIL CM bias).

Results saved to:
  environments/NeuralFoil/results/ld_adjoint_cm130_from_GA_att5/
"""

import json
import os
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, REPO_DIR)

NEURALFOIL_SRC = os.path.join(REPO_DIR, "environments", "NeuralFoil", "neuralfoil_src")
if os.path.isdir(NEURALFOIL_SRC) and NEURALFOIL_SRC not in sys.path:
    sys.path.insert(0, NEURALFOIL_SRC)

from environments.NeuralFoil.adjoint.optimize_ld_framework_bounds import run_ld_adjoint_framework_bounds

ADJOINT_DIR = os.path.dirname(os.path.abspath(__file__))
INITIAL_DESIGN = os.path.join(ADJOINT_DIR, "best_GA_att5.json")

OUTPUT_DIR = os.path.join(REPO_DIR, "environments", "NeuralFoil", "results",
                           "ld_adjoint_cm130_from_GA_att5")


def main():
    with open(INITIAL_DESIGN) as f:
        initial_kulfan = json.load(f)

    print("=" * 60)
    print("IPOPT adjoint: framework bounds, CM >= -0.130")
    print(f"Warm-start: GA att5 (NeuralFoil L/D=351.9)")
    print(f"Bounds: upper[-0.30,0.60], lower[-0.30,0.30], LE[-0.50,0.50], TE[0,0.01]")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 60)

    result = run_ld_adjoint_framework_bounds(
        initial_kulfan=initial_kulfan,
        alpha=5.0,
        re=1e7,
        mach=0.2,
        model_size="large",
        output_dir=OUTPUT_DIR,
        name="ld_adjoint_cm130_GA_att5",
    )

    print(f"\nResult:")
    print(f"  L/D          = {result['L_D']:.4f}")
    print(f"  CL           = {result['CL']:.4f}")
    print(f"  CD           = {result['CD']:.6f}")
    print(f"  CM           = {result['CM']:.4f}  (bound: {result['cm_lower_bound']})")
    print(f"  TE_thickness = {result['TE_thickness']:.5f}")
    print(f"  conf         = {result['analysis_confidence']:.4f}")
    print(f"  feasible     = {result['feasible']}")
    print(f"  solver_ok    = {result['solver_success']}")
    print(f"  n_iters      = {result['n_iters']}")
    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
