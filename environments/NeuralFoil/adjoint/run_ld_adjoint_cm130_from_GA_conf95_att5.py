"""
Run IPOPT adjoint optimization with CM >= -0.130, warm-started from the best
conf95 GA att5 design (NeuralFoil L/D=314.4, XFOIL L/D=295.6, XFOIL CM=-0.1266).

Purpose: test whether the adjoint can polish the conf95 design into a better
XFOIL-validated result. The conf95 design has only 6% NF->XFOIL L/D bias
(vs 10-15% for conf90 designs), so the adjoint result should be more reliable.

Results saved to:
  environments/NeuralFoil/results/ld_adjoint_cm130_from_GA_conf95_att5/
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

from environments.NeuralFoil.adjoint.optimize_ld_cm130 import run_ld_adjoint_optimization_cm130

ADJOINT_DIR = os.path.dirname(os.path.abspath(__file__))
INITIAL_DESIGN = os.path.join(ADJOINT_DIR, "best_GA_conf95_att5.json")

OUTPUT_DIR = os.path.join(REPO_DIR, "environments", "NeuralFoil", "results",
                           "ld_adjoint_cm130_from_GA_conf95_att5")


def main():
    with open(INITIAL_DESIGN) as f:
        initial_kulfan = json.load(f)

    print("=" * 60)
    print("IPOPT adjoint: CM >= -0.130, warm-start from GA conf95 att5")
    print(f"Initial design: {INITIAL_DESIGN}")
    print(f"  NeuralFoil L/D=314.4, XFOIL L/D=295.6, XFOIL CM=-0.1266")
    print(f"Output dir:     {OUTPUT_DIR}")
    print("=" * 60)

    result = run_ld_adjoint_optimization_cm130(
        initial_kulfan=initial_kulfan,
        alpha=5.0,
        re=1e7,
        mach=0.2,
        model_size="large",
        output_dir=OUTPUT_DIR,
        name="ld_adjoint_cm130_GA_conf95_att5",
    )

    print(f"\nResult:")
    print(f"  L/D          = {result['L_D']:.4f}")
    print(f"  CL           = {result['CL']:.4f}")
    print(f"  CD           = {result['CD']:.6f}")
    print(f"  CM           = {result['CM']:.4f}  (bound: {result['cm_lower_bound']})")
    print(f"  conf         = {result['analysis_confidence']:.4f}")
    print(f"  feasible     = {result['feasible']}")
    print(f"  solver_ok    = {result['solver_success']}")
    print(f"  n_iters      = {result['n_iters']}")
    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
