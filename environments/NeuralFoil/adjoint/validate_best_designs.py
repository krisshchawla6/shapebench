"""
Validate best designs found by NeuralFoil-based optimizers by warm-starting
the IPOPT adjoint from each design.

This serves two purposes:
  1. Confirms the NeuralFoil L/D under the same model used by the adjoint
     (should agree closely since both use NeuralFoil).
  2. Checks whether IPOPT can improve from the found starting point, probing
     whether these designs are at a local optimum or can be pushed further.
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

from environments.NeuralFoil.adjoint.optimize_ld import run_ld_adjoint_optimization

ADJOINT_DIR = os.path.dirname(os.path.abspath(__file__))

DESIGNS = [
    ("best_GA_att5",    os.path.join(ADJOINT_DIR, "best_GA_att5.json")),
    ("best_lbfgsb_s0",  os.path.join(ADJOINT_DIR, "best_lbfgsb_s0.json")),
    ("best_lbfgsb_s3",  os.path.join(ADJOINT_DIR, "best_lbfgsb_s3.json")),
]

OUTPUT_BASE = os.path.join(REPO_DIR, "environments", "NeuralFoil", "results",
                            "ld_adjoint_warmstart_validation")

def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    summary = []

    for name, design_path in DESIGNS:
        print(f"\n{'='*60}")
        print(f"Running adjoint warm-start from: {name}")
        print(f"{'='*60}")

        with open(design_path) as f:
            initial_kulfan = json.load(f)

        output_dir = os.path.join(OUTPUT_BASE, name)
        result = run_ld_adjoint_optimization(
            initial_kulfan=initial_kulfan,
            alpha=5.0,
            re=1e7,
            mach=0.2,
            model_size="large",
            output_dir=output_dir,
            name=name,
        )

        print(f"  L/D          = {result['L_D']:.4f}")
        print(f"  CL           = {result['CL']:.4f}")
        print(f"  CD           = {result['CD']:.6f}")
        print(f"  CM           = {result['CM']:.4f}")
        print(f"  conf         = {result['analysis_confidence']:.4f}")
        print(f"  feasible     = {result['feasible']}")
        print(f"  solver_ok    = {result['solver_success']}")
        print(f"  n_iters      = {result['n_iters']}")

        summary.append({
            "name": name,
            "L_D": result["L_D"],
            "CL": result["CL"],
            "CD": result["CD"],
            "CM": result["CM"],
            "analysis_confidence": result["analysis_confidence"],
            "feasible": result["feasible"],
            "solver_success": result["solver_success"],
            "n_iters": result["n_iters"],
        })

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Design':<20} {'L/D':>8} {'CM':>8} {'conf':>6} {'feasible':>8} {'solver_ok':>10}")
    for s in summary:
        print(f"{s['name']:<20} {s['L_D']:>8.3f} {s['CM']:>8.4f} {s['analysis_confidence']:>6.3f} {str(s['feasible']):>8} {str(s['solver_success']):>10}")

    with open(os.path.join(OUTPUT_BASE, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_BASE}")

if __name__ == "__main__":
    main()
