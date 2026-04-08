"""
Run IPOPT adjoint (CM>=-0.130, original bounds) warm-started from the top
distinct designs found in the GA/lbfgsb scale-up runs.

CM>=-0.130 (not -0.133) compensates for the known NeuralFoil->XFOIL CM bias
of ~0.003: NeuralFoil predicts CM=-0.130 -> XFOIL gives ~CM=-0.133 (feasible).

Candidates chosen for diversity in NF performance / CM profile:
  lbfgsb_s7  : NF=332.9, XF=312.5, CM_XF=-0.122  (different basin — CM well inside limit)
  GA_att20   : NF=316.1, XF=320.6, CM_XF=-0.137  (best raw XF from scale-up)
  lbfgsb_s26 : NF=337.3, XF=318.3, CM_XF=-0.134
  lbfgsb_s25 : NF=324.6, XF=317.6, CM_XF=-0.135
  lbfgsb_s20 : NF=355.4, XF=310.1, CM_XF=-0.132  (highest NF from scale-up)

Results saved to:
  environments/NeuralFoil/results/ld_adjoint_cm130_scaleup_{label}/
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

ADJOINT_DIR = os.path.join(REPO_DIR, "environments", "NeuralFoil", "adjoint")
RESULTS_DIR = os.path.join(REPO_DIR, "environments", "NeuralFoil", "results")

CANDIDATES = [
    ("lbfgsb_s7",  os.path.join(ADJOINT_DIR, "best_lbfgsb_s7.json")),
    ("GA_att20",   os.path.join(ADJOINT_DIR, "best_GA_att20.json")),
    ("lbfgsb_s26", os.path.join(ADJOINT_DIR, "best_lbfgsb_s26.json")),
    ("lbfgsb_s25", os.path.join(ADJOINT_DIR, "best_lbfgsb_s25.json")),
    ("lbfgsb_s20", os.path.join(ADJOINT_DIR, "best_lbfgsb_s20.json")),
]


def run_one(args):
    label, warm_start_path = args
    output_dir = os.path.join(RESULTS_DIR, f"ld_adjoint_cm130_scaleup_{label}")
    print(f"[{label}] starting...", flush=True)

    with open(warm_start_path) as f:
        warm_start = json.load(f)

    result = run_ld_adjoint_optimization_cm130(
        initial_kulfan=warm_start,
        model_size="large",
        output_dir=output_dir,
        name=f"ld_adjoint_cm130_{label}",
    )

    print(f"[{label}] L/D={result['L_D']:.4f}  CM={result['CM']:.5f}  "
          f"conf={result['analysis_confidence']:.5f}  feasible={result['feasible']}  "
          f"solver_ok={result['solver_success']}  n_iters={result['n_iters']}", flush=True)
    return label, result


def main():
    import multiprocessing
    with multiprocessing.Pool(processes=len(CANDIDATES)) as pool:
        results = pool.map(run_one, CANDIDATES)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Label':<16} {'NF L/D':>8} {'CM':>8} {'conf':>6} {'ok?':>5}")
    for label, r in results:
        ok = "YES" if r['feasible'] else "NO"
        print(f"{label:<16} {r['L_D']:>8.3f} {r['CM']:>8.5f} {r['analysis_confidence']:>6.4f} {ok:>5}")


if __name__ == "__main__":
    main()
