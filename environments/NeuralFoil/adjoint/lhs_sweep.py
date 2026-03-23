"""
Latin Hypercube Sampling sweep over the Kulfan parameter space.

Samples N airfoils using LHS, scores each with the same reward metric as the
adjoint (weighted mean CD over 6 HPA CL targets), then runs the adjoint
optimizer to convergence from each starting point.

Results written to:
  {OUTPUT_DIR}/results.csv
  {OUTPUT_DIR}/results.json  (full per-sample records for downstream analysis)

Usage:
  python lhs_sweep.py [--n 500] [--output-dir /path/to/output] [--seed 42]
  python lhs_sweep.py --n 50   # quick test
"""

import argparse
import csv
import json
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import aerosandbox as asb
import neuralfoil as nf
from scipy.optimize import brentq
from scipy.stats.qmc import LatinHypercube

from environments.NeuralFoil.rewards.multipoint_hpa import (
    CL_TARGETS,
    CL_WEIGHTS,
    MACH,
    _re_schedule,
)
from environments.NeuralFoil.adjoint.optimize import run_adjoint_optimization
from environments.NeuralFoil.design_actions import (
    UPPER_BOUNDS,
    LOWER_BOUNDS,
    LE_BOUNDS,
    TE_BOUNDS,
    N_CST,
)

# ── Kulfan search bounds matching the PSO/GA benchmark (design_actions.py) ────
#   upper_weights × 8 : [-0.30, 0.60]
#   lower_weights × 8 : [-0.30, 0.30]
#   leading_edge_weight: [-0.50, 0.50]
#   TE_thickness       : [ 0.00, 0.01]
_LB = np.array([UPPER_BOUNDS[0]] * N_CST + [LOWER_BOUNDS[0]] * N_CST
               + [LE_BOUNDS[0], TE_BOUNDS[0]])
_UB = np.array([UPPER_BOUNDS[1]] * N_CST + [LOWER_BOUNDS[1]] * N_CST
               + [LE_BOUNDS[1], TE_BOUNDS[1]])

CSV_COLS = [
    "idx",
    "upper_weights",
    "lower_weights",
    "leading_edge_weight",
    "TE_thickness",
    "initial_weighted_cd",
    "final_weighted_cd",
    "n_iters",
    "solver_success",
    "feasible",
    "error",
]


def _score_initial(upper_weights, lower_weights, leading_edge_weight,
                   model_size="large") -> float | None:
    """Compute weighted CD for an airfoil at the 6 HPA CL targets.

    Returns None if any CL target is unreachable.
    """
    kulfan = {
        "upper_weights": np.array(upper_weights, dtype=float),
        "lower_weights": np.array(lower_weights, dtype=float),
        "leading_edge_weight": float(leading_edge_weight),
        "TE_thickness": 0.0,
    }
    CDs = []
    for cl_target in CL_TARGETS:
        re = _re_schedule(cl_target)

        def residual(alpha):
            aero = nf.get_aero_from_kulfan_parameters(
                kulfan_parameters=kulfan,
                alpha=float(alpha),
                Re=re,
                mach=MACH,
                model_size=model_size,
            )
            return float(np.squeeze(aero["CL"])) - cl_target

        try:
            f_lo, f_hi = residual(-5.0), residual(18.0)
            if f_lo * f_hi > 0:
                for lo2, hi2 in [(-10.0, 25.0), (-15.0, 30.0)]:
                    if residual(lo2) * residual(hi2) <= 0:
                        f_lo, f_hi = lo2, hi2
                        break
                else:
                    return None
                alpha_sol = float(brentq(residual, f_lo, f_hi, xtol=1e-3, maxiter=60))
            else:
                alpha_sol = float(brentq(residual, -5.0, 18.0, xtol=1e-3, maxiter=60))
        except Exception:
            return None

        aero = nf.get_aero_from_kulfan_parameters(
            kulfan_parameters=kulfan,
            alpha=alpha_sol,
            Re=re,
            mach=MACH,
            model_size=model_size,
        )
        CDs.append(float(np.squeeze(aero["CD"])))

    return float(np.mean(np.array(CDs) * CL_WEIGHTS))


def run_sweep(n: int = 500, output_dir: str = None, seed: int = 42,
              model_size: str = "large"):
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(__file__), "..", "results", "adjoint", "lhs_sweep"
        )
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "results.csv")
    json_path = os.path.join(output_dir, "results.json")

    # ── LHS sample ────────────────────────────────────────────────────────────
    n_dims = 2 * N_CST + 2  # upper×8 + lower×8 + LE + TE
    sampler = LatinHypercube(d=n_dims, seed=seed)
    unit_samples = sampler.random(n=n)
    params = _LB + unit_samples * (_UB - _LB)

    all_records = []

    with open(csv_path, "w", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_COLS)
        writer.writeheader()

        for i, p in enumerate(params):
            upper_weights = p[:N_CST].tolist()
            lower_weights = p[N_CST:2*N_CST].tolist()
            le = float(p[2*N_CST])
            te = float(p[2*N_CST + 1])

            record = {
                "idx": i,
                "upper_weights": upper_weights,
                "lower_weights": lower_weights,
                "leading_edge_weight": le,
                "TE_thickness": te,
                "initial_weighted_cd": None,
                "final_weighted_cd": None,
                "n_iters": None,
                "solver_success": None,
                "feasible": None,
                "error": None,
            }

            initial_cd = _score_initial(upper_weights, lower_weights, le,
                                        model_size=model_size)
            record["initial_weighted_cd"] = initial_cd

            initial_kulfan = {
                "upper_weights": upper_weights,
                "lower_weights": lower_weights,
                "leading_edge_weight": le,
                "TE_thickness": te,
            }

            try:
                result = run_adjoint_optimization(
                    initial_kulfan=initial_kulfan,
                    model_size=model_size,
                    name=f"lhs_{i:04d}",
                )
                record["final_weighted_cd"] = result["weighted_cd"]
                record["n_iters"] = result["n_iters"]
                record["solver_success"] = result["solver_success"]
                record["feasible"] = result["feasible"]
            except Exception as e:
                record["error"] = str(e)[:200]
                traceback.print_exc()

            # Write CSV row (serialize list fields as JSON strings)
            row = dict(record)
            row["upper_weights"] = json.dumps(record["upper_weights"])
            row["lower_weights"] = json.dumps(record["lower_weights"])
            writer.writerow(row)
            csv_f.flush()

            all_records.append(record)

            init_str = f"{initial_cd:.5f}" if initial_cd is not None else "N/A"
            final_str = (
                f"{record['final_weighted_cd']:.5f}"
                if record["final_weighted_cd"] is not None
                else "FAILED"
            )
            print(
                f"[{i+1:>4}/{n}] init={init_str}  final={final_str}"
                f"  iters={record['n_iters']}  ok={record['solver_success']}"
                f"  feas={record['feasible']}"
            )

    with open(json_path, "w") as jf:
        json.dump(all_records, jf, indent=2)

    feasible_finals = [
        r["final_weighted_cd"]
        for r in all_records
        if r.get("feasible") and r["final_weighted_cd"] is not None
    ]
    print(f"\n=== Sweep complete: {n} samples ===")
    print(f"  Feasible converged : {len(feasible_finals)}/{n}")
    if feasible_finals:
        print(f"  Best final CD      : {min(feasible_finals):.6f}")
        print(f"  Mean final CD      : {np.mean(feasible_finals):.6f}")
        print(f"  Std  final CD      : {np.std(feasible_finals):.6f}")
    print(f"  Results saved to   : {output_dir}")
    return all_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LHS adjoint sweep")
    parser.add_argument("--n", type=int, default=500, help="Number of LHS samples")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-size", type=str, default="large")
    args = parser.parse_args()

    run_sweep(n=args.n, output_dir=args.output_dir, seed=args.seed,
              model_size=args.model_size)
