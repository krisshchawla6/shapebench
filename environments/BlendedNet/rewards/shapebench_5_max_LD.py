"""
Multi-point L/D maximisation for BWB.

    max_x  mean_LD(x)

    mean_LD = (1/N) * sum_i  (CL^(i) / CD(x; CL^(i)))
    CL^(i) in {0.206, 0.206, 0.206, 0.185, 0.227}

Alpha for each CL operating point is found by bisection over the Transolver
surrogate (CL_approx = -Cp_mean), identical to shapebench_5.  The difference
is the aggregation: instead of averaging CD, we average L/D = CL_target/CD at
each trimmed condition.  With CL fixed by bisection the optimizer cannot game
alpha (unlike ld_ratio.py which used a free alpha).

Reward (maximised by framework):
    R = mean_LD  (dimensionless, typically ~20-60 for a BWB)
"""

import json
import os

import numpy as np

from environments.base_reward import BaseReward

FAIL_REWARD = -5.0

CL_TARGETS = [0.206, 0.206, 0.206, 0.185, 0.227]

BISECT_ITERS = 8
ALPHA_LO = -5.0
ALPHA_HI = 12.0


class ShapeBench5MaxLDReward(BaseReward):
    """Multi-point L/D maximisation for BWB."""

    def __init__(self, mach=0.3, re=1.0e7, **kwargs):
        self.mach = float(mach)
        self.re = float(re)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--mach', type=float, default=0.3,
                            help='Freestream Mach number')
        parser.add_argument('--re', type=float, default=1.0e7,
                            help='Reynolds number')

    # ── helpers ───────────────────────────────────────────────────────

    def _run_at_alpha(self, run_sim, design_path, case_dir, alpha):
        """Call run_sim with a specific alpha, overriding the design file."""
        os.makedirs(case_dir, exist_ok=True)
        with open(design_path) as f:
            params = json.load(f)
        params['alpha'] = alpha
        temp_path = os.path.join(case_dir, 'design.json')
        with open(temp_path, 'w') as f:
            json.dump(params, f, indent=2)
        return run_sim(temp_path, case_dir,
                       mach=self.mach, re=self.re, alpha=alpha)

    def _bisect_cl(self, run_sim, design_path, work_dir, target_cl, tag):
        """Find alpha that yields target_cl via bisection on CL = -Cp_mean."""
        lo, hi = ALPHA_LO, ALPHA_HI
        r_last = None
        for i in range(BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            r = self._run_at_alpha(
                run_sim, design_path,
                os.path.join(work_dir, f'{tag}_{i}'), mid)
            cl = -r["Cp_mean"]
            r_last = r
            if cl < target_cl:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi), r_last

    # ── main evaluation ──────────────────────────────────────────────

    def evaluate(self, run_sim, design_path: str, case_dir: str) -> tuple:
        try:
            return self._evaluate(run_sim, design_path, case_dir)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[shapebench_5_max_LD] FAILED: {e}")
            return float(FAIL_REWARD), {
                "metrics": {"reward": FAIL_REWARD},
                "images": [],
                "feedback": "Simulation failed.",
            }

    def _evaluate(self, run_sim, design_path: str, case_dir: str) -> tuple:
        os.makedirs(case_dir, exist_ok=True)
        work = os.path.join(case_dir, "_mp")

        # --- solve unique CL targets via bisection ---
        unique_cls = sorted(set(CL_TARGETS))
        solved = {}
        for cl_t in unique_cls:
            alpha_sol, r = self._bisect_cl(
                run_sim, design_path, work, cl_t,
                tag=f'cl{cl_t:.3f}')
            solved[cl_t] = (alpha_sol, r)

        # --- mean L/D across all operating points ---
        lds = [cl_t / solved[cl_t][1]["Cfx_mean"] for cl_t in CL_TARGETS]
        mean_ld = float(np.mean(lds))

        reward = mean_ld

        # --- representative images from cruise condition ---
        cruise_cl = CL_TARGETS[0]
        cruise_alpha = solved[cruise_cl][0]
        cruise_r = solved[cruise_cl][1]
        images = cruise_r.get("images", [])

        # --- per-point breakdown ---
        with open(design_path) as f:
            orig_params = json.load(f)

        op_points = []
        for cl_t in CL_TARGETS:
            a_i, r_i = solved[cl_t]
            cd = r_i["Cfx_mean"]
            op_points.append({
                "cl_target": cl_t,
                "alpha_solved": a_i,
                "CL_approx": -r_i["Cp_mean"],
                "CD_approx": cd,
                "LD": cl_t / cd,
            })

        results_dict = {
            "design": orig_params,
            "operating_points": op_points,
            "mean_LD": mean_ld,
            "cruise_alpha": cruise_alpha,
            "reward": reward,
        }

        save_dir = os.path.join(case_dir, "save")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "results.json"), "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"[shapebench_5_max_LD] mean_LD={mean_ld:.4f}  reward={reward:.4f}")

        return float(reward), {
            "metrics": {
                "mean_LD": mean_ld,
                "cruise_alpha": cruise_alpha,
                "reward": reward,
            },
            "images": images,
            "feedback": "",
        }
