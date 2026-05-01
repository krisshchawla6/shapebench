"""
Multi-point L/D maximisation for BWB using total drag, with AR feasibility
constraint.

Identical to shapebench_5_max_LD_total_drag, with one additional check:

    AR = 4 * B3 / C2  ≥  AR_MIN  (default 2.5)

See shapebench_5_total_drag_constrained.py for full motivation.
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

_CD_MIN = 1e-6
AR_MIN  = 2.5


class ShapeBench5MaxLDTotalDragConstrainedReward(BaseReward):
    """Multi-point L/D maximisation for BWB using total drag, AR ≥ 2.5."""

    def __init__(self, mach=0.3, re=1.0e7, ar_min=AR_MIN, **kwargs):
        self.mach   = float(mach)
        self.re     = float(re)
        self.ar_min = float(ar_min)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--mach',   type=float, default=0.3)
        parser.add_argument('--re',     type=float, default=1.0e7)
        parser.add_argument('--ar_min', type=float, default=AR_MIN,
                            help='Minimum aspect ratio AR=4*B3/C2 (feasibility)')

    # ── helpers ───────────────────────────────────────────────────────

    def _run_at_alpha(self, run_sim, design_path, case_dir, alpha):
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
            print(f"[shapebench_5_max_LD_total_drag_constrained] FAILED: {e}")
            return float(FAIL_REWARD), {
                "metrics": {"reward": FAIL_REWARD},
                "images": [],
                "feedback": "Simulation failed.",
            }

    def _evaluate(self, run_sim, design_path: str, case_dir: str) -> tuple:
        os.makedirs(case_dir, exist_ok=True)

        with open(design_path) as f:
            orig_params = json.load(f)

        b3 = float(orig_params.get("B3", 0.0))
        c2 = float(orig_params.get("C2", 1.0))
        ar = 4.0 * b3 / c2 if c2 > 0 else 0.0

        if ar < self.ar_min:
            print(f"[shapebench_5_max_LD_total_drag_constrained] INFEASIBLE: "
                  f"AR={ar:.3f} < {self.ar_min} (B3={b3}, C2={c2})")
            return float(FAIL_REWARD), {
                "metrics": {"reward": FAIL_REWARD, "AR": ar},
                "images": [],
                "feedback": f"AR={ar:.3f} < {self.ar_min} (infeasible).",
            }

        work = os.path.join(case_dir, "_mp")
        unique_cls = sorted(set(CL_TARGETS))
        solved = {}
        for cl_t in unique_cls:
            alpha_sol, r = self._bisect_cl(
                run_sim, design_path, work, cl_t,
                tag=f'cl{cl_t:.3f}')
            solved[cl_t] = (alpha_sol, r)

        lds = [cl_t / max(solved[cl_t][1]["CD_integrated"], _CD_MIN)
               for cl_t in CL_TARGETS]
        mean_ld = float(np.mean(lds))
        reward  = mean_ld

        cruise_cl    = CL_TARGETS[0]
        cruise_alpha = solved[cruise_cl][0]
        cruise_r     = solved[cruise_cl][1]
        images       = cruise_r.get("images", [])

        op_points = []
        for cl_t in CL_TARGETS:
            a_i, r_i = solved[cl_t]
            cd = max(r_i["CD_integrated"], _CD_MIN)
            op_points.append({
                "cl_target":     cl_t,
                "alpha_solved":  a_i,
                "CL_approx":     -r_i["Cp_mean"],
                "CD_integrated": r_i["CD_integrated"],
                "Cfx_mean":      r_i["Cfx_mean"],
                "LD":            cl_t / cd,
            })

        results_dict = {
            "design":           orig_params,
            "AR":               ar,
            "operating_points": op_points,
            "mean_LD":          mean_ld,
            "cruise_alpha":     cruise_alpha,
            "reward":           reward,
        }

        save_dir = os.path.join(case_dir, "save")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "results.json"), "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"[shapebench_5_max_LD_total_drag_constrained] "
              f"AR={ar:.2f}  mean_LD={mean_ld:.4f}  reward={reward:.4f}")

        return float(reward), {
            "metrics": {
                "mean_LD":      mean_ld,
                "cruise_alpha": cruise_alpha,
                "AR":           ar,
                "reward":       reward,
            },
            "images": images,
            "feedback": "",
        }
