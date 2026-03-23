import json
import os

from environments.base_reward import BaseReward

FAIL_REWARD = -5.0
TARGET_CL = 0.206


class ShapeBench1Reward(BaseReward):
    """Minimize drag while matching target lift with a soft AoA-range penalty."""

    def __init__(
        self,
        mach=0.3,
        re=1.0e7,
        alpha=5.0,
        lambda_cl=None,
        lambda_alpha_range=None,
        cl_target=TARGET_CL,
        alpha_pref_min=-3.0,
        alpha_pref_max=3.0,
        **kwargs,
    ):
        if lambda_cl is None:
            raise ValueError("--lambda_cl is required for sahpebench_1")
        if lambda_alpha_range is None:
            raise ValueError("--lambda_alpha_range is required for sahpebench_1")
        if float(alpha_pref_min) >= float(alpha_pref_max):
            raise ValueError("alpha_pref_min must be smaller than alpha_pref_max")
        self.mach = mach
        self.re = re
        self.alpha = alpha
        self.lambda_cl = float(lambda_cl)
        self.lambda_alpha_range = float(lambda_alpha_range)
        self.cl_target = float(cl_target)
        self.alpha_pref_min = float(alpha_pref_min)
        self.alpha_pref_max = float(alpha_pref_max)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--mach', type=float, default=0.3,
                            help='Mach number')
        parser.add_argument('--re', type=float, default=1.0e7,
                            help='Reynolds number')
        parser.add_argument('--alpha', type=float, default=5.0,
                            help='Fallback angle of attack (deg) if the design does not provide one')
        parser.add_argument('--lambda_cl', type=float, required=True,
                            help='Penalty weight for cruise CL tracking')
        parser.add_argument('--lambda_alpha_range', type=float, required=True,
                            help='Penalty weight for violating the preferred AoA range')
        parser.add_argument('--cl_target', type=float, default=TARGET_CL,
                            help='Target cruise lift coefficient')
        parser.add_argument('--alpha_pref_min', type=float, default=-3.0,
                            help='Preferred lower bound for AoA (soft constraint)')
        parser.add_argument('--alpha_pref_max', type=float, default=3.0,
                            help='Preferred upper bound for AoA (soft constraint)')

    def evaluate(self, run_sim, design_path: str, case_dir: str) -> tuple:
        try:
            r = run_sim(design_path, case_dir,
                        mach=self.mach, re=self.re, alpha=self.alpha)

            cl_approx = -r["Cp_mean"]
            cd_approx = r["Cfx_mean"]
            alpha_val = float(r["alpha"])
            cl_error = abs(cl_approx - self.cl_target)
            alpha_violation = max(
                self.alpha_pref_min - alpha_val,
                0.0,
                alpha_val - self.alpha_pref_max,
            )
            reward = -(
                cd_approx
                + self.lambda_cl * cl_error
                + self.lambda_alpha_range * alpha_violation
            )

            results_dict = {
                "design": r["params"],
                "Re": r["Re"],
                "Mach": r["Mach"],
                "alpha": r["alpha"],
                "Cp_mean": r["Cp_mean"],
                "Cfx_mean": r["Cfx_mean"],
                "Cfz_mean": r["Cfz_mean"],
                "CL_approx": cl_approx,
                "CD_approx": cd_approx,
                "cl_target": self.cl_target,
                "lambda_cl": self.lambda_cl,
                "cl_error": cl_error,
                "alpha_pref_min": self.alpha_pref_min,
                "alpha_pref_max": self.alpha_pref_max,
                "lambda_alpha_range": self.lambda_alpha_range,
                "alpha_violation": alpha_violation,
                "reward": reward,
            }
            with open(os.path.join(r["save_dir"], "results.json"), "w") as f:
                json.dump(results_dict, f, indent=2)

            return float(reward), {
                "metrics": {
                    "Cp_mean": r["Cp_mean"],
                    "Cfx_mean": r["Cfx_mean"],
                    "Cfz_mean": r["Cfz_mean"],
                    "CL_approx": cl_approx,
                    "CD_approx": cd_approx,
                    "cl_target": self.cl_target,
                    "lambda_cl": self.lambda_cl,
                    "cl_error": cl_error,
                    "alpha": alpha_val,
                    "alpha_pref_min": self.alpha_pref_min,
                    "alpha_pref_max": self.alpha_pref_max,
                    "lambda_alpha_range": self.lambda_alpha_range,
                    "alpha_violation": alpha_violation,
                    "reward": reward,
                },
                "images": r["images"],
                "feedback": "",
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[sim] FAILED: {e}")
            return float(FAIL_REWARD), {
                "metrics": {"reward": FAIL_REWARD},
                "images": [],
                "feedback": "Simulation failed.",
            }
