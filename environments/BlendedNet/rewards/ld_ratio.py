import json
import os

from environments.base_reward import BaseReward

FAIL_REWARD = -5.0


class LDRatioReward(BaseReward):
    """Maximize approximate L/D ratio from Cp and Cfx field averages.

    Runs a single surrogate at the specified flight condition.
    reward = CL_approx / CD_approx  (mean field approximation)
    """

    def __init__(self, mach=0.3, re=1.0e7, alpha=5.0, **kwargs):
        self.mach = mach
        self.re = re
        self.alpha = alpha

    @staticmethod
    def add_args(parser):
        parser.add_argument('--mach', type=float, default=0.3,
                            help='Mach number')
        parser.add_argument('--re', type=float, default=1.0e7,
                            help='Reynolds number')
        parser.add_argument('--alpha', type=float, default=5.0,
                            help='Angle of attack (deg)')

    def evaluate(self, run_sim, design_path: str, case_dir: str) -> tuple:
        try:
            r = run_sim(design_path, case_dir,
                        mach=self.mach, re=self.re, alpha=self.alpha)

            CL_approx = -r["Cp_mean"]
            CD_approx = r["Cfx_mean"]
            LD = CL_approx / CD_approx if abs(CD_approx) > 1e-12 else 0.0
            reward = LD

            results_dict = {
                "design": r["params"],
                "Re": r["Re"], "Mach": r["Mach"], "alpha": r["alpha"],
                "Cp_mean": r["Cp_mean"], "Cfx_mean": r["Cfx_mean"],
                "Cfz_mean": r["Cfz_mean"],
                "CL_approx": CL_approx, "CD_approx": CD_approx,
                "L_D": LD, "reward": reward,
            }
            with open(os.path.join(r["save_dir"], "results.json"), "w") as f:
                json.dump(results_dict, f, indent=2)

            return float(reward), {
                "metrics": {
                    "Cp_mean": r["Cp_mean"], "Cfx_mean": r["Cfx_mean"],
                    "Cfz_mean": r["Cfz_mean"],
                    "CL_approx": CL_approx, "CD_approx": CD_approx,
                    "L_D": LD, "reward": reward,
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
