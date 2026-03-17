import json
import os

from environments.base_reward import BaseReward

FAIL_REWARD = -5.0


class MaxCLReward(BaseReward):
    """Maximize lift coefficient: reward = CL - w_cd * CD.

    Useful for high-lift device design or landing configuration optimization.
    Set w_cd > 0 to add a drag penalty that prevents unbounded CL at extreme
    camber (which would produce very high drag).
    """

    def __init__(self, alpha=10.0, re=1e6, model_size="large", n_crit=9.0,
                 w_cd=0.0, **kwargs):
        self.alpha = alpha
        self.re = re
        self.model_size = model_size
        self.n_crit = n_crit
        self.w_cd = w_cd

    @staticmethod
    def add_args(parser):
        parser.add_argument('--alpha', type=float, default=10.0,
                            help='Angle of attack (deg)')
        parser.add_argument('--re', type=float, default=1e6,
                            help='Reynolds number')
        parser.add_argument('--model_size', type=str, default='large',
                            help='NeuralFoil model size (xxsmall … xxxlarge)')
        parser.add_argument('--n_crit', type=float, default=9.0,
                            help='Critical amplification factor (e^9 method)')
        parser.add_argument('--w_cd', type=float, default=0.0,
                            help='Drag penalty weight (reward = CL - w_cd * CD)')

    def evaluate(self, run_sim, design_path: str, case_dir: str) -> tuple:
        try:
            r = run_sim(design_path, case_dir,
                        alpha=self.alpha, re=self.re,
                        model_size=self.model_size, n_crit=self.n_crit)

            CL, CD = r["CL"], r["CD"]
            confidence = r["analysis_confidence"]
            reward = CL - self.w_cd * CD

            print(f"[max_cl] CL={CL:.4f} CD={CD:.6f} conf={confidence:.2f} reward={reward:.4f}")

            _write_results(r, {"reward": reward})

            return float(reward), {
                "metrics": {
                    "CL": CL, "CD": CD, "CM": r["CM"],
                    "Top_Xtr": r["Top_Xtr"], "Bot_Xtr": r["Bot_Xtr"],
                    "analysis_confidence": confidence,
                    "reward": reward,
                },
                "images": r["images"],
                "feedback": "",
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[max_cl] FAILED: {e}")
            return float(FAIL_REWARD), {
                "metrics": {"reward": FAIL_REWARD},
                "images": [],
                "feedback": "Simulation failed.",
            }


def _write_results(r: dict, extra: dict):
    d = {
        "design": r["kulfan_params"],
        "CL": r["CL"], "CD": r["CD"], "CM": r["CM"],
        "Top_Xtr": r["Top_Xtr"], "Bot_Xtr": r["Bot_Xtr"],
        "analysis_confidence": r["analysis_confidence"],
    }
    d.update(extra)
    with open(os.path.join(r["save_dir"], "results.json"), "w") as f:
        json.dump(d, f, indent=2)
