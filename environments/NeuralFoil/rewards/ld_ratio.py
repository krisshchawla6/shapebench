import json
import os

from environments.base_reward import BaseReward

FAIL_REWARD = -5.0
_LOW_CONFIDENCE_THRESHOLD = 0.5


class LDRatioReward(BaseReward):
    """Maximize lift-to-drag ratio: reward = CL / CD.

    Runs a single NeuralFoil evaluation at the specified flight condition.
    Designs with analysis_confidence < threshold print a warning but are not
    penalized, allowing the optimizer to explore uncertain regions freely.
    """

    def __init__(self, alpha=5.0, re=1e6, model_size="large", n_crit=9.0, **kwargs):
        self.alpha = alpha
        self.re = re
        self.model_size = model_size
        self.n_crit = n_crit

    @staticmethod
    def add_args(parser):
        parser.add_argument('--alpha', type=float, default=5.0,
                            help='Angle of attack (deg)')
        parser.add_argument('--re', type=float, default=1e6,
                            help='Reynolds number')
        parser.add_argument('--model_size', type=str, default='large',
                            help='NeuralFoil model size (xxsmall … xxxlarge)')
        parser.add_argument('--n_crit', type=float, default=9.0,
                            help='Critical amplification factor (e^9 method)')

    def evaluate(self, run_sim, design_path: str, case_dir: str) -> tuple:
        try:
            r = run_sim(design_path, case_dir,
                        alpha=self.alpha, re=self.re,
                        model_size=self.model_size, n_crit=self.n_crit)

            CL, CD = r["CL"], r["CD"]
            confidence = r["analysis_confidence"]

            if confidence < _LOW_CONFIDENCE_THRESHOLD:
                print(f"[ld_ratio] Low confidence ({confidence:.2f}) — design may be out of distribution")

            if CD > 1e-9:
                reward = CL / CD
            else:
                reward = FAIL_REWARD

            LD = reward if CD > 1e-9 else 0.0
            print(f"[ld_ratio] CL={CL:.4f} CD={CD:.6f} L/D={LD:.2f} conf={confidence:.2f} reward={reward:.4f}")

            _write_results(r, {"L_D": LD, "reward": reward})

            return float(reward), {
                "metrics": {
                    "CL": CL, "CD": CD, "CM": r["CM"],
                    "L_D": LD,
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
            print(f"[ld_ratio] FAILED: {e}")
            return float(FAIL_REWARD), {
                "metrics": {"reward": FAIL_REWARD},
                "images": [],
                "feedback": "Simulation failed.",
            }


def _write_results(r: dict, extra: dict):
    """Write final results.json including reward and any extra computed values."""
    d = {
        "design": r["kulfan_params"],
        "CL": r["CL"], "CD": r["CD"], "CM": r["CM"],
        "Top_Xtr": r["Top_Xtr"], "Bot_Xtr": r["Bot_Xtr"],
        "analysis_confidence": r["analysis_confidence"],
    }
    d.update(extra)
    with open(os.path.join(r["save_dir"], "results.json"), "w") as f:
        json.dump(d, f, indent=2)
