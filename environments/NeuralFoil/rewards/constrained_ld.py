import json
import os
import sys

ENV_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEURALFOIL_SRC = os.path.join(ENV_DIR, 'neuralfoil_src')
if os.path.isdir(NEURALFOIL_SRC) and NEURALFOIL_SRC not in sys.path:
    sys.path.insert(0, NEURALFOIL_SRC)

from environments.base_reward import BaseReward
from . import _constrained_ld_prompt_blocks as _pb

FAIL_REWARD = -5.0
_LOW_CONFIDENCE_THRESHOLD = 0.5


class ConstrainedLDReward(BaseReward):
    """Maximize L/D with a pitching moment constraint: reward = CL/CD - w_cm * (CM - cm_target)^2.

    Designed for reflex/flying-wing airfoil sections where CM near zero is required
    for trim stability. The penalty term steers the optimizer toward moment-neutral
    shapes while still maximizing aerodynamic efficiency.
    """

    def __init__(self, alpha=5.0, re=1e6, model_size="large", n_crit=9.0,
                 w_cm=10.0, cm_target=0.0, **kwargs):
        self.alpha = alpha
        self.re = re
        self.model_size = model_size
        self.n_crit = n_crit
        self.w_cm = w_cm
        self.cm_target = cm_target

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
        parser.add_argument('--w_cm', type=float, default=10.0,
                            help='Pitching moment constraint weight')
        parser.add_argument('--cm_target', type=float, default=0.0,
                            help='Target pitching moment coefficient')

    def get_prompt_blocks(self) -> dict:
        return {
            "format_context": _pb.format_context,
            "format_response_instructions": _pb.format_response_instructions,
            "CONTEXT_FORMAT": _pb.CONTEXT_FORMAT,
            "DESIGN_ENTRY": _pb.DESIGN_ENTRY,
            "RESPONSE_FORMAT": _pb.RESPONSE_FORMAT,
        }

    def evaluate(self, run_sim, design_path: str, case_dir: str) -> tuple:
        try:
            r = run_sim(design_path, case_dir,
                        alpha=self.alpha, re=self.re,
                        model_size=self.model_size, n_crit=self.n_crit)

            CL, CD, CM = r["CL"], r["CD"], r["CM"]
            confidence = r["analysis_confidence"]

            if confidence < _LOW_CONFIDENCE_THRESHOLD:
                print(f"[constrained_ld] Low confidence ({confidence:.2f})")

            LD = CL / CD if CD > 1e-9 else 0.0
            cm_penalty = self.w_cm * (CM - self.cm_target) ** 2
            reward = LD - cm_penalty if CD > 1e-9 else FAIL_REWARD

            print(f"[constrained_ld] CL={CL:.4f} CD={CD:.6f} CM={CM:.4f} "
                  f"L/D={LD:.2f} CM_penalty={cm_penalty:.4f} reward={reward:.4f}")

            _write_results(r, {"L_D": LD, "CM_penalty": cm_penalty, "reward": reward})

            return float(reward), {
                "metrics": {
                    "CL": CL, "CD": CD, "CM": CM,
                    "L_D": LD, "CM_penalty": cm_penalty,
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
            print(f"[constrained_ld] FAILED: {e}")
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
