import json
import os

from environments.base_reward import BaseReward

FAIL_REWARD = -5.0


class CdOnlyReward(BaseReward):
    """Minimize drag coefficient: reward = -Cd."""

    def __init__(self, rho=1.25, u=40.0, area_ref=2.37, **kwargs):
        self.rho = rho
        self.u = u
        self.area_ref = area_ref

    @staticmethod
    def add_args(parser):
        parser.add_argument('--rho', type=float, default=1.25,
                            help='Freestream density (kg/m^3)')
        parser.add_argument('--u', type=float, default=40.0,
                            help='Freestream speed (m/s)')
        parser.add_argument('--area_ref', type=float, default=2.37,
                            help='Reference frontal area (m^2)')

    def evaluate(self, run_sim, design_path: str, case_dir: str) -> tuple:
        try:
            r = run_sim(design_path, case_dir)

            q = 0.5 * self.rho * self.u ** 2
            Cd = r['drag'] / (q * self.area_ref) if q * self.area_ref > 1e-12 else 0.0
            reward = -Cd

            results_dict = {
                "design": r["params"],
                "drag": r["drag"], "drag_pressure": r["drag_p"], "drag_shear": r["drag_w"],
                "lift": r["lift"], "lift_pressure": r["lift_p"], "lift_shear": r["lift_w"],
                "Cd": Cd, "reward": reward,
                "rho": self.rho, "u": self.u, "area_ref": self.area_ref,
            }
            with open(os.path.join(r["save_dir"], "results.json"), "w") as f:
                json.dump(results_dict, f, indent=2)

            return float(reward), {
                "metrics": {
                    "drag": r["drag"], "Cd": Cd, "lift": r["lift"],
                    "drag_pressure": r["drag_p"], "drag_shear": r["drag_w"],
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
