import json
import os

from environments.base_reward import BaseReward

FAIL_REWARD = -5.0


class DownforceEfficiencyReward(BaseReward):
    """Maximise downforce efficiency: reward = |Cl| / Cd = -Cl / Cd.

    Since Cl < 0 for downforce, -Cl > 0 and the reward is positive for any
    design that generates downforce.  Designs with positive lift (Cl > 0) get
    a negative reward, steering the optimizer away from lift-generating shapes.

    Unlike pure downforce maximisation (reward = -Cl), this objective penalises
    designs that generate downforce at the cost of high drag, making it more
    representative of a performance road-car aero target.
    """

    def __init__(self, rho=1.25, u=40.0, area_ref=2.37, cd_floor=1e-4, **kwargs):
        self.rho = rho
        self.u = u
        self.area_ref = area_ref
        self.cd_floor = cd_floor  # guard against near-zero Cd

    @staticmethod
    def add_args(parser):
        parser.add_argument('--rho', type=float, default=1.25,
                            help='Freestream density (kg/m^3)')
        parser.add_argument('--u', type=float, default=40.0,
                            help='Freestream speed (m/s)')
        parser.add_argument('--area_ref', type=float, default=2.37,
                            help='Reference frontal area (m^2)')
        parser.add_argument('--cd_floor', type=float, default=1e-4,
                            help='Minimum Cd denominator to avoid division by zero')

    def evaluate(self, run_sim, design_path: str, case_dir: str) -> tuple:
        try:
            r = run_sim(design_path, case_dir)

            q = 0.5 * self.rho * self.u ** 2
            denom = q * self.area_ref if q * self.area_ref > 1e-12 else 1.0
            Cd = r['drag'] / denom
            Cl = r['lift'] / denom

            efficiency = -Cl / max(Cd, self.cd_floor)   # |Cl|/Cd; negative when Cl > 0
            reward = efficiency

            results_dict = {
                "design": r["params"],
                "drag": r["drag"], "drag_pressure": r["drag_p"], "drag_shear": r["drag_w"],
                "lift": r["lift"], "lift_pressure": r["lift_p"], "lift_shear": r["lift_w"],
                "Cd": Cd, "Cl": Cl, "efficiency": efficiency, "reward": reward,
                "rho": self.rho, "u": self.u, "area_ref": self.area_ref,
            }
            with open(os.path.join(r["save_dir"], "results.json"), "w") as f:
                json.dump(results_dict, f, indent=2)

            return float(reward), {
                "metrics": {
                    "drag": r["drag"], "Cd": Cd,
                    "lift": r["lift"], "Cl": Cl,
                    "efficiency": efficiency,
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
