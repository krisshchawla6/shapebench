import json
import os

from environments.base_reward import BaseReward

FAIL_REWARD = -5.0


class CdClConstrainedReward(BaseReward):
    """Minimize Cd subject to Cl <= cl_threshold (downforce constraint).

    reward = -Cd - lam * max(0, Cl - cl_threshold)

    At convergence (Cl <= cl_threshold) the reward equals -Cd, so the
    objective remains purely drag-based in the feasible region — avoiding
    reliance on Cl surrogate accuracy outside the training distribution.

    Default cl_threshold=-0.10 requires modest downforce; the constraint is
    binding (unconstrained Cd-optimal designs have Cl ~ -0.08, violating it)
    while the feasible region (Cl <= -0.10) is well-represented in the
    LHS training data.
    """

    def __init__(self, rho=1.25, u=40.0, area_ref=2.37, lam=1.0,
                 cl_threshold=-0.10, **kwargs):
        self.rho = rho
        self.u = u
        self.area_ref = area_ref
        self.lam = lam
        self.cl_threshold = cl_threshold

    @staticmethod
    def add_args(parser):
        parser.add_argument('--rho', type=float, default=1.25,
                            help='Freestream density (kg/m^3)')
        parser.add_argument('--u', type=float, default=40.0,
                            help='Freestream speed (m/s)')
        parser.add_argument('--area_ref', type=float, default=2.37,
                            help='Reference frontal area (m^2)')
        parser.add_argument('--lam', type=float, default=1.0,
                            help='Penalty coefficient for Cl constraint violation')
        parser.add_argument('--cl_threshold', type=float, default=-0.10,
                            help='Cl upper bound; designs with Cl > cl_threshold are penalised')

    def evaluate(self, run_sim, design_path: str, case_dir: str) -> tuple:
        try:
            r = run_sim(design_path, case_dir)

            q = 0.5 * self.rho * self.u ** 2
            denom = q * self.area_ref if q * self.area_ref > 1e-12 else 1.0
            Cd = r['drag'] / denom
            Cl = r['lift'] / denom

            violation = max(0.0, Cl - self.cl_threshold)
            reward = -Cd - self.lam * violation

            results_dict = {
                "design": r["params"],
                "drag": r["drag"], "drag_pressure": r["drag_p"], "drag_shear": r["drag_w"],
                "lift": r["lift"], "lift_pressure": r["lift_p"], "lift_shear": r["lift_w"],
                "Cd": Cd, "Cl": Cl, "cl_threshold": self.cl_threshold,
                "violation": violation, "reward": reward,
                "rho": self.rho, "u": self.u, "area_ref": self.area_ref, "lam": self.lam,
            }
            with open(os.path.join(r["save_dir"], "results.json"), "w") as f:
                json.dump(results_dict, f, indent=2)

            feasible = Cl <= self.cl_threshold
            feedback = (
                ""
                if feasible
                else (f"Constraint violated: Cl={Cl:.4f} > {self.cl_threshold:.2f}"
                      f"  (penalty={self.lam * violation:.4f})")
            )

            return float(reward), {
                "metrics": {
                    "drag": r["drag"], "Cd": Cd,
                    "lift": r["lift"], "Cl": Cl,
                    "violation": violation,
                    "drag_pressure": r["drag_p"], "drag_shear": r["drag_w"],
                    "reward": reward,
                },
                "images": r["images"],
                "feedback": feedback,
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
