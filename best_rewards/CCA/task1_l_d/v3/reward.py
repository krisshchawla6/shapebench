from environments.base_reward import BaseReward
from ._cca_reward_common import FAIL_REWARD, CD_MIN, cd_floor, save_reward_json


class CcaLdRatioSingleReward(BaseReward):
    """Single-point L/D maximization."""

    def __init__(self, **kwargs):
        pass

    @staticmethod
    def add_args(parser):
        pass

    def evaluate(self, run_sim, design_path: str, case_dir: str) -> tuple:
        try:
            raw = run_sim(design_path, case_dir)
            cl = float(raw.get("CL", 0.0))
            cd = cd_floor(float(raw.get("CD", 0.0)), CD_MIN)
            ld = cl / cd if cd > 0 else 0.0
            reward = ld
            save_reward_json(case_dir, {"reward": reward, "CL": cl, "CD": cd, "L_D": ld})
            return reward, {
                "metrics": {"CL": cl, "CD": cd, "L_D": ld, "reward": reward},
                "images": raw.get("images", []),
                "feedback": f"single-point L/D={ld:.3f}",
            }
        except Exception as e:
            print(f"[cca_ld_ratio_single] FAILED: {e}")
            return float(FAIL_REWARD), {"metrics": {"reward": FAIL_REWARD}, "images": [], "feedback": "Simulation failed."}
