"""Reward: maximise Lift-to-Drag ratio for Mixed_integer_yiren."""

import math
from environments.base_reward import BaseReward

FAIL_REWARD = -5.0


class LdRatioReward(BaseReward):
    """Maximise LtoD at the fixed operating point."""

    def __init__(self, **kwargs):
        pass

    @staticmethod
    def add_args(parser):
        pass

    def evaluate(self, run_sim, design_path: str, case_dir: str) -> tuple:
        try:
            raw = run_sim(design_path, case_dir)
            cl = raw.get("CL", 0.0)
            cdi = raw.get("CDi", 0.0)
            ld = raw.get("LtoD", 0.0)
            reward = float(ld) if math.isfinite(ld) and ld > 0 else FAIL_REWARD
        except Exception as e:
            print(f"[ld_ratio] simulation failed: {e}")
            cl, cdi, ld = 0.0, 0.0, 0.0
            reward = FAIL_REWARD
            raw = {}

        return reward, {
            "metrics": {"CL": cl, "CDi": cdi, "LtoD": ld, "reward": reward},
            "images": raw.get("images", []),
            "feedback": f"L/D={ld:.3f} CL={cl:.4f} CDi={cdi:.6f}",
        }

    def get_prompt_blocks(self):
        return None
