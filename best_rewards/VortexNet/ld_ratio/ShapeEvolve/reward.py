"""Reward: maximise Lift-to-Drag ratio (L/D) using VortexNet-corrected VLM."""

import math
from environments.base_reward import BaseReward

FAIL_REWARD = -5.0


class LdRatioReward(BaseReward):
    """Maximise corrected L/D = CL / CDi at the design condition."""

    def __init__(self, **kwargs):
        pass

    @staticmethod
    def add_args(parser):
        pass

    def evaluate(self, run_sim, design_path: str, case_dir: str) -> tuple:
        try:
            raw = run_sim(design_path, case_dir)
        except Exception as e:
            print(f'  Simulation error: {e}')
            return FAIL_REWARD, {
                'metrics':  {'L_D': FAIL_REWARD, 'CL': 0.0, 'CDi': 0.0, 'CM': 0.0},
                'images':   [],
                'feedback': f'Simulation failed: {e}',
            }

        cl  = raw.get('cl', 0.0)
        cdi = raw.get('cdi', 0.0)
        cm  = raw.get('cm', 0.0)
        ld  = raw.get('ld', 0.0)

        if not _finite(ld) or cdi <= 0 or cl <= 0:
            reward = FAIL_REWARD
        else:
            reward = float(ld)

        metrics = {
            'L_D': reward if reward > FAIL_REWARD else 0.0,
            'CL':  cl,
            'CDi': cdi,
            'CM':  cm,
        }
        feedback = (
            f"L/D = {ld:.3f}  |  CL = {cl:.4f}  CDi = {cdi:.6f}  CM = {cm:.4f}"
        )
        return reward, {
            'metrics':  metrics,
            'images':   raw.get('images', []),
            'feedback': feedback,
        }

    def get_prompt_blocks(self):
        return None


def _finite(v):
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False
