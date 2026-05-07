"""Reward: single-point L/D relative to baseline."""

from environments.base_reward import BaseReward

BASELINE_LD = 50.0
FAIL_REWARD = -5.0


class SinglePtLDReward(BaseReward):
    """Single-point L/D reward = LtoD - BASELINE_LD."""

    def __init__(self, baseline_ld=BASELINE_LD, **kwargs):
        self.baseline_ld = baseline_ld

    @staticmethod
    def add_args(parser):
        parser.add_argument('--baseline-ld', type=float, default=BASELINE_LD,
                            help='Baseline L/D subtracted from reward')

    def evaluate(self, run_sim, design_path: str, case_dir: str) -> tuple:
        try:
            raw = run_sim(design_path, case_dir)
            CL   = raw.get('CL',   0.0)
            CDi  = raw.get('CDi',  0.0)
            LtoD = raw.get('LtoD', 0.0)
            reward = LtoD - self.baseline_ld
            print(f'  [single_pt_ld] CL={CL:.4f} CDi={CDi:.5f} '
                  f'L/D={LtoD:.3f} reward={reward:.4f}')
        except Exception as e:
            print(f'  [single_pt_ld] FAILED: {e}')
            CL = CDi = LtoD = 0.0
            reward = FAIL_REWARD
            raw = {}

        return float(reward), {
            'metrics': {
                'CL': CL, 'CDi': CDi, 'LtoD': LtoD, 'reward': reward,
            },
            'images': raw.get('images', []),
            'feedback': f'L/D={LtoD:.3f} baseline={self.baseline_ld} reward={reward:.4f}',
        }

    def get_prompt_blocks(self):
        return None
