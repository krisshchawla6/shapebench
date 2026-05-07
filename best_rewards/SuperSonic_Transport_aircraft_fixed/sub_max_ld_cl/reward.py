"""Reward: maximize L/D with CL target penalty."""

from environments.base_reward import BaseReward

CL_TARGET = 0.10
FAIL_REWARD = -5.0


class SubMaxLDCl(BaseReward):
    """Maximise L/D at the fixed operating point, with a quadratic CL penalty.

    Reward = LtoD - w_cl * (CL - CL_TARGET)^2
    """

    def __init__(self, cl_target=CL_TARGET, w_cl=10.0, **kwargs):
        self.cl_target = cl_target
        self.w_cl = w_cl

    @staticmethod
    def add_args(parser):
        g = parser.add_argument_group('SubMaxLDCl')
        g.add_argument('--cl-target', type=float, default=CL_TARGET,
                       help='Target CL for quadratic penalty')
        g.add_argument('--w-cl', type=float, default=10.0,
                       help='Penalty weight on CL deviation')

    def evaluate(self, run_sim, design_path: str, case_dir: str) -> tuple:
        try:
            raw  = run_sim(design_path, case_dir)
            CL   = raw.get('CL',   0.0)
            CDi  = raw.get('CDi',  0.0)
            LtoD = raw.get('LtoD', 0.0)
            reward = LtoD - self.w_cl * (CL - self.cl_target) ** 2
            print(f'  [sub_max_ld_cl] CL={CL:.4f} CDi={CDi:.5f} '
                  f'L/D={LtoD:.3f} reward={reward:.4f}')
        except Exception as e:
            print(f'  [sub_max_ld_cl] FAILED: {e}')
            CL = CDi = LtoD = 0.0
            reward = FAIL_REWARD
            raw = {}

        return float(reward), {
            'metrics': {
                'CL': CL, 'CDi': CDi, 'LtoD': LtoD, 'reward': reward,
            },
            'images': raw.get('images', []),
            'feedback': (f'L/D={LtoD:.3f} CL={CL:.4f} '
                         f'cl_target={self.cl_target} reward={reward:.4f}'),
        }

    def get_prompt_blocks(self):
        return None
