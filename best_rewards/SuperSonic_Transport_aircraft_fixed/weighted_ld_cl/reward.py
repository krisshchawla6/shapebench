"""Reward: weighted sum of L/D and normalized CL."""

from environments.base_reward import BaseReward

CL_REF = 0.10   # normalisation reference
FAIL_REWARD = -5.0


class WeightedLdClReward(BaseReward):
    """Weighted combination of L/D and normalised lift.

    Reward = w_ld * LtoD + w_cl * (CL / CL_REF)

    Encourages designs that are both aerodynamically efficient (high L/D)
    and generate adequate lift (CL near or above reference).
    """

    def __init__(self, w_ld=0.6, w_cl=0.4, cl_ref=CL_REF, **kwargs):
        self.w_ld  = w_ld
        self.w_cl  = w_cl
        self.cl_ref = cl_ref

    @staticmethod
    def add_args(parser):
        g = parser.add_argument_group('WeightedLdCl')
        g.add_argument('--w-ld',   type=float, default=0.6,
                       help='Weight for L/D term')
        g.add_argument('--w-cl',   type=float, default=0.4,
                       help='Weight for CL / CL_REF term')
        g.add_argument('--cl-ref', type=float, default=CL_REF,
                       help='Reference CL for normalisation')

    def evaluate(self, run_sim, design_path: str, case_dir: str) -> tuple:
        try:
            raw  = run_sim(design_path, case_dir)
            CL   = raw.get('CL',   0.0)
            CDi  = raw.get('CDi',  0.0)
            LtoD = raw.get('LtoD', 0.0)
            cl_norm = CL / max(self.cl_ref, 1e-9)
            reward = self.w_ld * LtoD + self.w_cl * cl_norm
            print(f'  [weighted_ld_cl] CL={CL:.4f} CDi={CDi:.5f} '
                  f'L/D={LtoD:.3f} reward={reward:.4f}')
        except Exception as e:
            print(f'  [weighted_ld_cl] FAILED: {e}')
            CL = CDi = LtoD = 0.0
            reward = FAIL_REWARD
            raw = {}

        return float(reward), {
            'metrics': {
                'CL': CL, 'CDi': CDi, 'LtoD': LtoD, 'reward': reward,
            },
            'images': raw.get('images', []),
            'feedback': (f'L/D={LtoD:.3f} CL={CL:.4f} '
                         f'w_ld={self.w_ld} w_cl={self.w_cl} reward={reward:.4f}'),
        }

    def get_prompt_blocks(self):
        return None
