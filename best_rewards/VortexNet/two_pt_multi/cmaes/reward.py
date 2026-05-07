"""Two-point subsonic multi-objective reward for VortexNet.

Optimises across two flight conditions within VortexNet training bounds:
  - Cruise:   low AoA (default 5°),  Ma=0.42, Re=8e6
  - Maneuver: high AoA (default 14°), Ma=0.42, Re=8e6

Reward = w_cr * (L/D)_cruise + w_mn * (L/D)_maneuver
       - penalty if CL < CL_min at maneuver point (need lift)

VortexNet training bounds: AoA [0-20°], Ma [0.35-0.50], Re [6.5-10]×10⁶
"""

import os

from environments.base_reward import BaseReward

MACH    = 0.42       # mid of training range [0.35, 0.50]
RE      = 8.0e6      # mid of training range [6.5e6, 10e6]
FAIL_REWARD = -5.0


class TwoPtMultiReward(BaseReward):
    """Two-point subsonic reward: cruise L/D + maneuver L/D weighted sum."""

    def __init__(self, aoa_cr=5.0, aoa_mn=14.0,
                 w_cr=0.4, w_mn=0.6, cl_min=0.3, **kwargs):
        self.aoa_cr = aoa_cr
        self.aoa_mn = aoa_mn
        self.w_cr   = w_cr
        self.w_mn   = w_mn
        self.cl_min = cl_min

    @staticmethod
    def add_args(parser):
        g = parser.add_argument_group('Two-point subsonic flight conditions')
        g.add_argument('--aoa-cr',  type=float, default=5.0,
                       help='Cruise AoA (deg), training range 0-20')
        g.add_argument('--aoa-mn',  type=float, default=14.0,
                       help='Maneuver AoA (deg), training range 0-20')
        g.add_argument('--w-cr',    type=float, default=0.4,
                       help='Weight for cruise L/D')
        g.add_argument('--w-mn',    type=float, default=0.6,
                       help='Weight for maneuver L/D')
        g.add_argument('--cl-min',  type=float, default=0.3,
                       help='Minimum CL required at maneuver point')

    def get_prompt_blocks(self):
        return None

    def evaluate(self, run_sim, design_path: str, case_dir: str) -> tuple:
        cr_dir = os.path.join(case_dir, 'cruise')
        mn_dir = os.path.join(case_dir, 'maneuver')

        try:
            r_cr = run_sim(design_path, cr_dir, aoa=self.aoa_cr, mach=MACH, re=RE)
            r_mn = run_sim(design_path, mn_dir, aoa=self.aoa_mn, mach=MACH, re=RE)

            ld_cr = r_cr.get('ld', 0.0)
            ld_mn = r_mn.get('ld', 0.0)
            cl_mn = r_mn.get('cl', 0.0)

            if ld_cr <= 0 or ld_mn <= 0 or cl_mn < self.cl_min:
                reward = FAIL_REWARD
            else:
                reward = self.w_cr * ld_cr + self.w_mn * ld_mn

        except Exception as e:
            print(f'[2pt] FAILED: {e}')
            ld_cr = ld_mn = cl_mn = 0.0
            r_cr = r_mn = {}
            reward = FAIL_REWARD

        return float(reward), {
            'metrics': {
                'L_D_cruise':   ld_cr,
                'CL_cruise':    r_cr.get('cl', 0.0),
                'CDi_cruise':   r_cr.get('cdi', 0.0),
                'L_D_maneuver': ld_mn,
                'CL_maneuver':  cl_mn,
                'CDi_maneuver': r_mn.get('cdi', 0.0),
                'reward':       reward,
            },
            'images': r_cr.get('images', []) + r_mn.get('images', []),
            'feedback': (f"L/D cruise={ld_cr:.3f} maneuver={ld_mn:.3f} "
                         f"CL_mn={cl_mn:.4f} reward={reward:.4f}"),
        }
