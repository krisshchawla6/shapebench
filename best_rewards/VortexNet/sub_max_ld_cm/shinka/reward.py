"""Reward 7: Maximize subsonic L/D at fixed AoA with soft CM = 0 penalty."""
from environments.base_reward import BaseReward

MACH = 0.42
RE = 8.0e6
FAIL_REWARD = -5.0


class SubMaxLDCm(BaseReward):
    """Maximize subsonic L/D at a fixed AoA, penalizing pitch-moment deviation.

    Unlike SubMaxLDCl (which bisects for fixed CL), this operates at a
    prescribed AoA and uses a soft constraint to push CM toward zero,
    encouraging naturally trimmed geometries.
    """

    def __init__(self, aoa=10.0, w_cm=100.0, **kwargs):
        self.aoa = aoa
        self.w_cm = w_cm

    @staticmethod
    def add_args(parser):
        g = parser.add_argument_group('SubMaxLDCm')
        g.add_argument('--aoa', type=float, default=10.0,
                        help='Fixed subsonic AoA (deg)')
        g.add_argument('--w-cm', type=float, default=100.0,
                        help='Pitch-moment penalty weight')

    def evaluate(self, run_sim, design_path, case_dir):
        try:
            r = run_sim(design_path, case_dir, aoa=self.aoa, mach=MACH, re=RE)
            cl, cdi, cm = r['cl'], r['cdi'], r['cm']
            ld = cl / cdi if abs(cdi) > 1e-12 else 0.0
            reward = ld - self.w_cm * cm ** 2
            print(f'[sub_max_ld_cm] AoA={self.aoa} CL={cl:.4f} CDi={cdi:.5f} '
                  f'CM={cm:.4f} L/D={ld:.3f} reward={reward:.4f}')
            images = r.get('images', [])
        except Exception as e:
            print(f'[sub_max_ld_cm] FAILED: {e}')
            cl = cdi = cm = ld = 0.0
            reward = FAIL_REWARD
            images = []

        return float(reward), {
            'metrics': {'CL': cl, 'CDi': cdi, 'CM': cm, 'L_D': ld, 'reward': reward},
            'images': images,
            'feedback': '',
        }
