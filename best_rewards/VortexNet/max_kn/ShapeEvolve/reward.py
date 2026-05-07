"""Reward 9: Maximize static margin Kn at subsonic conditions, CDi as soft penalty."""
import os
from environments.base_reward import BaseReward

MACH = 0.42
RE = 8.0e6
DELTA_AOA = 0.01
FAIL_REWARD = -5.0


class MaxKn(BaseReward):
    """Maximize longitudinal static margin, using drag as a soft penalty.

    Runs two simulations at AoA and AoA+delta_aoa to estimate Kn via
    finite difference:
        Kn = -dCM/dCL ≈ -(CM' - CM) / (CL' - CL)
    reward = Kn - w_cd * CDi
    This objective drives the optimizer to find inherently stable delta-wing
    geometries (large Kn) while staying aerodynamically efficient.
    """

    def __init__(self, aoa=10.0, delta_aoa=DELTA_AOA, w_cd=10.0, **kwargs):
        self.aoa = aoa
        self.delta_aoa = delta_aoa
        self.w_cd = w_cd

    @staticmethod
    def add_args(parser):
        g = parser.add_argument_group('MaxKn')
        g.add_argument('--aoa', type=float, default=10.0,
                        help='Baseline subsonic AoA (deg)')
        g.add_argument('--delta-aoa', type=float, default=DELTA_AOA,
                        help='AoA perturbation for Kn finite difference (deg)')
        g.add_argument('--w-cd', type=float, default=10.0,
                        help='CDi penalty weight')

    def _static_margin(self, cl, cm, cl_p, cm_p):
        dcl = cl_p - cl
        if abs(dcl) < 1e-12:
            return 0.0
        return -(cm_p - cm) / dcl

    def evaluate(self, run_sim, design_path, case_dir):
        try:
            r = run_sim(design_path, case_dir,
                        aoa=self.aoa, mach=MACH, re=RE)
            r_p = run_sim(design_path, os.path.join(case_dir, 'kn'),
                          aoa=self.aoa + self.delta_aoa, mach=MACH, re=RE)

            cl, cdi, cm = r['cl'], r['cdi'], r['cm']
            kn = self._static_margin(cl, cm, r_p['cl'], r_p['cm'])
            reward = kn - self.w_cd * cdi

            print(f'[max_kn] AoA={self.aoa} CL={cl:.4f} CDi={cdi:.5f} '
                  f'Kn={kn:.4f} reward={reward:.4f}')
            images = r.get('images', [])
        except Exception as e:
            print(f'[max_kn] FAILED: {e}')
            cl = cdi = cm = kn = 0.0
            reward = FAIL_REWARD
            images = []

        return float(reward), {
            'metrics': {'CL': cl, 'CDi': cdi, 'CM': cm, 'Kn': kn, 'reward': reward},
            'images': images,
            'feedback': '',
        }
