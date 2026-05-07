"""Reward 8: Min CDi subsonic with CL = 0.6933 and static-margin Kn >= 0.05."""
import os
from environments.base_reward import BaseReward
from . import _cl_bisect as _cb

MACH = 0.42
RE = 8.0e6
CL_TARGET = 0.6933
AOA_LO = 1.0
AOA_HI = 19.0
KN_TARGET = 0.05
DELTA_AOA = 0.01
FAIL_REWARD = -5.0


class SubMinCdKn(BaseReward):
    """Minimize subsonic induced drag at fixed CL with a static-margin constraint.

    Bisects AoA to CL_TARGET, then runs a second simulation at AoA+delta_aoa
    to compute the longitudinal static margin via finite difference:
        Kn = -dCM/dCL ≈ -(CM' - CM) / (CL' - CL)
    A one-sided penalty is applied if Kn falls below kn_target.
    """

    def __init__(self, cl_target=CL_TARGET, kn_target=KN_TARGET,
                 delta_aoa=DELTA_AOA, w_cl=10.0, w_kn=100.0,
                 aoa_lo=AOA_LO, aoa_hi=AOA_HI,
                 cl_tol=1e-3, bisect_iters=10, **kwargs):
        self.cl_target = cl_target
        self.kn_target = kn_target
        self.delta_aoa = delta_aoa
        self.w_cl = w_cl
        self.w_kn = w_kn
        self.aoa_lo = aoa_lo
        self.aoa_hi = aoa_hi
        self.cl_tol = cl_tol
        self.bisect_iters = bisect_iters

    @staticmethod
    def add_args(parser):
        g = parser.add_argument_group('SubMinCdKn')
        g.add_argument('--cl-target', type=float, default=CL_TARGET)
        g.add_argument('--kn-target', type=float, default=KN_TARGET)
        g.add_argument('--delta-aoa', type=float, default=DELTA_AOA)
        g.add_argument('--w-cl', type=float, default=10.0)
        g.add_argument('--w-kn', type=float, default=100.0)
        g.add_argument('--aoa-lo', type=float, default=AOA_LO)
        g.add_argument('--aoa-hi', type=float, default=AOA_HI)
        g.add_argument('--cl-tol', type=float, default=1e-3)
        g.add_argument('--bisect-iters', type=int, default=10)

    def _static_margin(self, cl, cm, cl_p, cm_p):
        dcl = cl_p - cl
        if abs(dcl) < 1e-12:
            return 0.0
        return -(cm_p - cm) / dcl

    def evaluate(self, run_sim, design_path, case_dir):
        try:
            result, aoa = _cb.bisect_cl(
                run_sim, design_path, case_dir, 'sub',
                MACH, RE, self.cl_target,
                self.aoa_lo, self.aoa_hi, self.cl_tol, self.bisect_iters,
            )
            if result is None:
                raise ValueError('CL target not bracketed')
            cl, cdi, cm = result['cl'], result['cdi'], result['cm']

            r_kn = run_sim(design_path, os.path.join(case_dir, 'kn'),
                           aoa=aoa + self.delta_aoa, mach=MACH, re=RE)
            kn = self._static_margin(cl, cm, r_kn['cl'], r_kn['cm'])

            reward = -cdi - self.w_cl * (cl - self.cl_target) ** 2
            if kn < self.kn_target:
                reward -= self.w_kn * (self.kn_target - kn) ** 2

            print(f'[sub_min_cd_kn] AoA={aoa:.3f} CL={cl:.4f} CDi={cdi:.5f} '
                  f'Kn={kn:.4f} reward={reward:.4f}')
            images = result.get('images', [])
        except Exception as e:
            print(f'[sub_min_cd_kn] FAILED: {e}')
            cl = cdi = cm = aoa = kn = 0.0
            reward = FAIL_REWARD
            images = []

        return float(reward), {
            'metrics': {'CL': cl, 'CDi': cdi, 'CM': cm, 'AoA': aoa,
                        'Kn': kn, 'reward': reward},
            'images': images,
            'feedback': '',
        }
