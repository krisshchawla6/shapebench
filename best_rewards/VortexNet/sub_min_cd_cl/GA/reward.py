"""Reward 2: Min CDi at subsonic cruise with CL = 0.6933 (equality via bisection)."""
from environments.base_reward import BaseReward
from . import _cl_bisect as _cb

MACH = 0.42
RE = 8.0e6
CL_TARGET = 0.6933
AOA_LO = 1.0
AOA_HI = 19.0
FAIL_REWARD = -5.0


class SubMinCdCl(BaseReward):
    """Minimize subsonic induced drag at a fixed CL.

    Bisects AoA to hit CL=CL_TARGET at M=0.42, then rewards -CDi with a
    quadratic penalty for residual CL error.
    """

    def __init__(self, cl_target=CL_TARGET, w_cl=10.0,
                 aoa_lo=AOA_LO, aoa_hi=AOA_HI,
                 cl_tol=1e-3, bisect_iters=10, **kwargs):
        self.cl_target = cl_target
        self.w_cl = w_cl
        self.aoa_lo = aoa_lo
        self.aoa_hi = aoa_hi
        self.cl_tol = cl_tol
        self.bisect_iters = bisect_iters

    @staticmethod
    def add_args(parser):
        g = parser.add_argument_group('SubMinCdCl')
        g.add_argument('--cl-target', type=float, default=CL_TARGET)
        g.add_argument('--w-cl', type=float, default=10.0)
        g.add_argument('--aoa-lo', type=float, default=AOA_LO)
        g.add_argument('--aoa-hi', type=float, default=AOA_HI)
        g.add_argument('--cl-tol', type=float, default=1e-3)
        g.add_argument('--bisect-iters', type=int, default=10)

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
            reward = -cdi - self.w_cl * (cl - self.cl_target) ** 2
            print(f'[sub_min_cd_cl] AoA={aoa:.3f} CL={cl:.4f} CDi={cdi:.5f} '
                  f'CM={cm:.4f} reward={reward:.4f}')
            images = result.get('images', [])
        except Exception as e:
            print(f'[sub_min_cd_cl] FAILED: {e}')
            cl = cdi = cm = aoa = 0.0
            reward = FAIL_REWARD
            images = []

        return float(reward), {
            'metrics': {'CL': cl, 'CDi': cdi, 'CM': cm, 'AoA': aoa, 'reward': reward},
            'images': images,
            'feedback': '',
        }
