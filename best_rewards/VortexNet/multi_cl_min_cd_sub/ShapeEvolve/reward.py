"""Reward 13: min Σ wi·CDi(x, M=0.42, CLi) — 3 CL targets at subsonic cruise."""
import os
from environments.base_reward import BaseReward
from . import _cl_bisect as _cb

MACH = 0.42
RE = 8.0e6
AOA_LO = 1.0
AOA_HI = 19.0

# CL targets spanning low, nominal, and high lift — all within VortexNet's
# achievable range at Ma=0.42, AoA∈[1°,19°] (validated: CL≈0.60–0.82)
_CL_TARGETS = [0.65, 0.72, 0.80]
_WEIGHTS = [0.25, 0.50, 0.25]

FAIL_REWARD = -5.0


class MultiClMinCdSub(BaseReward):
    """Minimize weighted drag across a range of subsonic CL targets.

    At M=0.42, delta wings operate across a lift range from low-lift cruise to
    high-lift maneuver. This objective prevents designs only optimal at one
    subsonic condition.

    reward = -Σ wi·CDi_i  -  w_cl·Σ wi·(CLi - CL*_i)²
    """

    def __init__(self, w_cl=10.0, cl_tol=1e-3, bisect_iters=10, **kwargs):
        self.w_cl = w_cl
        self.cl_tol = cl_tol
        self.bisect_iters = bisect_iters

    @staticmethod
    def add_args(parser):
        g = parser.add_argument_group('MultiClMinCdSub')
        g.add_argument('--w-cl', type=float, default=10.0)
        g.add_argument('--cl-tol', type=float, default=1e-3)
        g.add_argument('--bisect-iters', type=int, default=10)

    def evaluate(self, run_sim, design_path, case_dir):
        metrics = {}
        images = []
        try:
            reward = 0.0
            for cl_target, w in zip(_CL_TARGETS, _WEIGHTS):
                tag = f'cl{cl_target:.4f}'.replace('.', 'p')
                pt_dir = os.path.join(case_dir, tag)
                result, aoa = _cb.bisect_cl(
                    run_sim, design_path, pt_dir, tag,
                    MACH, RE, cl_target,
                    AOA_LO, AOA_HI, self.cl_tol, self.bisect_iters,
                )
                if result is None:
                    raise ValueError(f'CL target {cl_target} not bracketed')
                cl, cdi, cm = result['cl'], result['cdi'], result['cm']
                reward -= w * cdi + self.w_cl * w * (cl - cl_target) ** 2
                metrics[f'CL_{tag}'] = cl
                metrics[f'CDi_{tag}'] = cdi
                metrics[f'AoA_{tag}'] = aoa
                images += result.get('images', [])
                print(f'[multi_cl_min_cd_sub] CL*={cl_target} AoA={aoa:.3f} '
                      f'CL={cl:.4f} CDi={cdi:.5f}')

            metrics['reward'] = reward
            print(f'[multi_cl_min_cd_sub] reward={reward:.4f}')
        except Exception as e:
            print(f'[multi_cl_min_cd_sub] FAILED: {e}')
            reward = FAIL_REWARD
            metrics['reward'] = reward

        return float(reward), {'metrics': metrics, 'images': images, 'feedback': ''}
