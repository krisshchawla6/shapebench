"""
HPA-style strict multipoint at low Reynolds number (MAV/drone scale).

Identical structure to reward_exact_notebook but with:
  CL_TARGETS = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
  Re schedule: Re = 1e5 * (CL / 0.8)^(-0.5)  (lower Re, appropriate for small aircraft)
  MACH = 0.03

All 6 CL targets must be solved (strict mode).
"""
import numpy as np
from .reward_exact_notebook import RewardExactNotebook

_CL_TARGETS = np.array([0.4, 0.6, 0.8, 1.0, 1.2, 1.4], dtype=float)
_CL_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)


def _low_re_schedule(cl_target):
    return float(1e5 * (cl_target / 0.8) ** (-0.5))


class LowReMultipointReward(RewardExactNotebook):
    """Strict 6-point HPA at low Re; overrides _re_schedule used in _find_alpha."""

    def __init__(self, cl_targets=None, cl_weights=None, **kwargs):
        super().__init__(
            cl_targets=cl_targets if cl_targets is not None else _CL_TARGETS.tolist(),
            cl_weights=cl_weights if cl_weights is not None else _CL_WEIGHTS.tolist(),
            **kwargs,
        )

    def _find_alpha(self, kulfan, cl_target, re, **kwargs):
        re_i = _low_re_schedule(cl_target)
        return super()._find_alpha(kulfan, cl_target, re_i, **kwargs)
