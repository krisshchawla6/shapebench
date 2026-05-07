"""
HPA variant: original CL targets/weights with tighter aero thresholds (CM>=-0.130, conf>0.95).
"""
import numpy as np
from .reward_exact_notebook import RewardExactNotebook

_CL_TARGETS = np.array([0.8, 1.0, 1.2, 1.4, 1.5, 1.6], dtype=float)
_CL_WEIGHTS = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=float)


class RewardHPAStrict(RewardExactNotebook):
    def __init__(self, cl_targets=None, cl_weights=None,
                 cm_limit=-0.13, conf_limit=0.95, **kwargs):
        super().__init__(
            cl_targets=cl_targets if cl_targets is not None else _CL_TARGETS.tolist(),
            cl_weights=cl_weights if cl_weights is not None else _CL_WEIGHTS.tolist(),
            cm_limit=cm_limit,
            conf_limit=conf_limit,
            **kwargs,
        )
