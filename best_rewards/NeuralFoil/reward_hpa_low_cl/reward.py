"""
HPA variant: lower CL targets [0.4–1.4], equal weights. Light-loading regime.
"""
import numpy as np
from .reward_exact_notebook import RewardExactNotebook

_CL_TARGETS = np.array([0.4, 0.6, 0.8, 1.0, 1.2, 1.4], dtype=float)
_CL_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)


class RewardHPALowCL(RewardExactNotebook):
    def __init__(self, cl_targets=None, cl_weights=None,
                 cm_limit=-0.133, conf_limit=0.9, **kwargs):
        super().__init__(
            cl_targets=cl_targets if cl_targets is not None else _CL_TARGETS.tolist(),
            cl_weights=cl_weights if cl_weights is not None else _CL_WEIGHTS.tolist(),
            cm_limit=cm_limit,
            conf_limit=conf_limit,
            **kwargs,
        )
