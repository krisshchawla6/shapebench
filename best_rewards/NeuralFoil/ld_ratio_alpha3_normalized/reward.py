"""
L/D reward at alpha=3 deg: low-incidence operating point.
"""
from .ld_ratio_constrained_m02_re1e7_normalized import LDRatioConstrainedM02Re1e7NormalizedReward


class LDRatioAlpha3NormalizedReward(LDRatioConstrainedM02Re1e7NormalizedReward):
    def __init__(self, alpha=3.0, mach=0.2, re=10000000.0,
                 lambda_penalty=500.0,
                 cm_limit=-0.133, conf_limit=0.9, **kwargs):
        super().__init__(alpha=alpha, mach=mach, re=re,
                         lambda_penalty=lambda_penalty,
                         cm_limit=cm_limit, conf_limit=conf_limit, **kwargs)
