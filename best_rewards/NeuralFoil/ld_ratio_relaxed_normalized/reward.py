"""
L/D reward with relaxed constraints (CM>=-0.15, conf>0.85, lambda=200) for wider search.
"""
from .ld_ratio_constrained_m02_re1e7_normalized import LDRatioConstrainedM02Re1e7NormalizedReward


class LDRatioRelaxedNormalizedReward(LDRatioConstrainedM02Re1e7NormalizedReward):
    def __init__(self, alpha=5.0, mach=0.2, re=10000000.0,
                 lambda_penalty=200.0,
                 cm_limit=-0.15, conf_limit=0.85, **kwargs):
        super().__init__(alpha=alpha, mach=mach, re=re,
                         lambda_penalty=lambda_penalty,
                         cm_limit=cm_limit, conf_limit=conf_limit, **kwargs)
