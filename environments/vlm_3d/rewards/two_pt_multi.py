import os

from environments.base_reward import BaseReward
from . import _two_pt_prompt_blocks as _pb

# Paper Table 3 flight conditions
MACH_SUP = 1.8
MACH_SUB = 0.3
RE_SUP = 80.4e6
RE_SUB = 101.8e6
CL_TARGET_SUP = 0.1665
CL_TARGET_SUB = 0.6933
KN_TARGET_DEFAULT = 0.05
FAIL_REWARD = -5.0


class TwoPtMultiReward(BaseReward):
    """Two-point multi-objective reward per Yiren et al.

    Runs three simulations (supersonic, subsonic, subsonic+delta_aoa for static
    margin) and computes: reward = -CD_sup - penalty terms for constraint violations.

    Constraints:
        CL = CL* at both conditions, CM = 0 at both, Kn >= Kn* (subsonic).
    """

    def __init__(self, aoa_sup=0.0, aoa_sub=10.0,
                 kn_target=KN_TARGET_DEFAULT, delta_aoa=0.01,
                 w_cl=10.0, w_cm=100.0, w_kn=100.0, **kwargs):
        self.aoa_sup = aoa_sup
        self.aoa_sub = aoa_sub
        self.kn_target = kn_target
        self.delta_aoa = delta_aoa
        self.w_cl = w_cl
        self.w_cm = w_cm
        self.w_kn = w_kn

    @staticmethod
    def add_args(parser):
        g = parser.add_argument_group('Two-point flight conditions (Yiren et al.)')
        g.add_argument('--aoa-sup', type=float, default=0.0,
                       help='Supersonic AOA, deg (paper range: -3 to 3)')
        g.add_argument('--aoa-sub', type=float, default=10.0,
                       help='Subsonic AOA, deg (paper range: -5 to 20)')
        g.add_argument('--kn-target', type=float, default=KN_TARGET_DEFAULT,
                       help='Static margin target (fraction, e.g. 0.05 = 5%%)')
        g.add_argument('--delta-aoa', type=float, default=0.01,
                       help='AOA perturbation for Kn finite difference (deg)')
        g = parser.add_argument_group('Penalty weights')
        g.add_argument('--w-cl', type=float, default=10.0,
                       help='Weight for CL target miss penalty')
        g.add_argument('--w-cm', type=float, default=100.0,
                       help='Weight for CM != 0 penalty')
        g.add_argument('--w-kn', type=float, default=100.0,
                       help='Weight for static margin violation penalty')

    def get_prompt_blocks(self) -> dict:
        return {
            'format_context': _pb.format_context,
            'format_response_instructions': _pb.format_response_instructions,
            'CONTEXT_FORMAT': _pb.CONTEXT_FORMAT,
            'DESIGN_ENTRY': _pb.DESIGN_ENTRY,
            'RESPONSE_FORMAT': _pb.RESPONSE_FORMAT,
        }

    def evaluate(self, run_sim, design_path: str, case_dir: str) -> tuple:
        sup_dir = os.path.join(case_dir, 'sup')
        sub_dir = os.path.join(case_dir, 'sub')
        kn_dir = os.path.join(case_dir, 'sub_kn')

        try:
            r_sup = run_sim(design_path, sup_dir,
                            aoa=self.aoa_sup, mach=MACH_SUP, re=RE_SUP)
            r_sub = run_sim(design_path, sub_dir,
                            aoa=self.aoa_sub, mach=MACH_SUB, re=RE_SUB)
            r_kn = run_sim(design_path, kn_dir,
                           aoa=self.aoa_sub + self.delta_aoa, mach=MACH_SUB, re=RE_SUB)

            cl_sup, cdi_sup, cm_sup = r_sup['cl'], r_sup['cdi'], r_sup['cm']
            cl_sub, cdi_sub, cm_sub = r_sub['cl'], r_sub['cdi'], r_sub['cm']
            cl_sub_p, cm_sub_p = r_kn['cl'], r_kn['cm']

            kn = self._static_margin(cl_sub, cm_sub, cl_sub_p, cm_sub_p)
            reward = self._compute_reward(cdi_sup, cl_sup, cl_sub, cm_sup, cm_sub, kn)

            print(f"[2pt] sup: CL={cl_sup:.4f} CDi={cdi_sup:.5f} CM={cm_sup:.4f}")
            print(f"[2pt] sub: CL={cl_sub:.4f} CDi={cdi_sub:.5f} CM={cm_sub:.4f}")
            print(f"[2pt] Kn={kn:.4f}  reward={reward:.4f}")

            images = r_sup.get('images', []) + r_sub.get('images', [])

        except Exception as e:
            print(f"[2pt] FAILED: {e}")
            cl_sup = cdi_sup = cm_sup = 0.0
            cl_sub = cdi_sub = cm_sub = 0.0
            kn = 0.0
            reward = FAIL_REWARD
            images = []

        return float(reward), {
            'metrics': {
                'CL_sup': cl_sup, 'CDi_sup': cdi_sup, 'CM_sup': cm_sup,
                'CL_sub': cl_sub, 'CDi_sub': cdi_sub, 'CM_sub': cm_sub,
                'Kn': kn, 'reward': reward,
            },
            'images': images,
            'feedback': '',
        }

    def _static_margin(self, cl, cm, cl_p, cm_p) -> float:
        cl_alpha = (cl_p - cl) / self.delta_aoa
        cm_alpha = (cm_p - cm) / self.delta_aoa
        if abs(cl_alpha) < 1e-12:
            return 0.0
        return -cm_alpha / cl_alpha

    def _compute_reward(self, cd_sup, cl_sup, cl_sub, cm_sup, cm_sub, kn) -> float:
        reward = -cd_sup
        reward -= self.w_cl * (cl_sup - CL_TARGET_SUP) ** 2
        reward -= self.w_cl * (cl_sub - CL_TARGET_SUB) ** 2
        reward -= self.w_cm * cm_sup ** 2
        reward -= self.w_cm * cm_sub ** 2
        if kn < self.kn_target:
            reward -= self.w_kn * (self.kn_target - kn) ** 2
        return reward
