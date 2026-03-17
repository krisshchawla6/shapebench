from environments.base_reward import BaseReward

BASELINE_LD = 5.45
FAIL_REWARD = -5.0


class SinglePtLDReward(BaseReward):
    """Single flight-condition L/D reward (original vlm_3d objective).

    Runs one VLM simulation and returns reward = L/D - baseline_L/D.
    """

    def __init__(self, aoa=10.0, mach=0.3, re=3.0e6, **kwargs):
        self.aoa = aoa
        self.mach = mach
        self.re = re

    @staticmethod
    def add_args(parser):
        parser.add_argument('--aoa', type=float, default=10.0,
                            help='Angle of attack (deg)')
        parser.add_argument('--mach', type=float, default=0.3,
                            help='Mach number')
        parser.add_argument('--re', type=float, default=3.0e6,
                            help='Reynolds number')

    def evaluate(self, run_sim, design_path: str, case_dir: str) -> tuple:
        try:
            r = run_sim(design_path, case_dir, aoa=self.aoa, mach=self.mach, re=self.re)
            cl, cdi, cm = r['cl'], r['cdi'], r['cm']
            ld = cl / cdi if abs(cdi) > 1e-12 else 0.0
            reward = ld - BASELINE_LD
            print(f"[single_pt] CL={cl:.4f} CDi={cdi:.5f} CM={cm:.4f} "
                  f"L/D={ld:.3f} reward={reward:.4f}")
        except Exception as e:
            print(f"[single_pt] FAILED: {e}")
            cl, cdi, cm, ld, reward = 0.0, 0.0, 0.0, 0.0, FAIL_REWARD
            r = {'images': []}

        return float(reward), {
            'metrics': {'CL': cl, 'CDi': cdi, 'CM': cm, 'L_D': ld, 'reward': reward},
            'images': r.get('images', []),
            'feedback': '',
        }
