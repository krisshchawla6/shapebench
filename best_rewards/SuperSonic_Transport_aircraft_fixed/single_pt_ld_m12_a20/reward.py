from ._flight_utils import FAIL_REWARD, run_point, metric_payload, _NoPromptReward


class SinglePtLDM12A20Reward(_NoPromptReward):
    def __init__(self, baseline_ld=35.0, aoa=2.0, mach=1.2, **kwargs):
        self.baseline_ld = baseline_ld
        self.aoa = aoa
        self.mach = mach

    @staticmethod
    def add_args(parser):
        parser.add_argument('--baseline-ld', type=float, default=35.0)

    def evaluate(self, run_sim, design_path, case_dir):
        try:
            raw, cl, cdi, ld = run_point(run_sim, design_path, case_dir, self.aoa, self.mach)
            reward = ld - self.baseline_ld
        except Exception as e:
            print(f"[single_pt_ld_m12_a20] FAILED: {e}")
            raw, cl, cdi, ld, reward = {}, 0.0, 0.0, 0.0, FAIL_REWARD
        return float(reward), metric_payload(raw, cl, cdi, ld, reward, baseline_ld=self.baseline_ld, aoa=self.aoa, mach=self.mach)
