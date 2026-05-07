from ._flight_utils import FAIL_REWARD, run_point, metric_payload, _NoPromptReward


class LdRatioM12A20Reward(_NoPromptReward):
    def __init__(self, aoa=2.0, mach=1.2, **kwargs):
        self.aoa = aoa
        self.mach = mach

    @staticmethod
    def add_args(parser):
        pass

    def evaluate(self, run_sim, design_path, case_dir):
        try:
            raw, cl, cdi, ld = run_point(run_sim, design_path, case_dir, self.aoa, self.mach)
            reward = ld
        except Exception as e:
            print(f"[ld_ratio_m12_a20] FAILED: {e}")
            raw, cl, cdi, ld, reward = {}, 0.0, 0.0, 0.0, FAIL_REWARD
        return float(reward), metric_payload(raw, cl, cdi, ld, reward, aoa=self.aoa, mach=self.mach)
