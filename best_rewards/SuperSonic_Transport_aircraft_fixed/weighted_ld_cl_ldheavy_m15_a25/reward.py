from ._flight_utils import FAIL_REWARD, run_point, metric_payload, _NoPromptReward


class WeightedLdClLdheavyM15A25Reward(_NoPromptReward):
    def __init__(self, w_ld=0.8, w_cl=0.2, cl_ref=0.10, aoa=2.5, mach=1.5, **kwargs):
        self.w_ld = w_ld
        self.w_cl = w_cl
        self.cl_ref = cl_ref
        self.aoa = aoa
        self.mach = mach

    @staticmethod
    def add_args(parser):
        pass

    def evaluate(self, run_sim, design_path, case_dir):
        try:
            raw, cl, cdi, ld = run_point(run_sim, design_path, case_dir, self.aoa, self.mach)
            reward = self.w_ld * ld + self.w_cl * (cl / max(self.cl_ref, 1e-9))
        except Exception as e:
            print(f"[weighted_ld_cl_ldheavy_m15_a25] FAILED: {e}")
            raw, cl, cdi, ld, reward = {}, 0.0, 0.0, 0.0, FAIL_REWARD
        return float(reward), metric_payload(raw, cl, cdi, ld, reward, w_ld=self.w_ld, w_cl=self.w_cl, cl_ref=self.cl_ref, aoa=self.aoa, mach=self.mach)
