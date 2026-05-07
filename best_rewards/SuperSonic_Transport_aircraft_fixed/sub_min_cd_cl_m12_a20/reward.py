from ._flight_utils import FAIL_REWARD, run_point, metric_payload, _NoPromptReward


class SubMinCdClM12A20Reward(_NoPromptReward):
    def __init__(self, cl_target=0.08, w_cl=10.0, aoa=2.0, mach=1.2, **kwargs):
        self.cl_target = cl_target
        self.w_cl = w_cl
        self.aoa = aoa
        self.mach = mach

    @staticmethod
    def add_args(parser):
        pass

    def evaluate(self, run_sim, design_path, case_dir):
        try:
            raw, cl, cdi, ld = run_point(run_sim, design_path, case_dir, self.aoa, self.mach)
            reward = -cdi - self.w_cl * (cl - self.cl_target) ** 2
        except Exception as e:
            print(f"[sub_min_cd_cl_m12_a20] FAILED: {e}")
            raw, cl, cdi, ld, reward = {}, 0.0, 0.0, 0.0, FAIL_REWARD
        return float(reward), metric_payload(raw, cl, cdi, ld, reward, cl_target=self.cl_target, w_cl=self.w_cl, aoa=self.aoa, mach=self.mach)
