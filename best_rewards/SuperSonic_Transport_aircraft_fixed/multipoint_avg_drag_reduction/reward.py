"""Multipoint reward: average induced-drag reduction across flight points."""

from ._flight_utils import FAIL_REWARD, run_point, _NoPromptReward


class MultipointAvgDragReductionReward(_NoPromptReward):
    def __init__(self, **kwargs):
        # (aoa, mach, baseline_cdi)
        self.points = [
            (2.0, 1.2, 0.0045),
            (2.5, 1.5, 0.0040),
            (3.5, 1.5, 0.0050),
        ]

    @staticmethod
    def add_args(parser):
        pass

    def evaluate(self, run_sim, design_path, case_dir):
        point_rewards = []
        point_metrics = {}
        images = []
        try:
            for idx, (aoa, mach, base_cdi) in enumerate(self.points):
                raw, cl, cdi, ld = run_point(run_sim, design_path, case_dir, aoa, mach)
                reduction = (base_cdi - cdi) / max(base_cdi, 1e-9)
                point_rewards.append(reduction)
                point_metrics[f'CL_p{idx+1}'] = cl
                point_metrics[f'CDi_p{idx+1}'] = cdi
                point_metrics[f'LtoD_p{idx+1}'] = ld
                point_metrics[f'drag_reduction_p{idx+1}'] = reduction
                if idx == 0:
                    images = raw.get('images', [])
            reward = float(sum(point_rewards) / len(point_rewards))
        except Exception as e:
            print(f"[multipoint_avg_drag_reduction] FAILED: {e}")
            reward = FAIL_REWARD

        return float(reward), {
            'metrics': {'reward': reward, **point_metrics},
            'images': images,
            'feedback': ''
        }
