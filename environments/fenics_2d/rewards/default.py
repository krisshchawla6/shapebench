from environments.base_reward import BaseReward

FAIL_REWARD = -5.0


class DefaultReward(BaseReward):
    """Pass-through reward for the FEniCS 2D environment.

    The FEniCS solver computes the reward internally (penalized drag/lift).
    This reward simply forwards the solver's value.
    """

    def evaluate(self, run_sim, design_path: str, case_dir: str) -> tuple:
        try:
            r = run_sim(design_path, case_dir)
            reward = r.get('reward', 0.0)
            return float(reward), {
                'metrics': r.get('metrics', {}),
                'images': r.get('images', []),
                'feedback': r.get('feedback', ''),
                'shape_image': r.get('shape_image'),
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[sim] FAILED: {e}")
            return float(FAIL_REWARD), {
                'metrics': {'reward': FAIL_REWARD},
                'images': [],
                'feedback': 'Simulation failed.',
            }
