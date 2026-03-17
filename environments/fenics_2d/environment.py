import os
import sys
import shutil
import numpy as np

from environments.base import BaseEnvironment
from environments.base_reward import BaseReward

ENV_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ENV_DIR)

from .config import get_default_config
from . import prompt_blocks


class _SimEnvSingleton:
    """Lazy-loaded singleton for the heavy FEniCS environment."""
    _instance = None

    @classmethod
    def get(cls, env_root):
        if cls._instance is None:
            from .shapes_utils import Shape  # noqa: F401
            from .meshes_utils import Mesh   # noqa: F401
            _orig_env_mod = _import_original_env()
            cfg = get_default_config(env_root)
            cls._instance = _orig_env_mod.env(
                cfg['nb_pts_to_move'], cfg['pts_to_move'],
                cfg['nb_ctrls_per_episode'], cfg['nb_episodes'],
                cfg['max_deformation'],
                cfg['restart_from_cylinder'], cfg['replace_shape'],
                cfg['comp_dir'], cfg['restore_model'],
                cfg['saving_model_period'],
                cfg['final_time'], cfg['cfl'], cfg['reynolds'],
                cfg['output_vtu'],
                cfg['shape_h'], cfg['domain_h'], cfg['cell_limit'],
                cfg['reset_dir'],
                cfg['xmin'], cfg['xmax'], cfg['ymin'], cfg['ymax'],
            )
        return cls._instance


def _import_original_env():
    """Import the original environment module from this package directory."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fenics_2d_orig_env",
        os.path.join(ENV_DIR, "orig_environment.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FEniCS2DEnvironment(BaseEnvironment):
    """2D FEniCS CFD environment for airfoil shape optimization."""

    def __init__(self, reward: BaseReward, **kwargs):
        self._env_root = ENV_DIR
        self.reward = reward

    @staticmethod
    def add_args(parser):
        pass

    def _run_sim(self, design_path: str, case_dir: str) -> dict:
        """Run a single FEniCS simulation; return raw outputs including reward."""
        sim_env = _SimEnvSingleton.get(self._env_root)

        airfoil_shape = np.loadtxt(design_path, delimiter=',')
        if airfoil_shape.ndim == 2:
            airfoil_shape = airfoil_shape[0]

        from .shapes_utils import Shape  # noqa: F811
        sim_env.shape.index = 0
        sim_env.reset()

        _next_state, _terminal, reward = sim_env.execute(airfoil_shape)

        case_save_dir = os.path.join(case_dir, 'save')
        if os.path.exists('./save'):
            if os.path.exists(case_save_dir):
                shutil.rmtree(case_save_dir)
            shutil.copytree('./save', case_save_dir)

        return self._post_process(case_save_dir, reward)

    def _post_process(self, save_dir, reward=None) -> dict:
        drag, lift = 0.0, 0.0
        dl_path = os.path.join(save_dir, 'drag_lift')
        if os.path.exists(dl_path):
            with open(dl_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    last = lines[-1].split()
                    if len(last) >= 3:
                        drag = float(last[1])
                        lift = float(last[2])

        if reward is None:
            reward_file = os.path.join(save_dir, 'reward_penalization')
            if os.path.exists(reward_file):
                with open(reward_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        parts = lines[-1].split()
                        if len(parts) >= 2:
                            reward = float(parts[1])

        sol_dir = os.path.join(save_dir, 'sol')
        sol_images = [
            os.path.join(sol_dir, '1_p.png'),
            os.path.join(sol_dir, '1_u.png'),
            os.path.join(sol_dir, '1_v.png'),
        ]

        png_dir = os.path.join(save_dir, 'png')
        shape_image = None
        if os.path.exists(png_dir):
            shape_pngs = sorted(
                f for f in os.listdir(png_dir)
                if f.startswith('shape_') and f.endswith('.png')
            )
            if shape_pngs:
                shape_image = os.path.join(png_dir, shape_pngs[-1])

        analysis_text = ""
        try:
            from .analysis_llm import run_simulation_analysis
            metrics = {'drag': drag, 'lift': lift}
            if reward is not None:
                metrics['reward'] = reward
            analysis_text = run_simulation_analysis(sol_images, metrics)
        except Exception:
            pass

        return {
            'reward': float(reward) if reward is not None else 0.0,
            'metrics': {'drag': drag, 'lift': lift},
            'images': sol_images,
            'feedback': analysis_text,
            'shape_image': shape_image,
        }

    def simulate(self, design_path: str, case_dir: str, **kwargs) -> tuple:
        os.makedirs(case_dir, exist_ok=True)
        return self.reward.evaluate(self._run_sim, design_path, case_dir)

    def build_context_entry(self, db_entry) -> dict:
        csv_path = db_entry[0]
        vec = np.loadtxt(csv_path, delimiter=',')
        if vec.ndim == 2:
            vec = vec[0]

        results = db_entry[3] if len(db_entry) > 3 else {}
        metrics = results.get('metrics', {}) if isinstance(results, dict) else {}
        images_list = results.get('images', []) if isinstance(results, dict) else []
        feedback = results.get('feedback', '') if isinstance(results, dict) else ''
        shape_image = results.get('shape_image') if isinstance(results, dict) else None

        parent_images = []
        if shape_image and os.path.exists(shape_image):
            parent_images.append(shape_image)
        for sol_img in images_list:
            if sol_img and os.path.exists(sol_img):
                parent_images.append(sol_img)

        return {
            'vector': vec.tolist(),
            'reward': db_entry[2],
            'ranking': db_entry[1],
            'drag': metrics.get('drag', 0),
            'lift': metrics.get('lift', 0),
            'feedback': feedback,
            'images': parent_images,
        }

    def get_prompt_blocks(self) -> dict:
        reward_pb = self.reward.get_prompt_blocks()
        if reward_pb is not None:
            return reward_pb
        return {
            'format_context': prompt_blocks.format_context,
            'format_response_instructions': prompt_blocks.format_response_instructions,
            'CONTEXT_FORMAT': prompt_blocks.CONTEXT_FORMAT,
            'DESIGN_ENTRY': prompt_blocks.DESIGN_ENTRY,
            'RESPONSE_FORMAT': prompt_blocks.RESPONSE_FORMAT,
        }

    def run_llm_action(self, action, context_entries, output_dir, name,
                       debug_dir=None, parent_path=None, scratchpad=""):
        from . import agent
        pb = self.get_prompt_blocks()
        agent.set_env_format_context(pb['format_context'])
        return agent.run_llm_action(
            action, context_entries, output_dir,
            base_csv=parent_path, name=name, skip_vis=True,
            debug_dir=debug_dir, scratchpad=scratchpad)

    def get_reflection_inputs(self, design_path: str, case_dir: str):
        import json
        import numpy as np
        import subprocess
        import sys

        debug_dir = os.path.join(case_dir, 'context')

        params_path = os.path.join(debug_dir, 'llm_params.json')
        if not os.path.exists(params_path):
            return None

        with open(params_path) as f:
            intended_params = json.load(f)

        designer_analysis = ""
        analysis_path = os.path.join(debug_dir, 'llm_analysis.txt')
        if os.path.exists(analysis_path):
            with open(analysis_path) as f:
                designer_analysis = f.read().strip()

        designer_reasoning = ""
        reasoning_path = os.path.join(debug_dir, 'llm_rationale.txt')
        if os.path.exists(reasoning_path):
            with open(reasoning_path) as f:
                designer_reasoning = f.read().strip()

        try:
            actual_action = np.loadtxt(design_path, delimiter=',').flatten().tolist()
        except Exception:
            actual_action = []

        geometry_image_path = None
        try:
            vis_dir = os.path.join(case_dir, 'geometry')
            os.makedirs(vis_dir, exist_ok=True)
            test_script = os.path.join(ENV_DIR, 'test_modification.py')
            if os.path.exists(test_script):
                subprocess.run(
                    [sys.executable, test_script, design_path, '-o', vis_dir],
                    capture_output=True, text=True, timeout=60,
                )
                for fname in os.listdir(vis_dir):
                    if fname.endswith('_geometry.png'):
                        geometry_image_path = os.path.join(vis_dir, fname)
                        break
        except Exception as e:
            print(f"  Geometry visualization failed (non-fatal): {e}")

        return {
            'intended_params': intended_params,
            'actual_action': actual_action,
            'designer_analysis': designer_analysis,
            'designer_reasoning': designer_reasoning,
            'geometry_image_path': geometry_image_path,
        }

    def get_results_csv_columns(self):
        return ['drag', 'lift']

    def get_results_csv_row(self, metrics):
        return [
            f"{metrics.get('drag', 0):.6f}",
            f"{metrics.get('lift', 0):.6f}",
        ]
