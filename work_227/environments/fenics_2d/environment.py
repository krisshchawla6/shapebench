import os
import sys
import shutil
import numpy as np

from environments.base import BaseEnvironment

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
            # The original environment.py uses star-imports and a class called `env`.
            # We import it dynamically so the heavy FEniCS dep is only loaded when needed.
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

    def __init__(self, **kwargs):
        self._env_root = ENV_DIR

    @staticmethod
    def add_args(parser):
        pass

    def simulate(self, design_path: str, case_dir: str, **kwargs) -> tuple:
        sim_env = _SimEnvSingleton.get(self._env_root)

        airfoil_shape = np.loadtxt(design_path, delimiter=',')
        if airfoil_shape.ndim == 2:
            airfoil_shape = airfoil_shape[0]

        # Reset to baseline before each run
        from .shapes_utils import Shape  # noqa: F811
        sim_env.shape.index = 0
        sim_env.reset()

        _next_state, _terminal, reward = sim_env.execute(airfoil_shape)

        case_save_dir = os.path.join(case_dir, 'save')
        if os.path.exists('./save'):
            if os.path.exists(case_save_dir):
                shutil.rmtree(case_save_dir)
            shutil.copytree('./save', case_save_dir)

        results = self._post_process(case_save_dir, reward)
        return float(reward), results

    def _post_process(self, save_dir, reward=None):
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
            'metrics': {'drag': drag, 'lift': lift},
            'images': sol_images,
            'feedback': analysis_text,
            'shape_image': shape_image,
        }

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
        return {
            'format_context': prompt_blocks.format_context,
            'format_response_instructions': prompt_blocks.format_response_instructions,
            'CONTEXT_FORMAT': prompt_blocks.CONTEXT_FORMAT,
            'DESIGN_ENTRY': prompt_blocks.DESIGN_ENTRY,
            'RESPONSE_FORMAT': prompt_blocks.RESPONSE_FORMAT,
        }
