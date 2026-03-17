import os
import json
import importlib.util
import numpy as np

from environments.base import BaseEnvironment
from environments.base_reward import BaseReward
from . import prompt_blocks

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS_DIR = os.path.join(REPO_ROOT, 'scripts')

_pipeline_mod = None


def _get_pipeline():
    global _pipeline_mod
    if _pipeline_mod is None:
        spec = importlib.util.spec_from_file_location(
            "generate_design_corrected",
            os.path.join(SCRIPTS_DIR, "generate_design_corrected.py"),
        )
        _pipeline_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_pipeline_mod)
    return _pipeline_mod


class VLM3DEnvironment(BaseEnvironment):
    """3D VLM + VortexNet corrected delta wing environment."""

    def __init__(self, reward: BaseReward, **kwargs):
        self.reward = reward

    def _run_sim(self, design_path: str, case_dir: str, aoa=10.0, mach=0.3, re=3.0e6) -> dict:
        """Run a single VLM simulation; return raw aerodynamic outputs."""
        os.makedirs(case_dir, exist_ok=True)
        pipeline = _get_pipeline()
        with open(design_path) as f:
            params = json.load(f)
        vlm_res, corr_res, predicted_dcp, geometry = pipeline.run_full_pipeline(
            params, aoa, mach, re, case_dir,
        )
        cl = float(np.squeeze(corr_res.CL))
        cdi = float(np.squeeze(corr_res.CDi))
        cm = float(np.squeeze(corr_res.CM))
        pipeline.save_results_json(params, aoa, mach, re, vlm_res, corr_res, case_dir)
        pipeline.save_geometry_png(params, aoa, mach, vlm_res, corr_res,
                                   predicted_dcp, geometry, case_dir)
        geometry_png = os.path.join(case_dir, 'geometry.png')
        images = [geometry_png] if os.path.exists(geometry_png) else []
        return {'cl': cl, 'cdi': cdi, 'cm': cm, 'images': images}

    def simulate(self, design_path: str, case_dir: str, **kwargs) -> tuple:
        os.makedirs(case_dir, exist_ok=True)
        return self.reward.evaluate(self._run_sim, design_path, case_dir)

    def build_context_entry(self, db_entry) -> dict:
        json_path = db_entry[0]
        with open(json_path) as f:
            d = json.load(f)

        params = {
            'le_sweep': d.get('le_sweep'),
            'root_chord_in': d.get('root_chord_in', 25.734),
            'twist_root': d.get('twist_root', 0.0),
            'twist_tip': d.get('twist_tip', 0.0),
            'dihedral': d.get('dihedral', 0.0),
            'naca_m': d.get('naca', {}).get('m', 0),
            'naca_p': d.get('naca', {}).get('p', 0),
            'naca_t': d.get('naca', {}).get('t', 12),
        }

        results = db_entry[3] if len(db_entry) > 3 else {}
        metrics = results.get('metrics', {}) if isinstance(results, dict) else {}
        images_list = results.get('images', []) if isinstance(results, dict) else []
        feedback = results.get('feedback', '') if isinstance(results, dict) else ''
        parent_images = [p for p in images_list if isinstance(p, str) and os.path.exists(p)]

        return {
            'params': params,
            'reward': db_entry[2],
            'ranking': db_entry[1],
            'metrics': metrics,
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
        return agent.run_llm_action_3d(
            action, context_entries, output_dir, name=name, debug_dir=debug_dir)

    def set_llm_backend(self, backend, image_analyzer=None):
        from . import agent
        agent.set_llm_backend(backend, image_analyzer)

    def get_results_csv_columns(self):
        return ['CL', 'CDi', 'CM', 'L_D']

    def get_results_csv_row(self, metrics):
        return [
            f"{metrics.get('CL', 0):.6f}",
            f"{metrics.get('CDi', 0):.6f}",
            f"{metrics.get('CM', 0):.6f}",
            f"{metrics.get('L_D', 0):.4f}",
        ]
