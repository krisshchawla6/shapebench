import os
import sys
import json
import importlib.util
import numpy as np

from environments.base import BaseEnvironment
from . import prompt_blocks

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS_DIR = os.path.join(REPO_ROOT, 'scripts')

BASELINE_LD = 5.45
FAIL_REWARD = -5.0

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

    def __init__(self, aoa=10.0, mach=0.3, re=3.0e6, **kwargs):
        self.aoa = aoa
        self.mach = mach
        self.re = re

    @staticmethod
    def add_args(parser):
        parser.add_argument('--aoa', type=float, default=10.0, help='Angle of attack (deg)')
        parser.add_argument('--mach', type=float, default=0.3, help='Mach number')
        parser.add_argument('--re', type=float, default=3.0e6, help='Reynolds number')

    def simulate(self, design_path: str, case_dir: str, **kwargs) -> tuple:
        os.makedirs(case_dir, exist_ok=True)
        pipeline = _get_pipeline()

        with open(design_path) as f:
            params = json.load(f)

        try:
            vlm_res, corr_res, predicted_dcp, geometry = pipeline.run_full_pipeline(
                params, self.aoa, self.mach, self.re, case_dir,
            )

            cl = float(np.squeeze(corr_res.CL))
            cdi = float(np.squeeze(corr_res.CDi))
            cm = float(np.squeeze(corr_res.CM))
            ld = cl / cdi if abs(cdi) > 1e-12 else 0.0
            reward = ld - BASELINE_LD

            pipeline.save_results_json(params, self.aoa, self.mach, self.re,
                                       vlm_res, corr_res, case_dir)
            pipeline.save_geometry_png(params, self.aoa, self.mach, vlm_res,
                                       corr_res, predicted_dcp, geometry, case_dir)
        except Exception as e:
            print(f"[sim] FAILED: {e}")
            cl, cdi, cm, ld = 0.0, 0.0, 0.0, 0.0
            reward = FAIL_REWARD

        results = self._post_process(case_dir, cl, cdi, cm, ld, reward)
        return float(reward), results

    def _post_process(self, case_dir, cl, cdi, cm, ld, reward):
        geometry_png = os.path.join(case_dir, 'geometry.png')
        images = [geometry_png] if os.path.exists(geometry_png) else []
        return {
            'metrics': {'CL': cl, 'CDi': cdi, 'CM': cm, 'L_D': ld, 'reward': reward},
            'images': images,
            'feedback': '',
        }

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
        images_list = results.get('images', []) if isinstance(results, dict) else []
        feedback = results.get('feedback', '') if isinstance(results, dict) else ''

        parent_images = [p for p in images_list if isinstance(p, str) and os.path.exists(p)]

        return {
            'params': params,
            'reward': db_entry[2],
            'ranking': db_entry[1],
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
