import os
import json
import importlib.util
import numpy as np

from environments.base import BaseEnvironment
from . import prompt_blocks

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS_DIR = os.path.join(REPO_ROOT, 'scripts')

FAIL_REWARD = -5.0

# Paper Table 3 flight conditions
MACH_SUP = 1.8
MACH_SUB = 0.3
RE_SUP = 80.4e6
RE_SUB = 101.8e6
CL_TARGET_SUP = 0.1665
CL_TARGET_SUB = 0.6933
# Paper Table 5: CM = 0 (equality) at both conditions
CM_TARGET = 0.0
# Static margin default target (paper tests 0%, 5%, 10%)
KN_TARGET_DEFAULT = 0.05

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


def _run_single(pipeline, params, aoa, mach, re, out_dir):
    """Run pipeline for one flight condition, return (CL, CDi, CM) or raise."""
    os.makedirs(out_dir, exist_ok=True)
    vlm_res, corr_res, predicted_dcp, geometry = pipeline.run_full_pipeline(
        params, aoa, mach, re, out_dir,
    )
    cl = float(np.squeeze(corr_res.CL))
    cdi = float(np.squeeze(corr_res.CDi))
    cm = float(np.squeeze(corr_res.CM))
    pipeline.save_results_json(params, aoa, mach, re, vlm_res, corr_res, out_dir)
    pipeline.save_geometry_png(params, aoa, mach, vlm_res, corr_res,
                               predicted_dcp, geometry, out_dir)
    return cl, cdi, cm


class VLM3D2PtEnvironment(BaseEnvironment):
    """Two-point evaluation environment per Yiren et al.

    Each design is evaluated at supersonic and subsonic conditions.
    Reward = -CD_sup - penalty terms for constraint violations.
    Constraints (from the paper):
      CL = CL* at both conditions, CM = 0 at both, Kn >= Kn* (subsonic).
    """

    def __init__(self, aoa_sup=0.0, aoa_sub=10.0,
                 kn_target=KN_TARGET_DEFAULT, delta_aoa=0.01,
                 w_cl=10.0, w_cm=100.0, w_kn=100.0, **kwargs):
        self.aoa_sup = aoa_sup
        self.aoa_sub = aoa_sub
        self.kn_target = kn_target
        self.delta_aoa = delta_aoa
        self.w_cl = w_cl
        self.w_cm = w_cm
        self.w_kn = w_kn

    @staticmethod
    def add_args(parser):
        g = parser.add_argument_group('Two-point flight conditions (Yiren et al.)')
        g.add_argument('--aoa-sup', type=float, default=0.0,
                       help='Supersonic AOA, deg (paper range: -3 to 3)')
        g.add_argument('--aoa-sub', type=float, default=10.0,
                       help='Subsonic AOA, deg (paper range: -5 to 20)')
        g.add_argument('--kn-target', type=float, default=KN_TARGET_DEFAULT,
                       help='Static margin target (fraction, e.g. 0.05 = 5%%)')
        g.add_argument('--delta-aoa', type=float, default=0.01,
                       help='AOA perturbation for Kn finite difference (deg)')
        g = parser.add_argument_group('Penalty weights')
        g.add_argument('--w-cl', type=float, default=10.0,
                       help='Weight for CL target miss penalty')
        g.add_argument('--w-cm', type=float, default=100.0,
                       help='Weight for CM != 0 penalty')
        g.add_argument('--w-kn', type=float, default=100.0,
                       help='Weight for static margin violation penalty')

    # ------------------------------------------------------------------
    def simulate(self, design_path: str, case_dir: str, **kwargs) -> tuple:
        os.makedirs(case_dir, exist_ok=True)
        pipeline = _get_pipeline()

        with open(design_path) as f:
            params = json.load(f)

        sup_dir = os.path.join(case_dir, 'sup')
        sub_dir = os.path.join(case_dir, 'sub')
        kn_dir = os.path.join(case_dir, 'sub_kn')

        try:
            cl_sup, cdi_sup, cm_sup = _run_single(
                pipeline, params, self.aoa_sup, MACH_SUP, RE_SUP, sup_dir)
            cl_sub, cdi_sub, cm_sub = _run_single(
                pipeline, params, self.aoa_sub, MACH_SUB, RE_SUB, sub_dir)

            # Third sim for static margin (subsonic, perturbed AOA)
            cl_sub_p, _, cm_sub_p = _run_single(
                pipeline, params,
                self.aoa_sub + self.delta_aoa, MACH_SUB, RE_SUB, kn_dir)

            kn = self._compute_static_margin(
                cl_sub, cm_sub, cl_sub_p, cm_sub_p, self.delta_aoa)

            reward = self._compute_reward(
                cdi_sup, cl_sup, cl_sub, cm_sup, cm_sub, kn)

            print(f"[2pt] sup: CL={cl_sup:.4f} CDi={cdi_sup:.5f} CM={cm_sup:.4f}")
            print(f"[2pt] sub: CL={cl_sub:.4f} CDi={cdi_sub:.5f} CM={cm_sub:.4f}")
            print(f"[2pt] Kn={kn:.4f}  reward={reward:.4f}")

        except Exception as e:
            print(f"[2pt] FAILED: {e}")
            cl_sup = cdi_sup = cm_sup = 0.0
            cl_sub = cdi_sub = cm_sub = 0.0
            kn = 0.0
            reward = FAIL_REWARD

        metrics = {
            'CL_sup': cl_sup, 'CDi_sup': cdi_sup, 'CM_sup': cm_sup,
            'CL_sub': cl_sub, 'CDi_sub': cdi_sub, 'CM_sub': cm_sub,
            'Kn': kn, 'reward': reward,
        }
        results = self._post_process(case_dir, metrics)
        return float(reward), results

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_static_margin(cl, cm, cl_p, cm_p, delta_aoa):
        cl_alpha = (cl_p - cl) / delta_aoa
        cm_alpha = (cm_p - cm) / delta_aoa
        if abs(cl_alpha) < 1e-12:
            return 0.0
        return -cm_alpha / cl_alpha

    def _compute_reward(self, cd_sup, cl_sup, cl_sub, cm_sup, cm_sub, kn):
        reward = -cd_sup
        reward -= self.w_cl * (cl_sup - CL_TARGET_SUP) ** 2
        reward -= self.w_cl * (cl_sub - CL_TARGET_SUB) ** 2
        reward -= self.w_cm * cm_sup ** 2
        reward -= self.w_cm * cm_sub ** 2
        if kn < self.kn_target:
            reward -= self.w_kn * (self.kn_target - kn) ** 2
        return reward

    # ------------------------------------------------------------------
    def _post_process(self, case_dir, metrics):
        images = []
        for subdir in ('sup', 'sub'):
            png = os.path.join(case_dir, subdir, 'geometry.png')
            if os.path.exists(png):
                images.append(png)
        return {'metrics': metrics, 'images': images, 'feedback': ''}

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
        return {
            'format_context': prompt_blocks.format_context,
            'format_response_instructions': prompt_blocks.format_response_instructions,
            'CONTEXT_FORMAT': prompt_blocks.CONTEXT_FORMAT,
            'DESIGN_ENTRY': prompt_blocks.DESIGN_ENTRY,
            'RESPONSE_FORMAT': prompt_blocks.RESPONSE_FORMAT,
        }
