import os
import sys
import json
import numpy as np

from environments.base import BaseEnvironment
from environments.base_reward import BaseReward
from . import prompt_blocks

ENV_DIR = os.path.dirname(os.path.abspath(__file__))
NEURALFOIL_SRC = os.path.join(ENV_DIR, 'neuralfoil_src')

if os.path.isdir(NEURALFOIL_SRC) and NEURALFOIL_SRC not in sys.path:
    sys.path.insert(0, NEURALFOIL_SRC)

import neuralfoil as nf
import aerosandbox as asb

FAIL_REWARD = -5.0


def _save_images(coords, save_dir, alpha):
    """Generate only the airfoil shape plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    shape_path = os.path.join(save_dir, 'shape.png')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(coords[:, 0], coords[:, 1], 'b-', linewidth=1.5)
    ax.fill(coords[:, 0], coords[:, 1], alpha=0.15, color='steelblue')
    ax.set_aspect('equal')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.set_title(f'Airfoil Shape  (α = {alpha}°)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(shape_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

    # Remove stale cp image from prior runs; this env no longer emits Cp plots.
    cp_path = os.path.join(save_dir, 'cp.png')
    if os.path.exists(cp_path):
        os.remove(cp_path)

    return [shape_path]


class NeuralFoilEnvironment(BaseEnvironment):
    """NeuralFoil 2D airfoil aerodynamics environment.

    Design files are JSON with 18 Kulfan (CST) parameters:
        upper_weights      -- 8 upper-surface CST coefficients
        lower_weights      -- 8 lower-surface CST coefficients
        leading_edge_weight -- leading-edge modification weight
        TE_thickness        -- trailing-edge thickness (fraction of chord)
    """

    def __init__(self, reward: BaseReward, **kwargs):
        self.reward = reward

    @staticmethod
    def add_args(parser):
        pass

    def _run_sim(self, design_path: str, case_dir: str,
                 alpha=5.0, re=1e6, model_size="large", n_crit=9.0) -> dict:
        """Run NeuralFoil for one set of flight conditions; return raw aero outputs."""
        os.makedirs(case_dir, exist_ok=True)
        save_dir = os.path.join(case_dir, 'save')
        os.makedirs(save_dir, exist_ok=True)

        with open(design_path) as f:
            params = json.load(f)

        kulfan_params = {
            "upper_weights": np.array(params["upper_weights"], dtype=float),
            "lower_weights": np.array(params["lower_weights"], dtype=float),
            "leading_edge_weight": float(params["leading_edge_weight"]),
            "TE_thickness": float(params["TE_thickness"]),
        }

        aero = nf.get_aero_from_kulfan_parameters(
            kulfan_parameters=kulfan_params,
            alpha=alpha,
            Re=re,
            n_crit=n_crit,
            model_size=model_size,
        )

        CL = float(np.squeeze(aero["CL"]))
        CD = float(np.squeeze(aero["CD"]))
        CM = float(np.squeeze(aero["CM"]))
        Top_Xtr = float(np.squeeze(aero["Top_Xtr"]))
        Bot_Xtr = float(np.squeeze(aero["Bot_Xtr"]))
        analysis_confidence = float(np.squeeze(aero["analysis_confidence"]))

        airfoil = asb.KulfanAirfoil(
            upper_weights=kulfan_params["upper_weights"],
            lower_weights=kulfan_params["lower_weights"],
            leading_edge_weight=kulfan_params["leading_edge_weight"],
            TE_thickness=kulfan_params["TE_thickness"],
        )
        coords = np.array(airfoil.coordinates)

        images = _save_images(coords, save_dir, alpha)

        return {
            "CL": CL, "CD": CD, "CM": CM,
            "Top_Xtr": Top_Xtr, "Bot_Xtr": Bot_Xtr,
            "analysis_confidence": analysis_confidence,
            "kulfan_params": params,
            "save_dir": save_dir,
            "images": images,
        }

    def simulate(self, design_path: str, case_dir: str, **kwargs) -> tuple:
        os.makedirs(case_dir, exist_ok=True)
        return self.reward.evaluate(self._run_sim, design_path, case_dir)

    def build_context_entry(self, db_entry) -> dict:
        json_path = db_entry[0]
        with open(json_path) as f:
            d = json.load(f)

        params = {
            "upper_weights": d.get("upper_weights", []),
            "lower_weights": d.get("lower_weights", []),
            "leading_edge_weight": d.get("leading_edge_weight", 0.0),
            "TE_thickness": d.get("TE_thickness", 0.0),
        }

        results = db_entry[3] if len(db_entry) > 3 else {}
        metrics = results.get("metrics", {}) if isinstance(results, dict) else {}
        images_list = results.get("images", []) if isinstance(results, dict) else []
        feedback = results.get("feedback", "") if isinstance(results, dict) else ""
        parent_images = [p for p in images_list if isinstance(p, str) and os.path.exists(p)]

        return {
            "params": params,
            "reward": db_entry[2],
            "ranking": db_entry[1],
            "metrics": metrics,
            "feedback": feedback,
            "images": parent_images,
        }

    def get_prompt_blocks(self) -> dict:
        reward_pb = self.reward.get_prompt_blocks()
        if reward_pb is not None:
            return reward_pb
        return {
            "format_context": prompt_blocks.format_context,
            "format_response_instructions": prompt_blocks.format_response_instructions,
            "CONTEXT_FORMAT": prompt_blocks.CONTEXT_FORMAT,
            "DESIGN_ENTRY": prompt_blocks.DESIGN_ENTRY,
            "RESPONSE_FORMAT": prompt_blocks.RESPONSE_FORMAT,
        }

    def run_llm_action(self, action, context_entries, output_dir, name,
                       debug_dir=None, parent_path=None, scratchpad=""):
        from . import agent
        pb = self.get_prompt_blocks()
        agent.set_env_format_context(pb['format_context'])
        return agent.run_llm_action_neuralfoil(
            action, context_entries, output_dir, name=name,
            debug_dir=debug_dir, scratchpad=scratchpad)

    def get_reflection_inputs(self, design_path: str, case_dir: str):
        import json

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

        # Actual params: load from the design JSON file
        actual_action = {}
        try:
            with open(design_path) as f:
                actual_action = json.load(f)
        except Exception:
            pass

        # Shape image is already generated by simulate(); no subprocess needed
        geometry_image_path = None
        shape_path = os.path.join(case_dir, 'save', 'shape.png')
        if os.path.exists(shape_path):
            geometry_image_path = shape_path

        return {
            'intended_params': intended_params,
            'actual_action': actual_action,
            'designer_analysis': designer_analysis,
            'designer_reasoning': designer_reasoning,
            'geometry_image_path': geometry_image_path,
        }

    def sample_gaussian(self, mean_params: dict, output_dir: str, name: str,
                        std_scale: float = 1.0) -> str:
        from .design_actions import gaussian_neuralfoil
        return gaussian_neuralfoil(
            params=mean_params, out_dir=output_dir, name=name, std_scale=std_scale)

    def get_param_bounds(self):
        from .design_actions import UPPER_BOUNDS, LOWER_BOUNDS, LE_BOUNDS, TE_BOUNDS, N_CST
        lb = np.array([UPPER_BOUNDS[0]] * N_CST + [LOWER_BOUNDS[0]] * N_CST
                      + [LE_BOUNDS[0], TE_BOUNDS[0]])
        ub = np.array([UPPER_BOUNDS[1]] * N_CST + [LOWER_BOUNDS[1]] * N_CST
                      + [LE_BOUNDS[1], TE_BOUNDS[1]])
        return lb, ub

    def get_named_param_bounds(self):
        from .design_actions import UPPER_BOUNDS, LOWER_BOUNDS, LE_BOUNDS, TE_BOUNDS
        return {
            'upper_weights':       UPPER_BOUNDS,
            'lower_weights':       LOWER_BOUNDS,
            'leading_edge_weight': LE_BOUNDS,
            'TE_thickness':        TE_BOUNDS,
        }

    def write_design(self, x, output_dir: str, name: str) -> str:
        from .design_actions import save_design_json, N_CST
        os.makedirs(output_dir, exist_ok=True)
        params = {
            "upper_weights":       [float(v) for v in x[:N_CST]],
            "lower_weights":       [float(v) for v in x[N_CST:2 * N_CST]],
            "leading_edge_weight": float(x[2 * N_CST]),
            "TE_thickness":        float(x[2 * N_CST + 1]),
            "name":                name,
        }
        path = os.path.join(output_dir, f"{name}.json")
        return save_design_json(path, params)

    def set_llm_backend(self, backend, image_analyzer=None):
        from . import agent
        agent.set_llm_backend(backend, image_analyzer)

    def set_designer_model(self, name: str):
        from . import agent
        agent.set_designer_model(name)

    def get_results_csv_columns(self):
        return [
            'CL',
            'CD',
            'L_D',
            'CM',
            'weighted_CD_mean',
            'weighted_CD_mean_solved',
            'fitness_objective',
            'fitness_penalty',
            'fitness_total',
            'total_violation',
            'feasible',
            'n_solved',
            'n_unreachable',
        ]

    def get_results_csv_row(self, metrics):
        CL = metrics.get('CL', 0.0)
        CD = metrics.get('CD', 0.0)
        CM = metrics.get('CM', 0.0)
        LD = (CL / CD) if CD != 0 else 0.0
        weighted_cd_mean = metrics.get('weighted_CD_mean')
        weighted_cd_mean_solved = metrics.get('weighted_CD_mean_solved')
        fitness_objective = float(metrics.get('fitness_objective', 0.0))
        fitness_penalty = float(metrics.get('fitness_penalty', 0.0))
        fitness_total = float(metrics.get('fitness_total', metrics.get('reward', 0.0)))
        total_violation = float(metrics.get('total_violation', 0.0))
        feasible = bool(metrics.get('feasible', False))
        n_solved = len(metrics.get('alphas', [])) if isinstance(metrics.get('alphas', []), list) else 0
        n_unreachable = int(metrics.get('constraint_violations', {}).get('cl_target_unreachable', 0.0)) \
            if isinstance(metrics.get('constraint_violations', {}), dict) else 0

        def _fmt_optional(v, fmt):
            return '' if v is None else fmt.format(v)

        return [
            f"{CL:.6f}",
            f"{CD:.8f}",
            f"{LD:.4f}",
            f"{CM:.6f}",
            _fmt_optional(weighted_cd_mean, "{:.8f}"),
            _fmt_optional(weighted_cd_mean_solved, "{:.8f}"),
            f"{fitness_objective:.8f}",
            f"{fitness_penalty:.8f}",
            f"{fitness_total:.8f}",
            f"{total_violation:.8f}",
            '1' if feasible else '0',
            str(n_solved),
            str(n_unreachable),
        ]
