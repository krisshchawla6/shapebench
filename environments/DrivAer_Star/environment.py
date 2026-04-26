import fcntl
import os
import sys
import json
import numpy as np

from environments.base import BaseEnvironment
from environments.base_reward import BaseReward
from . import prompt_blocks
from .mesh_generator import deform_vtk, extract_model_input, PARAM_KEYS

ENV_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ENV_DIR, "model")

sys.path.insert(0, MODEL_DIR)

# Serialise the torchvision + Transolver import across all processes on the
# same node so concurrent jobs don't race on the __pycache__ write.
# /tmp is node-local, making fcntl.lockf reliable here.
_LOCK_FILE = "/tmp/drivaerstar_torchvision_import.lock"
with open(_LOCK_FILE, "a") as _lf:
    fcntl.lockf(_lf, fcntl.LOCK_EX)
    try:
        import torchvision.extension  # loads _C before _meta_registrations runs
        from Transolver import Model as _TransolverModel
    finally:
        fcntl.lockf(_lf, fcntl.LOCK_UN)

_model = None
_norm = None
_device = None
_norm_stats_path_loaded = None


def _load_model(norm_stats_path=None):
    global _model, _norm, _device, _norm_stats_path_loaded
    if norm_stats_path is None:
        norm_stats_path = os.path.join(MODEL_DIR, "norm_stats.pt")
    # Re-load if a different norm_stats file is requested (e.g. switching body style).
    if _model is not None and _norm_stats_path_loaded == norm_stats_path:
        return

    import torch
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(norm_stats_path):
        raise FileNotFoundError(
            f"{norm_stats_path} not found. Run: "
            f"python model/setup_model.py --vtk-dir <path> --norm-output {norm_stats_path}")
    _norm = torch.load(norm_stats_path, map_location=_device, weights_only=True)

    _model = _TransolverModel(
        space_dim=7, n_layers=4, n_hidden=64, dropout=0.0,
        n_head=4, act='gelu', mlp_ratio=1, fun_dim=0, out_dim=4,
        slice_num=16, ref=8, unified_pos=0,
    )
    ckpt = torch.load(os.path.join(MODEL_DIR, "transolver_best.pt"),
                      map_location=_device, weights_only=True)
    _model.load_state_dict(ckpt)
    _model.to(_device)
    _model.eval()
    _norm_stats_path_loaded = norm_stats_path


def _run_surrogate(mesh, norm_stats_path=None):
    """Run Transolver inference on a VTK mesh, return predicted fields."""
    import torch
    _load_model(norm_stats_path)
    x_np = extract_model_input(mesh)
    x_t = torch.from_numpy(x_np).unsqueeze(0).to(_device)

    x_norm = (x_t - _norm["x_mean"]) / (_norm["x_std"] + 1e-7)

    with torch.no_grad():
        out_norm = _model(x_norm)

    out = out_norm * (_norm["y_std"] + 1e-7) + _norm["y_mean"]
    out = out.squeeze(0).cpu().numpy()

    return {
        "x": x_np,
        "pressure": out[:, 0],
        "wss_i": out[:, 1],
        "wss_j": out[:, 2],
        "wss_k": out[:, 3],
    }


def _compute_drag(x, result):
    """Compute drag force from predicted fields.

    drag = sum(pressure * area * normal_x) + sum(wss_x * area)
    """
    areas = x[:, 3]
    normal_x = x[:, 4]
    drag_press = float(np.sum(result["pressure"] * areas * normal_x))
    drag_wss = float(np.sum(result["wss_i"] * areas))
    return drag_press + drag_wss, drag_press, drag_wss


def _compute_lift(x, result):
    """Compute lift force from predicted fields."""
    areas = x[:, 3]
    normal_z = x[:, 6]
    lift_press = float(np.sum(result["pressure"] * areas * normal_z))
    lift_wss = float(np.sum(result["wss_k"] * areas))
    return lift_press + lift_wss, lift_press, lift_wss


def _render_field(mesh, scalar, out_path, title, cmap, clim, view, fmt="%.3f"):
    import pyvista as pv
    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    pl.set_background("white")
    pl.add_mesh(mesh, scalars=scalar, cmap=cmap, clim=clim,
                smooth_shading=True, show_edges=False,
                scalar_bar_args={"title": title, "shadow": True,
                                 "fmt": fmt, "n_labels": 9,
                                 "title_font_size": 18, "label_font_size": 14,
                                 "position_x": 0.80, "width": 0.14,
                                 "color": "black"})
    if view == "top":
        pl.view_xy()
        pl.camera.zoom(1.3)
    elif view == "side":
        pl.view_xz()
        pl.camera.zoom(1.2)
    else:
        pl.camera_position = [(2.5, -4.0, 1.5), (2.0, 0.0, 0.5), (0, 0, 1)]
    pl.add_axes(color="black")
    pl.screenshot(out_path, transparent_background=False)
    pl.close()


def _save_sol_images(mesh, result, sol_dir):
    from scipy.spatial import cKDTree
    centers = mesh.cell_centers().points
    tree = cKDTree(result["x"][:, :3])
    _, idx = tree.query(centers.astype(np.float32), k=1)

    mesh.cell_data["Pressure"] = result["pressure"][idx]
    mesh.cell_data["WSSx"] = result["wss_i"][idx]

    P = result["pressure"][idx]
    W = result["wss_i"][idx]
    p_clim = (float(np.percentile(P, 2)), float(np.percentile(P, 98)))
    w_lim = float(np.percentile(np.abs(W), 98))
    w_clim = (-w_lim, w_lim)

    images = []
    for scalar, title, cmap, clim, fmt in [
        ("Pressure", "Pressure", "coolwarm", p_clim, "%.1f"),
        ("WSSx", "Wall Shear Stress (x)", "seismic", w_clim, "%.2f"),
    ]:
        for view in ["iso", "top", "side"]:
            path = os.path.join(sol_dir, f"{scalar}_{view}.png")
            _render_field(mesh, scalar, path, f"{title} ({view})", cmap, clim, view, fmt)
            images.append(path)

    return images


class DrivAerStarEnvironment(BaseEnvironment):
    """Transolver surrogate for DrivAerStar vehicle aerodynamics."""

    def __init__(self, reward: BaseReward, base_vtk=None, render_images=False,
                 save_fields=False, norm_stats_path=None, **kwargs):
        self.reward = reward
        self.base_vtk = base_vtk
        self.render_images = render_images
        self.save_fields = save_fields
        self.norm_stats_path = norm_stats_path  # None → default model/norm_stats.pt

    @staticmethod
    def add_args(parser):
        default_vtk = os.path.join(ENV_DIR, "data", "vtk_E", "00000.vtk")
        parser.add_argument('--base_vtk', type=str, default=default_vtk,
                            help='Path to base VTK mesh for FFD deformation')
        parser.add_argument('--norm_stats_path', type=str, default=None,
                            help='Path to norm_stats.pt for Transolver normalisation '
                                 '(default: model/norm_stats.pt, computed from vtk_E). '
                                 'Use model/norm_stats_F.pt or norm_stats_N.pt for other body styles.')
        parser.add_argument('--render_images', action='store_true', default=False,
                            help='Render solution images per evaluation (slow; off by default)')
        parser.add_argument('--save_fields', action='store_true', default=False,
                            help='Save fields.npz per evaluation (large; off by default)')

    def _run_sim(self, design_path: str, case_dir: str) -> dict:
        """Run surrogate on design; return raw forces and field outputs."""
        save_dir = os.path.join(case_dir, "save")
        sol_dir = os.path.join(save_dir, "sol")
        os.makedirs(sol_dir, exist_ok=True)

        with open(design_path) as f:
            params = json.load(f)

        vtk_path = params.get("vtk_path", self.base_vtk)
        if not vtk_path or not os.path.exists(vtk_path):
            raise FileNotFoundError(
                "No base VTK mesh. Set --base_vtk or include 'vtk_path' in design JSON.")

        mesh = deform_vtk(vtk_path, params)
        result = _run_surrogate(mesh, self.norm_stats_path)
        drag, drag_p, drag_w = _compute_drag(result["x"], result)
        lift, lift_p, lift_w = _compute_lift(result["x"], result)

        if self.save_fields:
            np.savez(os.path.join(save_dir, "fields.npz"),
                     x=result["x"],
                     pressure=result["pressure"],
                     wss_i=result["wss_i"],
                     wss_j=result["wss_j"],
                     wss_k=result["wss_k"])

        images = _save_sol_images(mesh, result, sol_dir) if self.render_images else []

        return {
            "params": params,
            "drag": drag, "drag_p": drag_p, "drag_w": drag_w,
            "lift": lift, "lift_p": lift_p, "lift_w": lift_w,
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

        params = {k: d.get(k) for k in PARAM_KEYS}
        results = db_entry[3] if len(db_entry) > 3 else {}
        images_list = results.get("images", []) if isinstance(results, dict) else []
        feedback = results.get("feedback", "") if isinstance(results, dict) else ""
        parent_images = [p for p in images_list if isinstance(p, str) and os.path.exists(p)]

        return {
            "params": params,
            "reward": db_entry[2],
            "ranking": db_entry[1],
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
        return agent.run_llm_action_drivaer(
            action, context_entries, output_dir, name=name, debug_dir=debug_dir)

    def sample_gaussian(self, mean_params: dict, output_dir: str, name: str,
                        std_scale: float = 1.0) -> str:
        from .design_actions import gaussian_drivaer
        return gaussian_drivaer(
            params=mean_params, out_dir=output_dir, name=name, std_scale=std_scale)

    def get_param_bounds(self):
        from .design_actions import CONTINUOUS_KEYS, BOUNDS
        lb = np.array([BOUNDS[k][0] for k in CONTINUOUS_KEYS])
        ub = np.array([BOUNDS[k][1] for k in CONTINUOUS_KEYS])
        return lb, ub

    def write_design(self, x, output_dir: str, name: str) -> str:
        from .design_actions import CONTINUOUS_KEYS, save_design_json
        os.makedirs(output_dir, exist_ok=True)
        params = {k: float(x[i]) for i, k in enumerate(CONTINUOUS_KEYS)}
        params['name'] = name
        path = os.path.join(output_dir, f'{name}.json')
        return save_design_json(path, params)

    def set_llm_backend(self, backend, image_analyzer=None):
        from . import agent
        agent.set_llm_backend(backend, image_analyzer)

    def get_results_csv_columns(self):
        return ['drag', 'Cd', 'lift']

    def get_results_csv_row(self, metrics):
        return [
            f"{metrics.get('drag', 0):.4f}",
            f"{metrics.get('Cd', 0):.8f}",
            f"{metrics.get('lift', 0):.4f}",
        ]
