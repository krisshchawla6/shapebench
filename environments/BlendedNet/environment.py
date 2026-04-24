import os
import sys
import fcntl
import json
import math
import numpy as np
import torch

from environments.base import BaseEnvironment
from environments.base_reward import BaseReward
from . import prompt_blocks
from .mesh_generator import generate_mesh

ENV_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ENV_DIR, "model")
# /tmp is always local to the compute node — fcntl.lockf is reliable there.
# Serialises the torchvision + Transolver import across all processes on the
# same node so that concurrent jobs don't race on the __pycache__ write.
_LOCK_FILE = "/tmp/blendednet_torchvision_import.lock"

sys.path.insert(0, MODEL_DIR)

with open(_LOCK_FILE, "a") as _lf:
    fcntl.lockf(_lf, fcntl.LOCK_EX)
    try:
        import torchvision.extension  # loads _C before _meta_registrations runs
        from Transolver import Model as _TransolverModel
    finally:
        fcntl.lockf(_lf, fcntl.LOCK_UN)

GEOM_KEYS = ["B1", "B2", "B3", "C2", "C3", "C4", "S1", "S2", "S3"]
N_POINTS = 8192

_model = None
_norm = None
_device = None

# In-process mesh cache: geometry key → pyvista.PolyData
# Avoids regenerating the identical mesh for each of the 24 bisection calls
# (only alpha changes within a design evaluation, not the geometry).
_mesh_cache: dict = {}


def _load_model():
    global _model, _norm, _device
    if _model is not None:
        return

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _norm = torch.load(os.path.join(MODEL_DIR, "norm_stats.pt"),
                       map_location=_device, weights_only=True)

    args = type("Args", (), {
        "fun_dim": 15, "space_dim": 3, "out_dim": 3,
        "n_hidden": 256, "n_layers": 8, "n_heads": 8,
        "act": "gelu", "mlp_ratio": 2, "slice_num": 32,
        "ref": 8, "dropout": 0.0, "geotype": "unstructured",
        "unified_pos": False, "time_input": False, "shapelist": None,
    })()

    _model = _TransolverModel(args)
    ckpt = torch.load(os.path.join(MODEL_DIR, "transolver_best.pt"),
                      map_location=_device, weights_only=True)
    _model.load_state_dict(ckpt)
    _model.to(_device)
    _model.eval()


def _run_surrogate(mesh, geom_params, flight_params):
    """Run Transolver inference on a generated surface mesh."""
    _load_model()

    normals_mesh = mesh.compute_normals(
        point_normals=True, cell_normals=False, auto_orient_normals=True)
    pts = np.array(normals_mesh.points, dtype=np.float32)
    norms = np.array(normals_mesh.point_data["Normals"], dtype=np.float32)

    rng = np.random.RandomState(42)
    n_pts = pts.shape[0]
    idx = rng.choice(n_pts, N_POINTS, replace=(n_pts < N_POINTS))

    pos = pts[idx]
    point_normals = norms[idx]

    global_feat = np.array(geom_params + flight_params, dtype=np.float32)
    global_tiled = np.tile(global_feat, (N_POINTS, 1))
    fx = np.concatenate([global_tiled, point_normals], axis=-1)

    pos_t = torch.from_numpy(pos).unsqueeze(0).to(_device)
    fx_t = torch.from_numpy(fx).unsqueeze(0).to(_device)
    fx_t[:, :, :12] = (fx_t[:, :, :12] - _norm["fx_mean"]) / _norm["fx_std"]

    with torch.no_grad():
        out = _model(pos_t, fx_t)

    out = out * _norm["y_std"] + _norm["y_mean"]
    out = out.squeeze(0).cpu().numpy()

    return {
        "pos": pos, "idx": idx,
        "Cp": out[:, 0], "Cfx": out[:, 1], "Cfz": out[:, 2],
    }


def _render_field(mesh, scalar, out_path, title, cmap, clim, view, fmt="%.3f"):
    import pyvista as pv
    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    pl.set_background("white")
    pl.add_mesh(mesh, scalars=scalar, cmap=cmap, clim=clim,
                smooth_shading=True, show_edges=False,
                scalar_bar_args={"title": title, "shadow": True,
                                 "fmt": fmt, "n_labels": 9,
                                 "title_font_size": 18,
                                 "label_font_size": 14,
                                 "position_x": 0.80, "width": 0.14,
                                 "color": "black"})
    if view == "top":
        pl.view_xy()
        pl.camera.zoom(1.3)
    else:
        pl.camera_position = [(0.5, -1.6, 0.5), (0.5, 0.0, 0.0), (0, 0, 1)]
    pl.add_axes(color="black")
    pl.screenshot(out_path, transparent_background=False)
    pl.close()


def _map_fields_to_full_mesh(mesh, result):
    """Map subsampled predictions back onto the full mesh via nearest-neighbor."""
    from scipy.spatial import cKDTree
    tree = cKDTree(result["pos"])
    _, idx = tree.query(np.array(mesh.points, dtype=np.float32), k=1)
    mesh.point_data["Cp"] = result["Cp"][idx]
    mesh.point_data["Cfx"] = result["Cfx"][idx]
    return mesh


def _save_sol_images(mesh, result, sol_dir):
    mesh = _map_fields_to_full_mesh(mesh, result)

    Cp = np.array(mesh.point_data["Cp"])
    Cfx = np.array(mesh.point_data["Cfx"])

    cp_clim = (float(np.percentile(Cp, 2)), float(np.percentile(Cp, 98)))
    cfx_lim = float(np.percentile(np.abs(Cfx), 98))
    cfx_clim = (-cfx_lim, cfx_lim)

    images = []
    for scalar, title, cmap, clim, fmt in [
        ("Cp",  "Cp",  "coolwarm", cp_clim,  "%.3f"),
        ("Cfx", "Cfx", "seismic",  cfx_clim, "%.4f"),
    ]:
        for view in ["iso", "top"]:
            path = os.path.join(sol_dir, f"{scalar}_{view}.png")
            _render_field(mesh, scalar, path, f"{title} ({view})", cmap, clim, view, fmt)
            images.append(path)

    return images


class BlendedNetEnvironment(BaseEnvironment):
    """Transolver surrogate for BlendedNet blended-wing-body aerodynamics."""

    def __init__(self, reward: BaseReward, save_fields=False, render_images=False, **kwargs):
        self.reward = reward
        self.save_fields = save_fields
        self.render_images = render_images

    @staticmethod
    def add_args(parser):
        parser.add_argument('--save_fields', action='store_true', default=False,
                            help='Save fields.npz per evaluation (large; off by default)')
        parser.add_argument('--render_images', action='store_true', default=False,
                            help='Render solution images per evaluation (slow; off by default)')

    def _run_sim(self, design_path: str, case_dir: str,
                 mach=0.3, re=1.0e7, alpha=5.0) -> dict:
        """Run surrogate for given flight conditions; return raw field outputs."""
        save_dir = os.path.join(case_dir, "save")
        sol_dir = os.path.join(save_dir, "sol")
        os.makedirs(sol_dir, exist_ok=True)

        with open(design_path) as f:
            params = json.load(f)

        geom_vals = [float(params[k]) for k in GEOM_KEYS]
        re_val = float(params.get("Re", re))
        mach_val = float(params.get("Mach", mach))
        alpha_val = float(params.get("alpha", alpha))
        flight_vals = [math.log10(re_val), mach_val, alpha_val]

        # Cache mesh by geometry key: all 24 bisection calls within one design
        # evaluation share identical geometry (only alpha changes), so generate
        # the mesh once and reuse it. Cache lives for the process lifetime.
        geom_key = tuple(geom_vals)
        if geom_key not in _mesh_cache:
            _mesh_cache[geom_key] = generate_mesh(params)
        mesh = _mesh_cache[geom_key]
        result = _run_surrogate(mesh, geom_vals, flight_vals)

        if self.save_fields:
            np.savez(os.path.join(save_dir, "fields.npz"),
                     pos=result["pos"], Cp=result["Cp"],
                     Cfx=result["Cfx"], Cfz=result["Cfz"])

        images = _save_sol_images(mesh, result, sol_dir) if self.render_images else []

        return {
            "params": params,
            "Re": re_val, "Mach": mach_val, "alpha": alpha_val,
            "Cp_mean": float(result["Cp"].mean()),
            "Cfx_mean": float(result["Cfx"].mean()),
            "Cfz_mean": float(result["Cfz"].mean()),
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

        params = {k: d.get(k) for k in GEOM_KEYS}
        params["Re"] = d.get("Re")
        params["Mach"] = d.get("Mach")
        params["alpha"] = d.get("alpha")

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
        return agent.run_llm_action_bwb(
            action, context_entries, output_dir, name=name, debug_dir=debug_dir)

    def sample_gaussian(self, mean_params: dict, output_dir: str, name: str,
                        std_scale: float = 1.0) -> str:
        from .design_actions import gaussain_bwb
        return gaussain_bwb(
            params=mean_params, out_dir=output_dir, name=name, std_scale=std_scale)

    def get_param_bounds(self):
        from .design_actions import CONTINUOUS_KEYS, BOUNDS
        lb = np.array([BOUNDS[k][0] for k in CONTINUOUS_KEYS])
        ub = np.array([BOUNDS[k][1] for k in CONTINUOUS_KEYS])
        return lb, ub

    def get_named_param_bounds(self):
        from .design_actions import BOUNDS
        return dict(BOUNDS)

    def write_design(self, x, output_dir: str, name: str) -> str:
        from .design_actions import CONTINUOUS_KEYS, save_design_json
        os.makedirs(output_dir, exist_ok=True)
        params = {k: float(x[i]) for i, k in enumerate(CONTINUOUS_KEYS)}
        params['name'] = name
        path = os.path.join(output_dir, f'{name}.json')
        return save_design_json(path, params)

    def read_design(self, design_path: str):
        from .design_actions import CONTINUOUS_KEYS
        with open(design_path) as f:
            params = json.load(f)
        return np.array([float(params[k]) for k in CONTINUOUS_KEYS])

    def set_llm_backend(self, backend, image_analyzer=None):
        from . import agent
        agent.set_llm_backend(backend, image_analyzer)

    def get_results_csv_columns(self):
        return ['Cp_mean', 'Cfx_mean', 'L_D']

    def get_results_csv_row(self, metrics):
        return [
            f"{metrics.get('Cp_mean', 0):.6f}",
            f"{metrics.get('Cfx_mean', 0):.6f}",
            f"{metrics.get('L_D', 0):.4f}",
        ]
