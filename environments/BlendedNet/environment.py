import os
import sys
import json
import math
import numpy as np
import torch

from environments.base import BaseEnvironment
from . import prompt_blocks
from .mesh_generator import generate_mesh

ENV_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ENV_DIR, "model")

sys.path.insert(0, MODEL_DIR)

GEOM_KEYS = ["B1", "B2", "B3", "C2", "C3", "C4", "S1", "S2", "S3"]
N_POINTS = 8192
FAIL_REWARD = -5.0

_model = None
_norm = None
_device = None


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

    from Transolver import Model
    _model = Model(args)
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

    def __init__(self, mach=0.3, re=1.0e7, alpha=5.0, **kwargs):
        self.mach = mach
        self.re = re
        self.alpha = alpha

    @staticmethod
    def add_args(parser):
        parser.add_argument('--mach', type=float, default=0.3, help='Mach number')
        parser.add_argument('--re', type=float, default=1.0e7, help='Reynolds number')
        parser.add_argument('--alpha', type=float, default=5.0, help='Angle of attack (deg)')

    def simulate(self, design_path: str, case_dir: str, **kwargs) -> tuple:
        save_dir = os.path.join(case_dir, "save")
        sol_dir = os.path.join(save_dir, "sol")
        os.makedirs(sol_dir, exist_ok=True)

        with open(design_path) as f:
            params = json.load(f)

        try:
            geom_vals = [float(params[k]) for k in GEOM_KEYS]
            re_val = float(params.get("Re", self.re))
            mach = float(params.get("Mach", self.mach))
            alpha = float(params.get("alpha", self.alpha))
            flight_vals = [math.log10(re_val), mach, alpha]

            mesh = generate_mesh(params)
            result = _run_surrogate(mesh, geom_vals, flight_vals)

            Cp_mean = float(result["Cp"].mean())
            Cfx_mean = float(result["Cfx"].mean())
            Cfz_mean = float(result["Cfz"].mean())
            CL_approx = float(-result["Cp"].mean())
            CD_approx = float(result["Cfx"].mean())
            LD = CL_approx / CD_approx if abs(CD_approx) > 1e-12 else 0.0
            reward = LD

            np.savez(os.path.join(save_dir, "fields.npz"),
                     pos=result["pos"], Cp=result["Cp"],
                     Cfx=result["Cfx"], Cfz=result["Cfz"])

            results_dict = {
                "design": params,
                "Re": re_val, "Mach": mach, "alpha": alpha,
                "Cp_mean": Cp_mean, "Cfx_mean": Cfx_mean, "Cfz_mean": Cfz_mean,
                "CL_approx": CL_approx, "CD_approx": CD_approx, "L_D": LD,
                "reward": reward,
            }
            with open(os.path.join(save_dir, "results.json"), "w") as f:
                json.dump(results_dict, f, indent=2)

            images = _save_sol_images(mesh, result, sol_dir)

            return float(reward), {
                "metrics": {
                    "Cp_mean": Cp_mean, "Cfx_mean": Cfx_mean, "Cfz_mean": Cfz_mean,
                    "CL_approx": CL_approx, "CD_approx": CD_approx, "L_D": LD,
                    "reward": reward,
                },
                "images": images,
                "feedback": "",
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[sim] FAILED: {e}")
            return float(FAIL_REWARD), self._fail_results(FAIL_REWARD)

    def _fail_results(self, reward):
        return {
            "metrics": {"reward": reward},
            "images": [],
            "feedback": "Simulation failed.",
        }

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
        return {
            "format_context": prompt_blocks.format_context,
            "format_response_instructions": prompt_blocks.format_response_instructions,
            "CONTEXT_FORMAT": prompt_blocks.CONTEXT_FORMAT,
            "DESIGN_ENTRY": prompt_blocks.DESIGN_ENTRY,
            "RESPONSE_FORMAT": prompt_blocks.RESPONSE_FORMAT,
        }
