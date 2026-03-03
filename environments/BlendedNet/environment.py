import os
import sys
import json
import math
import numpy as np
import torch

ENV_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ENV_DIR, "model")

sys.path.insert(0, MODEL_DIR)

GEOM_KEYS = ["B1", "B2", "B3", "C2", "C3", "C4", "S1", "S2", "S3"]
N_POINTS = 8192

_model = None
_norm = None
_device = None


def _load_model():
    global _model, _norm, _device
    if _model is not None:
        return

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _norm = torch.load(os.path.join(MODEL_DIR, "norm_stats.pt"), map_location=_device, weights_only=True)

    args = type("Args", (), {
        "fun_dim": 15, "space_dim": 3, "out_dim": 3,
        "n_hidden": 256, "n_layers": 8, "n_heads": 8,
        "act": "gelu", "mlp_ratio": 2, "slice_num": 32,
        "ref": 8, "dropout": 0.0, "geotype": "unstructured",
        "unified_pos": False, "time_input": False, "shapelist": None,
    })()

    from Transolver import Model
    _model = Model(args)
    ckpt = torch.load(os.path.join(MODEL_DIR, "transolver_best.pt"), map_location=_device, weights_only=True)
    _model.load_state_dict(ckpt)
    _model.to(_device)
    _model.eval()


def predict(params):
    """Run surrogate inference.

    Parameters
    ----------
    params : dict  with keys:
        "vtk_path" : str  - path to VTK surface mesh
        "B1" .. "B3", "C2" .. "C4", "S1" .. "S3" : float  - 9 planform params
        "Re" : float  - Reynolds number (raw, e.g. 1.3e7)
        "Mach" : float
        "alpha" : float  - angle of attack in degrees

    Returns
    -------
    dict with keys "pos" (N,3), "Cp" (N,), "Cfx" (N,), "Cfz" (N,)
    """
    _load_model()
    import pyvista as pv

    mesh = pv.read(params["vtk_path"])
    pts = np.array(mesh.points, dtype=np.float32)

    normals_mesh = mesh.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
    norms = np.array(normals_mesh.point_data["Normals"], dtype=np.float32)

    rng = np.random.RandomState(42)
    n_pts = pts.shape[0]
    if n_pts >= N_POINTS:
        idx = rng.choice(n_pts, N_POINTS, replace=False)
    else:
        idx = rng.choice(n_pts, N_POINTS, replace=True)

    pos = pts[idx]
    point_normals = norms[idx]

    geom_vals = [float(params[k]) for k in GEOM_KEYS]
    flight_vals = [math.log10(float(params["Re"])), float(params["Mach"]), float(params["alpha"])]
    global_feat = np.array(geom_vals + flight_vals, dtype=np.float32)
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
        "pos": pos,
        "Cp": out[:, 0],
        "Cfx": out[:, 1],
        "Cfz": out[:, 2],
    }


def predict_full_mesh(params):
    """Run inference on the full mesh (no subsampling) by batching chunks of N_POINTS."""
    _load_model()
    import pyvista as pv

    mesh = pv.read(params["vtk_path"])
    pts = np.array(mesh.points, dtype=np.float32)
    normals_mesh = mesh.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
    norms = np.array(normals_mesh.point_data["Normals"], dtype=np.float32)

    n_pts = pts.shape[0]
    geom_vals = [float(params[k]) for k in GEOM_KEYS]
    flight_vals = [math.log10(float(params["Re"])), float(params["Mach"]), float(params["alpha"])]
    global_feat = np.array(geom_vals + flight_vals, dtype=np.float32)

    all_out = np.zeros((n_pts, 3), dtype=np.float32)

    chunk_size = N_POINTS
    for start in range(0, n_pts, chunk_size):
        end = min(start + chunk_size, n_pts)
        chunk_pts = pts[start:end]
        chunk_norms = norms[start:end]
        actual = chunk_pts.shape[0]

        if actual < chunk_size:
            pad = chunk_size - actual
            chunk_pts = np.concatenate([chunk_pts, np.zeros((pad, 3), dtype=np.float32)])
            chunk_norms = np.concatenate([chunk_norms, np.zeros((pad, 3), dtype=np.float32)])

        global_tiled = np.tile(global_feat, (chunk_size, 1))
        fx = np.concatenate([global_tiled, chunk_norms], axis=-1)

        pos_t = torch.from_numpy(chunk_pts).unsqueeze(0).to(_device)
        fx_t = torch.from_numpy(fx).unsqueeze(0).to(_device)
        fx_t[:, :, :12] = (fx_t[:, :, :12] - _norm["fx_mean"]) / _norm["fx_std"]

        with torch.no_grad():
            out = _model(pos_t, fx_t)
        out = out * _norm["y_std"] + _norm["y_mean"]
        all_out[start:end] = out.squeeze(0).cpu().numpy()[:actual]

    return {
        "pos": pts,
        "Cp": all_out[:, 0],
        "Cfx": all_out[:, 1],
        "Cfz": all_out[:, 2],
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python environment.py <input.json> [--full-mesh] [--output result.json]")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        params = json.load(f)

    full_mesh = "--full-mesh" in sys.argv

    if full_mesh:
        result = predict_full_mesh(params)
    else:
        result = predict(params)

    out_path = None
    if "--output" in sys.argv:
        out_path = sys.argv[sys.argv.index("--output") + 1]

    summary = {
        "n_points": int(result["pos"].shape[0]),
        "Cp_mean": float(result["Cp"].mean()),
        "Cp_std": float(result["Cp"].std()),
        "Cfx_mean": float(result["Cfx"].mean()),
        "Cfx_std": float(result["Cfx"].std()),
        "Cfz_mean": float(result["Cfz"].mean()),
        "Cfz_std": float(result["Cfz"].std()),
    }

    if out_path:
        np.savez(out_path,
                 pos=result["pos"],
                 Cp=result["Cp"],
                 Cfx=result["Cfx"],
                 Cfz=result["Cfz"])
        summary["saved_to"] = out_path

    print(json.dumps(summary, indent=2))
