#!/usr/bin/env python3
"""Extract Transolver weights from Lightning checkpoint and optionally compute norm stats.

Usage:
    python setup_model.py                           # extract weights only
    python setup_model.py --vtk-dir /path/to/vtk/   # also compute norm stats from VTK data
"""

import os
import sys
import argparse
import torch
import numpy as np

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.dirname(MODEL_DIR)
CKPT_DIR = os.path.join(ENV_DIR, "ckpts", "Transolver")
CKPT_PATH = os.path.join(CKPT_DIR, "lightning_logs", "version_0", "checkpoints",
                          "epoch=490-step=1473000.ckpt")


def extract_model_weights(ckpt_path=CKPT_PATH, output_path=None):
    output_path = output_path or os.path.join(MODEL_DIR, "transolver_best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    model_state = {}
    for k, v in state.items():
        if k.startswith("model."):
            model_state[k[len("model."):]] = v
    torch.save(model_state, output_path)
    print(f"Saved {len(model_state)} weight tensors to {output_path}")
    return model_state


def compute_norm_stats(vtk_dir, output_path=None, max_files=500):
    """Compute input/output normalization stats from a directory of VTK files."""
    import pyvista as pv
    output_path = output_path or os.path.join(MODEL_DIR, "norm_stats.pt")

    files = sorted(f for f in os.listdir(vtk_dir) if f.endswith(".vtk"))[:max_files]
    if not files:
        print(f"No VTK files found in {vtk_dir}")
        return

    x_means, x_stds, y_means, y_stds = [], [], [], []

    for fname in files:
        path = os.path.join(vtk_dir, fname)
        try:
            mesh = pv.read(path)
            centers = mesh.cell_centers().points
            areas = mesh.cell_data.get("Area")
            normals = mesh.cell_data.get("Normals")
            pressure = mesh.cell_data.get("Pressure")
            wss_i = mesh.cell_data.get("WallShearStressi")
            wss_j = mesh.cell_data.get("WallShearStressj")
            wss_k = mesh.cell_data.get("WallShearStressk")

            if any(v is None for v in [areas, normals, pressure, wss_i, wss_j, wss_k]):
                continue
            if np.any(np.abs(pressure) > 20000):
                continue

            x = np.hstack([centers, areas.reshape(-1, 1), normals])
            y = np.hstack([pressure.reshape(-1, 1), wss_i.reshape(-1, 1),
                           wss_j.reshape(-1, 1), wss_k.reshape(-1, 1)])

            if np.isnan(x).any() or np.isnan(y).any():
                continue

            x_t = torch.from_numpy(x.astype(np.float32))
            y_t = torch.from_numpy(y.astype(np.float32))
            x_means.append(x_t.mean(dim=0))
            x_stds.append(x_t.std(dim=0))
            y_means.append(y_t.mean(dim=0))
            y_stds.append(y_t.std(dim=0))
            print(f"  Processed {fname}: {x_t.shape[0]} cells")
        except Exception as e:
            print(f"  Skipping {fname}: {e}")

    if not x_means:
        print("No valid VTK files processed.")
        return

    stats = {
        "x_mean": torch.stack(x_means).mean(dim=0),
        "x_std": torch.stack(x_stds).mean(dim=0),
        "y_mean": torch.stack(y_means).mean(dim=0),
        "y_std": torch.stack(y_stds).mean(dim=0),
    }
    torch.save(stats, output_path)
    print(f"\nSaved norm stats to {output_path}")
    print(f"  x_mean: {stats['x_mean']}")
    print(f"  x_std:  {stats['x_std']}")
    print(f"  y_mean: {stats['y_mean']}")
    print(f"  y_std:  {stats['y_std']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=CKPT_PATH)
    parser.add_argument("--vtk-dir", default=None,
                        help="Directory of VTK files for computing normalization stats")
    args = parser.parse_args()

    extract_model_weights(args.ckpt)
    if args.vtk_dir:
        compute_norm_stats(args.vtk_dir)
    else:
        print("\nNote: Run with --vtk-dir <path> to compute normalization stats.")
        print("Without norm_stats.pt, the environment will not function correctly.")
