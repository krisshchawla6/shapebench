"""Generate BWB surface meshes from planform parameters via OpenVSP.

The openvsp C extension is linked against Python 3.10, so mesh generation
runs as a subprocess under the ``openvsp310`` conda environment.  The
subprocess exports an STL (in mm), which is read back with PyVista and
converted to meters to match the original CFD dataset convention.
"""
import json
import os
import subprocess
import tempfile

import numpy as np
import pyvista as pv

ENV_DIR = os.path.dirname(os.path.abspath(__file__))
_VSP3_PATH = os.path.join(ENV_DIR, "data", "model.vsp3")
_GEN_SCRIPT = os.path.join(ENV_DIR, "_vsp_generate_stl.py")
_PYTHON310 = os.path.expanduser("~/miniconda3/envs/openvsp310/bin/python")
_MM_TO_M = 1.0 / 1000.0

GEOM_KEYS = ["B1", "B2", "B3", "C2", "C3", "C4", "S1", "S2", "S3"]


def generate_mesh(params: dict, cache_dir: str | None = None) -> pv.PolyData:
    """Generate a BWB surface mesh for the given planform + flight parameters.

    Parameters
    ----------
    params : dict
        Must contain keys B1, B2, B3, C2, C3, C4, S1, S2, S3 (mm units,
        matching the vsp3 model convention).
    cache_dir : str, optional
        Directory to cache generated STL files.  If None a temp file is used.

    Returns
    -------
    pv.PolyData
        Triangulated surface mesh with coordinates in **meters**.
    """
    geom_args = []
    for k in GEOM_KEYS:
        v = params.get(k)
        if v is None:
            raise ValueError(f"Missing geometry parameter: {k}")
        geom_args.append(f"{k}={float(v)}")

    _temp_stl = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        tag = "_".join(f"{k}{float(params[k]):.1f}" for k in GEOM_KEYS)
        stl_path = os.path.join(cache_dir, f"{tag}.stl")
    else:
        fd, stl_path = tempfile.mkstemp(suffix=".stl")
        os.close(fd)
        _temp_stl = stl_path

    try:
        if not os.path.exists(stl_path) or os.path.getsize(stl_path) == 0:
            cmd = [_PYTHON310, _GEN_SCRIPT, _VSP3_PATH, stl_path] + geom_args
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                raise RuntimeError(
                    f"OpenVSP mesh generation failed:\n{result.stderr}\n{result.stdout}"
                )
            info = json.loads(result.stdout.strip().split("\n")[-1])
            if "error" in info:
                raise RuntimeError(f"OpenVSP error: {info['error']}")

        mesh = pv.read(stl_path)
        mesh.points *= _MM_TO_M
        return mesh
    finally:
        if _temp_stl is not None:
            try:
                os.unlink(_temp_stl)
            except OSError:
                pass
