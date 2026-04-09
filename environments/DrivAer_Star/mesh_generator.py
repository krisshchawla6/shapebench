"""FFD-based mesh deformation for DrivAerStar vehicle geometry.

Translates the Blender FFD deformation logic into pure Python/NumPy so that
base VTK meshes can be parametrically deformed without Blender.

The deformation operates on cell centers directly.  Each parameter controls
a region-specific displacement, mirroring the Blender lattice functions.
After deformation, cell normals and areas are recomputed by PyVista.
"""

import math
import os
import numpy as np
import pyvista as pv

# Cache base mesh objects (topology + original points) keyed by absolute path.
# Avoids repeated pv.read() disk I/O — the mesh is read once per process and
# reused for all subsequent deform_vtk() calls with the same base file.
_BASE_MESH_CACHE = {}

PARAM_KEYS = [
    "car_size", "car_width", "car_len", "ramp_angle",
    "front_bumper_length", "wind_screen_x", "wind_screen_z",
    "side_mirrors_x", "side_mirrors_z", "rear_window_x",
    "rear_window_z", "trunklid_angle", "trunklid_x", "trunklid_z",
    "diffusor_angle", "car_green_house_angle",
    "car_front_hood_angle", "car_air_intake_angle",
    "tires_diameter", "tires_width",
]

BOUNDS = {
    "car_size":             (0.8,    1.2),
    "car_width":            (-0.1,   0.1),
    "car_len":              (-0.1,   0.1),
    "ramp_angle":           (-8.0,   8.0),
    "front_bumper_length":  (-0.1,   0.1),
    "wind_screen_x":        (-0.05,  0.05),
    "wind_screen_z":        (-0.05,  0.05),
    "side_mirrors_x":       (-0.05,  0.05),
    "side_mirrors_z":       (-0.05,  0.05),
    "rear_window_x":        (-0.05,  0.05),
    "rear_window_z":        (-0.05,  0.05),
    "trunklid_angle":       (-8.0,   8.0),
    "trunklid_x":           (-0.05,  0.05),
    "trunklid_z":           (-0.05,  0.05),
    "diffusor_angle":       (-8.0,   8.0),
    "car_green_house_angle":(-8.0,   8.0),
    "car_front_hood_angle": (-8.0,   8.0),
    "car_air_intake_angle": (-8.0,   8.0),
    "tires_diameter":       (-0.013, 0.013),
    "tires_width":          (-0.015, 0.015),
}


def _smooth_mask(values, lo, hi, width=0.02):
    """Smooth 0-1 mask with sigmoid-like falloff at boundaries."""
    mask = np.ones_like(values)
    if width > 0:
        mask *= 1.0 / (1.0 + np.exp(-(values - lo) / width))
        mask *= 1.0 / (1.0 + np.exp((values - hi) / width))
    else:
        mask = ((values >= lo) & (values <= hi)).astype(float)
    return mask


def _extent(pts):
    """Return bounding box quantities."""
    xmin, ymin, zmin = pts.min(axis=0)
    xmax, ymax, zmax = pts.max(axis=0)
    return xmin, xmax, ymin, ymax, zmin, zmax


def apply_ffd(pts, params):
    """Apply DrivAerStar FFD deformations to an array of 3D points.

    Parameters
    ----------
    pts : ndarray [N, 3]
        Original point coordinates (meters).
    params : dict
        Design parameters (keys from PARAM_KEYS, values as floats).

    Returns
    -------
    ndarray [N, 3]
        Deformed point coordinates.
    """
    pts = pts.copy().astype(np.float64)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    xmin, xmax, ymin, ymax, zmin, zmax = _extent(pts)
    xlen = xmax - xmin
    ylen = ymax - ymin
    zlen = zmax - zmin

    scale = float(params.get("car_size", 1.0))
    if abs(scale - 1.0) > 1e-9:
        pts *= scale
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        xmin, xmax, ymin, ymax, zmin, zmax = _extent(pts)
        xlen = xmax - xmin
        ylen = ymax - ymin
        zlen = zmax - zmin

    # car_width
    w = float(params.get("car_width", 0.0))
    if abs(w) > 1e-9:
        ymean = (ymin + ymax) / 2.0
        yl = ylen / 7.0
        w_half = w / (2.2 * 2.0)
        inner = _smooth_mask(y, ymean - yl, ymean + yl)
        outer_lo = (y < ymean - yl).astype(float)
        outer_hi = (y > ymean + yl).astype(float)
        frac = np.clip((y - ymean) / max(yl, 1e-9), -1, 1)
        y += inner * frac * w_half + outer_lo * (-w_half) + outer_hi * w_half

    # car_len
    cl = float(params.get("car_len", 0.0))
    if abs(cl) > 1e-9:
        move = cl / 5.0
        piv = xmin + xlen * (9 / 31)
        ll = xlen * (12 / 31)
        mid = _smooth_mask(x, piv, piv + ll)
        rear = (x > piv + ll).astype(float)
        frac_mid = np.clip((x - piv) / max(ll, 1e-9), 0, 1)
        x += mid * frac_mid * move + rear * move

    # ramp_angle (front underbody)
    ra = float(params.get("ramp_angle", 0.0))
    if abs(ra) > 1e-6:
        angle = math.radians(ra)
        center = xmin + xlen * (4 / 31)
        move_z = math.tan(angle) * xlen * (4 / 31)
        region = _smooth_mask(x, xmin, center) * _smooth_mask(z, zmin, zmin + zlen * (2 / 7))
        frac = np.clip((center - x) / max(center - xmin, 1e-9), 0, 1) * (-1)
        z += region * frac * move_z

    # front_bumper_length
    fb = float(params.get("front_bumper_length", 0.0))
    if abs(fb) > 1e-9:
        move_x = fb / 5.0
        center = xmin + xlen * (4 / 31)
        region = _smooth_mask(x, xmin, center)
        frac = np.clip((center - x) / max(center - xmin, 1e-9), 0, 1)
        x -= region * frac * move_x

    # wind_screen_x
    ws_x = float(params.get("wind_screen_x", 0.0))
    if abs(ws_x) > 1e-9:
        region = (_smooth_mask(x, xmin + xlen * (10 / 31), xmin + xlen * (14 / 31)) *
                  _smooth_mask(y, ymin + 0.01, ymax - 0.01) *
                  _smooth_mask(z, zmin + zlen * (5 / 7), zmin + zlen * (6.5 / 7)))
        x += region * ws_x / 5.0

    # wind_screen_z
    ws_z = float(params.get("wind_screen_z", 0.0))
    if abs(ws_z) > 1e-9:
        region = (_smooth_mask(x, xmin + xlen * (10 / 31), xmin + xlen * (14 / 31)) *
                  _smooth_mask(y, ymin + 0.01, ymax - 0.01) *
                  _smooth_mask(z, zmin + zlen * (5 / 7), zmin + zlen * (6.5 / 7)))
        z += region * ws_z / 1.7

    # side_mirrors_x/z
    sm_x = float(params.get("side_mirrors_x", 0.0))
    sm_z = float(params.get("side_mirrors_z", 0.0))
    if abs(sm_x) > 1e-9 or abs(sm_z) > 1e-9:
        side_mask = ((y < ymin + ylen * 0.05) | (y > ymax - ylen * 0.05)).astype(float)
        region = (_smooth_mask(x, xmin + xlen * (11 / 31), xmin + xlen * (13 / 31)) *
                  side_mask *
                  _smooth_mask(z, zmin + zlen * (4 / 7), zmin + zlen * (6 / 7)))
        x += region * sm_x / 5.0
        z += region * sm_z / 1.7

    # rear_window_x/z
    rw_x = float(params.get("rear_window_x", 0.0))
    rw_z = float(params.get("rear_window_z", 0.0))
    if abs(rw_x) > 1e-9 or abs(rw_z) > 1e-9:
        region = (_smooth_mask(x, xmin + xlen * (15 / 31), xmin + xlen * (27 / 31)) *
                  _smooth_mask(y, ymin + 0.01, ymax - 0.01) *
                  _smooth_mask(z, zmin + zlen * (5 / 7), zmin + zlen * (6.5 / 7)))
        x += region * rw_x / 5.0
        z += region * rw_z / 1.7

    # trunklid_angle
    ta = float(params.get("trunklid_angle", 0.0))
    if abs(ta) > 1e-6:
        angle = math.radians(ta)
        center = xmax - xlen * (8 / 31)
        move_z = -math.tan(angle) * xlen * (8 / 31) * 2
        region = _smooth_mask(x, center, xmax) * _smooth_mask(z, zmin + zlen * (4 / 7), zmax)
        frac = np.clip((x - center) / max(xmax - center, 1e-9), 0, 1)
        z += region * frac * move_z

    # trunklid_x/z
    tl_x = float(params.get("trunklid_x", 0.0))
    tl_z = float(params.get("trunklid_z", 0.0))
    if abs(tl_x) > 1e-9 or abs(tl_z) > 1e-9:
        region = (_smooth_mask(x, xmin + xlen * (27 / 31), xmin + xlen * (30 / 31)) *
                  _smooth_mask(y, ymin + 0.01, ymax - 0.01) *
                  _smooth_mask(z, zmin + zlen * (1 / 7), zmin + zlen * (6 / 7)))
        x += region * tl_x / 5.0
        z += region * tl_z / 1.7

    # diffusor_angle
    da = float(params.get("diffusor_angle", 0.0))
    if abs(da) > 1e-6:
        angle = math.radians(da)
        center = xmax - xlen * (6 / 31)
        move_z = math.tan(angle) * xlen * (6 / 31) * 1.5
        region = _smooth_mask(x, center, xmax) * _smooth_mask(z, zmin, zmin + zlen * (3 / 7))
        frac = np.clip((x - center) / max(xlen * (6 / 31), 1e-9), 0, 1)
        z += region * frac * move_z

    # car_green_house_angle
    gha = float(params.get("car_green_house_angle", 0.0))
    if abs(gha) > 1e-6:
        angle = math.radians(gha)
        z_roof = zmin + 0.6 * zlen
        ymean = (ymin + ymax) / 2.0
        yl = (ymax - ymean)
        region = _smooth_mask(z, z_roof, zmax)
        frac_z = np.clip((z - z_roof) / max(zmax - z_roof, 1e-9), 0, 1)
        frac_y = (y - ymean) / max(yl, 1e-9)
        y += region * frac_z * math.tan(angle) * frac_y

    # car_front_hood_angle
    fha = float(params.get("car_front_hood_angle", 0.0))
    if abs(fha) > 1e-6:
        angle = math.radians(fha)
        center = xmin + xlen * (9 / 31)
        move_z = math.tan(angle) * xlen * (9 / 31) * 1.7
        region = (_smooth_mask(x, xmin, center) *
                  _smooth_mask(z, zmin + zlen * (4 / 7), zmin + zlen * (5 / 7)))
        frac = np.clip((center - x) / max(center - xmin, 1e-9), 0, 1)
        z += region * frac * move_z

    # car_air_intake_angle
    aia = float(params.get("car_air_intake_angle", 0.0))
    if abs(aia) > 1e-6:
        angle = math.radians(aia)
        z_top = zmin + zlen * (3 / 7)
        move_x = math.tan(angle) * zlen * (3 / 7) * 1.7 / 5.0
        region = (_smooth_mask(x, xmin, xmin + xlen * (3 / 31)) *
                  _smooth_mask(z, zmin, z_top))
        frac = np.clip((z_top - z) / max(z_top - zmin, 1e-9), 0, 1)
        x += region * frac * move_x

    pts[:, 0], pts[:, 1], pts[:, 2] = x, y, z
    return pts


def deform_vtk(base_vtk_path, params, cache_dir=None):
    """Load a base VTK, apply FFD deformation, recompute cell data.

    Returns a PyVista mesh with updated cell_centers, Area, Normals.
    """
    abs_path = os.path.abspath(base_vtk_path)
    if abs_path not in _BASE_MESH_CACHE:
        _BASE_MESH_CACHE[abs_path] = pv.read(abs_path)
    mesh = _BASE_MESH_CACHE[abs_path].copy()
    original_points = np.array(mesh.points, dtype=np.float64)
    deformed_points = apply_ffd(original_points, params)
    mesh.points = deformed_points

    mesh = mesh.compute_cell_sizes(length=False, area=True, volume=False)
    mesh = mesh.compute_normals(
        cell_normals=True, point_normals=False,
        auto_orient_normals=False, consistent_normals=False, inplace=False)

    return mesh


def extract_model_input(mesh):
    """Extract the 7-channel input tensor from a VTK mesh.

    Returns: ndarray [N, 7] = [centers(3), area(1), normals(3)]
    """
    centers = mesh.cell_centers().points
    areas = mesh.cell_data.get("Area")
    normals = mesh.cell_data.get("Normals")

    if areas is None:
        mesh = mesh.compute_cell_sizes(length=False, area=True, volume=False)
        areas = mesh.cell_data["Area"]
    if normals is None:
        mesh = mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)
        normals = mesh.cell_data["Normals"]

    return np.hstack([
        centers.astype(np.float32),
        areas.reshape(-1, 1).astype(np.float32),
        normals.astype(np.float32),
    ])
