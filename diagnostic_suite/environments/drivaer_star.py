from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from diagnostic_suite.config import (
    DRIVAER_EXPECTED_IMAGE_SUFFIXES,
    DRIVAER_PARAM_BOUNDS,
    DRIVAER_REQUIRED_METRIC_KEYS,
    DRIVAER_REQUIRED_PARAM_KEYS,
    DEFAULT_CONFIG,
)


def infer_body_style(base_vtk_path: Optional[str]) -> str:
    """Infer DrivAer body style from base VTK path."""
    if not base_vtk_path:
        return "unknown"
    path = base_vtk_path.replace("\\", "/")
    if "/vtk_F/" in path or path.endswith("_F.vtk"):
        return "F"
    if "/vtk_N/" in path or path.endswith("_N.vtk"):
        return "N"
    if "/vtk_E/" in path or path.endswith("_E.vtk"):
        return "E"
    return "unknown"


def expected_norm_stats_filename(body_style: str) -> Optional[str]:
    """Norm-stats file expected by style based on current environment implementation."""
    if body_style == "E":
        return "norm_stats.pt"
    if body_style == "F":
        return "norm_stats_F.pt"
    if body_style == "N":
        return "norm_stats_N.pt"
    return None


def norm_stats_style_compatible(base_vtk_path: Optional[str], norm_stats_path: Optional[str]) -> Optional[bool]:
    """Return True/False when style is inferrable, else None if unknown."""
    if not norm_stats_path:
        return None
    style = infer_body_style(base_vtk_path)
    expected = expected_norm_stats_filename(style)
    if expected is None:
        return None
    return os.path.basename(norm_stats_path) == expected


def expected_image_paths(images: List[str]) -> Dict[str, bool]:
    """Check expected DrivAer render image presence by suffix."""
    suffix_present = {suffix: False for suffix in DRIVAER_EXPECTED_IMAGE_SUFFIXES}
    for image in images:
        name = os.path.basename(image)
        for suffix in suffix_present:
            if name.endswith(suffix):
                suffix_present[suffix] = True
    return suffix_present


def get_environment_metadata() -> Dict[str, Any]:
    """Environment-specific metadata consumed by deterministic checks."""
    th = DEFAULT_CONFIG.thresholds
    return {
        "name": "DrivAer_Star",
        "required_param_keys": DRIVAER_REQUIRED_PARAM_KEYS,
        "param_bounds": DRIVAER_PARAM_BOUNDS,
        "required_metric_keys": DRIVAER_REQUIRED_METRIC_KEYS,
        "expected_image_suffixes": DRIVAER_EXPECTED_IMAGE_SUFFIXES,
        "thresholds": {
            "feasibility_tol": th.feasibility_tol,
            "near_bound_fraction_warn": th.near_bound_fraction_warn,
            "near_bound_margin_ratio": th.near_bound_margin_ratio,
            "combined_angle_abs_sum_warn": th.combined_angle_abs_sum_warn,
            "drag_decomposition_rel_tol_warn": th.drag_decomposition_rel_tol_warn,
            "cd_min_warn": th.cd_min_warn,
            "cd_max_warn": th.cd_max_warn,
            "lift_abs_warn": th.lift_abs_warn,
        },
    }
