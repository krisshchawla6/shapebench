"""MVP configuration for DrivAerStar diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# Kept in sync with environments/DrivAer_Star/design_actions.py
DRIVAER_REQUIRED_PARAM_KEYS: List[str] = [
    "car_size",
    "car_width",
    "car_len",
    "ramp_angle",
    "front_bumper_length",
    "wind_screen_x",
    "wind_screen_z",
    "side_mirrors_x",
    "side_mirrors_z",
    "rear_window_x",
    "rear_window_z",
    "trunklid_angle",
    "trunklid_x",
    "trunklid_z",
    "diffusor_angle",
    "car_green_house_angle",
    "car_front_hood_angle",
    "car_air_intake_angle",
    "tires_diameter",
    "tires_width",
]

DRIVAER_PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "car_size": (0.8, 1.2),
    "car_width": (-0.1, 0.1),
    "car_len": (-0.1, 0.1),
    "ramp_angle": (-8.0, 8.0),
    "front_bumper_length": (-0.1, 0.1),
    "wind_screen_x": (-0.05, 0.05),
    "wind_screen_z": (-0.05, 0.05),
    "side_mirrors_x": (-0.05, 0.05),
    "side_mirrors_z": (-0.05, 0.05),
    "rear_window_x": (-0.05, 0.05),
    "rear_window_z": (-0.05, 0.05),
    "trunklid_angle": (-8.0, 8.0),
    "trunklid_x": (-0.05, 0.05),
    "trunklid_z": (-0.05, 0.05),
    "diffusor_angle": (-8.0, 8.0),
    "car_green_house_angle": (-8.0, 8.0),
    "car_front_hood_angle": (-8.0, 8.0),
    "car_air_intake_angle": (-8.0, 8.0),
    "tires_diameter": (-0.013, 0.013),
    "tires_width": (-0.015, 0.015),
}

DRIVAER_REQUIRED_METRIC_KEYS: List[str] = [
    "drag",
    "Cd",
    "lift",
    "drag_pressure",
    "drag_shear",
]

DRIVAER_EXPECTED_IMAGE_SUFFIXES: List[str] = [
    "Pressure_iso.png",
    "Pressure_top.png",
    "Pressure_side.png",
    "WSSx_iso.png",
    "WSSx_top.png",
    "WSSx_side.png",
]


@dataclass
class DrivAerThresholds:
    # Feasibility
    feasibility_tol: float = 1e-9
    # Geometry heuristics
    near_bound_fraction_warn: float = 0.60
    near_bound_margin_ratio: float = 0.05
    combined_angle_abs_sum_warn: float = 26.0
    # Aero plausibility
    drag_decomposition_rel_tol_warn: float = 0.02
    cd_min_warn: float = 0.0
    cd_max_warn: float = 1.5
    lift_abs_warn: float = 2.0e5
    # LLM
    llm_model_name: str = "gemini-2.5-flash"
    llm_temperature: float = 0.2
    llm_max_output_tokens: int = 2000
    prompt_version: str = "drivaer_mvp_v1"


@dataclass
class DiagnosticConfig:
    version: str = "0.1.0"
    environment_name: str = "DrivAer_Star"
    strict_schema_validation: bool = True
    allow_partial_evidence: bool = True
    thresholds: DrivAerThresholds = field(default_factory=DrivAerThresholds)


DEFAULT_CONFIG = DiagnosticConfig()
