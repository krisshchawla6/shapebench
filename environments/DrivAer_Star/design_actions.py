import os
import json
import numpy as np

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

CONTINUOUS_KEYS = list(BOUNDS.keys())

STD_FRACTIONS = {
    "car_size":             0.05,
    "car_width":            0.10,
    "car_len":              0.10,
    "ramp_angle":           0.12,
    "front_bumper_length":  0.10,
    "wind_screen_x":        0.12,
    "wind_screen_z":        0.12,
    "side_mirrors_x":       0.12,
    "side_mirrors_z":       0.12,
    "rear_window_x":        0.12,
    "rear_window_z":        0.12,
    "trunklid_angle":       0.12,
    "trunklid_x":           0.12,
    "trunklid_z":           0.12,
    "diffusor_angle":       0.12,
    "car_green_house_angle":0.12,
    "car_front_hood_angle": 0.12,
    "car_air_intake_angle": 0.12,
    "tires_diameter":       0.10,
    "tires_width":          0.10,
}


def gaussian_sampling_drivaer(mean_params: dict, std_scale: float = 1.0) -> dict:
    sampled = {}
    for key in CONTINUOUS_KEYS:
        lo, hi = BOUNDS[key]
        mean = float(mean_params.get(key, (lo + hi) / 2))
        std = (hi - lo) * STD_FRACTIONS.get(key, 0.1) * std_scale
        sampled[key] = float(np.clip(np.random.normal(mean, std), lo, hi))
    return sampled


def save_design_json(path: str, params: dict) -> str:
    design = {k: params[k] for k in CONTINUOUS_KEYS if k in params}
    design['name'] = params.get('name', 'design')
    with open(path, 'w') as f:
        json.dump(design, f, indent=2)
    return path


def gaussian_drivaer(params: dict, out_dir: str = './output', name: str = 'design',
                     std_scale: float = 1.0) -> str:
    os.makedirs(out_dir, exist_ok=True)
    sampled = gaussian_sampling_drivaer(params, std_scale=std_scale)
    sampled['name'] = params.get('name', name)
    path = os.path.join(out_dir, f'{name}.json')
    return save_design_json(path, sampled)


def run_action_drivaer(action: str, **kwargs) -> str:
    if action in ('gaussain', 'gaussian'):
        return gaussian_drivaer(**kwargs)
    raise ValueError(f"Unknown DrivAerStar action: {action}")
