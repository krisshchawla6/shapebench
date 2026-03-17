import os
import json
import numpy as np

C1 = 1000.0

RATIO_BOUNDS = {
    'B1': (0.10, 0.20),
    'B2': (0.05, 0.20),
    'B3': (0.20, 0.70),
    'C2': (0.55, 0.85),
    'C3': (0.18, 0.28),
    'C4': (0.06, 0.09),
}

BOUNDS = {
    'B1': (100.0, 200.0),
    'B2': (50.0, 200.0),
    'B3': (200.0, 700.0),
    'C2': (550.0, 850.0),
    'C3': (180.0, 280.0),
    'C4': (60.0, 90.0),
    'S1': (40.0, 60.0),
    'S2': (40.0, 60.0),
    'S3': (24.0, 40.0),
}

CONTINUOUS_KEYS = list(BOUNDS.keys())

STD_FRACTIONS = {
    'B1': 0.10,
    'B2': 0.10,
    'B3': 0.08,
    'C2': 0.08,
    'C3': 0.10,
    'C4': 0.10,
    'S1': 0.10,
    'S2': 0.10,
    'S3': 0.12,
}


def gaussian_sampling_bwb(mean_params: dict, std_scale: float = 1.0) -> dict:
    """Gaussian-sample a new BWB design around the LLM-proposed mean values."""
    sampled = {}
    for key in CONTINUOUS_KEYS:
        lo, hi = BOUNDS[key]
        mean = float(mean_params.get(key, (lo + hi) / 2))
        std = (hi - lo) * STD_FRACTIONS.get(key, 0.1) * std_scale
        sampled[key] = float(np.clip(np.random.normal(mean, std), lo, hi))
    return sampled


def save_design_json(path: str, params: dict) -> str:
    design = {k: params[k] for k in CONTINUOUS_KEYS}
    design['name'] = params.get('name', 'design')
    with open(path, 'w') as f:
        json.dump(design, f, indent=2)
    return path


def gaussain_bwb(params: dict, out_dir: str = './output', name: str = 'design',
                 std_scale: float = 1.0) -> str:
    """Gaussian-sample around LLM-proposed means, save as JSON."""
    os.makedirs(out_dir, exist_ok=True)
    sampled = gaussian_sampling_bwb(params, std_scale=std_scale)
    sampled['name'] = params.get('name', name)
    path = os.path.join(out_dir, f'{name}.json')
    return save_design_json(path, sampled)


def run_action_bwb(action: str, **kwargs) -> str:
    if action in ('gaussain', 'gaussian'):
        return gaussain_bwb(**kwargs)
    raise ValueError(f"Unknown BWB action: {action}")
