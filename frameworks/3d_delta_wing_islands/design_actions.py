import os
import json
import numpy as np

BOUNDS = {
    'le_sweep':      (45.0, 80.0),
    'root_chord_in': (10.0, 50.0),
    'twist_root':    (-10.0, 10.0),
    'twist_tip':     (-10.0, 10.0),
    'dihedral':      (-15.0, 15.0),
}

VALID_M = [0, 2, 4]
VALID_P = [0, 4]
NACA_T_RANGE = (6, 24)

CONTINUOUS_KEYS = list(BOUNDS.keys())

# Std-dev as fraction of parameter range
STD_FRACTIONS = {
    'le_sweep':      0.08,
    'root_chord_in': 0.08,
    'twist_root':    0.15,
    'twist_tip':     0.15,
    'dihedral':      0.15,
    'naca_t':        0.12,
}


def _snap_discrete(value, valid):
    return min(valid, key=lambda v: abs(v - value))


def gaussian_sampling_3d(mean_params: dict, std_scale: float = 1.0) -> dict:
    """Gaussian-sample a new design around the LLM-proposed mean values."""
    sampled = {}

    for key in CONTINUOUS_KEYS:
        lo, hi = BOUNDS[key]
        mean = float(mean_params.get(key, (lo + hi) / 2))
        std = (hi - lo) * STD_FRACTIONS.get(key, 0.1) * std_scale
        sampled[key] = float(np.clip(np.random.normal(mean, std), lo, hi))

    # naca_t — integer, gaussian then round+clip
    t_mean = float(mean_params.get('naca_t', 12))
    t_lo, t_hi = NACA_T_RANGE
    t_std = (t_hi - t_lo) * STD_FRACTIONS.get('naca_t', 0.12) * std_scale
    sampled['naca_t'] = int(np.clip(round(np.random.normal(t_mean, t_std)), t_lo, t_hi))

    # naca_m — discrete {0,2,4}, gaussian then snap
    m_mean = float(mean_params.get('naca_m', 0))
    sampled['naca_m'] = _snap_discrete(np.random.normal(m_mean, 1.0 * std_scale), VALID_M)

    # naca_p — discrete {0,4}, constrained by m
    p_mean = float(mean_params.get('naca_p', 0))
    sampled['naca_p'] = _snap_discrete(np.random.normal(p_mean, 1.5 * std_scale), VALID_P)
    if sampled['naca_m'] == 0:
        sampled['naca_p'] = 0

    return sampled


def save_design_json(path: str, params: dict) -> str:
    design = {
        'design_name': params.get('name', 'design'),
        'le_sweep':      params['le_sweep'],
        'root_chord_in': params['root_chord_in'],
        'twist_root':    params['twist_root'],
        'twist_tip':     params['twist_tip'],
        'dihedral':      params['dihedral'],
        'naca': {
            'm': params['naca_m'],
            'p': params['naca_p'],
            't': params['naca_t'],
            'chord_length': 1.0,
        },
    }
    with open(path, 'w') as f:
        json.dump(design, f, indent=2)
    return path


def gaussain_3d(params: dict, out_dir: str = './output', name: str = 'design') -> str:
    """Entry point: Gaussian-sample around LLM-proposed means, save as JSON."""
    os.makedirs(out_dir, exist_ok=True)
    sampled = gaussian_sampling_3d(params)
    sampled['name'] = params.get('name', name)
    path = os.path.join(out_dir, f'{name}.json')
    return save_design_json(path, sampled)


def run_action_3d(action: str, **kwargs) -> str:
    if action in ('gaussain', 'gaussian'):
        return gaussain_3d(**kwargs)
    raise ValueError(f"Unknown 3D action: {action}")
