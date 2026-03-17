import os
import json
import numpy as np

# Bounds for each Kulfan (CST) parameter
UPPER_BOUNDS = (-0.30, 0.60)   # upper surface CST coefficients
LOWER_BOUNDS = (-0.30, 0.30)   # lower surface CST coefficients
LE_BOUNDS    = (-0.50, 0.50)   # leading edge weight
TE_BOUNDS    = (0.000, 0.010)  # trailing edge thickness (fraction of chord)

N_CST = 8  # number of CST coefficients per surface

# Gaussian std as fraction of parameter range
STD_FRACTION_WEIGHTS = 0.12
STD_FRACTION_LE      = 0.10
STD_FRACTION_TE      = 0.15

# Baseline (approximate NACA 4412)
DEFAULT_UPPER = [0.17, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04]
DEFAULT_LOWER = [-0.10, -0.07, -0.05, -0.04, -0.03, -0.02, -0.01, -0.005]
DEFAULT_LE    = 0.0
DEFAULT_TE    = 0.002


def gaussian_sampling_neuralfoil(mean_params: dict, std_scale: float = 1.0) -> dict:
    ulo, uhi = UPPER_BOUNDS
    llo, lhi = LOWER_BOUNDS
    le_lo, le_hi = LE_BOUNDS
    te_lo, te_hi = TE_BOUNDS

    upper_std = (uhi - ulo) * STD_FRACTION_WEIGHTS * std_scale
    lower_std = (lhi - llo) * STD_FRACTION_WEIGHTS * std_scale
    le_std    = (le_hi - le_lo) * STD_FRACTION_LE * std_scale
    te_std    = (te_hi - te_lo) * STD_FRACTION_TE * std_scale

    mean_upper = list(mean_params.get("upper_weights", DEFAULT_UPPER))
    mean_lower = list(mean_params.get("lower_weights", DEFAULT_LOWER))

    # Ensure exactly N_CST elements by padding/truncating
    while len(mean_upper) < N_CST:
        mean_upper.append(DEFAULT_UPPER[len(mean_upper) % N_CST])
    while len(mean_lower) < N_CST:
        mean_lower.append(DEFAULT_LOWER[len(mean_lower) % N_CST])

    upper_weights = [
        float(np.clip(np.random.normal(v, upper_std), ulo, uhi))
        for v in mean_upper[:N_CST]
    ]
    lower_weights = [
        float(np.clip(np.random.normal(v, lower_std), llo, lhi))
        for v in mean_lower[:N_CST]
    ]

    le = float(mean_params.get("leading_edge_weight", DEFAULT_LE))
    te = float(mean_params.get("TE_thickness", DEFAULT_TE))

    return {
        "upper_weights":       upper_weights,
        "lower_weights":       lower_weights,
        "leading_edge_weight": float(np.clip(np.random.normal(le, le_std), le_lo, le_hi)),
        "TE_thickness":        float(np.clip(np.random.normal(te, te_std), te_lo, te_hi)),
    }


def save_design_json(path: str, params: dict) -> str:
    design = {
        "upper_weights":       [float(v) for v in params["upper_weights"]],
        "lower_weights":       [float(v) for v in params["lower_weights"]],
        "leading_edge_weight": float(params["leading_edge_weight"]),
        "TE_thickness":        float(params["TE_thickness"]),
    }
    design["name"] = params.get("name", "design")
    with open(path, "w") as f:
        json.dump(design, f, indent=2)
    return path


def gaussian_neuralfoil(params: dict, out_dir: str = "./output", name: str = "design",
                        std_scale: float = 1.0) -> str:
    os.makedirs(out_dir, exist_ok=True)
    sampled = gaussian_sampling_neuralfoil(params, std_scale=std_scale)
    sampled["name"] = params.get("name", name)
    path = os.path.join(out_dir, f"{name}.json")
    return save_design_json(path, sampled)


def run_action_neuralfoil(action: str, **kwargs) -> str:
    if action in ("gaussain", "gaussian"):
        return gaussian_neuralfoil(**kwargs)
    raise ValueError(f"Unknown NeuralFoil action: {action}")
