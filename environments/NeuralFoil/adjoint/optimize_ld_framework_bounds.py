"""
Variant of optimize_ld_cm130.py using design space bounds that match the
GA/lbfgsb framework (environments/NeuralFoil/design_actions.py):

    upper_weights:       [-0.30, 0.60]   (vs [-0.25, 0.50] in original adjoint)
    lower_weights:       [-0.30, 0.30]   (vs [-0.50, 0.25] in original adjoint)
    leading_edge_weight: [-0.50, 0.50]   (vs [-1.00, 1.00] in original adjoint)
    TE_thickness:        [0.000, 0.010]  (vs fixed 0.0 in original adjoint)

The wider upper_weights upper bound (0.60 vs 0.50) may allow higher-camber
upper surfaces and a better L/D optimum. CM bound remains -0.130 to compensate
for the NeuralFoil->XFOIL CM bias of ~0.003.
"""

import json
import os

import aerosandbox.numpy as anp
import aerosandbox as asb
from environments.NeuralFoil.rewards.multipoint_hpa import _wiggliness

from .optimize import _save_shape, _to_kulfan_airfoil

_DEFAULT_LD_IPOPT_OPTIONS = {
    "ipopt.mu_strategy": "monotone",
    "ipopt.start_with_resto": "yes",
}

CM_LOWER_BOUND = -0.130


def _sym_wiggliness(af):
    return sum(
        anp.sum(anp.diff(anp.diff(arr)) ** 2)
        for arr in [af.lower_weights, af.upper_weights]
    )


def run_ld_adjoint_framework_bounds(
    initial_kulfan=None,
    alpha=5.0,
    re=1e7,
    mach=0.2,
    model_size="large",
    output_dir=None,
    name="ld_adjoint_framework_bounds",
    solver_options=None,
) -> dict:
    """
    Run constrained one-point L/D IPOPT optimization with bounds matching the
    GA/lbfgsb framework and CM >= -0.130.
    """
    import numpy as np

    initial_airfoil = _to_kulfan_airfoil(initial_kulfan)

    opti = asb.Opti()
    optimized_airfoil = asb.KulfanAirfoil(
        name="OptimizedLD_fwbounds",
        lower_weights=opti.variable(
            init_guess=initial_airfoil.lower_weights,
            lower_bound=-0.30,
            upper_bound=0.30,
        ),
        upper_weights=opti.variable(
            init_guess=initial_airfoil.upper_weights,
            lower_bound=-0.30,
            upper_bound=0.60,
        ),
        leading_edge_weight=opti.variable(
            init_guess=initial_airfoil.leading_edge_weight,
            lower_bound=-0.50,
            upper_bound=0.50,
        ),
        TE_thickness=opti.variable(
            init_guess=float(np.array(initial_airfoil.TE_thickness))
                       if hasattr(initial_airfoil, 'TE_thickness') else 0.0,
            lower_bound=0.000,
            upper_bound=0.010,
        ),
    )

    aero = optimized_airfoil.get_aero_from_neuralfoil(
        alpha=float(alpha),
        Re=float(re),
        mach=float(mach),
        model_size=model_size,
    )

    opti.subject_to(
        [
            aero["analysis_confidence"] > 0.90,
            aero["CM"] >= CM_LOWER_BOUND,
            optimized_airfoil.local_thickness(x_over_c=0.33) >= 0.128,
            optimized_airfoil.local_thickness(x_over_c=0.90) >= 0.014,
            optimized_airfoil.TE_angle() >= 6.03,
            optimized_airfoil.lower_weights[0] < -0.05,
            optimized_airfoil.upper_weights[0] > 0.05,
            optimized_airfoil.local_thickness() > 0,
            _sym_wiggliness(optimized_airfoil) < 2 * _wiggliness(initial_airfoil),
        ]
    )

    ld = aero["CL"] / anp.maximum(aero["CD"], 1e-8)
    reg = (
        anp.sum((optimized_airfoil.upper_weights - initial_airfoil.upper_weights) ** 2)
        + anp.sum((optimized_airfoil.lower_weights - initial_airfoil.lower_weights) ** 2)
        + (optimized_airfoil.leading_edge_weight - initial_airfoil.leading_edge_weight) ** 2
    )
    opti.minimize(-ld + 1e-6 * reg)

    opts = dict(_DEFAULT_LD_IPOPT_OPTIONS)
    if solver_options:
        opts.update(solver_options)

    sol = opti.solve(behavior_on_failure="return_last", options=opts)
    stats = sol._sol.stats()
    solver_success = bool(stats.get("success", False))
    n_iters = int(stats.get("iter_count", -1))

    optimized_airfoil = sol(optimized_airfoil)
    aero_sol = sol(aero)

    upper_weights = [float(v) for v in np.array(optimized_airfoil.upper_weights)]
    lower_weights = [float(v) for v in np.array(optimized_airfoil.lower_weights)]
    leading_edge_weight = float(np.array(optimized_airfoil.leading_edge_weight))
    te_thickness = float(np.array(optimized_airfoil.TE_thickness))

    cl = float(np.array(aero_sol["CL"]))
    cd = float(np.array(aero_sol["CD"]))
    cm = float(np.array(aero_sol["CM"]))
    conf = float(np.array(aero_sol["analysis_confidence"]))
    ld_value = float(cl / cd) if cd > 1e-9 else float("-inf")

    feasible = (
        conf > 0.90
        and cm >= CM_LOWER_BOUND
        and upper_weights[0] > 0.05
        and lower_weights[0] < -0.05
        and float(np.array(optimized_airfoil.local_thickness(0.33))) >= 0.128
        and float(np.array(optimized_airfoil.local_thickness(0.90))) >= 0.014
        and float(np.array(optimized_airfoil.TE_angle())) >= 6.03
    )

    result = {
        "upper_weights": upper_weights,
        "lower_weights": lower_weights,
        "leading_edge_weight": leading_edge_weight,
        "TE_thickness": te_thickness,
        "name": name,
        "alpha": float(alpha),
        "re": float(re),
        "mach": float(mach),
        "model_size": model_size,
        "CL": cl,
        "CD": cd,
        "CM": cm,
        "analysis_confidence": conf,
        "L_D": ld_value,
        "solver_success": solver_success,
        "n_iters": n_iters,
        "feasible": feasible,
        "cm_lower_bound": CM_LOWER_BOUND,
    }

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_dir = os.path.join(output_dir, "save")
        os.makedirs(save_dir, exist_ok=True)

        design = {
            "upper_weights": upper_weights,
            "lower_weights": lower_weights,
            "leading_edge_weight": leading_edge_weight,
            "TE_thickness": te_thickness,
            "name": name,
        }
        with open(os.path.join(output_dir, f"{name}.json"), "w") as f:
            json.dump(design, f, indent=2)

        with open(os.path.join(save_dir, "results.json"), "w") as f:
            json.dump(result, f, indent=2)

        _save_shape(optimized_airfoil, os.path.join(save_dir, "shape.png"))

    return result
