"""
Variant of optimize.py using design space bounds that match the GA/lbfgsb
framework (environments/NeuralFoil/design_actions.py):

    upper_weights:       [-0.30, 0.60]   (vs [-0.25, 0.50] in original adjoint)
    lower_weights:       [-0.30, 0.30]   (vs [-0.50, 0.25] in original adjoint)
    leading_edge_weight: [-0.50, 0.50]   (vs [-1.00, 1.00] in original adjoint)
    TE_thickness:        [0.000, 0.010]  (vs fixed 0.0 in original adjoint)

Objective and constraints are identical to optimize.py / reward_exact_notebook:
  Minimize  mean(CD_i * w_i)  over 6 CL targets
  Subject to CM >= -0.133, conf > 0.90, thickness, TE angle, alpha monotonicity.
"""

import json
import os

import aerosandbox as asb
import aerosandbox.numpy as anp

from environments.NeuralFoil.rewards.multipoint_hpa import (
    CL_TARGETS,
    CL_WEIGHTS,
    MACH,
    _wiggliness,
)

from .optimize import _sym_wiggliness, _save_shape, _to_kulfan_airfoil

_DEFAULT_IPOPT_OPTIONS = {
    "ipopt.mu_strategy": "monotone",
    "ipopt.start_with_resto": "yes",
}


def run_adjoint_multipoint_framework_bounds(
    initial_kulfan=None,
    model_size="large",
    output_dir=None,
    name="adjoint_multipoint_fwbounds",
    solver_options=None,
) -> dict:
    """Run HPA multipoint IPOPT optimization with framework-matching bounds.

    Bounds match environments/NeuralFoil/design_actions.py (same as GA/lbfgsb):
      upper_weights:       [-0.30, 0.60]
      lower_weights:       [-0.30, 0.30]
      leading_edge_weight: [-0.50, 0.50]
      TE_thickness:        [0.000, 0.010]
    """
    import numpy as np

    initial_airfoil = _to_kulfan_airfoil(initial_kulfan)

    Re = 500e3 * (CL_TARGETS / 1.25) ** -0.5

    opti = asb.Opti()

    optimized_airfoil = asb.KulfanAirfoil(
        name="Optimized_fwbounds",
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

    alpha = opti.variable(
        init_guess=anp.degrees(CL_TARGETS / (2 * anp.pi)),
        lower_bound=-5,
        upper_bound=18,
    )

    aero = optimized_airfoil.get_aero_from_neuralfoil(
        alpha=alpha,
        Re=Re,
        mach=MACH,
        model_size=model_size,
    )

    opti.subject_to(
        [
            aero["analysis_confidence"] > 0.90,
            aero["CL"] == CL_TARGETS,
            anp.diff(alpha) > 0,
            aero["CM"] >= -0.133,
            optimized_airfoil.local_thickness(x_over_c=0.33) >= 0.128,
            optimized_airfoil.local_thickness(x_over_c=0.90) >= 0.014,
            optimized_airfoil.TE_angle() >= 6.03,
            optimized_airfoil.lower_weights[0] < -0.05,
            optimized_airfoil.upper_weights[0] > 0.05,
            optimized_airfoil.local_thickness() > 0,
        ]
    )

    opti.subject_to(
        _sym_wiggliness(optimized_airfoil) < 2 * _wiggliness(initial_airfoil)
    )

    opti.minimize(anp.mean(aero["CD"] * CL_WEIGHTS))

    opts = dict(_DEFAULT_IPOPT_OPTIONS)
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
    alphas_sol = [float(v) for v in np.array(sol(alpha))]
    CDs_sol = [float(v) for v in np.array(aero_sol["CD"])]
    CMs_sol = [float(v) for v in np.array(aero_sol["CM"])]
    confs_sol = [float(v) for v in np.array(aero_sol["analysis_confidence"])]
    weighted_cd = float(np.mean(np.array(aero_sol["CD"]) * CL_WEIGHTS))

    feasible = (
        all(c > 0.90 for c in confs_sol)
        and all(cm >= -0.133 for cm in CMs_sol)
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
        "alphas": alphas_sol,
        "CDs": CDs_sol,
        "CMs": CMs_sol,
        "analysis_confidences": confs_sol,
        "weighted_cd": weighted_cd,
        "solver_success": solver_success,
        "feasible": feasible,
        "n_iters": n_iters,
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

        full_result = dict(result)
        full_result["CL_targets"] = CL_TARGETS.tolist()
        full_result["CL_weights"] = CL_WEIGHTS.tolist()
        full_result["mach"] = float(MACH)
        with open(os.path.join(save_dir, "results.json"), "w") as f:
            json.dump(full_result, f, indent=2)

        _save_shape(optimized_airfoil, os.path.join(save_dir, "shape.png"))

    return result
