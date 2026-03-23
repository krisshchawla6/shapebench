"""
Gradient-based (IPOPT) adjoint optimizer for the HPA multipoint problem.

Wraps the exact AeroSandbox opti formulation from the NeuralFoil notebook:
  Minimize  mean(CD_i * w_i)  over 6 CL targets
Subject to the same constraints used by multipoint_hpa / reward_exact_notebook.

The returned dict is compatible with the NeuralFoil environment design format
(loadable by run_benchmark as an initial seed or saved as a standalone result).
"""

import json
import os
import sys

import aerosandbox as asb
import aerosandbox.numpy as anp

_ENV_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NEURALFOIL_SRC = os.path.join(_ENV_DIR, "neuralfoil_src")
if os.path.isdir(_NEURALFOIL_SRC) and _NEURALFOIL_SRC not in sys.path:
    sys.path.insert(0, _NEURALFOIL_SRC)

from environments.NeuralFoil.rewards.multipoint_hpa import (
    CL_TARGETS,
    CL_WEIGHTS,
    MACH,
    _wiggliness,
)


def _sym_wiggliness(af):
    """CasADi-safe wiggliness using anp.diff for symbolic variables."""
    return sum(
        anp.sum(anp.diff(anp.diff(arr)) ** 2)
        for arr in [af.lower_weights, af.upper_weights]
    )

_DEFAULT_IPOPT_OPTIONS = {
    "ipopt.mu_strategy": "monotone",
    "ipopt.start_with_resto": "yes",
}


def _to_kulfan_airfoil(initial_kulfan) -> asb.KulfanAirfoil:
    """Convert a dict or KulfanAirfoil to KulfanAirfoil; None → NACA0012."""
    if initial_kulfan is None:
        return asb.KulfanAirfoil("naca0012")
    if isinstance(initial_kulfan, asb.KulfanAirfoil):
        return initial_kulfan
    import numpy as np
    return asb.KulfanAirfoil(
        upper_weights=np.array(initial_kulfan["upper_weights"], dtype=float),
        lower_weights=np.array(initial_kulfan["lower_weights"], dtype=float),
        leading_edge_weight=float(initial_kulfan["leading_edge_weight"]),
        TE_thickness=0.0,
    )


def _save_shape(airfoil: asb.KulfanAirfoil, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    coords = np.array(airfoil.coordinates)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(coords[:, 0], coords[:, 1], "b-", linewidth=1.5)
    ax.fill(coords[:, 0], coords[:, 1], alpha=0.15, color="steelblue")
    ax.set_aspect("equal")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title("Adjoint-Optimized Airfoil")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def run_adjoint_optimization(
    initial_kulfan=None,
    model_size="large",
    output_dir=None,
    name="adjoint_optimized",
    solver_options=None,
) -> dict:
    """Run the HPA multipoint IPOPT optimization.

    Parameters
    ----------
    initial_kulfan:
        Starting point — a dict with keys ``upper_weights``, ``lower_weights``,
        ``leading_edge_weight``, ``TE_thickness`` (env JSON format), an
        ``asb.KulfanAirfoil``, or ``None`` to start from NACA0012.
    model_size:
        NeuralFoil model size string (e.g. ``"large"``, ``"xxxlarge"``).
    output_dir:
        If provided, saves:
          ``{output_dir}/{name}.json``         — design params (env-compatible)
          ``{output_dir}/save/results.json``   — full result dict
          ``{output_dir}/save/shape.png``      — airfoil shape plot
    name:
        Design name stored in the output JSON.
    solver_options:
        Dict of IPOPT options merged on top of the defaults
        ``{"ipopt.mu_strategy": "monotone", "ipopt.start_with_resto": "yes"}``.

    Returns
    -------
    dict with keys:
        ``upper_weights``, ``lower_weights``, ``leading_edge_weight``, ``TE_thickness``,
        ``name``, ``alphas``, ``CDs``, ``CMs``, ``analysis_confidences``,
        ``weighted_cd``, ``solver_success``, ``feasible``.
    """
    import numpy as np

    initial_airfoil = _to_kulfan_airfoil(initial_kulfan)

    Re = 500e3 * (CL_TARGETS / 1.25) ** -0.5

    opti = asb.Opti()

    optimized_airfoil = asb.KulfanAirfoil(
        name="Optimized",
        lower_weights=opti.variable(
            init_guess=initial_airfoil.lower_weights,
            lower_bound=-0.5,
            upper_bound=0.25,
        ),
        upper_weights=opti.variable(
            init_guess=initial_airfoil.upper_weights,
            lower_bound=-0.25,
            upper_bound=0.5,
        ),
        leading_edge_weight=opti.variable(
            init_guess=initial_airfoil.leading_edge_weight,
            lower_bound=-1,
            upper_bound=1,
        ),
        TE_thickness=0,
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

    solver_success = True
    sol = opti.solve(behavior_on_failure="return_last", options=opts)
    stats = sol._sol.stats()
    solver_success = bool(stats.get("success", False))
    n_iters = int(stats.get("iter_count", -1))

    optimized_airfoil = sol(optimized_airfoil)
    aero_sol = sol(aero)

    upper_weights = [float(v) for v in np.array(optimized_airfoil.upper_weights)]
    lower_weights = [float(v) for v in np.array(optimized_airfoil.lower_weights)]
    leading_edge_weight = float(np.array(optimized_airfoil.leading_edge_weight))
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
        "TE_thickness": 0.0,
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
            "TE_thickness": 0.0,
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
