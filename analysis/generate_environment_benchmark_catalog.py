from __future__ import annotations

import ast
import csv
import html
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = ROOT / "environments"
OUT_CSV = ROOT / "analysis" / "environment_benchmark_catalog.csv"
OUT_HTML = ROOT / "analysis" / "environment_benchmark_catalog.html"


TAIL_TYPE_LABELS = {0: "conventional", 1: "T-tail"}
ENGINE_POSITION_LABELS = {1: "under-wing", 2: "rear-fuselage"}
LAYOUT_LABELS = {1: "under-wing", 2: "behind-fuselage"}


REWARD_PROFILES: dict[str, dict[str, str]] = {
    "BlendedNet/rewards/ld_ratio.py": {
        "reward_name": "L/D Ratio",
        "reward_summary": "Single-point blended-wing-body aerodynamic efficiency.",
        "reward_formula": "reward = CL_approx / CD_approx, with CL_approx = -Cp_mean and CD_approx = Cfx_mean.",
        "reward_python_compact": "CL_approx = -Cp_mean; CD_approx = Cfx_mean; reward = CL_approx / CD_approx if abs(CD_approx) > 1e-12 else FAIL_REWARD",
    },
    "BlendedNet/rewards/sahpebench_1.py": {
        "reward_name": "ShapeBench-1",
        "reward_summary": "Minimize drag while matching a target lift coefficient and preferred alpha range.",
        "reward_formula": "reward = -(CD_approx + lambda_cl*|CL_approx - CL_target| + lambda_alpha_range*alpha_violation).",
        "reward_python_compact": "CL_approx = -Cp_mean; CD_approx = Cfx_mean; cl_error = abs(CL_approx - cl_target); reward = -(CD_approx + lambda_cl*cl_error + lambda_alpha_range*alpha_violation)",
    },
    "BlendedNet/rewards/shapebench_5.py": {
        "reward_name": "ShapeBench-5",
        "reward_summary": "Multipoint drag minimization across fixed target CL conditions.",
        "reward_formula": "reward = -mean(CD_approx_i) after solving alpha_i for each CL_target_i.",
        "reward_python_compact": "alphas = solve_alpha_for_targets(CL_targets); cds = [CD_approx_i for each solved point]; reward = -mean(cds)",
    },
    "CERAS/rewards/fuel_mass.py": {
        "reward_name": "Fuel Mass",
        "reward_summary": "Mission fuel burn with static-margin feasibility penalty.",
        "reward_formula": "reward = -fuel_mass - penalty(static_margin outside [0.05, 0.10]).",
        "reward_python_compact": "penalty = sm_penalty(static_margin, lo=0.05, hi=0.10); reward = -fuel_mass - penalty",
    },
    "DrivAer_Star/rewards/cd_only.py": {
        "reward_name": "Drag Coefficient",
        "reward_summary": "Vehicle drag minimization from predicted force coefficients.",
        "reward_formula": "reward = -Cd, where Cd = drag / (0.5*rho*u^2*area_ref).",
        "reward_python_compact": "q = 0.5 * rho * u**2; Cd = drag / (q * area_ref) if q * area_ref > 1e-12 else 0.0; reward = -Cd",
    },
    "Dragon_ARCHIVE/rewards/fuel_mass.py": {
        "reward_name": "Fuel Mass",
        "reward_summary": "Mission fuel burn with penalties for span, takeoff, fan-span, and climb constraints.",
        "reward_formula": "reward = -fuel_mass - penalty(span, TOFL, fan_span, climb_duration, climb_slope violations).",
        "reward_python_compact": "penalty = p_span + p_tofl + p_fan + p_climb_duration + p_climb_slope; reward = -fuel_mass - penalty",
    },
    "fenics_2d/rewards/default.py": {
        "reward_name": "Solver Reward",
        "reward_summary": "Pass-through reward from the FEniCS solver's internal penalized objective.",
        "reward_formula": "reward = solver_computed_reward.",
        "reward_python_compact": "reward = solver_reward",
    },
    "Mixed_integer_yiren/rewards/ld_ratio.py": {
        "reward_name": "L/D Ratio",
        "reward_summary": "Supersonic transport lift-to-drag ratio at the fixed SUAVE VLM cruise condition.",
        "reward_formula": "reward = L/D if finite and positive; otherwise FAIL_REWARD.",
        "reward_python_compact": "reward = LtoD if isfinite(LtoD) and LtoD > 0 else FAIL_REWARD",
    },
    "NeuralFoil/rewards/ld_ratio.py": {
        "reward_name": "L/D Ratio",
        "reward_summary": "Single-point airfoil aerodynamic efficiency.",
        "reward_formula": "reward = CL / CD.",
        "reward_python_compact": "reward = CL / CD if CD > 1e-9 else FAIL_REWARD",
    },
    "NeuralFoil/rewards/constrained_ld.py": {
        "reward_name": "Constrained L/D",
        "reward_summary": "Airfoil efficiency penalized away from a target pitching moment.",
        "reward_formula": "reward = CL/CD - w_cm*(CM - cm_target)^2.",
        "reward_python_compact": "LD = CL / CD if CD > 1e-9 else 0.0; reward = LD - w_cm * (CM - cm_target)**2 if CD > 1e-9 else FAIL_REWARD",
    },
    "NeuralFoil/rewards/max_cl.py": {
        "reward_name": "Max CL",
        "reward_summary": "Lift maximization with optional drag penalty.",
        "reward_formula": "reward = CL - w_cd*CD.",
        "reward_python_compact": "reward = CL - w_cd * CD",
    },
    "NeuralFoil/rewards/ld_ratio_constrained_m02_re1e7.py": {
        "reward_name": "Constrained L/D at M0.2 Re1e7",
        "reward_summary": "One-point constrained airfoil benchmark with weighted feasibility penalties.",
        "reward_formula": "reward = L/D - weighted_constraint_violation_sum.",
        "reward_python_compact": "LD = CL / CD; total_violation = sum(weight_i * violation_i); reward = LD - total_violation",
    },
    "NeuralFoil/rewards/ld_ratio_constrained_m02_re1e7_normalized.py": {
        "reward_name": "Normalized Constrained L/D at M0.2 Re1e7",
        "reward_summary": "One-point constrained benchmark with a single normalized penalty scale.",
        "reward_formula": "reward = L/D - lambda_penalty*sum(fractional_violations).",
        "reward_python_compact": "LD = CL / CD; total_violation = sum(violations.values()); reward = LD - lambda_penalty * total_violation",
    },
    "NeuralFoil/rewards/ld_ratio_constrained_m02_re1e7_normalized_conf95.py": {
        "reward_name": "Normalized Constrained L/D at M0.2 Re1e7 (conf95)",
        "reward_summary": "Normalized constrained benchmark with tightened confidence and CM bounds.",
        "reward_formula": "reward = L/D - lambda_penalty*sum(fractional_violations), with conf > 0.95 and CM >= -0.130 targets.",
        "reward_python_compact": "LD = CL / CD; total_violation = sum(violations.values()); reward = LD - lambda_penalty * total_violation",
    },
    "NeuralFoil/rewards/multipoint_hpa.py": {
        "reward_name": "Multipoint HPA",
        "reward_summary": "Weighted multipoint human-powered-aircraft drag objective with feasibility penalties.",
        "reward_formula": "reward = -weighted_mean(CD_i over solved CL targets) - weighted_penalties.",
        "reward_python_compact": "objective = -weighted_mean(CD_i, weights_i); penalty = sum(weight_i * violation_i); reward = objective - penalty",
    },
    "NeuralFoil/rewards/reward_exact_notebook.py": {
        "reward_name": "Exact Notebook Multipoint",
        "reward_summary": "Strict notebook-faithful multipoint objective; all CL targets must be solved.",
        "reward_formula": "reward = -mean(CD_i*w_i) - weighted_penalties, with strict failure if any target is unreachable.",
        "reward_python_compact": "objective = -mean(CD_i * w_i) if all_targets_solved else 0.0; reward = objective - penalty if all_targets_solved else FAIL_REWARD",
    },
    "SuperWing/rewards/ld_ratio.py": {
        "reward_name": "L/D Ratio",
        "reward_summary": "Transonic wing efficiency at the specified Mach and angle of attack.",
        "reward_formula": "reward = CL / CD when CD > 0 and CL is physically plausible; otherwise FAIL_REWARD.",
        "reward_python_compact": "LD = CL / CD if CD > 1e-9 and CL >= 0.05 else 0.0; reward = LD if CD > 1e-9 and CL >= 0.05 else FAIL_REWARD",
    },
    "SuperWing/rewards/min_cd_cl055.py": {
        "reward_name": "Min Cd at CL 0.55",
        "reward_summary": "Transonic wing drag minimization with a quadratic penalty for missing CL = 0.55.",
        "reward_formula": "reward = -CD - lambda_cl*(CL - CL_target)^2, with CL_target = 0.55 by default.",
        "reward_python_compact": "cl_penalty = lambda_cl * (CL - cl_target)**2; reward = -CD - cl_penalty if CD > 1e-9 else FAIL_REWARD",
    },
    "SuperWing/rewards/multipoint_avg_cd.py": {
        "reward_name": "Multipoint Avg Cd",
        "reward_summary": "Transonic wing drag minimization across a variable set of target CL operating points.",
        "reward_formula": "reward = -mean(CD_i) after solving AoA_i for each target CL_i at fixed Mach.",
        "reward_python_compact": "alphas = solve_aoa_for_targets(CL_targets, mach); cds = [CD_i for each solved target]; reward = -mean(cds) if all_targets_solved else FAIL_REWARD",
    },
    "SuperWing/rewards/range_optimization.py": {
        "reward_name": "Range Optimization",
        "reward_summary": "Transonic range proxy with a quadratic lift-loading penalty.",
        "reward_formula": "reward = Mach*CL/CD - lambda_constraint*(Mach^2*CL - l)^2, where l = Mach*target_cl.",
        "reward_python_compact": "range_proxy = mach * CL / CD; l_value = mach * target_cl; penalty = lambda_constraint * (mach**2 * CL - l_value)**2; reward = range_proxy - penalty if CD > 1e-9 else FAIL_REWARD",
    },
    "SuperWing/rewards/multipoint_mach_range_optimization.py": {
        "reward_name": "Multipoint Mach Range Optimization",
        "reward_summary": "Average the range-optimization reward across several Mach points at fixed CL target.",
        "reward_formula": "reward = mean_i[Mach_i*CL_i/CD_i - lambda_constraint*(Mach_i^2*CL_i - Mach_i*target_cl)^2] after solving AoA_i for the fixed target_cl.",
        "reward_python_compact": "point_rewards = [mach_i * CL_i / CD_i - lambda_constraint * (mach_i**2 * CL_i - mach_i * target_cl)**2 for each solved Mach point]; reward = mean(point_rewards) if all_points_solved else FAIL_REWARD",
    },
    "SuperWing/rewards/weighted_cl_range_optimization.py": {
        "reward_name": "Weighted CL Range Optimization",
        "reward_summary": "Weighted average of the range-optimization reward across several CL targets at fixed Mach.",
        "reward_formula": "reward = weighted_mean_i[Mach*CL_i/CD_i - lambda_constraint*(Mach^2*CL_i - Mach*CL_target_i)^2] after solving AoA_i for each target CL_i.",
        "reward_python_compact": "point_rewards = [mach * CL_i / CD_i - lambda_constraint * (mach**2 * CL_i - mach * cl_target_i)**2 for each solved CL target]; reward = weighted_mean(point_rewards, cl_weights) if all_points_solved else FAIL_REWARD",
    },
    "SuperWing/rewards/min_cd_alternative_variation.py": {
        "reward_name": "Min Cd Alternative Variation",
        "reward_summary": "Single-point drag minimization with a one-sided CL >= target constraint penalty.",
        "reward_formula": "reward = -CD - lambda_cl*max(0, CL_target - CL)^2.",
        "reward_python_compact": "cl_shortfall = max(0.0, cl_target - CL); reward = -CD - lambda_cl * cl_shortfall**2 if CD > 1e-9 else FAIL_REWARD",
    },
    "SuperWing/rewards/min_cd_avf_altenrtaive.py": {
        "reward_name": "Min Cd Avf Altenrtaive",
        "reward_summary": "Average minimum feasible drag across several Mach points with CL >= target enforced at each point.",
        "reward_formula": "reward = -mean(CD_i) after solving the boundary-feasible CL_target condition at each Mach point.",
        "reward_python_compact": "cds = [CD_i for each Mach point solved at CL_target]; reward = -mean(cds) if all_points_solved else FAIL_REWARD",
    },
    "vlm_3d/rewards/single_pt_ld.py": {
        "reward_name": "Single-Point L/D",
        "reward_summary": "3D delta-wing lift-to-drag ratio relative to a fixed baseline.",
        "reward_formula": "reward = L/D - 5.45.",
        "reward_python_compact": "LD = CL / CDi if abs(CDi) > 1e-12 else 0.0; reward = LD - 5.45",
    },
    "vlm_3d/rewards/two_pt_multi.py": {
        "reward_name": "Two-Point Multi-Objective",
        "reward_summary": "Supersonic/subsonic multi-condition objective with CL, CM, and static-margin penalties.",
        "reward_formula": "reward = -CDi_sup - w_cl[(CL_sup-CL*_sup)^2 + (CL_sub-CL*_sub)^2] - w_cm(CM_sup^2 + CM_sub^2) - w_kn*max(0, Kn_target - Kn)^2.",
        "reward_python_compact": "reward = -CDi_sup - w_cl*((CL_sup-CL_target_sup)**2 + (CL_sub-CL_target_sub)**2) - w_cm*(CM_sup**2 + CM_sub**2) - w_kn*max(0.0, kn_target-Kn)**2",
    },
}


def hpa_re(cl_target: float) -> float:
    return 500_000.0 * (cl_target / 1.25) ** -0.5


BENCHMARK_SPECS: list[dict[str, Any]] = [
    {
        "environment": "BlendedNet",
        "reward_file": "BlendedNet/rewards/ld_ratio.py",
        "generator": "grid",
        "axes": {
            "mach": [0.20, 0.25, 0.30, 0.35, 0.40],
            "reynolds": [3.0e6, 1.0e7, 3.0e7],
            "alpha_deg": [-3.0, 0.0, 3.0, 5.0],
        },
        "condition_basis": "Reward defaults and BlendedNet README flight inputs.",
    },
    {
        "environment": "BlendedNet",
        "reward_file": "BlendedNet/rewards/sahpebench_1.py",
        "generator": "grid",
        "axes": {
            "mach": [0.25, 0.30, 0.35],
            "reynolds": [1.0e7],
            "alpha_deg": [0.0, 3.0, 5.0],
            "cl_target": [0.185, 0.206, 0.227],
        },
        "fixed": {
            "alpha_pref_min": -3.0,
            "alpha_pref_max": 3.0,
            "lambda_cl": "required",
            "lambda_alpha_range": "required",
        },
        "condition_basis": "ShapeBench1 arguments, centered on the cruise CL values used elsewhere in the BWB benchmark family.",
        "notes": "Penalty weights are intentionally metadata, not sweep axes.",
    },
    {
        "environment": "BlendedNet",
        "reward_file": "BlendedNet/rewards/shapebench_5.py",
        "generator": "multipoint",
        "axes": {
            "mach": [0.25, 0.30, 0.35],
            "reynolds": [1.0e7, 3.0e7],
        },
        "targets": [
            {"cl_target": 0.206},
            {"cl_target": 0.206},
            {"cl_target": 0.206},
            {"cl_target": 0.185},
            {"cl_target": 0.227},
        ],
        "condition_basis": "ShapeBench5 CL target list with a modest outer Mach/Re sweep for benchmark organization.",
    },
    {
        "environment": "CERAS",
        "reward_file": "CERAS/rewards/fuel_mass.py",
        "generator": "grid",
        "axes": {
            "cruise_altitude_ft": [30000, 32000, 34000, 36000],
            "engine_count": [2, 3, 4],
            "tail_type": [0, 1],
            "engine_position": [1, 2],
        },
        "notes": "Mission-style environment; these are discrete mission/layout conditions rather than Mach-Re-alpha points.",
        "condition_basis": "Exact integer and categorical levels from CERAS design_actions.py.",
    },
    {
        "environment": "DrivAer_Star",
        "reward_file": "DrivAer_Star/rewards/cd_only.py",
        "generator": "grid",
        "axes": {
            "rho_kg_m3": [1.15, 1.20, 1.25],
            "speed_m_s": [25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0],
        },
        "fixed": {
            "area_ref_m2": 2.37,
        },
        "condition_basis": "CdOnly reward arguments; road-vehicle cases use density and speed instead of Mach/Re.",
    },
    {
        "environment": "fenics_2d",
        "reward_file": "fenics_2d/rewards/default.py",
        "generator": "grid",
        "axes": {
            "reynolds": [1.0, 5.0, 10.0, 20.0, 50.0, 100.0],
            "cfl": [0.25, 0.50, 1.00],
            "final_time": [10.0, 15.0, 20.0],
        },
        "condition_basis": "Expanded from the exposed FEniCS config defaults; Reynolds is the only aerodynamic state variable here.",
    },
    {
        "environment": "Mixed_integer_yiren",
        "reward_file": "Mixed_integer_yiren/rewards/ld_ratio.py",
        "generator": "grid",
        "axes": {
            "mach": [1.5],
            "alpha_deg": [2.5],
            "altitude_ft": [0.0],
            "sideslip_deg": [0.0],
        },
        "condition_basis": "Hard-coded SUAVE VLM condition in Mixed_integer_yiren.environment._run_vlm().",
        "notes": "Only one true reward case exists because the environment fixes its operating condition internally.",
    },
    {
        "environment": "NeuralFoil",
        "reward_file": "NeuralFoil/rewards/ld_ratio.py",
        "generator": "grid",
        "axes": {
            "alpha_deg": [0.0, 2.5, 5.0, 7.5, 10.0],
            "reynolds": [5.0e5, 1.0e6, 3.0e6, 1.0e7, 3.0e7],
            "model_size": ["small", "large"],
        },
        "fixed": {"n_crit": 9.0},
        "condition_basis": "Reward arguments plus a compact low-to-high Reynolds sweep.",
    },
    {
        "environment": "NeuralFoil",
        "reward_file": "NeuralFoil/rewards/constrained_ld.py",
        "generator": "grid",
        "axes": {
            "alpha_deg": [0.0, 2.5, 5.0, 7.5, 10.0],
            "reynolds": [5.0e5, 1.0e6, 3.0e6, 1.0e7, 3.0e7],
            "model_size": ["small", "large"],
        },
        "fixed": {"n_crit": 9.0, "w_cm": 10.0, "cm_target": 0.0},
        "condition_basis": "Same pointwise grid as ld_ratio with the moment target fixed at its code default.",
    },
    {
        "environment": "NeuralFoil",
        "reward_file": "NeuralFoil/rewards/max_cl.py",
        "generator": "grid",
        "axes": {
            "alpha_deg": [6.0, 8.0, 10.0, 12.0, 14.0],
            "reynolds": [5.0e5, 1.0e6, 3.0e6, 1.0e7, 3.0e7],
            "model_size": ["small", "large"],
        },
        "fixed": {"n_crit": 9.0, "w_cd": 0.0},
        "condition_basis": "High-lift NeuralFoil cases using the reward's alpha-rich operating region.",
    },
    {
        "environment": "NeuralFoil",
        "reward_file": "NeuralFoil/rewards/ld_ratio_constrained_m02_re1e7.py",
        "generator": "grid",
        "axes": {
            "alpha_deg": [5.0],
            "reynolds": [1.0e7],
            "mach": [0.2],
            "model_size": ["large"],
        },
        "condition_basis": "Single canonical operating point encoded directly in the reward.",
    },
    {
        "environment": "NeuralFoil",
        "reward_file": "NeuralFoil/rewards/ld_ratio_constrained_m02_re1e7_normalized.py",
        "generator": "grid",
        "axes": {
            "alpha_deg": [5.0],
            "reynolds": [1.0e7],
            "mach": [0.2],
            "model_size": ["large"],
        },
        "fixed": {"lambda_penalty": 500.0},
        "condition_basis": "Single canonical normalized-penalty operating point encoded directly in the reward.",
    },
    {
        "environment": "NeuralFoil",
        "reward_file": "NeuralFoil/rewards/ld_ratio_constrained_m02_re1e7_normalized_conf95.py",
        "generator": "grid",
        "axes": {
            "alpha_deg": [5.0],
            "reynolds": [1.0e7],
            "mach": [0.2],
            "model_size": ["large"],
        },
        "fixed": {"lambda_penalty": 500.0},
        "condition_basis": "Single canonical conf95 operating point encoded directly in the reward.",
    },
    {
        "environment": "NeuralFoil",
        "reward_file": "NeuralFoil/rewards/multipoint_hpa.py",
        "generator": "multipoint",
        "axes": {"model_size": ["large"]},
        "fixed": {"mach": 0.03, "n_crit": 9.0},
        "targets": [
            {"cl_target": 0.8, "cl_weight": 5.0, "reynolds": hpa_re(0.8)},
            {"cl_target": 1.0, "cl_weight": 6.0, "reynolds": hpa_re(1.0)},
            {"cl_target": 1.2, "cl_weight": 7.0, "reynolds": hpa_re(1.2)},
            {"cl_target": 1.4, "cl_weight": 8.0, "reynolds": hpa_re(1.4)},
            {"cl_target": 1.5, "cl_weight": 9.0, "reynolds": hpa_re(1.5)},
            {"cl_target": 1.6, "cl_weight": 10.0, "reynolds": hpa_re(1.6)},
        ],
        "condition_basis": "Exact HPA notebook CL targets, weights, Mach, and Reynolds schedule.",
    },
    {
        "environment": "NeuralFoil",
        "reward_file": "NeuralFoil/rewards/reward_exact_notebook.py",
        "generator": "multipoint",
        "axes": {"model_size": ["large"]},
        "fixed": {"mach": 0.03, "n_crit": 9.0},
        "targets": [
            {"cl_target": 0.8, "cl_weight": 5.0, "reynolds": hpa_re(0.8)},
            {"cl_target": 1.0, "cl_weight": 6.0, "reynolds": hpa_re(1.0)},
            {"cl_target": 1.2, "cl_weight": 7.0, "reynolds": hpa_re(1.2)},
            {"cl_target": 1.4, "cl_weight": 8.0, "reynolds": hpa_re(1.4)},
            {"cl_target": 1.5, "cl_weight": 9.0, "reynolds": hpa_re(1.5)},
            {"cl_target": 1.6, "cl_weight": 10.0, "reynolds": hpa_re(1.6)},
        ],
        "condition_basis": "Exact-notebook reward reusing the same fixed multipoint schedule as MultipointHPA.",
    },
    {
        "environment": "SuperWing",
        "reward_file": "SuperWing/rewards/ld_ratio.py",
        "permutation_file": "environments/SuperWing/rewards/ld_ratio.permutations.json",
    },
    {
        "environment": "SuperWing",
        "reward_file": "SuperWing/rewards/min_cd_cl055.py",
        "permutation_file": "environments/SuperWing/rewards/min_cd_cl055.permutations.json",
    },
    {
        "environment": "SuperWing",
        "reward_file": "SuperWing/rewards/multipoint_avg_cd.py",
        "permutation_file": "environments/SuperWing/rewards/multipoint_avg_cd.permutations.json",
    },
    {
        "environment": "SuperWing",
        "reward_file": "SuperWing/rewards/range_optimization.py",
        "permutation_file": "environments/SuperWing/rewards/range_optimization.permutations.json",
    },
    {
        "environment": "SuperWing",
        "reward_file": "SuperWing/rewards/multipoint_mach_range_optimization.py",
        "permutation_file": "environments/SuperWing/rewards/multipoint_mach_range_optimization.permutations.json",
    },
    {
        "environment": "SuperWing",
        "reward_file": "SuperWing/rewards/weighted_cl_range_optimization.py",
        "permutation_file": "environments/SuperWing/rewards/weighted_cl_range_optimization.permutations.json",
    },
    {
        "environment": "SuperWing",
        "reward_file": "SuperWing/rewards/min_cd_alternative_variation.py",
        "permutation_file": "environments/SuperWing/rewards/min_cd_alternative_variation.permutations.json",
    },
    {
        "environment": "SuperWing",
        "reward_file": "SuperWing/rewards/min_cd_avf_altenrtaive.py",
        "permutation_file": "environments/SuperWing/rewards/min_cd_avf_altenrtaive.permutations.json",
    },
    {
        "environment": "vlm_3d",
        "reward_file": "vlm_3d/rewards/single_pt_ld.py",
        "generator": "grid",
        "axes": {
            "alpha_deg": [0.0, 5.0, 10.0, 15.0, 20.0],
            "mach": [0.2, 0.3, 0.4, 0.5],
            "reynolds": [1.0e6, 3.0e6, 1.0e7, 3.0e7],
        },
        "condition_basis": "SinglePtLD reward arguments plus the exact saved test point at alpha 10, Mach 0.3, Re 3e6.",
    },
    {
        "environment": "vlm_3d",
        "reward_file": "vlm_3d/rewards/two_pt_multi.py",
        "generator": "two_point",
        "axes": {
            "aoa_sup_deg": [-3.0, -1.5, 0.0, 1.5, 3.0],
            "aoa_sub_deg": [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0],
        },
        "fixed": {
            "mach_sup": 1.8,
            "mach_sub": 0.3,
            "reynolds_sup": 80.4e6,
            "reynolds_sub": 101.8e6,
            "cl_target_sup": 0.1665,
            "cl_target_sub": 0.6933,
            "delta_aoa_deg": 0.01,
        },
        "condition_basis": "Two-point Yiren-style constants from the reward plus its documented AoA ranges.",
    },
]


COLUMNS = [
    "environment",
    "reward_formula",
    "reward_python_compact",
    "reward_summary",
    "case_kind",
    "point_role",
    "benchmark_mode",
    "permutation_index",
    "run_status",
    "conditions_compact",
    "mach",
    "mach_point_count",
    "mach_points_all",
    "reynolds",
    "alpha_deg",
    "aoa_sup_deg",
    "aoa_sub_deg",
    "delta_aoa_deg",
    "altitude_ft",
    "sideslip_deg",
    "cl_target",
    "cl_target_index",
    "cl_target_count",
    "cl_targets_all",
    "cl_weight",
    "cl_weights_all",
    "rho_kg_m3",
    "speed_m_s",
    "area_ref_m2",
    "model_size",
    "n_crit",
    "lambda_penalty",
    "lambda_constraint",
    "w_cm",
    "cm_target",
    "w_cd",
    "lambda_cl",
    "lambda_alpha_range",
    "alpha_pref_min",
    "alpha_pref_max",
    "cruise_altitude_ft",
    "engine_count",
    "tail_type",
    "tail_type_label",
    "engine_position",
    "engine_position_label",
    "architecture",
    "layout",
    "layout_label",
    "fan_opr",
    "climb_vspeed_ft_min",
    "climb_slope_rad",
    "tofl_sizing_m",
    "cfl",
    "final_time",
    "condition_basis",
    "run_artifact",
    "notes",
    "reward_name",
    "reward_file",
]


def relpath(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def dict_product(axes: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(axes.keys())
    if not keys:
        return [{}]
    values = [axes[k] for k in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".10g")
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=True, sort_keys=isinstance(value, dict))
    return str(value)


def resolve_spec(spec: dict[str, Any]) -> dict[str, Any]:
    permutation_file = spec.get("permutation_file")
    if not permutation_file:
        return spec

    data = json.loads((ROOT / permutation_file).read_text())
    resolved = dict(spec)
    resolved["default_case"] = data.get("default_case", {})
    resolved["condition_basis"] = data.get("condition_basis", resolved.get("condition_basis", ""))
    resolved["notes"] = data.get("notes", resolved.get("notes", ""))

    all_permutations = data.get("all_permutations", {})
    for key, value in all_permutations.items():
        resolved[key] = value

    return resolved


def reward_metadata(reward_file: str) -> dict[str, str]:
    text = (ENV_DIR / reward_file).read_text()
    tree = ast.parse(text)
    reward_class = ""
    reward_module = Path(reward_file).stem
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == "BaseReward":
                    reward_class = node.name
                    break
                if isinstance(base, ast.Attribute) and base.attr == "BaseReward":
                    reward_class = node.name
                    break
        if reward_class:
            break
    profile = REWARD_PROFILES[reward_file]
    return {
        "_reward_module": reward_module,
        "_reward_class": reward_class,
        "reward_name": profile["reward_name"],
        "reward_summary": profile["reward_summary"],
        "reward_formula": profile["reward_formula"],
        "reward_python_compact": profile["reward_python_compact"],
    }


def base_row(spec: dict[str, Any], reward_meta: dict[str, str]) -> dict[str, Any]:
    row = {col: "" for col in COLUMNS}
    row.update(reward_meta)
    row["environment"] = spec["environment"]
    row["reward_file"] = spec["reward_file"]
    row["condition_basis"] = spec.get("condition_basis", "")
    row["notes"] = spec.get("notes", "")
    return row


def build_case_id(environment: str, reward_module: str, payload: dict[str, Any]) -> str:
    parts = [environment.lower(), reward_module]
    for key, value in sorted(payload.items()):
        if value in ("", None):
            continue
        parts.append(f"{key}={stringify(value)}")
    return "|".join(parts)


def add_labels(row: dict[str, Any]) -> None:
    if row.get("tail_type") != "":
        row["tail_type_label"] = TAIL_TYPE_LABELS.get(int(row["tail_type"]), "")
    if row.get("engine_position") != "":
        row["engine_position_label"] = ENGINE_POSITION_LABELS.get(int(row["engine_position"]), "")
    if row.get("layout") != "":
        row["layout_label"] = LAYOUT_LABELS.get(int(row["layout"]), "")


def conditions_compact(row: dict[str, Any]) -> str:
    keys = [
        ("mach", "Mach"),
        ("mach_point_count", "nMach"),
        ("reynolds", "Re"),
        ("alpha_deg", "alpha"),
        ("aoa_sup_deg", "aoa_sup"),
        ("aoa_sub_deg", "aoa_sub"),
        ("delta_aoa_deg", "delta_aoa"),
        ("altitude_ft", "alt_ft"),
        ("sideslip_deg", "beta"),
        ("cl_target", "CL*"),
        ("cl_target_count", "nCL"),
        ("lambda_constraint", "lambda_range"),
        ("rho_kg_m3", "rho"),
        ("speed_m_s", "U"),
        ("model_size", "model"),
        ("cruise_altitude_ft", "cruise_alt_ft"),
        ("engine_count", "engines"),
        ("tail_type_label", "tail"),
        ("engine_position_label", "engine_pos"),
        ("architecture", "arch"),
        ("layout_label", "layout"),
        ("fan_opr", "fan_opr"),
        ("climb_vspeed_ft_min", "climb_vspeed"),
        ("climb_slope_rad", "climb_slope"),
        ("cfl", "cfl"),
        ("final_time", "t_final"),
    ]
    parts = []
    for key, label in keys:
        value = row.get(key, "")
        if value not in ("", None):
            parts.append(f"{label}={stringify(value)}")
    return "; ".join(parts)


def approx_eq(a: Any, b: Any, tol: float = 1e-9) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def classify_run(row: dict[str, Any]) -> tuple[str, str]:
    env = row["environment"]
    reward = row.get("_reward_module", "")

    if (
        env == "vlm_3d"
        and reward == "single_pt_ld"
        and approx_eq(row.get("alpha_deg"), 10.0)
        and approx_eq(row.get("mach"), 0.3)
        and approx_eq(row.get("reynolds"), 3.0e6)
    ):
        return "RUN_CONFIRMED", "environments/vlm_3d/test_ev/results.json"

    if (
        env == "SuperWing"
        and reward == "ld_ratio"
        and approx_eq(row.get("mach"), 0.82)
        and approx_eq(row.get("alpha_deg"), 3.0)
    ):
        return "RUN_CONFIRMED", "environments/SuperWing/analysis/corrected_n20_comparison/summary.json"

    if (
        env == "Mixed_integer_yiren"
        and reward == "ld_ratio"
        and approx_eq(row.get("mach"), 1.5)
        and approx_eq(row.get("alpha_deg"), 2.5)
    ):
        return "RUN_CONFIRMED", "environments/Mixed_integer_yiren/analysis/best_by_run.json"

    if (
        env == "NeuralFoil"
        and reward in {
            "ld_ratio_constrained_m02_re1e7",
            "ld_ratio_constrained_m02_re1e7_normalized",
            "ld_ratio_constrained_m02_re1e7_normalized_conf95",
        }
        and approx_eq(row.get("alpha_deg"), 5.0)
        and approx_eq(row.get("mach"), 0.2)
        and approx_eq(row.get("reynolds"), 1.0e7)
        and row.get("model_size") == "large"
    ):
        return "RELATED_RUN_FOUND", "environments/NeuralFoil/analysis/l_d_onepoint/summary.json"

    if env == "DrivAer_Star" and reward == "cd_only":
        return "TRAINING_ARTIFACT_ONLY", "environments/DrivAer_Star/ckpts"

    return "NOT_RUN_IN_REPO", ""


def finalize(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("target_cl", "") not in ("", None) and row.get("cl_target", "") in ("", None):
        row["cl_target"] = row["target_cl"]
    mach_points_all = row.get("mach_points_all")
    if isinstance(mach_points_all, list) and row.get("mach_point_count", "") in ("", None):
        row["mach_point_count"] = len(mach_points_all)
    cl_targets_all = row.get("cl_targets_all")
    if isinstance(cl_targets_all, list) and row.get("cl_target_count", "") in ("", None):
        row["cl_target_count"] = len(cl_targets_all)
    add_labels(row)
    row["conditions_compact"] = conditions_compact(row)
    row["run_status"], row["run_artifact"] = classify_run(row)
    return row


def default_point_role(generator: str) -> str:
    if generator == "grid":
        return "single_point"
    return "aggregate"


def make_default_rows(spec: dict[str, Any], reward_meta: dict[str, str]) -> list[dict[str, Any]]:
    default_case = dict(spec.get("default_case", {}))
    if not default_case:
        return []

    row = base_row(spec, reward_meta)
    row["case_kind"] = "default_case"
    row["point_role"] = default_point_role(spec["generator"])
    row["benchmark_mode"] = "default"
    row["permutation_index"] = 0
    row.update(dict(spec.get("fixed", {})))
    row.update(default_case)
    return [finalize(row)]


def make_grid_rows(spec: dict[str, Any], reward_meta: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    fixed = dict(spec.get("fixed", {}))
    for idx, combo in enumerate(dict_product(spec.get("axes", {})), start=1):
        row = base_row(spec, reward_meta)
        row["case_kind"] = "single_case"
        row["point_role"] = "single_point"
        row["benchmark_mode"] = "all_permutations"
        row["permutation_index"] = idx
        row.update(fixed)
        row.update(combo)
        rows.append(finalize(row))
    return rows


def make_multipoint_rows(spec: dict[str, Any], reward_meta: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    fixed = dict(spec.get("fixed", {}))
    targets = list(spec["targets"])
    cl_targets_all = [target["cl_target"] for target in targets]
    cl_weights_all = [target.get("cl_weight") for target in targets]

    for idx, combo in enumerate(dict_product(spec.get("axes", {})), start=1):
        common = fixed | combo

        aggregate = base_row(spec, reward_meta)
        aggregate["case_kind"] = "multipoint_aggregate"
        aggregate["point_role"] = "aggregate"
        aggregate["benchmark_mode"] = "all_permutations"
        aggregate["permutation_index"] = idx
        aggregate.update(common)
        aggregate["cl_targets_all"] = cl_targets_all
        aggregate["cl_weights_all"] = cl_weights_all
        rows.append(finalize(aggregate))

        for target_idx, target in enumerate(targets):
            point = base_row(spec, reward_meta)
            point["case_kind"] = "multipoint_target"
            point["point_role"] = f"target_{target_idx}"
            point["benchmark_mode"] = "all_permutations"
            point["permutation_index"] = idx
            point.update(common)
            point.update(target)
            point["cl_target_index"] = target_idx
            point["cl_targets_all"] = cl_targets_all
            point["cl_weights_all"] = cl_weights_all
            rows.append(finalize(point))

    return rows


def make_multipoint_target_pool_rows(spec: dict[str, Any], reward_meta: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    fixed = dict(spec.get("fixed", {}))
    target_pool = [float(value) for value in spec["target_pool"]]
    target_counts = [int(value) for value in spec["target_counts"]]
    target_sets = []
    for count in target_counts:
        target_sets.extend(list(itertools.combinations(target_pool, count)))

    permutation_index = 1
    for combo in dict_product(spec.get("axes", {})):
        common = fixed | combo
        for target_set in target_sets:
            cl_targets_all = [float(value) for value in target_set]

            aggregate = base_row(spec, reward_meta)
            aggregate["case_kind"] = "multipoint_aggregate"
            aggregate["point_role"] = "aggregate"
            aggregate["benchmark_mode"] = "all_permutations"
            aggregate["permutation_index"] = permutation_index
            aggregate.update(common)
            aggregate["cl_targets_all"] = cl_targets_all
            rows.append(finalize(aggregate))

            for target_idx, cl_target in enumerate(cl_targets_all):
                point = base_row(spec, reward_meta)
                point["case_kind"] = "multipoint_target"
                point["point_role"] = f"target_{target_idx}"
                point["benchmark_mode"] = "all_permutations"
                point["permutation_index"] = permutation_index
                point.update(common)
                point["cl_target"] = cl_target
                point["cl_target_index"] = target_idx
                point["cl_targets_all"] = cl_targets_all
                rows.append(finalize(point))

            permutation_index += 1

    return rows


def make_mach_point_pool_rows(spec: dict[str, Any], reward_meta: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    fixed = dict(spec.get("fixed", {}))
    mach_pool = [float(value) for value in spec["mach_pool"]]
    point_counts = [int(value) for value in spec["point_counts"]]
    mach_sets = []
    for count in point_counts:
        mach_sets.extend(list(itertools.combinations(mach_pool, count)))

    permutation_index = 1
    for combo in dict_product(spec.get("axes", {})):
        common = fixed | combo
        for mach_set in mach_sets:
            mach_points_all = [float(value) for value in mach_set]

            aggregate = base_row(spec, reward_meta)
            aggregate["case_kind"] = "multipoint_aggregate"
            aggregate["point_role"] = "aggregate"
            aggregate["benchmark_mode"] = "all_permutations"
            aggregate["permutation_index"] = permutation_index
            aggregate.update(common)
            aggregate["mach_points_all"] = mach_points_all
            rows.append(finalize(aggregate))

            for point_idx, mach in enumerate(mach_points_all):
                point = base_row(spec, reward_meta)
                point["case_kind"] = "multipoint_leg"
                point["point_role"] = f"mach_point_{point_idx}"
                point["benchmark_mode"] = "all_permutations"
                point["permutation_index"] = permutation_index
                point.update(common)
                point["mach"] = mach
                point["mach_points_all"] = mach_points_all
                rows.append(finalize(point))

            permutation_index += 1

    return rows


def _weights_for_targets(targets: list[float], weight_mode: str) -> list[float]:
    if weight_mode == "normalize_cl":
        denom = sum(targets)
        return [value / denom for value in targets]
    raise ValueError(f"Unknown weight_mode: {weight_mode}")


def make_weighted_cl_target_pool_rows(spec: dict[str, Any], reward_meta: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    fixed = dict(spec.get("fixed", {}))
    target_pool = [float(value) for value in spec["target_pool"]]
    target_counts = [int(value) for value in spec["target_counts"]]
    weight_mode = spec.get("weight_mode", "normalize_cl")
    target_sets = []
    for count in target_counts:
        target_sets.extend(list(itertools.combinations(target_pool, count)))

    permutation_index = 1
    for combo in dict_product(spec.get("axes", {})):
        common = fixed | combo
        for target_set in target_sets:
            cl_targets_all = [float(value) for value in target_set]
            cl_weights_all = _weights_for_targets(cl_targets_all, weight_mode)

            aggregate = base_row(spec, reward_meta)
            aggregate["case_kind"] = "multipoint_aggregate"
            aggregate["point_role"] = "aggregate"
            aggregate["benchmark_mode"] = "all_permutations"
            aggregate["permutation_index"] = permutation_index
            aggregate.update(common)
            aggregate["cl_targets_all"] = cl_targets_all
            aggregate["cl_weights_all"] = cl_weights_all
            rows.append(finalize(aggregate))

            for target_idx, cl_target in enumerate(cl_targets_all):
                point = base_row(spec, reward_meta)
                point["case_kind"] = "multipoint_target"
                point["point_role"] = f"target_{target_idx}"
                point["benchmark_mode"] = "all_permutations"
                point["permutation_index"] = permutation_index
                point.update(common)
                point["cl_target"] = cl_target
                point["cl_weight"] = cl_weights_all[target_idx]
                point["cl_target_index"] = target_idx
                point["cl_targets_all"] = cl_targets_all
                point["cl_weights_all"] = cl_weights_all
                rows.append(finalize(point))

            permutation_index += 1

    return rows


def make_two_point_rows(spec: dict[str, Any], reward_meta: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    fixed = dict(spec.get("fixed", {}))

    for idx, combo in enumerate(dict_product(spec["axes"]), start=1):
        common = fixed | combo

        aggregate = base_row(spec, reward_meta)
        aggregate["case_kind"] = "two_point_aggregate"
        aggregate["point_role"] = "aggregate"
        aggregate["benchmark_mode"] = "all_permutations"
        aggregate["permutation_index"] = idx
        aggregate.update(common)
        aggregate["cl_targets_all"] = [fixed["cl_target_sup"], fixed["cl_target_sub"]]
        rows.append(finalize(aggregate))

        sup = base_row(spec, reward_meta)
        sup["case_kind"] = "two_point_leg"
        sup["point_role"] = "supersonic_leg"
        sup["benchmark_mode"] = "all_permutations"
        sup["permutation_index"] = idx
        sup.update(common)
        sup["mach"] = fixed["mach_sup"]
        sup["reynolds"] = fixed["reynolds_sup"]
        sup["alpha_deg"] = combo["aoa_sup_deg"]
        sup["cl_target"] = fixed["cl_target_sup"]
        rows.append(finalize(sup))

        sub = base_row(spec, reward_meta)
        sub["case_kind"] = "two_point_leg"
        sub["point_role"] = "subsonic_leg"
        sub["benchmark_mode"] = "all_permutations"
        sub["permutation_index"] = idx
        sub.update(common)
        sub["mach"] = fixed["mach_sub"]
        sub["reynolds"] = fixed["reynolds_sub"]
        sub["alpha_deg"] = combo["aoa_sub_deg"]
        sub["cl_target"] = fixed["cl_target_sub"]
        rows.append(finalize(sub))

        pert = base_row(spec, reward_meta)
        pert["case_kind"] = "two_point_leg"
        pert["point_role"] = "subsonic_static_margin_perturbation"
        pert["benchmark_mode"] = "all_permutations"
        pert["permutation_index"] = idx
        pert.update(common)
        pert["mach"] = fixed["mach_sub"]
        pert["reynolds"] = fixed["reynolds_sub"]
        pert["alpha_deg"] = combo["aoa_sub_deg"] + fixed["delta_aoa_deg"]
        pert["cl_target"] = fixed["cl_target_sub"]
        rows.append(finalize(pert))

    return rows


def all_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_spec in BENCHMARK_SPECS:
        spec = resolve_spec(raw_spec)
        meta = reward_metadata(spec["reward_file"])
        rows.extend(make_default_rows(spec, meta))
        if spec["generator"] == "grid":
            rows.extend(make_grid_rows(spec, meta))
        elif spec["generator"] == "multipoint":
            rows.extend(make_multipoint_rows(spec, meta))
        elif spec["generator"] == "multipoint_target_pool":
            rows.extend(make_multipoint_target_pool_rows(spec, meta))
        elif spec["generator"] == "mach_point_pool":
            rows.extend(make_mach_point_pool_rows(spec, meta))
        elif spec["generator"] == "weighted_cl_target_pool":
            rows.extend(make_weighted_cl_target_pool_rows(spec, meta))
        elif spec["generator"] == "two_point":
            rows.extend(make_two_point_rows(spec, meta))
        else:
            raise ValueError(f"Unknown generator: {spec['generator']}")
    return rows


def write_csv(rows: list[dict[str, Any]]) -> None:
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: stringify(row.get(col, "")) for col in COLUMNS})


def row_class(run_status: str) -> str:
    return {
        "RUN_CONFIRMED": "run-confirmed",
        "RELATED_RUN_FOUND": "run-related",
        "TRAINING_ARTIFACT_ONLY": "run-related",
        "NOT_RUN_IN_REPO": "not-run",
    }[run_status]


def write_html(rows: list[dict[str, Any]]) -> None:
    headers = "".join(f"<th>{html.escape(col)}</th>" for col in COLUMNS)
    body_rows = []
    for row in rows:
        cells = "".join(
            f"<td>{html.escape(stringify(row.get(col, '')))}</td>" for col in COLUMNS
        )
        body_rows.append(f"<tr class='{row_class(row['run_status'])}'>{cells}</tr>")

    counts = Counter(row["run_status"] for row in rows)
    summary = "".join(
        f"<li><strong>{html.escape(status)}</strong>: {count}</li>"
        for status, count in sorted(counts.items())
    )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Environment Benchmark Catalog</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      color: #1f1f1f;
      background: #fbfbfb;
    }}
    h1 {{
      margin-bottom: 8px;
    }}
    .meta {{
      color: #555;
      margin-bottom: 12px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 12px;
    }}
    th, td {{
      border: 1px solid #d8d8d8;
      vertical-align: top;
      padding: 6px;
      word-break: break-word;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #14202b;
      color: white;
      z-index: 2;
    }}
    .run-confirmed {{
      background: #e5f7e8;
    }}
    .run-related {{
      background: #fff6d8;
    }}
    .not-run {{
      background: #f1f1f1;
      color: #7b7b7b;
    }}
  </style>
</head>
<body>
  <h1>Environment Benchmark Catalog</h1>
  <div class="meta">Paper-ready compact table generated from reward and environment code.</div>
  <ul>{summary}</ul>
  <table>
    <thead><tr>{headers}</tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
  </table>
</body>
</html>
"""
    OUT_HTML.write_text(html_text)


def main() -> None:
    rows = all_rows()
    write_csv(rows)
    write_html(rows)
    counts = Counter(row["run_status"] for row in rows)
    print(f"Wrote {relpath(OUT_CSV)}")
    print(f"Wrote {relpath(OUT_HTML)}")
    print(f"Total rows: {len(rows)}")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()
