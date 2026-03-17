"""
Exact-notebook multipoint reward for NeuralFoil.

This keeps the same constraint/penalty framework as multipoint_hpa, but uses the
exact notebook objective scalar:

  objective_notebook_raw = mean(CD_i * w_i)  over all 6 CL targets

The framework remains a maximize problem by sign inversion:

  fitness_objective = -objective_notebook_raw
  fitness_total     = fitness_objective + fitness_penalty

Strict mode: all CL targets must be solved. If any target is unreachable, this
reward returns FAIL_REWARD directly.
"""

import json
import os
import sys
import numpy as np

ENV_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEURALFOIL_SRC = os.path.join(ENV_DIR, "neuralfoil_src")
if os.path.isdir(NEURALFOIL_SRC) and NEURALFOIL_SRC not in sys.path:
    sys.path.insert(0, NEURALFOIL_SRC)

from environments.base_reward import BaseReward
from . import _multipoint_hpa_prompt_blocks as _pb
from . import multipoint_hpa as _hpa
import neuralfoil as nf
import aerosandbox as asb

FAIL_REWARD = -10.0

CL_TARGETS = _hpa.CL_TARGETS
CL_WEIGHTS = _hpa.CL_WEIGHTS
MACH = _hpa.MACH


class RewardExactNotebook(BaseReward):
    def __init__(
        self,
        model_size="large",
        n_crit=9.0,
        lambda_thick_all=1.0,
        lambda_t033=1.0,
        lambda_t090=1.0,
        lambda_te_angle=1.0,
        lambda_uw0=1.0,
        lambda_lw0=1.0,
        lambda_wiggliness=1.0,
        lambda_cm=1.0,
        lambda_conf=1.0,
        lambda_alpha_mono=1.0,
        lambda_unreach=1.0,
        **kwargs,
    ):
        self.model_size = model_size
        self.n_crit = n_crit
        self.penalty_weights = {
            "local_thickness_all": float(lambda_thick_all),
            "t033": float(lambda_t033),
            "t090": float(lambda_t090),
            "te_angle": float(lambda_te_angle),
            "upper_weights_0": float(lambda_uw0),
            "lower_weights_0": float(lambda_lw0),
            "wiggliness": float(lambda_wiggliness),
            "CM": float(lambda_cm),
            "analysis_confidence": float(lambda_conf),
            "alpha_monotonic": float(lambda_alpha_mono),
            "cl_target_unreachable": float(lambda_unreach),
        }

    def get_prompt_blocks(self) -> dict:
        return {
            "format_context": _pb.format_context,
            "format_response_instructions": _pb.format_response_instructions,
            "CONTEXT_FORMAT": _pb.CONTEXT_FORMAT,
            "DESIGN_ENTRY": _pb.DESIGN_ENTRY,
            "RESPONSE_FORMAT": _pb.RESPONSE_FORMAT,
        }

    @staticmethod
    def add_args(parser):
        _hpa.MultipointHPAReward.add_args(parser)

    def evaluate(self, run_sim, design_path: str, case_dir: str) -> tuple:
        try:
            return self._evaluate(run_sim, design_path, case_dir)
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"[reward_exact_notebook] FAILED: {e}")
            return float(FAIL_REWARD), {
                "metrics": {"reward": FAIL_REWARD},
                "images": [],
                "feedback": "Simulation failed.",
            }

    def _evaluate(self, run_sim, design_path: str, case_dir: str) -> tuple:
        os.makedirs(case_dir, exist_ok=True)
        save_dir = os.path.join(case_dir, "save")
        os.makedirs(save_dir, exist_ok=True)

        with open(design_path) as f:
            params = json.load(f)

        kulfan = {
            "upper_weights": np.array(params["upper_weights"], dtype=float),
            "lower_weights": np.array(params["lower_weights"], dtype=float),
            "leading_edge_weight": float(params["leading_edge_weight"]),
            "TE_thickness": 0.0,
        }
        airfoil = asb.KulfanAirfoil(
            upper_weights=kulfan["upper_weights"],
            lower_weights=kulfan["lower_weights"],
            leading_edge_weight=kulfan["leading_edge_weight"],
            TE_thickness=0.0,
        )

        violations = {}
        feedback_bits = []

        thickness_profile = np.asarray(airfoil.local_thickness())
        min_thickness = float(np.min(thickness_profile))
        _hpa._accumulate_violation(
            violations, "local_thickness_all", max(0.0, -min_thickness) / 0.01
        )
        if min_thickness <= 0:
            feedback_bits.append(f"self-intersection: min_thickness={min_thickness:.5f}")

        t033 = float(airfoil.local_thickness(0.33))
        _hpa._accumulate_violation(violations, "t033", (0.128 - t033) / 0.128)
        if t033 < 0.128:
            feedback_bits.append(f"t(0.33)={t033:.5f} < 0.128")

        t090 = float(airfoil.local_thickness(0.90))
        _hpa._accumulate_violation(violations, "t090", (0.014 - t090) / 0.014)
        if t090 < 0.014:
            feedback_bits.append(f"t(0.90)={t090:.5f} < 0.014")

        te_angle = float(airfoil.TE_angle())
        _hpa._accumulate_violation(violations, "te_angle", (6.03 - te_angle) / 6.03)
        if te_angle < 6.03:
            feedback_bits.append(f"TE_angle={te_angle:.4f} < 6.03")

        uw0 = float(kulfan["upper_weights"][0])
        _hpa._accumulate_violation(violations, "upper_weights_0", (0.05 - uw0) / 0.05)
        if uw0 <= 0.05:
            feedback_bits.append(f"upper_weights[0]={uw0:.5f} <= 0.05")

        lw0 = float(kulfan["lower_weights"][0])
        _hpa._accumulate_violation(violations, "lower_weights_0", (lw0 + 0.05) / 0.05)
        if lw0 >= -0.05:
            feedback_bits.append(f"lower_weights[0]={lw0:.5f} >= -0.05")

        w = _hpa._wiggliness(airfoil)
        w_limit = 2.0 * _hpa._NACA0012_WIGGLINESS
        _hpa._accumulate_violation(violations, "wiggliness", (w - w_limit) / max(w_limit, 1e-12))
        if w > w_limit:
            feedback_bits.append(f"wiggliness={w:.6f} > {w_limit:.6f}")

        alphas, CDs, CMs, confs = [], [], [], []
        solved_cls = []

        for idx, cl_t in enumerate(CL_TARGETS):
            re_i = _hpa._re_schedule(cl_t)
            alpha_i = self._find_alpha(kulfan, cl_t, re_i)
            if alpha_i is None:
                _hpa._accumulate_violation(violations, "cl_target_unreachable", 1.0)
                feedback_bits.append(f"CL target unreachable at CL={cl_t:.2f}")
                continue

            aero_i = nf.get_aero_from_kulfan_parameters(
                kulfan_parameters=kulfan,
                alpha=float(alpha_i),
                Re=re_i,
                n_crit=self.n_crit,
                model_size=self.model_size,
            )
            alphas.append(float(alpha_i))
            solved_cls.append(float(cl_t))
            CDs.append(float(np.squeeze(aero_i["CD"])))
            CMs.append(float(np.squeeze(aero_i["CM"])))
            confs.append(float(np.squeeze(aero_i["analysis_confidence"])))

        alphas = np.array(alphas)
        solved_cls = np.array(solved_cls)
        CDs = np.array(CDs)
        CMs = np.array(CMs)
        confs = np.array(confs)

        if len(alphas) >= 2:
            d_alpha = np.diff(alphas)
            mono_violation = np.maximum(0.0, -d_alpha)
            _hpa._accumulate_violation(violations, "alpha_monotonic", float(np.sum(mono_violation)))
            if np.any(d_alpha <= 0):
                feedback_bits.append("alpha monotonicity violated")

        if len(CMs) > 0:
            cm_violation = np.maximum(0.0, -0.133 - CMs)
            _hpa._accumulate_violation(violations, "CM", float(np.sum(cm_violation / 0.133)))
            if np.any(CMs < -0.133):
                worst_i = int(np.argmin(CMs))
                feedback_bits.append(
                    f"CM violated at CL={solved_cls[worst_i]:.2f}: CM={CMs[worst_i]:.5f} < -0.133"
                )

        if len(confs) > 0:
            conf_violation = np.maximum(0.0, 0.90 - confs)
            _hpa._accumulate_violation(
                violations, "analysis_confidence", float(np.sum(conf_violation / 0.90))
            )
            if np.any(confs <= 0.90):
                worst_i = int(np.argmin(confs))
                feedback_bits.append(
                    f"confidence violated at CL={solved_cls[worst_i]:.2f}: conf={confs[worst_i]:.5f} <= 0.90"
                )

        weighted_penalties = {}
        for key, p_i in violations.items():
            lam = self.penalty_weights.get(key, 1.0)
            weighted_penalties[key] = float(lam * p_i)

        total_penalty_magnitude = float(sum(weighted_penalties.values()))
        fitness_penalty = -total_penalty_magnitude

        n_targets = len(CL_TARGETS)
        all_targets_solved = len(CDs) == n_targets
        weighted_cd_solved = None
        objective_notebook_raw = None
        weighted_cd = None

        if all_targets_solved:
            # Exact notebook objective scalar: mean(CD_i * w_i)
            weighted_cd = float(np.mean(CDs * CL_WEIGHTS))
            objective_notebook_raw = weighted_cd
            # Keep maximize framing for the framework.
            fitness_objective = -objective_notebook_raw
            fitness_total = float(fitness_objective + fitness_penalty)
            reward_value = float(fitness_total)
            feasible = len(violations) == 0
        else:
            # Strict mode: unsolved target(s) immediately fail reward.
            fitness_objective = 0.0
            fitness_total = float(fitness_objective + fitness_penalty)
            reward_value = float(FAIL_REWARD)
            feasible = False
            feedback_bits.append("strict fail: not all CL targets solved")

        print(
            f"[reward_exact_notebook] f_obj={fitness_objective:.6f} "
            f"f_pen={fitness_penalty:.6f} f_total={fitness_total:.6f} reward={reward_value:.6f}"
        )
        if violations:
            print(f"  violations: {violations}")
            print(f"  weighted_penalties: {weighted_penalties}")

        images = []
        try:
            if len(alphas) > 0:
                rep_idx = len(alphas) // 2
                rep_alpha = float(alphas[rep_idx])
                rep_re = float(_hpa._re_schedule(solved_cls[rep_idx]))
            else:
                rep_alpha = 5.0
                rep_re = float(_hpa._re_schedule(CL_TARGETS[0]))

            r_img = run_sim(
                design_path,
                case_dir,
                alpha=rep_alpha,
                re=rep_re,
                model_size=self.model_size,
                n_crit=self.n_crit,
            )
            images = r_img.get("images", [])
            save_dir = r_img.get("save_dir", save_dir)
        except Exception as e:
            print(f"[reward_exact_notebook] Image generation skipped: {e}")

        _write_results(
            save_dir=save_dir,
            kulfan_params=params,
            alphas=alphas.tolist(),
            CDs=CDs.tolist(),
            CMs=CMs.tolist(),
            confs=confs.tolist(),
            weighted_cd=weighted_cd,
            weighted_cd_solved=weighted_cd_solved,
            objective_notebook_raw=objective_notebook_raw,
            fitness_objective=fitness_objective,
            fitness_penalty=fitness_penalty,
            fitness_total=fitness_total,
            feasible=feasible,
            total_violation=float(sum(violations.values())),
            violations=violations,
            weighted_penalties=weighted_penalties,
            reward=reward_value,
        )

        feedback = "" if feasible else "; ".join(feedback_bits[:4])
        if (not feasible) and (not feedback):
            feedback = "Constraint violation."

        return float(reward_value), {
            "metrics": {
                "weighted_CD_mean": weighted_cd,
                "weighted_CD_mean_solved": weighted_cd_solved,
                "objective_notebook_raw": objective_notebook_raw,
                "CDs": CDs.tolist(),
                "CMs": CMs.tolist(),
                "analysis_confidences": confs.tolist(),
                "alphas": alphas.tolist(),
                "fitness_objective": fitness_objective,
                "fitness_penalty": fitness_penalty,
                "fitness_total": fitness_total,
                "feasible": feasible,
                "strict_all_targets_solved": all_targets_solved,
                "total_violation": float(sum(violations.values())),
                "constraint_violations": violations,
                "weighted_penalties": weighted_penalties,
                "reward": reward_value,
            },
            "images": images,
            "feedback": "" if feasible else feedback,
        }

    def _find_alpha(self, kulfan, cl_target, re, lo=-5.0, hi=18.0, tol=1e-3, maxiter=60):
        from scipy.optimize import brentq

        def residual(alpha):
            aero = nf.get_aero_from_kulfan_parameters(
                kulfan_parameters=kulfan,
                alpha=float(alpha),
                Re=re,
                n_crit=self.n_crit,
                model_size=self.model_size,
            )
            return float(np.squeeze(aero["CL"])) - cl_target

        try:
            f_lo, f_hi = residual(lo), residual(hi)
            if f_lo * f_hi > 0:
                for lo2, hi2 in [(-10.0, 25.0), (-15.0, 30.0)]:
                    f_lo2, f_hi2 = residual(lo2), residual(hi2)
                    if f_lo2 * f_hi2 <= 0:
                        lo, hi = lo2, hi2
                        break
                else:
                    return None
            return float(brentq(residual, lo, hi, xtol=tol, maxiter=maxiter))
        except Exception:
            return None


def _write_results(
    save_dir,
    kulfan_params,
    alphas,
    CDs,
    CMs,
    confs,
    weighted_cd,
    weighted_cd_solved,
    objective_notebook_raw,
    fitness_objective,
    fitness_penalty,
    fitness_total,
    feasible,
    total_violation,
    violations,
    weighted_penalties,
    reward,
):
    d = {
        "design": kulfan_params,
        "CL_targets": CL_TARGETS.tolist(),
        "CL_weights": CL_WEIGHTS.tolist(),
        "mach": MACH,
        "alphas": alphas,
        "CDs": CDs,
        "CMs": CMs,
        "analysis_confidences": confs,
        "weighted_CD_mean": weighted_cd,
        "weighted_CD_mean_solved": weighted_cd_solved,
        "objective_notebook_raw": objective_notebook_raw,
        "fitness_objective": fitness_objective,
        "fitness_penalty": fitness_penalty,
        "fitness_total": fitness_total,
        "feasible": feasible,
        "total_violation": total_violation,
        "constraint_violations": violations,
        "weighted_penalties": weighted_penalties,
        "reward": reward,
    }
    if save_dir is not None:
        with open(os.path.join(save_dir, "results.json"), "w") as f:
            json.dump(d, f, indent=2)
