"""
Multipoint HPA optimization reward — exact problem from:

  AeroSandbox tutorial/06 - Aerodynamics/
  02 - AeroSandbox 2D Aerodynamics Tools/
  02 - NeuralFoil Optimization.ipynb
  https://github.com/peterdsharpe/AeroSandbox/blob/master/tutorial/06%20-%20Aerodynamics/
  02%20-%20AeroSandbox%202D%20Aerodynamics%20Tools/02%20-%20NeuralFoil%20Optimization.ipynb

Human Powered Aircraft (HPA) multipoint optimization:

  Minimize  mean(CD_i × w_i)  over 6 lift targets
  Subject to:
    CL_i == CL_targets[i]           (alpha found by bisection)
    np.diff(alpha) > 0              (monotonically increasing alpha vs CL)
    CM_i >= -0.133
    analysis_confidence_i > 0.90
    local_thickness(0.33) >= 0.128
    local_thickness(0.90) >= 0.014
    TE_angle >= 6.03 deg            (modified from Drela 6.25 to match DAE-11)
    upper_weights[0] > 0.05
    lower_weights[0] < -0.05
    local_thickness > 0 everywhere  (no self-intersection)
    wiggliness < 2 × wiggliness(NACA0012)

Fitness decomposition for GA:

  F(x) = F_objective(x) + P(x)

where:
  F_objective = -mean(CD_i * weight_i)  (exact notebook objective, sign-flipped for maximization)
  P(x)        = -sum(lambda_i * p_i(x)) (weighted normalized penalties)

If simulation/runtime fails, return -10 directly.
"""

import json
import os
import sys
import numpy as np

ENV_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEURALFOIL_SRC = os.path.join(ENV_DIR, 'neuralfoil_src')
if os.path.isdir(NEURALFOIL_SRC) and NEURALFOIL_SRC not in sys.path:
    sys.path.insert(0, NEURALFOIL_SRC)

from environments.base_reward import BaseReward
from . import _multipoint_hpa_prompt_blocks as _pb
import neuralfoil as nf
import aerosandbox as asb

FAIL_REWARD = -10.0

# ── Fixed operating conditions (from notebook) ───────────────────────────────
CL_TARGETS = np.array([0.8, 1.0, 1.2, 1.4, 1.5, 1.6])
CL_WEIGHTS = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
MACH = 0.03


def _re_schedule(cl_target):
    """Re = 500k × (CL/1.25)^-0.5  (Drela HPA Reynolds schedule)."""
    return 500e3 * (cl_target / 1.25) ** -0.5


def _wiggliness(af):
    """Discrete Wahba spline wiggliness: Σ ||Δ²weights||²."""
    return sum(np.sum(np.diff(np.diff(arr)) ** 2)
               for arr in [af.lower_weights, af.upper_weights])


def _accumulate_violation(violations, key, value):
    """Add nonnegative normalized violation to a bucket."""
    v = float(max(0.0, value))
    if v > 0.0:
        violations[key] = violations.get(key, 0.0) + v


# Pre-compute NACA0012 reference wiggliness once at module load
_NACA0012_WIGGLINESS = _wiggliness(asb.KulfanAirfoil("naca0012"))


class MultipointHPAReward(BaseReward):
    """HPA multipoint optimization — exact reward from AeroSandbox NeuralFoil notebook.

    Evaluates an airfoil at 6 CL targets (0.8–1.6) using the Drela HPA
    Reynolds schedule (Re = 500k × (CL/1.25)^-0.5, Mach = 0.03).
    For each target the angle of attack is found by bisection so that
    NeuralFoil returns exactly CL = CL_target.

    Fitness:
      F_total = F_objective + F_penalty
      F_objective = -weighted_mean(CD * weights) over solved CL points
                    (exact notebook objective when all 6 points are solved)
      F_penalty   = -sum(lambda_i * p_i) over normalized constraint violations
    """

    def __init__(self, model_size="large", n_crit=9.0,
                 lambda_thick_all=1.0, lambda_t033=1.0, lambda_t090=1.0,
                 lambda_te_angle=1.0, lambda_uw0=1.0, lambda_lw0=1.0,
                 lambda_wiggliness=1.0, lambda_cm=1.0, lambda_conf=1.0,
                 lambda_alpha_mono=1.0, lambda_unreach=1.0, **kwargs):
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
        parser.add_argument('--model_size', type=str, default='large',
                            help='NeuralFoil model size (xxsmall … xxxlarge)')
        parser.add_argument('--n_crit', type=float, default=9.0,
                            help='Critical amplification factor (e^9 method)')
        parser.add_argument('--lambda_thick_all', type=float, default=1.0,
                            help='Penalty weight for local_thickness() > 0 violation')
        parser.add_argument('--lambda_t033', type=float, default=1.0,
                            help='Penalty weight for thickness at x=0.33 violation')
        parser.add_argument('--lambda_t090', type=float, default=1.0,
                            help='Penalty weight for thickness at x=0.90 violation')
        parser.add_argument('--lambda_te_angle', type=float, default=1.0,
                            help='Penalty weight for TE angle violation')
        parser.add_argument('--lambda_uw0', type=float, default=1.0,
                            help='Penalty weight for upper_weights[0] violation')
        parser.add_argument('--lambda_lw0', type=float, default=1.0,
                            help='Penalty weight for lower_weights[0] violation')
        parser.add_argument('--lambda_wiggliness', type=float, default=1.0,
                            help='Penalty weight for wiggliness violation')
        parser.add_argument('--lambda_cm', type=float, default=1.0,
                            help='Penalty weight for CM >= -0.133 violation')
        parser.add_argument('--lambda_conf', type=float, default=1.0,
                            help='Penalty weight for analysis_confidence > 0.90 violation')
        parser.add_argument('--lambda_alpha_mono', type=float, default=1.0,
                            help='Penalty weight for alpha monotonicity violation')
        parser.add_argument('--lambda_unreach', type=float, default=1.0,
                            help='Penalty weight for unreachable CL targets')

    def evaluate(self, run_sim, design_path: str, case_dir: str) -> tuple:
        try:
            return self._evaluate(run_sim, design_path, case_dir)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[multipoint_hpa] FAILED: {e}")
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
            "TE_thickness": 0.0,  # fixed to 0 per problem definition
        }

        airfoil = asb.KulfanAirfoil(
            upper_weights=kulfan["upper_weights"],
            lower_weights=kulfan["lower_weights"],
            leading_edge_weight=kulfan["leading_edge_weight"],
            TE_thickness=0.0,
        )

        violations = {}
        feedback_bits = []

        # ── Hard geometric constraints from notebook ─────────────────────────
        # optimized_airfoil.local_thickness() > 0
        thickness_profile = np.asarray(airfoil.local_thickness())
        min_thickness = float(np.min(thickness_profile))
        _accumulate_violation(violations, "local_thickness_all", max(0.0, -min_thickness) / 0.01)
        if min_thickness <= 0:
            feedback_bits.append(f"self-intersection: min_thickness={min_thickness:.5f}")

        # optimized_airfoil.local_thickness(x_over_c=0.33) >= 0.128
        t033 = float(airfoil.local_thickness(0.33))
        _accumulate_violation(violations, "t033", (0.128 - t033) / 0.128)
        if t033 < 0.128:
            feedback_bits.append(f"t(0.33)={t033:.5f} < 0.128")

        # optimized_airfoil.local_thickness(x_over_c=0.90) >= 0.014
        t090 = float(airfoil.local_thickness(0.90))
        _accumulate_violation(violations, "t090", (0.014 - t090) / 0.014)
        if t090 < 0.014:
            feedback_bits.append(f"t(0.90)={t090:.5f} < 0.014")

        # optimized_airfoil.TE_angle() >= 6.03
        te_angle = float(airfoil.TE_angle())
        _accumulate_violation(violations, "te_angle", (6.03 - te_angle) / 6.03)
        if te_angle < 6.03:
            feedback_bits.append(f"TE_angle={te_angle:.4f} < 6.03")

        # optimized_airfoil.upper_weights[0] > 0.05
        uw0 = float(kulfan["upper_weights"][0])
        _accumulate_violation(violations, "upper_weights_0", (0.05 - uw0) / 0.05)
        if uw0 <= 0.05:
            feedback_bits.append(f"upper_weights[0]={uw0:.5f} <= 0.05")

        # optimized_airfoil.lower_weights[0] < -0.05
        lw0 = float(kulfan["lower_weights"][0])
        _accumulate_violation(violations, "lower_weights_0", (lw0 + 0.05) / 0.05)
        if lw0 >= -0.05:
            feedback_bits.append(f"lower_weights[0]={lw0:.5f} >= -0.05")

        # get_wiggliness(optimized_airfoil) < 2 * get_wiggliness(initial_guess_airfoil)
        w = _wiggliness(airfoil)
        w_limit = 2.0 * _NACA0012_WIGGLINESS
        _accumulate_violation(violations, "wiggliness", (w - w_limit) / max(w_limit, 1e-12))
        if w > w_limit:
            feedback_bits.append(f"wiggliness={w:.6f} > {w_limit:.6f}")

        # ── Aerodynamic evaluation at each operating point ────────────────────
        alphas, CDs, CMs, confs = [], [], [], []
        solved_weights = []
        solved_cls = []

        for idx, cl_t in enumerate(CL_TARGETS):
            re_i = _re_schedule(cl_t)
            alpha_i = self._find_alpha(kulfan, cl_t, re_i)
            if alpha_i is None:
                _accumulate_violation(violations, "cl_target_unreachable", 1.0)
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
            solved_weights.append(float(CL_WEIGHTS[idx]))
            solved_cls.append(float(cl_t))
            CDs.append(float(np.squeeze(aero_i["CD"])))
            CMs.append(float(np.squeeze(aero_i["CM"])))
            confs.append(float(np.squeeze(aero_i["analysis_confidence"])))

        alphas = np.array(alphas)
        solved_weights = np.array(solved_weights)
        solved_cls = np.array(solved_cls)
        CDs = np.array(CDs)
        CMs = np.array(CMs)
        confs = np.array(confs)

        # ── Hard aerodynamic constraints from notebook ───────────────────────
        if len(alphas) >= 2:
            # np.diff(alpha) > 0
            d_alpha = np.diff(alphas)
            mono_violation = np.maximum(0.0, -d_alpha)
            _accumulate_violation(violations, "alpha_monotonic", float(np.sum(mono_violation)))
            if np.any(d_alpha <= 0):
                feedback_bits.append("alpha monotonicity violated")

        if len(CMs) > 0:
            # aero["CM"] >= -0.133
            cm_violation = np.maximum(0.0, -0.133 - CMs)
            _accumulate_violation(violations, "CM", float(np.sum(cm_violation / 0.133)))
            if np.any(CMs < -0.133):
                worst_i = int(np.argmin(CMs))
                feedback_bits.append(
                    f"CM violated at CL={solved_cls[worst_i]:.2f}: CM={CMs[worst_i]:.5f} < -0.133"
                )

        if len(confs) > 0:
            # aero["analysis_confidence"] > 0.90
            conf_violation = np.maximum(0.0, 0.90 - confs)
            _accumulate_violation(violations, "analysis_confidence", float(np.sum(conf_violation / 0.90)))
            if np.any(confs <= 0.90):
                worst_i = int(np.argmin(confs))
                feedback_bits.append(
                    f"confidence violated at CL={solved_cls[worst_i]:.2f}: conf={confs[worst_i]:.5f} <= 0.90"
                )

        # ── Objective + penalty decomposition ─────────────────────────────────
        # Exact notebook objective when all points solved; otherwise use solved-point
        # weighted mean to keep objective signal for GA while p_unreach penalizes misses.
        weighted_cd = float(np.mean(CDs * CL_WEIGHTS)) if len(CDs) == len(CL_WEIGHTS) else None
        weighted_cd_solved = None
        if len(CDs) > 0 and len(solved_weights) == len(CDs):
            denom = float(np.sum(solved_weights))
            if denom > 0:
                weighted_cd_solved = float(np.sum(CDs * solved_weights) / denom)
        fitness_objective = -weighted_cd_solved if weighted_cd_solved is not None else 0.0

        weighted_penalties = {}
        for key, p_i in violations.items():
            lam = self.penalty_weights.get(key, 1.0)
            weighted_penalties[key] = float(lam * p_i)

        total_penalty_magnitude = float(sum(weighted_penalties.values()))
        fitness_penalty = -total_penalty_magnitude
        fitness_total = float(fitness_objective + fitness_penalty)
        feasible = (weighted_cd is not None) and (len(violations) == 0)

        print(
            f"[multipoint_hpa] f_obj={fitness_objective:.6f} "
            f"f_pen={fitness_penalty:.6f} f_total={fitness_total:.6f}"
        )
        if violations:
            print(f"  violations: {violations}")
            print(f"  weighted_penalties: {weighted_penalties}")

        # Always try to generate representative images (shape + Cp),
        # even for infeasible designs, to aid qualitative debugging.
        images = []
        try:
            if len(alphas) > 0:
                # If at least one point solved, use a solved condition near the center.
                rep_idx = len(alphas) // 2
                rep_alpha = float(alphas[rep_idx])
                rep_re = float(_re_schedule(solved_cls[rep_idx]))
            else:
                # Fallback condition if no CL target was solved.
                rep_alpha = 5.0
                rep_re = float(_re_schedule(CL_TARGETS[0]))

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
            print(f"[multipoint_hpa] Image generation skipped: {e}")

        _write_results(
            save_dir=save_dir,
            kulfan_params=params,
            alphas=alphas.tolist(),
            CDs=CDs.tolist(),
            CMs=CMs.tolist(),
            confs=confs.tolist(),
            weighted_cd=weighted_cd,
            weighted_cd_solved=weighted_cd_solved,
            fitness_objective=fitness_objective,
            fitness_penalty=fitness_penalty,
            fitness_total=fitness_total,
            feasible=feasible,
            total_violation=float(sum(violations.values())),
            violations=violations,
            weighted_penalties=weighted_penalties,
            reward=fitness_total,
        )

        feedback = "" if feasible else "; ".join(feedback_bits[:4])
        if (not feasible) and (not feedback):
            feedback = "Constraint violation."

        return float(fitness_total), {
            "metrics": {
                "weighted_CD_mean": weighted_cd,
                "weighted_CD_mean_solved": weighted_cd_solved,
                "CDs": CDs.tolist(),
                "CMs": CMs.tolist(),
                "analysis_confidences": confs.tolist(),
                "alphas": alphas.tolist(),
                "fitness_objective": fitness_objective,
                "fitness_penalty": fitness_penalty,
                "fitness_total": fitness_total,
                "feasible": feasible,
                "total_violation": float(sum(violations.values())),
                "constraint_violations": violations,
                "weighted_penalties": weighted_penalties,
                "reward": fitness_total,
            },
            "images": images,
            "feedback": "" if feasible else feedback,
        }

    def _find_alpha(self, kulfan, cl_target, re,
                    lo=-5.0, hi=18.0, tol=1e-3, maxiter=60):
        """Brent's method: find α such that NeuralFoil gives CL(α) = cl_target."""
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


def _write_results(save_dir, kulfan_params, alphas, CDs, CMs, confs, weighted_cd, weighted_cd_solved,
                   fitness_objective, fitness_penalty, fitness_total,
                   feasible, total_violation, violations, weighted_penalties, reward):
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
