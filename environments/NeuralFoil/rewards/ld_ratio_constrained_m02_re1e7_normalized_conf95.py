import json
import os
import sys

import aerosandbox as asb
import numpy as np

ENV_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEURALFOIL_SRC = os.path.join(ENV_DIR, "neuralfoil_src")
if os.path.isdir(NEURALFOIL_SRC) and NEURALFOIL_SRC not in sys.path:
    sys.path.insert(0, NEURALFOIL_SRC)

from environments.base_reward import BaseReward
from . import _constrained_ld_prompt_blocks as _pb
from .multipoint_hpa import _wiggliness

FAIL_REWARD = -10.0
_NACA0012_WIGGLINESS = _wiggliness(asb.KulfanAirfoil("naca0012"))


class LDRatioConstrainedM02Re1e7NormalizedConf95Reward(BaseReward):
    """One-point L/D reward with normalized penalty and tightened conf >= 0.95.

    Identical to ld_ratio_constrained_m02_re1e7_normalized except the
    analysis_confidence threshold is raised from 0.90 to 0.95.  This reduces
    surrogate exploitation at the confidence boundary, keeps designs in
    NeuralFoil's well-trained region, and should reduce the ~10-15%
    NeuralFoil→XFOIL L/D discrepancy observed at conf=0.90.

    Also tightens CM >= -0.130 (vs -0.133) to compensate for the systematic
    ~0.003 NeuralFoil→XFOIL CM bias found during XFOIL validation.
    """

    def __init__(
        self,
        alpha=5.0,
        re=1e7,
        mach=0.2,
        model_size="large",
        lambda_penalty=500.0,
        **kwargs,
    ):
        self.alpha = float(alpha)
        self.re = float(re)
        self.mach = float(mach)
        self.model_size = model_size
        self.lambda_penalty = float(lambda_penalty)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--alpha", type=float, default=5.0)
        parser.add_argument("--re", type=float, default=1e7)
        parser.add_argument("--mach", type=float, default=0.2)
        parser.add_argument("--model_size", type=str, default="large")
        parser.add_argument("--lambda_penalty", type=float, default=500.0)

    def get_prompt_blocks(self) -> dict:
        def _format_context_with_scratchpad(context, scratchpad=""):
            return _pb.format_context(context)

        return {
            "format_context": _format_context_with_scratchpad,
            "format_response_instructions": _pb.format_response_instructions,
            "CONTEXT_FORMAT": _pb.CONTEXT_FORMAT,
            "DESIGN_ENTRY": _pb.DESIGN_ENTRY,
            "RESPONSE_FORMAT": _pb.RESPONSE_FORMAT,
        }

    def evaluate(self, run_sim, design_path: str, case_dir: str) -> tuple:
        try:
            return self._evaluate(design_path, case_dir)
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"[ld_ratio_constrained_m02_re1e7_normalized_conf95] FAILED: {e}")
            return float(FAIL_REWARD), {
                "metrics": {"reward": FAIL_REWARD},
                "images": [],
                "feedback": "Simulation failed.",
            }

    def _evaluate(self, design_path: str, case_dir: str) -> tuple:
        os.makedirs(case_dir, exist_ok=True)
        save_dir = os.path.join(case_dir, "save")
        os.makedirs(save_dir, exist_ok=True)

        with open(design_path) as f:
            params = json.load(f)

        kulfan = {
            "upper_weights": np.array(params["upper_weights"], dtype=float),
            "lower_weights": np.array(params["lower_weights"], dtype=float),
            "leading_edge_weight": float(params["leading_edge_weight"]),
            "TE_thickness": float(params.get("TE_thickness", 0.0)),
        }

        airfoil = asb.KulfanAirfoil(
            upper_weights=kulfan["upper_weights"],
            lower_weights=kulfan["lower_weights"],
            leading_edge_weight=kulfan["leading_edge_weight"],
            TE_thickness=kulfan["TE_thickness"],
        )

        # --- geometric constraints (fractional violations, clamped to >= 0) ---
        violations = {}

        thickness_profile = np.asarray(airfoil.local_thickness())
        min_thickness = float(np.min(thickness_profile))
        v = max(0.0, -min_thickness) / 0.01
        if v > 0.0:
            violations["local_thickness_all"] = v

        t033 = float(airfoil.local_thickness(0.33))
        v = max(0.0, 0.128 - t033) / 0.128
        if v > 0.0:
            violations["t033"] = v

        t090 = float(airfoil.local_thickness(0.90))
        v = max(0.0, 0.014 - t090) / 0.014
        if v > 0.0:
            violations["t090"] = v

        te_angle = float(airfoil.TE_angle())
        v = max(0.0, 6.03 - te_angle) / 6.03
        if v > 0.0:
            violations["te_angle"] = v

        uw0 = float(kulfan["upper_weights"][0])
        v = max(0.0, 0.05 - uw0) / 0.05
        if v > 0.0:
            violations["upper_weights_0"] = v

        lw0 = float(kulfan["lower_weights"][0])
        v = max(0.0, lw0 + 0.05) / 0.05
        if v > 0.0:
            violations["lower_weights_0"] = v

        w = _wiggliness(airfoil)
        w_limit = 2.0 * _NACA0012_WIGGLINESS
        v = max(0.0, w - w_limit) / max(w_limit, 1e-12)
        if v > 0.0:
            violations["wiggliness"] = v

        # --- aerodynamic evaluation ---
        aero = airfoil.get_aero_from_neuralfoil(
            alpha=self.alpha,
            Re=self.re,
            mach=self.mach,
            model_size=self.model_size,
        )
        CL = float(np.squeeze(aero["CL"]))
        CD = float(np.squeeze(aero["CD"]))
        CM = float(np.squeeze(aero["CM"]))
        conf = float(np.squeeze(aero["analysis_confidence"]))

        # Tightened CM bound: -0.130 instead of -0.133 (compensates for
        # ~0.003 NeuralFoil→XFOIL CM bias found during XFOIL validation)
        v = max(0.0, -0.130 - CM) / 0.130
        if v > 0.0:
            violations["CM"] = v

        # Tightened confidence threshold: 0.95 instead of 0.90
        v = max(0.0, 0.95 - conf) / 0.95
        if v > 0.0:
            violations["analysis_confidence"] = v

        if CD <= 1e-9:
            return float(FAIL_REWARD), {
                "metrics": {"reward": FAIL_REWARD},
                "images": [],
                "feedback": "Invalid CD <= 0.",
            }

        ld = CL / CD
        fitness_objective = float(ld)

        total_violation = sum(violations.values())
        fitness_penalty = -self.lambda_penalty * float(total_violation)
        reward = float(fitness_objective + fitness_penalty)

        feasible = (
            (min_thickness > 0.0)
            and (t033 >= 0.128)
            and (t090 >= 0.014)
            and (te_angle >= 6.03)
            and (uw0 > 0.05)
            and (lw0 < -0.05)
            and (w <= w_limit)
            and (CM >= -0.130)
            and (conf > 0.95)
        )

        feedback_bits = []
        if not feasible:
            for k in sorted(violations.keys()):
                feedback_bits.append(f"{k}:{violations[k]:.4f}")

        metrics = {
            "CL": CL,
            "CD": CD,
            "CM": CM,
            "L_D": ld,
            "analysis_confidence": conf,
            "fitness_objective": fitness_objective,
            "fitness_penalty": fitness_penalty,
            "fitness_total": reward,
            "total_violation": float(total_violation),
            "constraint_violations": violations,
            "feasible": bool(feasible),
            "reward": reward,
        }

        serializable_design = {
            "upper_weights": [float(v) for v in kulfan["upper_weights"]],
            "lower_weights": [float(v) for v in kulfan["lower_weights"]],
            "leading_edge_weight": float(kulfan["leading_edge_weight"]),
            "TE_thickness": float(kulfan["TE_thickness"]),
        }

        with open(os.path.join(save_dir, "results.json"), "w") as f:
            json.dump(
                {
                    "design": serializable_design,
                    "alpha": self.alpha,
                    "re": self.re,
                    "mach": self.mach,
                    "model_size": self.model_size,
                    "lambda_penalty": self.lambda_penalty,
                    **metrics,
                },
                f,
                indent=2,
            )

        return reward, {
            "metrics": metrics,
            "images": [],
            "feedback": "; ".join(feedback_bits),
        }
