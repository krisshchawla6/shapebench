"""
Maximize CL with soft CD penalty and CM constraint.

  reward = CL - w_cd * CD - lambda_cm * max(0, -CM - cm_limit)^2
           - lambda_penalty * geom_violations
"""
import json, os, sys
import numpy as np

ENV_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEURALFOIL_SRC = os.path.join(ENV_DIR, "neuralfoil_src")
if os.path.isdir(NEURALFOIL_SRC) and NEURALFOIL_SRC not in sys.path:
    sys.path.insert(0, NEURALFOIL_SRC)

import aerosandbox as asb
import neuralfoil as nf
from environments.base_reward import BaseReward
from . import _constrained_ld_prompt_blocks as _pb
from ._geom_constraints import compute_geom_violations

FAIL_REWARD = -10.0


class MaxCLConstrainedReward(BaseReward):
    def __init__(self, alpha=10.0, re=1e6, mach=0.2, model_size="large", n_crit=9.0,
                 w_cd=0.5, cm_limit=-0.133, lambda_cm=50.0, lambda_penalty=500.0, **kwargs):
        self.alpha = float(alpha)
        self.re = float(re)
        self.mach = float(mach)
        self.model_size = model_size
        self.n_crit = float(n_crit)
        self.w_cd = float(w_cd)
        self.cm_limit = float(cm_limit)
        self.lambda_cm = float(lambda_cm)
        self.lambda_penalty = float(lambda_penalty)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--alpha", type=float, default=10.0)
        parser.add_argument("--re", type=float, default=1e6)
        parser.add_argument("--w_cd", type=float, default=0.5)
        parser.add_argument("--lambda_penalty", type=float, default=500.0)

    def get_prompt_blocks(self):
        def _fc(context, scratchpad=""):
            return _pb.format_context(context)
        return {"format_context": _fc,
                "format_response_instructions": _pb.format_response_instructions,
                "CONTEXT_FORMAT": _pb.CONTEXT_FORMAT,
                "DESIGN_ENTRY": _pb.DESIGN_ENTRY,
                "RESPONSE_FORMAT": _pb.RESPONSE_FORMAT}

    def evaluate(self, run_sim, design_path, case_dir):
        try:
            return self._evaluate(design_path, case_dir)
        except Exception as e:
            import traceback; traceback.print_exc()
            return float(FAIL_REWARD), {"metrics": {"reward": FAIL_REWARD}, "images": [], "feedback": str(e)}

    def _evaluate(self, design_path, case_dir):
        os.makedirs(case_dir, exist_ok=True)
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
        aero = nf.get_aero_from_kulfan_parameters(
            kulfan_parameters=kulfan, alpha=self.alpha, Re=self.re,
            n_crit=self.n_crit, model_size=self.model_size)
        CL = float(np.squeeze(aero["CL"]))
        CD = float(np.squeeze(aero["CD"]))
        CM = float(np.squeeze(aero["CM"]))
        conf = float(np.squeeze(aero["analysis_confidence"]))

        violations = compute_geom_violations(airfoil, kulfan, conf=conf)
        cm_pen = self.lambda_cm * max(0.0, -self.cm_limit - CM)**2
        reward = CL - self.w_cd * CD - cm_pen - self.lambda_penalty * sum(violations.values())
        print(f"[max_cl_constrained] CL={CL:.4f} CD={CD:.6f} reward={reward:.4f}")
        os.makedirs(os.path.join(case_dir, "save"), exist_ok=True)
        import json as _json
        with open(os.path.join(case_dir, "save", "results.json"), "w") as f:
            _json.dump({"CL": CL, "CD": CD, "CM": CM, "conf": conf, "cm_penalty": cm_pen,
                        "violations": violations, "reward": reward}, f, indent=2)
        return float(reward), {"metrics": {"CL": CL, "reward": reward}, "images": [], "feedback": ""}
