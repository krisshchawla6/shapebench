"""
Average CD over CL targets (alpha found by bisection) at fixed Re.

  reward = -mean(CD_i) - lambda_penalty * geom_violations
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
from . import _multipoint_hpa_prompt_blocks as _pb
from ._geom_constraints import compute_geom_violations, find_alpha

FAIL_REWARD = -10.0
_DEFAULT_CL_TARGETS = [0.4, 0.6, 0.8, 1.0]


class MultipointClAvgCdReward(BaseReward):
    def __init__(self, re=1e6, mach=0.2, model_size="large", n_crit=9.0,
                 lambda_penalty=500.0, cl_targets=None, **kwargs):
        self.re = float(re)
        self.mach = float(mach)
        self.model_size = model_size
        self.n_crit = float(n_crit)
        self.lambda_penalty = float(lambda_penalty)
        self.cl_targets = list(cl_targets) if cl_targets is not None else list(_DEFAULT_CL_TARGETS)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--re", type=float, default=1e6)
        parser.add_argument("--lambda_penalty", type=float, default=500.0)

    def get_prompt_blocks(self):
        return {"format_context": _pb.format_context,
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
        violations = compute_geom_violations(airfoil, kulfan)

        CDs, n_miss = [], 0
        for cl_t in self.cl_targets:
            alpha = find_alpha(kulfan, cl_t, self.re, self.model_size, self.n_crit)
            if alpha is None:
                n_miss += 1
                continue
            aero = nf.get_aero_from_kulfan_parameters(
                kulfan_parameters=kulfan, alpha=alpha, Re=self.re,
                n_crit=self.n_crit, model_size=self.model_size)
            CD = float(np.squeeze(aero["CD"]))
            CM = float(np.squeeze(aero["CM"]))
            conf = float(np.squeeze(aero["analysis_confidence"]))
            for k, v in compute_geom_violations(airfoil, kulfan, CM=CM, conf=conf).items():
                violations[k] = violations.get(k, 0.0) + v / len(self.cl_targets)
            if CD > 1e-9:
                CDs.append(CD)

        if not CDs:
            return float(FAIL_REWARD), {"metrics": {"reward": FAIL_REWARD}, "images": [],
                                        "feedback": "no CL targets solved"}
        avg_cd = float(np.mean(CDs))
        miss_pen = n_miss / len(self.cl_targets)
        total_v = sum(violations.values()) + miss_pen
        reward = -avg_cd - self.lambda_penalty * total_v
        print(f"[multipoint_cl_avg_cd] avg_CD={avg_cd:.6f} n_miss={n_miss} reward={reward:.4f}")
        os.makedirs(os.path.join(case_dir, "save"), exist_ok=True)
        import json as _json
        with open(os.path.join(case_dir, "save", "results.json"), "w") as f:
            _json.dump({"avg_cd": avg_cd, "CDs": CDs, "cl_targets": self.cl_targets,
                        "n_miss": n_miss, "violations": violations, "reward": reward}, f, indent=2)
        return float(reward), {"metrics": {"avg_CD": avg_cd, "reward": reward}, "images": [], "feedback": ""}
