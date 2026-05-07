from __future__ import annotations

from typing import Any, Dict, List

from diagnostic_suite.checks.base_check import BaseCheck
from diagnostic_suite.types import CheckResult, DiagnosticInput


class ParamExtremenessRatioCheck(BaseCheck):
    check_id = "G001_param_extremeness_ratio"
    tier = "geometry"

    def run(self, diag_input: DiagnosticInput, env_meta: Dict[str, Any]) -> CheckResult:
        params = diag_input.design_params or {}
        bounds = env_meta["param_bounds"]
        margin_ratio = env_meta["thresholds"]["near_bound_margin_ratio"]
        warn_fraction = env_meta["thresholds"]["near_bound_fraction_warn"]

        near_count = 0
        considered = 0
        near_keys: List[str] = []
        for key, (lo, hi) in bounds.items():
            if key not in params:
                continue
            considered += 1
            value = float(params[key])
            margin = (hi - lo) * margin_ratio
            if value <= lo + margin or value >= hi - margin:
                near_count += 1
                near_keys.append(key)

        frac = (near_count / considered) if considered else 0.0
        if considered == 0:
            status = "missing"
            sev = 0.3
            msg = "No parameters available to compute extremeness ratio."
        elif frac >= warn_fraction:
            status = "warning"
            sev = min(1.0, 0.5 + 0.5 * frac)
            msg = f"High fraction of parameters near bounds ({frac:.2f})."
        else:
            status = "ok"
            sev = 0.0
            msg = f"Parameter extremeness ratio within nominal range ({frac:.2f})."

        return CheckResult(
            check_id=self.check_id,
            tier=self.tier,
            status=status,
            severity=sev,
            message=msg,
            value={"near_bound_fraction": frac, "near_bound_keys": near_keys},
            threshold={"warn_fraction": warn_fraction, "margin_ratio": margin_ratio},
        )


class CombinedAngleStressCheck(BaseCheck):
    check_id = "G002_combined_angle_stress"
    tier = "geometry"

    _ANGLE_KEYS = [
        "ramp_angle",
        "trunklid_angle",
        "diffusor_angle",
        "car_green_house_angle",
        "car_front_hood_angle",
        "car_air_intake_angle",
    ]

    def run(self, diag_input: DiagnosticInput, env_meta: Dict[str, Any]) -> CheckResult:
        params = diag_input.design_params or {}
        warn_sum = float(env_meta["thresholds"]["combined_angle_abs_sum_warn"])
        angle_sum = sum(abs(float(params.get(k, 0.0))) for k in self._ANGLE_KEYS)
        if angle_sum > warn_sum:
            status = "warning"
            sev = min(1.0, angle_sum / max(warn_sum * 2.0, 1e-9))
            msg = f"Combined angle stress is high ({angle_sum:.2f} deg abs-sum)."
        else:
            status = "ok"
            sev = 0.0
            msg = f"Combined angle stress within nominal range ({angle_sum:.2f} deg abs-sum)."
        return CheckResult(
            check_id=self.check_id,
            tier=self.tier,
            status=status,
            severity=sev,
            message=msg,
            value={"combined_abs_angle_sum": angle_sum},
            threshold={"warn_sum": warn_sum},
        )


class SizeWidthLengthCouplingCheck(BaseCheck):
    check_id = "G003_size_width_length_coupling"
    tier = "geometry"

    def run(self, diag_input: DiagnosticInput, env_meta: Dict[str, Any]) -> CheckResult:
        p = diag_input.design_params or {}
        car_size = float(p.get("car_size", 1.0))
        car_width = abs(float(p.get("car_width", 0.0)))
        car_len = abs(float(p.get("car_len", 0.0)))
        coupling_score = abs(car_size - 1.0) / 0.2 + car_width / 0.1 + car_len / 0.1

        if coupling_score >= 2.4:
            status = "warning"
            sev = min(1.0, coupling_score / 3.0)
            msg = "Global scale + width/length coupling is aggressive; geometry realism risk increased."
        else:
            status = "ok"
            sev = 0.0
            msg = "Global scale + width/length coupling within nominal range."

        return CheckResult(
            check_id=self.check_id,
            tier=self.tier,
            status=status,
            severity=sev,
            message=msg,
            value={
                "car_size": car_size,
                "abs_car_width": car_width,
                "abs_car_len": car_len,
                "coupling_score": coupling_score,
            },
            threshold={"warn_score": 2.4},
        )


def get_geometry_checks() -> List[BaseCheck]:
    return [
        ParamExtremenessRatioCheck(),
        CombinedAngleStressCheck(),
        SizeWidthLengthCouplingCheck(),
    ]
