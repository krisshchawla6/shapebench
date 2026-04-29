from __future__ import annotations

from typing import Any, Dict, List

from diagnostic_suite.checks.base_check import BaseCheck
from diagnostic_suite.environments.drivaer_star import expected_image_paths
from diagnostic_suite.types import CheckResult, DiagnosticInput


class DragDecompositionConsistencyCheck(BaseCheck):
    check_id = "A001_drag_decomposition_consistency"
    tier = "aero"

    def run(self, diag_input: DiagnosticInput, env_meta: Dict[str, Any]) -> CheckResult:
        m = diag_input.metrics or {}
        required = ["drag", "drag_pressure", "drag_shear"]
        if any(k not in m for k in required):
            return CheckResult(
                check_id=self.check_id,
                tier=self.tier,
                status="missing",
                severity=0.3,
                message="Missing drag decomposition metrics.",
                value={"available_keys": sorted(m.keys())},
            )

        drag = float(m["drag"])
        recon = float(m["drag_pressure"]) + float(m["drag_shear"])
        abs_err = abs(drag - recon)
        denom = max(abs(drag), 1.0)
        rel_err = abs_err / denom
        warn_tol = float(env_meta["thresholds"]["drag_decomposition_rel_tol_warn"])
        if rel_err > warn_tol:
            status = "warning"
            sev = min(1.0, rel_err / max(warn_tol * 3.0, 1e-9))
            msg = f"Drag decomposition inconsistency detected (rel_err={rel_err:.5f})."
        else:
            status = "ok"
            sev = 0.0
            msg = f"Drag decomposition consistent (rel_err={rel_err:.5f})."
        return CheckResult(
            check_id=self.check_id,
            tier=self.tier,
            status=status,
            severity=sev,
            message=msg,
            value={"drag": drag, "drag_pressure_plus_shear": recon, "rel_err": rel_err},
            threshold={"warn_rel_err": warn_tol},
        )


class CdPlausibleRangeCheck(BaseCheck):
    check_id = "A002_cd_plausible_range"
    tier = "aero"

    def run(self, diag_input: DiagnosticInput, env_meta: Dict[str, Any]) -> CheckResult:
        m = diag_input.metrics or {}
        if "Cd" not in m:
            return CheckResult(
                check_id=self.check_id,
                tier=self.tier,
                status="missing",
                severity=0.3,
                message="Cd metric missing.",
            )
        cd = float(m["Cd"])
        cd_min = float(env_meta["thresholds"]["cd_min_warn"])
        cd_max = float(env_meta["thresholds"]["cd_max_warn"])
        if cd < cd_min or cd > cd_max:
            status = "warning"
            sev = 0.7
            msg = f"Cd outside plausible warning band [{cd_min}, {cd_max}]."
        else:
            status = "ok"
            sev = 0.0
            msg = "Cd within plausible warning band."
        return CheckResult(
            check_id=self.check_id,
            tier=self.tier,
            status=status,
            severity=sev,
            message=msg,
            value=cd,
            threshold={"min": cd_min, "max": cd_max},
        )


class LiftPlausibleRangeCheck(BaseCheck):
    check_id = "A003_lift_plausible_range"
    tier = "aero"

    def run(self, diag_input: DiagnosticInput, env_meta: Dict[str, Any]) -> CheckResult:
        m = diag_input.metrics or {}
        if "lift" not in m:
            return CheckResult(
                check_id=self.check_id,
                tier=self.tier,
                status="missing",
                severity=0.3,
                message="Lift metric missing.",
            )
        lift = float(m["lift"])
        warn_abs = float(env_meta["thresholds"]["lift_abs_warn"])
        if abs(lift) > warn_abs:
            status = "warning"
            sev = min(1.0, abs(lift) / max(warn_abs * 3.0, 1e-9))
            msg = f"Lift magnitude exceeds warning threshold (|lift|={abs(lift):.2f})."
        else:
            status = "ok"
            sev = 0.0
            msg = "Lift magnitude within plausible warning range."
        return CheckResult(
            check_id=self.check_id,
            tier=self.tier,
            status=status,
            severity=sev,
            message=msg,
            value=lift,
            threshold={"warn_abs": warn_abs},
        )


class ImageAvailabilitySignalCheck(BaseCheck):
    check_id = "A004_image_availability_signal"
    tier = "aero"

    def run(self, diag_input: DiagnosticInput, env_meta: Dict[str, Any]) -> CheckResult:
        images = diag_input.images or []
        expected = expected_image_paths(images)
        present = sum(1 for v in expected.values() if v)
        total = len(expected)
        frac = (present / total) if total else 0.0
        if total == 0:
            status = "missing"
            sev = 0.3
            msg = "No expected image definitions found."
        elif present == 0:
            status = "missing"
            sev = 0.5
            msg = "No expected flow images found; LLM evidence quality will be reduced."
        elif frac < 1.0:
            status = "warning"
            sev = 0.3
            msg = f"Partial flow image coverage ({present}/{total})."
        else:
            status = "ok"
            sev = 0.0
            msg = "All expected flow images are available."

        return CheckResult(
            check_id=self.check_id,
            tier=self.tier,
            status=status,
            severity=sev,
            message=msg,
            value={"present": present, "total": total, "coverage": frac, "suffix_map": expected},
            threshold={"expected_total": total},
            evidence_refs=images[:6],
        )


def get_aero_checks() -> List[BaseCheck]:
    return [
        DragDecompositionConsistencyCheck(),
        CdPlausibleRangeCheck(),
        LiftPlausibleRangeCheck(),
        ImageAvailabilitySignalCheck(),
    ]
