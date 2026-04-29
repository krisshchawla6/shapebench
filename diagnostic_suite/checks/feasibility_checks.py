from __future__ import annotations

import math
import os
from typing import Any, Dict, List

from diagnostic_suite.checks.base_check import BaseCheck
from diagnostic_suite.environments.drivaer_star import (
    infer_body_style,
    norm_stats_style_compatible,
)
from diagnostic_suite.types import CheckResult, DiagnosticInput


class RequiredParamsPresentCheck(BaseCheck):
    check_id = "F001_required_params_present"
    tier = "feasibility"

    def run(self, diag_input: DiagnosticInput, env_meta: Dict[str, Any]) -> CheckResult:
        required = env_meta["required_param_keys"]
        params = diag_input.design_params or {}
        missing = [k for k in required if k not in params]
        status = "ok" if not missing else "issue"
        severity = 0.0 if not missing else 1.0
        return CheckResult(
            check_id=self.check_id,
            tier=self.tier,
            status=status,
            severity=severity,
            message="All required parameters present." if not missing else f"Missing parameters: {missing}",
            value={"missing": missing},
            evidence_refs=[diag_input.design_path] if diag_input.design_path else [],
        )


class ParamBoundsRespectedCheck(BaseCheck):
    check_id = "F002_param_bounds_respected"
    tier = "feasibility"

    def run(self, diag_input: DiagnosticInput, env_meta: Dict[str, Any]) -> CheckResult:
        bounds = env_meta["param_bounds"]
        tol = env_meta["thresholds"]["feasibility_tol"]
        params = diag_input.design_params or {}
        violations: List[Dict[str, Any]] = []
        for key, (lo, hi) in bounds.items():
            if key not in params:
                continue
            value = float(params[key])
            if value < lo - tol or value > hi + tol:
                violations.append({"key": key, "value": value, "bounds": [lo, hi]})
        status = "ok" if not violations else "issue"
        severity = 0.0 if not violations else min(1.0, 0.2 + 0.8 * (len(violations) / max(1, len(bounds))))
        return CheckResult(
            check_id=self.check_id,
            tier=self.tier,
            status=status,
            severity=severity,
            message="All parameter values within bounds." if not violations else f"Out-of-bounds parameters: {len(violations)}",
            value={"violations": violations},
            threshold="within configured bounds",
        )


class BaseVtkExistsCheck(BaseCheck):
    check_id = "F003_base_vtk_exists"
    tier = "feasibility"

    def run(self, diag_input: DiagnosticInput, env_meta: Dict[str, Any]) -> CheckResult:
        base_vtk_path = (diag_input.model_artifacts or {}).get("base_vtk_path")
        exists = bool(base_vtk_path) and os.path.exists(base_vtk_path)
        return CheckResult(
            check_id=self.check_id,
            tier=self.tier,
            status="ok" if exists else "issue",
            severity=0.0 if exists else 1.0,
            message="Base VTK exists." if exists else f"Missing base_vtk_path or file not found: {base_vtk_path}",
            value=base_vtk_path,
            evidence_refs=[base_vtk_path] if base_vtk_path else [],
        )


class NormStatsExistsCheck(BaseCheck):
    check_id = "F004_norm_stats_exists"
    tier = "feasibility"

    def run(self, diag_input: DiagnosticInput, env_meta: Dict[str, Any]) -> CheckResult:
        norm_stats_path = (diag_input.model_artifacts or {}).get("norm_stats_path")
        exists = bool(norm_stats_path) and os.path.exists(norm_stats_path)
        return CheckResult(
            check_id=self.check_id,
            tier=self.tier,
            status="ok" if exists else "issue",
            severity=0.0 if exists else 1.0,
            message="Norm stats file exists." if exists else f"Missing norm_stats_path or file not found: {norm_stats_path}",
            value=norm_stats_path,
            evidence_refs=[norm_stats_path] if norm_stats_path else [],
        )


class MetricsFiniteCheck(BaseCheck):
    check_id = "F005_metrics_finite"
    tier = "feasibility"

    def run(self, diag_input: DiagnosticInput, env_meta: Dict[str, Any]) -> CheckResult:
        metrics = diag_input.metrics or {}
        required = env_meta["required_metric_keys"]
        bad: List[str] = []
        missing: List[str] = []
        for key in required:
            if key not in metrics:
                missing.append(key)
                continue
            value = metrics.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                bad.append(key)
        status = "ok" if not bad and not missing else "issue"
        severity = 0.0 if status == "ok" else 1.0
        msg = "All required metrics are finite." if status == "ok" else f"Missing metrics={missing}, non-finite metrics={bad}"
        return CheckResult(
            check_id=self.check_id,
            tier=self.tier,
            status=status,
            severity=severity,
            message=msg,
            value={"missing": missing, "non_finite": bad},
        )


class BodyStyleNormCompatibilityCheck(BaseCheck):
    check_id = "F006_body_style_norm_compatibility"
    tier = "feasibility"

    def run(self, diag_input: DiagnosticInput, env_meta: Dict[str, Any]) -> CheckResult:
        artifacts = diag_input.model_artifacts or {}
        base_vtk = artifacts.get("base_vtk_path")
        norm_stats = artifacts.get("norm_stats_path")
        style = infer_body_style(base_vtk)
        compatibility = norm_stats_style_compatible(base_vtk, norm_stats)

        if compatibility is True:
            return CheckResult(
                check_id=self.check_id,
                tier=self.tier,
                status="ok",
                severity=0.0,
                message=f"Norm stats compatible with inferred body style '{style}'.",
                value={"style": style, "norm_stats_path": norm_stats},
            )
        if compatibility is False:
            return CheckResult(
                check_id=self.check_id,
                tier=self.tier,
                status="warning",
                severity=0.6,
                message=(
                    "Possible style/norm-stats mismatch. Inferred style from base VTK does not match "
                    "norm_stats filename convention."
                ),
                value={"style": style, "norm_stats_path": norm_stats},
            )
        return CheckResult(
            check_id=self.check_id,
            tier=self.tier,
            status="missing",
            severity=0.3,
            message="Could not infer style compatibility (missing or ambiguous artifact paths).",
            value={"style": style, "norm_stats_path": norm_stats},
        )


def get_feasibility_checks() -> List[BaseCheck]:
    return [
        RequiredParamsPresentCheck(),
        ParamBoundsRespectedCheck(),
        BaseVtkExistsCheck(),
        NormStatsExistsCheck(),
        MetricsFiniteCheck(),
        BodyStyleNormCompatibilityCheck(),
    ]
