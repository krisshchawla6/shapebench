from __future__ import annotations

from collections import Counter
from typing import Dict, List

from diagnostic_suite.checks.aero_checks import get_aero_checks
from diagnostic_suite.checks.feasibility_checks import get_feasibility_checks
from diagnostic_suite.checks.geometry_checks import get_geometry_checks
from diagnostic_suite.environments.registry import get_environment_metadata
from diagnostic_suite.types import CheckResult, DiagnosticInput, EvidenceBundle


def _summarize(results: List[CheckResult]) -> Dict[str, int]:
    counter = Counter(r.status for r in results)
    return {
        "ok": counter.get("ok", 0),
        "warning": counter.get("warning", 0),
        "issue": counter.get("issue", 0),
        "error": counter.get("error", 0),
        "missing": counter.get("missing", 0),
    }


def _build_data_quality_notes(feasibility: List[CheckResult], aero: List[CheckResult]) -> List[str]:
    notes: List[str] = []
    for check in feasibility + aero:
        if check.status in ("missing", "error"):
            notes.append(f"{check.check_id}: {check.message}")
    return notes


def run_evidence(diag_input: DiagnosticInput) -> EvidenceBundle:
    """Run all deterministic evidence checks for one diagnostic payload."""
    env_meta = get_environment_metadata(diag_input.environment)

    feasibility_checks = get_feasibility_checks()
    geometry_checks = get_geometry_checks()
    aero_checks = get_aero_checks()

    feasibility_results = [chk.run_safe(diag_input, env_meta) for chk in feasibility_checks]
    geometry_results = [chk.run_safe(diag_input, env_meta) for chk in geometry_checks]
    aero_results = [chk.run_safe(diag_input, env_meta) for chk in aero_checks]

    summary = {
        "feasibility": _summarize(feasibility_results),
        "geometry": _summarize(geometry_results),
        "aero": _summarize(aero_results),
    }
    data_quality_notes = _build_data_quality_notes(feasibility_results, aero_results)

    return EvidenceBundle(
        environment=diag_input.environment,
        design_id=diag_input.design_id,
        feasibility=feasibility_results,
        geometry=geometry_results,
        aero=aero_results,
        summary=summary,
        data_quality_notes=data_quality_notes,
    )
