from __future__ import annotations

import json
from typing import Dict, List

from diagnostic_suite.types import DiagnosticInput, EvidenceBundle


def _compact_checks(checks: List[Dict]) -> List[Dict]:
    """Reduce payload size while keeping high-value evidence."""
    compact = []
    for c in checks:
        compact.append(
            {
                "check_id": c.get("check_id"),
                "status": c.get("status"),
                "severity": c.get("severity"),
                "message": c.get("message"),
                "value": c.get("value"),
                "threshold": c.get("threshold"),
            }
        )
    return compact


def build_user_prompt(diag_input: DiagnosticInput, evidence: EvidenceBundle) -> str:
    """Build structured user prompt from deterministic evidence and context."""
    inp = diag_input.to_dict()
    ev = evidence.to_dict()

    payload = {
        "environment": inp["environment"],
        "design_id": inp["design_id"],
        "problem_setup": inp.get("problem_setup", {}),
        "run_context": inp.get("run_context", {}),
        "design_params": inp.get("design_params", {}),
        "metrics": inp.get("metrics", {}),
        "field_stats": inp.get("field_stats", {}),
        "images": inp.get("images", []),
        "model_artifacts": inp.get("model_artifacts", {}),
        "evidence_summary": ev.get("summary", {}),
        "feasibility_checks": _compact_checks(ev.get("feasibility", [])),
        "geometry_checks": _compact_checks(ev.get("geometry", [])),
        "aero_checks": _compact_checks(ev.get("aero", [])),
        "data_quality_notes": ev.get("data_quality_notes", []),
    }

    response_schema = {
        "diagnostic_status": "complete | partial | llm_error",
        "overall_assessment": "string",
        "primary_failure_mechanisms": ["enum FailureMechanism"],
        "secondary_risks": ["enum FailureMechanism"],
        "surrogate_exploitation_risk": "low | medium | high | unknown",
        "physical_credibility": "low | medium | high | unknown",
        "confidence": "float in [0,1]",
        "evidence_weighting_rationale": "string",
        "recommended_mitigations": ["enum MitigationAction"],
        "recommended_next_tests": ["string"],
        "notes_on_missing_evidence": ["string"],
        "citations_to_evidence": ["check_id strings"],
        "model_name": "string | null",
        "prompt_version": "string | null",
        "raw_response": "string | null",
        "parser_warnings": ["string"],
    }

    return (
        "Diagnose this DrivAer_Star candidate using provided deterministic evidence.\n"
        "Return only valid JSON matching the schema.\n\n"
        f"INPUT_PAYLOAD:\n{json.dumps(payload, indent=2)}\n\n"
        f"REQUIRED_OUTPUT_SCHEMA:\n{json.dumps(response_schema, indent=2)}\n"
    )

