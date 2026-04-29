from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from diagnostic_suite.config import DEFAULT_CONFIG
from diagnostic_suite.failure_taxonomy import ALL_FAILURE_MECHANISMS
from diagnostic_suite.mitigation_catalog import ALL_MITIGATION_ACTIONS
from diagnostic_suite.schemas import validate_llm_report
from diagnostic_suite.types import LLMDiagnosticReport


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: extract first {...} block.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _ensure_list(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def _normalize_enum_list(values: List[Any], allowed: List[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        s = str(value)
        out.append(s if s in allowed else "OTHER")
    return out


def _bounded_confidence(value: Any) -> float:
    try:
        f = float(value)
    except Exception:
        return 0.0
    if f < 0.0:
        return 0.0
    if f > 1.0:
        return 1.0
    return f


def fallback_llm_report(
    reason: str,
    raw_response: Optional[str] = None,
    parser_warnings: Optional[List[str]] = None,
) -> LLMDiagnosticReport:
    warnings = list(parser_warnings or [])
    warnings.append(reason)
    return LLMDiagnosticReport(
        diagnostic_status="llm_error",
        overall_assessment=(
            "LLM diagnostic generation failed; deterministic evidence should be used as the primary "
            "interim basis until the LLM path is restored."
        ),
        primary_failure_mechanisms=["OTHER"],
        secondary_risks=[],
        surrogate_exploitation_risk="unknown",
        physical_credibility="unknown",
        confidence=0.0,
        evidence_weighting_rationale="Fallback report due to LLM/parse/schema failure.",
        recommended_mitigations=["OTHER"],
        recommended_next_tests=["Re-run LLM diagnosis with strict JSON output enabled."],
        notes_on_missing_evidence=["LLM report unavailable due to parser/model failure."],
        citations_to_evidence=[],
        model_name=DEFAULT_CONFIG.thresholds.llm_model_name,
        prompt_version=DEFAULT_CONFIG.thresholds.prompt_version,
        raw_response=raw_response,
        parser_warnings=warnings,
    )


def parse_llm_report(raw_text: str, model_name: Optional[str] = None) -> LLMDiagnosticReport:
    parser_warnings: List[str] = []
    obj = _extract_json_object(raw_text)
    if obj is None:
        return fallback_llm_report("Unable to parse JSON object from LLM output.", raw_response=raw_text)

    # Normalize aggressively to preserve robustness.
    diag_status = str(obj.get("diagnostic_status", "partial"))
    if diag_status not in {"complete", "partial", "llm_error"}:
        parser_warnings.append(f"Unknown diagnostic_status '{diag_status}', set to 'partial'.")
        diag_status = "partial"

    report_obj: Dict[str, Any] = {
        "diagnostic_status": diag_status,
        "overall_assessment": str(obj.get("overall_assessment", "")),
        "primary_failure_mechanisms": _normalize_enum_list(
            _ensure_list(obj.get("primary_failure_mechanisms")), ALL_FAILURE_MECHANISMS
        ),
        "secondary_risks": _normalize_enum_list(
            _ensure_list(obj.get("secondary_risks")), ALL_FAILURE_MECHANISMS
        ),
        "surrogate_exploitation_risk": str(obj.get("surrogate_exploitation_risk", "unknown")),
        "physical_credibility": str(obj.get("physical_credibility", "unknown")),
        "confidence": _bounded_confidence(obj.get("confidence", 0.0)),
        "evidence_weighting_rationale": str(obj.get("evidence_weighting_rationale", "")),
        "recommended_mitigations": _normalize_enum_list(
            _ensure_list(obj.get("recommended_mitigations")), ALL_MITIGATION_ACTIONS
        ),
        "recommended_next_tests": [str(x) for x in _ensure_list(obj.get("recommended_next_tests"))],
        "notes_on_missing_evidence": [str(x) for x in _ensure_list(obj.get("notes_on_missing_evidence"))],
        "citations_to_evidence": [str(x) for x in _ensure_list(obj.get("citations_to_evidence"))],
        "model_name": model_name or str(obj.get("model_name")) if obj.get("model_name") is not None else model_name,
        "prompt_version": obj.get("prompt_version", DEFAULT_CONFIG.thresholds.prompt_version),
        "raw_response": raw_text,
        "parser_warnings": [str(x) for x in _ensure_list(obj.get("parser_warnings"))] + parser_warnings,
    }

    # normalize risk labels
    if report_obj["surrogate_exploitation_risk"] not in {"low", "medium", "high", "unknown"}:
        report_obj["parser_warnings"].append(
            f"Invalid surrogate_exploitation_risk '{report_obj['surrogate_exploitation_risk']}', set to 'unknown'."
        )
        report_obj["surrogate_exploitation_risk"] = "unknown"
    if report_obj["physical_credibility"] not in {"low", "medium", "high", "unknown"}:
        report_obj["parser_warnings"].append(
            f"Invalid physical_credibility '{report_obj['physical_credibility']}', set to 'unknown'."
        )
        report_obj["physical_credibility"] = "unknown"

    errors = validate_llm_report(report_obj)
    if errors:
        return fallback_llm_report(
            "Schema validation failed for LLM report.",
            raw_response=raw_text,
            parser_warnings=errors + report_obj.get("parser_warnings", []),
        )

    return LLMDiagnosticReport(**report_obj)

