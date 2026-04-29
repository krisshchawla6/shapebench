"""JSON schemas and validation helpers for diagnostic MVP artifacts."""

from __future__ import annotations

from typing import Any, Dict, List

from .failure_taxonomy import ALL_FAILURE_MECHANISMS
from .mitigation_catalog import ALL_MITIGATION_ACTIONS


CHECK_RESULT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["check_id", "tier", "status", "severity", "message"],
    "properties": {
        "check_id": {"type": "string"},
        "tier": {"type": "string", "enum": ["feasibility", "geometry", "aero"]},
        "status": {"type": "string", "enum": ["ok", "warning", "issue", "error", "missing"]},
        "severity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "message": {"type": "string"},
        "value": {},
        "threshold": {},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
        "metadata": {"type": "object"},
    },
    "additionalProperties": False,
}

EVIDENCE_BUNDLE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["environment", "design_id", "feasibility", "geometry", "aero", "summary"],
    "properties": {
        "environment": {"type": "string"},
        "design_id": {"type": "string"},
        "feasibility": {"type": "array", "items": CHECK_RESULT_SCHEMA},
        "geometry": {"type": "array", "items": CHECK_RESULT_SCHEMA},
        "aero": {"type": "array", "items": CHECK_RESULT_SCHEMA},
        "summary": {"type": "object"},
        "data_quality_notes": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}

LLM_DIAGNOSTIC_REPORT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": [
        "diagnostic_status",
        "overall_assessment",
        "primary_failure_mechanisms",
        "secondary_risks",
        "surrogate_exploitation_risk",
        "physical_credibility",
        "confidence",
        "evidence_weighting_rationale",
        "recommended_mitigations",
        "recommended_next_tests",
        "notes_on_missing_evidence",
        "citations_to_evidence",
    ],
    "properties": {
        "diagnostic_status": {"type": "string", "enum": ["complete", "partial", "llm_error"]},
        "overall_assessment": {"type": "string"},
        "primary_failure_mechanisms": {
            "type": "array",
            "items": {"type": "string", "enum": ALL_FAILURE_MECHANISMS},
        },
        "secondary_risks": {
            "type": "array",
            "items": {"type": "string", "enum": ALL_FAILURE_MECHANISMS},
        },
        "surrogate_exploitation_risk": {"type": "string", "enum": ["low", "medium", "high", "unknown"]},
        "physical_credibility": {"type": "string", "enum": ["low", "medium", "high", "unknown"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "evidence_weighting_rationale": {"type": "string"},
        "recommended_mitigations": {
            "type": "array",
            "items": {"type": "string", "enum": ALL_MITIGATION_ACTIONS},
        },
        "recommended_next_tests": {"type": "array", "items": {"type": "string"}},
        "notes_on_missing_evidence": {"type": "array", "items": {"type": "string"}},
        "citations_to_evidence": {"type": "array", "items": {"type": "string"}},
        "model_name": {"type": ["string", "null"]},
        "prompt_version": {"type": ["string", "null"]},
        "raw_response": {"type": ["string", "null"]},
        "parser_warnings": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}

DIAGNOSTIC_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": [
        "version",
        "environment",
        "design_id",
        "timestamp_utc",
        "input_snapshot",
        "evidence_bundle",
        "llm_report",
    ],
    "properties": {
        "version": {"type": "string"},
        "environment": {"type": "string"},
        "design_id": {"type": "string"},
        "timestamp_utc": {"type": "string"},
        "input_snapshot": {"type": "object"},
        "evidence_bundle": EVIDENCE_BUNDLE_SCHEMA,
        "llm_report": LLM_DIAGNOSTIC_REPORT_SCHEMA,
        "trace": {"type": "object"},
        "provenance": {"type": "object"},
    },
    "additionalProperties": False,
}


def _validate_required_keys(payload: Dict[str, Any], required: List[str], path: str) -> List[str]:
    errors: List[str] = []
    for key in required:
        if key not in payload:
            errors.append(f"{path}: missing required key '{key}'")
    return errors


def validate_minimal(payload: Dict[str, Any], schema: Dict[str, Any], path: str = "root") -> List[str]:
    """Minimal validator (required keys + simple enum checks).

    This fallback avoids hard dependency on `jsonschema` while still enforcing
    the contract shape during MVP development.
    """
    errors: List[str] = []
    if schema.get("type") == "object":
        if not isinstance(payload, dict):
            return [f"{path}: expected object, got {type(payload).__name__}"]
        errors.extend(_validate_required_keys(payload, schema.get("required", []), path))
        properties = schema.get("properties", {})
        for key, rule in properties.items():
            if key not in payload:
                continue
            value = payload[key]
            if "enum" in rule and value not in rule["enum"]:
                errors.append(f"{path}.{key}: value '{value}' not in enum {rule['enum']}")
            if rule.get("type") == "array":
                if not isinstance(value, list):
                    errors.append(f"{path}.{key}: expected array, got {type(value).__name__}")
                else:
                    item_rule = rule.get("items")
                    if isinstance(item_rule, dict) and item_rule.get("enum"):
                        for idx, item in enumerate(value):
                            if item not in item_rule["enum"]:
                                errors.append(
                                    f"{path}.{key}[{idx}]: value '{item}' not in enum {item_rule['enum']}"
                                )
        return errors
    return [f"{path}: unsupported schema type {schema.get('type')}"]


def validate_with_jsonschema(payload: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """Validate with jsonschema when available, fallback otherwise."""
    try:
        import jsonschema
    except ImportError:
        return validate_minimal(payload, schema)

    try:
        jsonschema.validate(instance=payload, schema=schema)
        return []
    except jsonschema.ValidationError as exc:
        return [f"jsonschema validation error: {exc.message}"]


def validate_evidence_bundle(payload: Dict[str, Any]) -> List[str]:
    return validate_with_jsonschema(payload, EVIDENCE_BUNDLE_SCHEMA)


def validate_llm_report(payload: Dict[str, Any]) -> List[str]:
    return validate_with_jsonschema(payload, LLM_DIAGNOSTIC_REPORT_SCHEMA)


def validate_diagnostic_output(payload: Dict[str, Any]) -> List[str]:
    return validate_with_jsonschema(payload, DIAGNOSTIC_OUTPUT_SCHEMA)
