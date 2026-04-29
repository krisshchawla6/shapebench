from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from diagnostic_suite.evidence_runner import run_evidence
from diagnostic_suite.llm.judge import BackendFn, JudgeConfig, run_llm_judge
from diagnostic_suite.schemas import validate_diagnostic_output
from diagnostic_suite.types import DiagnosticInput, DiagnosticOutput
from diagnostic_suite.writers.json_writer import write_diagnostics_json


def _coerce_input(diag_input: DiagnosticInput | Dict[str, Any]) -> DiagnosticInput:
    """Accept dataclass input or dict and return DiagnosticInput."""
    if isinstance(diag_input, DiagnosticInput):
        return diag_input
    if isinstance(diag_input, dict):
        return DiagnosticInput(**diag_input)
    raise TypeError(f"diag_input must be DiagnosticInput or dict, got {type(diag_input).__name__}")


def _validate_input_shape(diag_input: DiagnosticInput) -> None:
    """Basic shape checks before running evidence."""
    if not diag_input.environment:
        raise ValueError("DiagnosticInput.environment is required.")
    if not diag_input.design_id:
        raise ValueError("DiagnosticInput.design_id is required.")
    if not isinstance(diag_input.design_params, dict):
        raise ValueError("DiagnosticInput.design_params must be a dict.")
    if not isinstance(diag_input.metrics, dict):
        raise ValueError("DiagnosticInput.metrics must be a dict.")
    if not isinstance(diag_input.images, list):
        raise ValueError("DiagnosticInput.images must be a list.")
    if not isinstance(diag_input.model_artifacts, dict):
        raise ValueError("DiagnosticInput.model_artifacts must be a dict.")
    if not isinstance(diag_input.problem_setup, dict):
        raise ValueError("DiagnosticInput.problem_setup must be a dict.")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_diagnostics(
    diag_input: DiagnosticInput | Dict[str, Any],
    out_path: Optional[str] = None,
    llm_backend: Optional[BackendFn] = None,
    llm_config: Optional[JudgeConfig] = None,
) -> Tuple[DiagnosticOutput, str]:
    """Run full diagnostic pipeline and persist diagnostics.json.

    Pipeline order:
    1) validate input shape
    2) run deterministic evidence
    3) run LLM judge
    4) assemble DiagnosticOutput
    5) schema-validate final output
    6) write diagnostics.json
    """
    inp = _coerce_input(diag_input)
    _validate_input_shape(inp)

    evidence_bundle = run_evidence(inp)
    llm_report = run_llm_judge(
        diag_input=inp,
        evidence_bundle=evidence_bundle,
        backend=llm_backend,
        config=llm_config,
    )

    diag_output = DiagnosticOutput(
        version="0.1.0",
        environment=inp.environment,
        design_id=inp.design_id,
        timestamp_utc=inp.timestamp_utc or _now_utc_iso(),
        input_snapshot=inp.to_dict(),
        evidence_bundle=evidence_bundle,
        llm_report=llm_report,
        trace={
            "pipeline_version": "phase4_mvp_v1",
            "llm_diagnostic_status": llm_report.diagnostic_status,
        },
        provenance={},
    )

    errors = validate_diagnostic_output(diag_output.to_dict())
    if errors:
        raise ValueError(f"Final DiagnosticOutput schema validation failed: {errors}")

    written_path = write_diagnostics_json(diag_output, out_path=out_path)
    return diag_output, written_path

