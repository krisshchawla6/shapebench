from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

from diagnostic_suite.pipeline import run_diagnostics
from diagnostic_suite.types import DiagnosticInput, DiagnosticOutput
from diagnostic_suite.llm.judge import BackendFn, JudgeConfig


def _infer_design_id(case_dir: Optional[str], design_path: Optional[str]) -> str:
    if case_dir:
        return os.path.basename(os.path.normpath(case_dir))
    if design_path:
        return os.path.splitext(os.path.basename(design_path))[0]
    return "unknown_design"


def _load_design_params(design_path: Optional[str]) -> Dict[str, Any]:
    if not design_path or not os.path.exists(design_path):
        return {}
    try:
        with open(design_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def build_drivaer_diagnostic_input(
    *,
    design_path: str,
    case_dir: str,
    metrics: Dict[str, Any],
    images: Optional[list[str]] = None,
    model_artifacts: Optional[Dict[str, Any]] = None,
    problem_setup: Optional[Dict[str, Any]] = None,
    run_context: Optional[Dict[str, Any]] = None,
    field_stats: Optional[Dict[str, Any]] = None,
    raw_feedback: str = "",
) -> DiagnosticInput:
    """Build DiagnosticInput from DrivAer_Star evaluation outputs.

    This helper is designed to match the current DrivAer environment output shape:
    - metrics from reward.evaluate(...) return payload
    - images list from reward/environment
    - model_artifacts from runtime args (base_vtk / norm_stats_path)
    """
    design_params = _load_design_params(design_path)
    design_id = _infer_design_id(case_dir=case_dir, design_path=design_path)

    return DiagnosticInput(
        environment="DrivAer_Star",
        design_id=design_id,
        design_path=design_path,
        case_dir=case_dir,
        design_params=design_params,
        metrics=metrics or {},
        images=images or [],
        field_stats=field_stats or {},
        model_artifacts=model_artifacts or {},
        problem_setup=problem_setup or {},
        run_context=run_context or {},
        raw_feedback=raw_feedback or "",
    )


def run_drivaer_diagnostics(
    *,
    design_path: str,
    case_dir: str,
    metrics: Dict[str, Any],
    images: Optional[list[str]] = None,
    model_artifacts: Optional[Dict[str, Any]] = None,
    problem_setup: Optional[Dict[str, Any]] = None,
    run_context: Optional[Dict[str, Any]] = None,
    field_stats: Optional[Dict[str, Any]] = None,
    raw_feedback: str = "",
    out_path: Optional[str] = None,
    llm_backend: Optional[BackendFn] = None,
    llm_config: Optional[JudgeConfig] = None,
) -> Tuple[DiagnosticOutput, str]:
    """One-call DrivAer integration helper.

    Builds DiagnosticInput from environment outputs and runs the full diagnostic
    pipeline, returning `(DiagnosticOutput, written_json_path)`.
    """
    diag_input = build_drivaer_diagnostic_input(
        design_path=design_path,
        case_dir=case_dir,
        metrics=metrics,
        images=images,
        model_artifacts=model_artifacts,
        problem_setup=problem_setup,
        run_context=run_context,
        field_stats=field_stats,
        raw_feedback=raw_feedback,
    )

    return run_diagnostics(
        diag_input=diag_input,
        out_path=out_path,
        llm_backend=llm_backend,
        llm_config=llm_config,
    )

