from __future__ import annotations

import json
import os
from typing import Optional

from diagnostic_suite.types import DiagnosticOutput


DEFAULT_FILENAME = "diagnostics.json"


def resolve_output_path(
    diag_output: DiagnosticOutput,
    out_path: Optional[str] = None,
) -> str:
    """Resolve target path for diagnostics JSON.

    Priority:
    1) explicit out_path
    2) <case_dir>/save/diagnostics.json if case_dir is present
    3) cwd/diagnostics.json
    """
    if out_path:
        return out_path

    case_dir = diag_output.input_snapshot.get("case_dir")
    if case_dir:
        return os.path.join(case_dir, "save", DEFAULT_FILENAME)

    return os.path.abspath(DEFAULT_FILENAME)


def write_diagnostics_json(
    diag_output: DiagnosticOutput,
    out_path: Optional[str] = None,
) -> str:
    """Write DiagnosticOutput to diagnostics.json and return absolute path."""
    target = resolve_output_path(diag_output, out_path=out_path)
    target = os.path.abspath(target)
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(target, "w", encoding="utf-8") as f:
        json.dump(diag_output.to_dict(), f, indent=2)

    return target

