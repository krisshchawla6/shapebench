from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from diagnostic_suite.types import CheckResult, DiagnosticInput


class BaseCheck(ABC):
    """Base class for deterministic diagnostic checks."""

    check_id: str = "UNSET_CHECK_ID"
    tier: str = "UNSET_TIER"

    @abstractmethod
    def run(self, diag_input: DiagnosticInput, env_meta: Dict[str, Any]) -> CheckResult:
        """Execute check and return one structured result."""

    def run_safe(self, diag_input: DiagnosticInput, env_meta: Dict[str, Any]) -> CheckResult:
        """Run check with non-fatal error capture."""
        try:
            result = self.run(diag_input, env_meta)
            if not result.evidence_refs:
                result.evidence_refs = self._default_evidence_refs(diag_input)
            return result
        except Exception as exc:  # pragma: no cover - safety path
            return CheckResult(
                check_id=self.check_id,
                tier=self.tier,
                status="error",
                severity=1.0,
                message=f"Check raised exception: {exc}",
                evidence_refs=self._default_evidence_refs(diag_input),
                metadata={"exception_type": type(exc).__name__},
            )

    @staticmethod
    def clamp_severity(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _default_evidence_refs(diag_input: DiagnosticInput) -> List[str]:
        refs: List[str] = []
        if diag_input.design_path:
            refs.append(diag_input.design_path)
        if diag_input.case_dir:
            refs.append(diag_input.case_dir)
        return refs
