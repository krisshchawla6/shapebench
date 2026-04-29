from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DiagnosticInput:
    """Single-design diagnostic input payload."""

    environment: str
    design_id: str
    design_path: Optional[str]
    case_dir: Optional[str]
    design_params: Dict[str, float]
    metrics: Dict[str, Any]
    images: List[str] = field(default_factory=list)
    field_stats: Dict[str, Any] = field(default_factory=dict)
    model_artifacts: Dict[str, Any] = field(default_factory=dict)
    problem_setup: Dict[str, Any] = field(default_factory=dict)
    run_context: Dict[str, Any] = field(default_factory=dict)
    raw_feedback: str = ""
    timestamp_utc: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CheckResult:
    """One deterministic check result."""

    check_id: str
    tier: str
    status: str
    severity: float
    message: str
    value: Any = None
    threshold: Any = None
    evidence_refs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceBundle:
    """Deterministic evidence generated before LLM integration."""

    environment: str
    design_id: str
    feasibility: List[CheckResult] = field(default_factory=list)
    geometry: List[CheckResult] = field(default_factory=list)
    aero: List[CheckResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    data_quality_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["feasibility"] = [c.to_dict() for c in self.feasibility]
        out["geometry"] = [c.to_dict() for c in self.geometry]
        out["aero"] = [c.to_dict() for c in self.aero]
        return out


@dataclass
class LLMDiagnosticReport:
    """Primary integrative diagnostic report produced by the LLM judge."""

    diagnostic_status: str
    overall_assessment: str
    primary_failure_mechanisms: List[str] = field(default_factory=list)
    secondary_risks: List[str] = field(default_factory=list)
    surrogate_exploitation_risk: str = "unknown"
    physical_credibility: str = "unknown"
    confidence: float = 0.0
    evidence_weighting_rationale: str = ""
    recommended_mitigations: List[str] = field(default_factory=list)
    recommended_next_tests: List[str] = field(default_factory=list)
    notes_on_missing_evidence: List[str] = field(default_factory=list)
    citations_to_evidence: List[str] = field(default_factory=list)
    model_name: Optional[str] = None
    prompt_version: Optional[str] = None
    raw_response: Optional[str] = None
    parser_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosticOutput:
    """Final persisted output artifact for one diagnosed design."""

    version: str
    environment: str
    design_id: str
    timestamp_utc: str
    input_snapshot: Dict[str, Any]
    evidence_bundle: EvidenceBundle
    llm_report: LLMDiagnosticReport
    trace: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["evidence_bundle"] = self.evidence_bundle.to_dict()
        out["llm_report"] = self.llm_report.to_dict()
        return out
