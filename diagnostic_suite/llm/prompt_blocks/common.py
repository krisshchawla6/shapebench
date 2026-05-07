"""Common prompt block content shared by environments."""

COMMON_RUBRIC = """
Use the deterministic evidence as primary factual input:
- feasibility checks (parameter/domain/artifact integrity),
- geometry checks (deformation realism heuristics),
- aerodynamic checks (force/field plausibility).
Visual artifacts may also be attached to this request; use them as additional evidence when present.

Do not fabricate measurements, files, or CFD facts.
If evidence is missing, explicitly state uncertainty and downgrade confidence.
Use controlled failure mechanisms and mitigations only from the provided catalogs.
"""

OUTPUT_RULES = """
Output MUST be valid JSON only.
No markdown, no prose outside JSON, no code fences.
All fields required by schema must be present.
For unknown categories, use the enum value "OTHER".
Confidence must be a float in [0, 1].
"""

EVIDENCE_PRIORITIZATION = """
Prioritize hard data in this order:
1) metric consistency and feasibility errors,
2) geometry risk signals,
3) image availability and visual evidence quality signals.
"""

