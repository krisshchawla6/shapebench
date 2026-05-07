from __future__ import annotations

from diagnostic_suite.failure_taxonomy import ALL_FAILURE_MECHANISMS
from diagnostic_suite.mitigation_catalog import ALL_MITIGATION_ACTIONS
from diagnostic_suite.llm.prompt_blocks.common import (
    COMMON_RUBRIC,
    EVIDENCE_PRIORITIZATION,
    OUTPUT_RULES,
)
from diagnostic_suite.llm.prompt_blocks.drivaer_star import (
    DRIVAER_CONTEXT,
    DRIVAER_FAILURE_INTERPRETATION,
)


def build_system_prompt() -> str:
    """System instruction for primary diagnostic judge."""
    return (
        "You are a scientific diagnostic judge for aerodynamic shape optimization.\n"
        "You must integrate deterministic evidence and visual context (flow fields, pressure maps, final design images, etc.) into a concise, auditable diagnosis.\n\n"
        f"{COMMON_RUBRIC.strip()}\n\n"
        f"{EVIDENCE_PRIORITIZATION.strip()}\n\n"
        f"{DRIVAER_CONTEXT.strip()}\n\n"
        f"{DRIVAER_FAILURE_INTERPRETATION.strip()}\n\n"
        f"{OUTPUT_RULES.strip()}\n\n"
        "Allowed failure mechanisms:\n"
        f"{ALL_FAILURE_MECHANISMS}\n\n"
        "Allowed mitigation actions:\n"
        f"{ALL_MITIGATION_ACTIONS}\n"
    )

