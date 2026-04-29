from __future__ import annotations

from typing import Any, Dict

from diagnostic_suite.environments import drivaer_star


def get_environment_metadata(environment_name: str) -> Dict[str, Any]:
    """Return metadata for the requested environment."""
    if environment_name == "DrivAer_Star":
        return drivaer_star.get_environment_metadata()
    raise ValueError(f"Unsupported environment for MVP: {environment_name}")
