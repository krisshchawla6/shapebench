"""ShapeEvolveState -- subclass of discover's State for design optimization.

Maps discover's generic fields to ShapeEvolve's design domain:
  construction -> sorted [[key, value], ...] pairs from design_params (hashable via tuple for PUCT dedup)
  code         -> LLM response text (reasoning + JSON)
  value        -> reward (e.g. L/D ratio), higher = better
  observation  -> simulation feedback + Gemini flow analysis
"""

from __future__ import annotations

import json
import sys
import os
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "discover"))
from ttt_discover.tinker_utils.state import State, to_json_serializable


class ShapeEvolveState(State):
    """Design state for any ShapeEvolve environment.

    Stores design parameters as sorted key-value pairs in `construction`
    so that discover's PUCTSampler can deduplicate via tuple(construction).
    The actual parameter keys are environment-specific and defined by each
    framework's agent.REQUIRED_KEYS -- nothing is hardcoded here.
    """

    def __init__(
        self,
        timestep: int = None,
        construction: list[Any] = None,
        code: str = None,
        value: float = None,
        parent_values: list[float] = None,
        parents: list[dict] = None,
        id: str = "",
        observation: str = "",
        design_path: str = None,
        image_paths: list[str] = None,
        gemini_analysis: str = "",
        **kwargs,
    ):
        super().__init__(
            timestep=timestep,
            construction=construction,
            code=code,
            value=value,
            parent_values=parent_values,
            parents=parents,
            id=id,
            observation=observation,
        )
        self.design_path = design_path
        self.image_paths = image_paths or []
        self.gemini_analysis = gemini_analysis

    @property
    def design_params(self) -> Dict[str, Any]:
        """Reconstruct design_params dict from sorted key-value pairs."""
        if not self.construction:
            return {}
        if isinstance(self.construction, dict):
            return self.construction
        return {k: v for k, v in self.construction}

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["type"] = "ShapeEvolveState"
        d["design_path"] = self.design_path
        d["image_paths"] = self.image_paths
        d["gemini_analysis"] = self.gemini_analysis
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ShapeEvolveState:
        return cls(
            timestep=d["timestep"],
            construction=d.get("construction", []),
            code=d.get("code", ""),
            value=d.get("value"),
            parent_values=d.get("parent_values", []),
            parents=d.get("parents", []),
            id=d.get("id"),
            observation=d.get("observation", ""),
            design_path=d.get("design_path", ""),
            image_paths=d.get("image_paths", []),
            gemini_analysis=d.get("gemini_analysis", ""),
        )

    def to_prompt(self, target=None, metric_name: str = "L/D", maximize: bool = True, language: str = ""):
        """Build prompt context from this state for the trainable LLM.

        Instead of showing code (as discover does for coding problems),
        we show the design parameters, reward, and flow analysis.
        """
        params = self.design_params
        parts = []

        if params:
            param_str = ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
            parts.append(f"Previous design parameters: {param_str}")

        if self.value is not None:
            parts.append(f"Previous {metric_name}: {self.value:.4f}")

            if self.parent_values:
                parent_val = self.parent_values[0]
                direction = "improvement" if (self.value > parent_val) == maximize else "regression"
                parts.append(f"Parent {metric_name}: {parent_val:.4f} ({direction})")

            if target is not None:
                gap = (target - self.value) if maximize else (self.value - target)
                parts.append(f"Target: {target}. Gap: {gap:.4f}.")

        if self.gemini_analysis:
            parts.append(f"\n--- Flow Field Analysis ---\n{self.gemini_analysis}\n--- End Analysis ---")

        if self.observation and self.observation.strip():
            obs = self.observation.strip()
            if len(obs) > 500:
                obs = "\n\t...(TRUNCATED)...\n" + obs[-500:]
            parts.append(f"\n--- Simulation Feedback ---\n{obs}\n--- End Feedback ---")

        return "\n".join(parts) if parts else "No previous design available."


def params_to_construction(params: Dict[str, Any]) -> list:
    """Convert a design_params dict to a sorted tuple list for State.construction.

    Uses tuples so PUCT can hash the construction for deduplication.
    """
    return sorted([(k, to_json_serializable(v)) for k, v in params.items()])


def make_initial_state(design_params: Optional[Dict[str, Any]] = None, value: float = 0.0) -> ShapeEvolveState:
    """Create a seed ShapeEvolveState for the PUCT archive."""
    construction = params_to_construction(design_params) if design_params else []
    return ShapeEvolveState(
        timestep=-1,
        construction=construction,
        code="",
        value=value,
        observation="",
    )
