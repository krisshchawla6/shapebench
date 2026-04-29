"""
Ready-to-run diagnostic entrypoint for DrivAer_Star completed runs.

Usage:
  1) Set Gemini key in `frameworks/.env` (GOOGLE_API_KEY=...).
  2) Edit the USER CONFIG section below.
  3) Run: python3 diagnostic_suite/run/run_drivaer_diagnostics.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from diagnostic_suite.config import DRIVAER_EXPECTED_IMAGE_SUFFIXES
from diagnostic_suite.integration_helper import run_drivaer_diagnostics


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict JSON in {path}, got {type(data).__name__}")
    return data


def _collect_expected_flow_images(case_dir: str) -> List[str]:
    """
    DrivAer_Star environment saves render images under:
      <case_dir>/save/sol/<suffix>

    where suffixes are:
      Pressure_iso.png, ..., WSSx_side.png
    """
    sol_dir = os.path.join(case_dir, "save", "sol")
    images: List[str] = []
    for suffix in DRIVAER_EXPECTED_IMAGE_SUFFIXES:
        p = os.path.join(sol_dir, suffix)
        if os.path.exists(p):
            images.append(p)
    return images


def _metrics_from_results_json(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    diagnostics expects:
      drag, Cd, lift, drag_pressure, drag_shear
    DrivAer_Star reward typically writes:
      drag, Cd, lift, drag_pressure, drag_shear
    """
    # Be forgiving: allow alternate naming if present.
    def g(*keys: str, default: Any = None) -> Any:
        for k in keys:
            if k in results:
                return results[k]
        return default

    return {
        "drag": g("drag", default=0.0),
        "Cd": g("Cd", default=0.0),
        "lift": g("lift", default=0.0),
        "drag_pressure": g("drag_pressure", default=g("drag_p", default=0.0)),
        "drag_shear": g("drag_shear", default=g("drag_w", default=0.0)),
    }


def main() -> None:
    # Resolve repo root so relative paths work when run from anywhere.
    repo_root = Path(__file__).resolve().parents[2]

    # Load shared project env file (Gemini API key).
    load_dotenv(repo_root / "frameworks" / ".env")

    # =========================
    # USER CONFIG (EDIT THESE)
    # =========================
    CASE_DIR = "/ABS/PATH/TO/RUN_CASE_DIR"  # e.g. ".../environments/DrivAer_Star/results/run_x/reward_y/design_17"
    DESIGN_PATH = "/ABS/PATH/TO/design_17.json"  # the design JSON with the 20 DrivAer params (required by deterministic checks)

    # Optional: provide a final geometry figure so the LLM can look for “weird rear / bumpy front”.
    FINAL_SHAPE_IMAGE_PATH: Optional[str] = "/ABS/PATH/TO/final_shape_comparison.png"

    # Optional: explicitly pass task setup / constraints to help the LLM interpret visuals.
    # Keep empty if you don't have it.
    PROBLEM_SETUP: Dict[str, Any] = {
        # Example:
        # "objective": {"name": "minimize Cd", "reward": "-Cd"},
        # "constraints": {"hard": ["param bounds", "required artifacts present"]},
        # "param_bounds": {"car_size": [0.8, 1.2]},
        # "operating_conditions": {"body_style": "F"},
    }

    # ==================================
    # END USER CONFIG
    # ==================================

    if not os.path.exists(CASE_DIR):
        raise FileNotFoundError(f"CASE_DIR not found: {CASE_DIR}")
    if not os.path.exists(DESIGN_PATH):
        raise FileNotFoundError(f"DESIGN_PATH not found: {DESIGN_PATH}")

    results_path = os.path.join(CASE_DIR, "save", "results.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"Expected DrivAer results.json at: {results_path}\n"
            f"If your run saved metrics elsewhere, update this script."
        )

    results = _load_json(results_path)
    metrics = _metrics_from_results_json(results)

    # Include canonical flow evidence images + optional final shape plot.
    images = _collect_expected_flow_images(CASE_DIR)
    if FINAL_SHAPE_IMAGE_PATH and os.path.exists(FINAL_SHAPE_IMAGE_PATH):
        images.append(FINAL_SHAPE_IMAGE_PATH)

    model_artifacts = {
        # Required by feasibility checks; best source is the design JSON's vtk_path.
        # If your design JSON doesn't include vtk_path, set BASE_VTK_PATH explicitly here.
        "base_vtk_path": results.get("vtk_path", None),
        "norm_stats_path": None,
    }

    # If design JSON includes vtk_path, use it.
    try:
        design_params = _load_json(DESIGN_PATH)
        if isinstance(design_params.get("vtk_path"), str):
            model_artifacts["base_vtk_path"] = design_params["vtk_path"]
    except Exception:
        # We'll just fall back to defaults below if vtk_path is missing.
        pass

    # Norm stats default (exists in repo).
    default_norm_stats = os.path.join(
        str(repo_root / "environments" / "DrivAer_Star" / "model" / "norm_stats.pt")
    )
    model_artifacts["norm_stats_path"] = default_norm_stats

    out, written = run_drivaer_diagnostics(
        design_path=DESIGN_PATH,
        case_dir=CASE_DIR,
        metrics=metrics,
        images=images,
        model_artifacts=model_artifacts,
        problem_setup=PROBLEM_SETUP,
    )

    print(f"diagnostics.json written to: {written}")
    print(f"llm diagnostic_status: {out.llm_report.diagnostic_status}")
    print(f"overall_assessment: {out.llm_report.overall_assessment}")


if __name__ == "__main__":
    main()

