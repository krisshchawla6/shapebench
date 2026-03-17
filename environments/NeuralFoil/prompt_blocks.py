"""Prompt blocks for 2D airfoil optimization with NeuralFoil (Kulfan parameterization)."""

from typing import List, Dict

CONTEXT_FORMAT = """# 2D Airfoil Design Context — NeuralFoil

## Parameter Space
Each airfoil is defined by 18 Kulfan (CST) parameters. The CST parameterization expresses the
airfoil shape as a weighted sum of Bernstein basis polynomials; coefficients closer to the leading
edge affect the nose shape while coefficients closer to the trailing edge affect aft camber/thickness.

| Parameter            | Type       | Typical Range     | Aerodynamic Meaning                                |
|----------------------|------------|-------------------|----------------------------------------------------|
| upper_weights[0..7]  | continuous | -0.30 – 0.60      | Upper surface shape (LE → TE); controls camber/thickness distribution |
| lower_weights[0..7]  | continuous | -0.30 – 0.30      | Lower surface shape (LE → TE); negative values give concave lower surface |
| leading_edge_weight  | continuous | -0.50 – 0.50      | Leading edge radius modification; larger = rounder nose |
| TE_thickness         | continuous |  0.000 – 0.010    | Trailing edge thickness as fraction of chord       |

Conventions:
  - A symmetric airfoil has upper_weights = -lower_weights and leading_edge_weight = 0.
  - Positive camber (lift) requires upper_weights > |lower_weights| on average.
  - Thick airfoils have large upper and lower weights with similar magnitude.
  - NACA 4412 approximate Kulfan: upper ≈ [0.17,0.12,0.10,0.08,0.07,0.06,0.05,0.04],
    lower ≈ [-0.10,-0.07,-0.05,-0.04,-0.03,-0.02,-0.01,-0.005], LE ≈ 0.0, TE ≈ 0.002

{action_scratchpad}
## Objective
{objective}

## Previous Designs
{design_history}

Note: If images are provided, they show the airfoil shape and Cp (pressure) distribution for a
representative design. Use these to understand the mapping between parameters and aerodynamics,
but generate a NEW design to advance the population.
"""

DESIGN_ENTRY = """Design {idx}:
  - upper_weights: {upper_weights}
  - lower_weights: {lower_weights}
  - leading_edge_weight: {leading_edge_weight:.4f}   TE_thickness: {TE_thickness:.5f}
  - CL: {CL:.4f}   CD: {CD:.6f}   CM: {CM:.4f}   confidence: {confidence:.2f}
  - Reward: {reward:.4f}   Rank: {rank}
"""

RESPONSE_FORMAT = """
You MUST respond with ONLY a valid JSON object (no markdown, no schema definitions).
Do NOT include $schema, title, description, type, properties, or required fields.

Example response format:
{example_json}
"""

_DEFAULT_OBJECTIVE = "Maximize the reward (defined by the selected reward function)."


def format_context(context: List[Dict], objective: str = _DEFAULT_OBJECTIVE,
                   scratchpad: str = "") -> str:
    if not context:
        design_history = "No previous designs available."
    else:
        history_lines = []
        for i, item in enumerate(context):
            p = item.get("params", {})
            m = item.get("metrics", {})

            uw = p.get("upper_weights", [])
            lw = p.get("lower_weights", [])
            uw_str = "[" + ", ".join(f"{v:.4f}" for v in uw) + "]"
            lw_str = "[" + ", ".join(f"{v:.4f}" for v in lw) + "]"

            entry = DESIGN_ENTRY.format(
                idx=i + 1,
                upper_weights=uw_str,
                lower_weights=lw_str,
                leading_edge_weight=float(p.get("leading_edge_weight", 0.0)),
                TE_thickness=float(p.get("TE_thickness", 0.0)),
                CL=float(m.get("CL", 0.0)),
                CD=float(m.get("CD", 0.0)),
                CM=float(m.get("CM", 0.0)),
                confidence=float(m.get("analysis_confidence", 0.0)),
                reward=float(item.get("reward", 0.0)),
                rank=item.get("ranking", "N/A"),
            )
            history_lines.append(entry)
        design_history = "\n".join(history_lines)

    if scratchpad and scratchpad.strip():
        scratchpad_block = (
            "## Parameter-Geometry Knowledge (from previous reflections)\n"
            + scratchpad.strip()
            + "\n\n"
        )
    else:
        scratchpad_block = ""

    return CONTEXT_FORMAT.format(
        objective=objective,
        design_history=design_history,
        action_scratchpad=scratchpad_block,
    )


def format_response_instructions(example_json: str) -> str:
    return RESPONSE_FORMAT.format(example_json=example_json)
