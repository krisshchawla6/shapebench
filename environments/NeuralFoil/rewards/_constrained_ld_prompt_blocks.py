"""Prompt blocks for the constrained L/D (pitching moment penalty) reward."""

from typing import List, Dict

CONTEXT_FORMAT = """# 2D Airfoil Design Context — Constrained L/D (Flying-Wing / Reflex Section)

## Parameter Space
Each airfoil is defined by 18 Kulfan (CST) parameters. The CST parameterization expresses the
airfoil shape as a weighted sum of Bernstein basis polynomials.

| Parameter            | Type       | Typical Range     | Aerodynamic Meaning                                |
|----------------------|------------|-------------------|----------------------------------------------------|
| upper_weights[0..7]  | continuous | -0.30 – 0.60      | Upper surface shape (LE → TE); controls camber/thickness distribution |
| lower_weights[0..7]  | continuous | -0.30 – 0.30      | Lower surface shape; reflexed (S-shaped) TE gives CM ≈ 0 |
| leading_edge_weight  | continuous | -0.50 – 0.50      | Leading edge radius; larger = rounder nose         |
| TE_thickness         | continuous |  0.000 – 0.010    | Trailing edge thickness as fraction of chord       |

Key insight for this objective: a **reflexed** (S-shaped) camber line moves the trailing edge
upward relative to a simple arc, which reduces the nose-down (negative) pitching moment.
This is achieved by making lower_weights[5..7] more negative (concave aft lower surface)
or upper_weights[5..7] smaller (flat/reflexed upper aft section).

## Objective
Maximize L/D while keeping CM close to target (CM_target). Reward = CL/CD - w_cm*(CM - CM_target)^2.
Higher reward = higher aerodynamic efficiency AND better pitching moment trim.

## Previous Designs
{design_history}

Note: If images are provided, they show the airfoil shape and Cp distribution. Use these to
understand the relationship between shape and CM (note any S-curvature in the camber line).
Generate a NEW design to advance the population.
"""

DESIGN_ENTRY = """Design {idx}:
  - upper_weights: {upper_weights}
  - lower_weights: {lower_weights}
  - leading_edge_weight: {leading_edge_weight:.4f}   TE_thickness: {TE_thickness:.5f}
  - CL: {CL:.4f}   CD: {CD:.6f}   CM: {CM:.4f}   L/D: {LD:.2f}
  - CM_penalty: {CM_penalty:.4f}   confidence: {confidence:.2f}
  - Reward: {reward:.4f}   Rank: {rank}
"""

RESPONSE_FORMAT = """
You MUST respond with ONLY a valid JSON object (no markdown, no schema definitions).
Do NOT include $schema, title, description, type, properties, or required fields.
ALL floats MUST have exactly 8 decimal places (e.g., 0.98521047 not 0.9852).

Example response format:
{example_json}
"""


def format_context(context: List[Dict]) -> str:
    if not context:
        return CONTEXT_FORMAT.format(design_history="No previous designs available.")

    history_lines = []
    for i, item in enumerate(context):
        p = item.get("params", {})
        m = item.get("metrics", {})

        uw = p.get("upper_weights", [])
        lw = p.get("lower_weights", [])
        uw_str = "[" + ", ".join(f"{v:.4f}" for v in uw) + "]"
        lw_str = "[" + ", ".join(f"{v:.4f}" for v in lw) + "]"

        CD = float(m.get("CD", 0.0))
        CL = float(m.get("CL", 0.0))
        LD = CL / CD if CD > 1e-9 else 0.0

        entry = DESIGN_ENTRY.format(
            idx=i + 1,
            upper_weights=uw_str,
            lower_weights=lw_str,
            leading_edge_weight=float(p.get("leading_edge_weight", 0.0)),
            TE_thickness=float(p.get("TE_thickness", 0.0)),
            CL=CL, CD=CD,
            CM=float(m.get("CM", 0.0)),
            LD=LD,
            CM_penalty=float(m.get("CM_penalty", 0.0)),
            confidence=float(m.get("analysis_confidence", 0.0)),
            reward=float(item.get("reward", 0.0)),
            rank=item.get("ranking", "N/A"),
        )
        history_lines.append(entry)

    return CONTEXT_FORMAT.format(design_history="\n".join(history_lines))


def format_response_instructions(example_json: str) -> str:
    return RESPONSE_FORMAT.format(example_json=example_json)
