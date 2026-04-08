"""Prompt blocks for the HPA multipoint optimization reward."""

from typing import List, Dict

CONTEXT_FORMAT = """# 2D Airfoil Design Context — HPA Multipoint Optimization

## Parameter Space
Each airfoil is defined by 17 Kulfan (CST) parameters. TE_thickness is fixed at 0 (sharp TE).

| Parameter            | Type       | Bounds            | Aerodynamic Meaning                                     |
|----------------------|------------|-------------------|---------------------------------------------------------|
| upper_weights[0..7]  | continuous | -0.30 – 0.60      | Upper surface shape (LE → TE); controls camber and thickness |
| lower_weights[0..7]  | continuous | -0.30 – 0.30      | Lower surface shape; negative = concave (typical for lift) |
| leading_edge_weight  | continuous | -0.50 – 0.50      | Leading edge radius; larger = rounder nose              |

## Geometric Constraints (penalized in fitness)
  - local_thickness > 0 everywhere  (no self-intersecting shapes)
  - upper_weights[0] > 0.05        (blunt upper LE — required for attached flow)
  - lower_weights[0] < -0.05       (sharp lower LE)
  - local_thickness(x=0.33) ≥ 0.128  (spar thickness for structural depth)
  - local_thickness(x=0.90) ≥ 0.014  (thin TE region minimum)
  - TE_angle ≥ 6.03°               (trailing edge wedge angle)
  - wiggliness < 2 × wiggliness(NACA0012)  (smooth surface)

## Aerodynamic Constraints (penalized in fitness)
  - CM ≥ -0.133 at all 6 operating points
  - analysis_confidence > 0.90 at all 6 operating points
  - alpha increases monotonically with CL  (stable aerodynamic behavior)

{action_scratchpad}
## Objective
Minimize the weighted-mean drag over 6 human-powered-aircraft lift targets:

  CL targets : [0.8, 1.0, 1.2, 1.4, 1.5, 1.6]
  Weights    : [5,   6,   7,   8,   9,  10  ]
  Re schedule: Re = 500k × (CL / 1.25)^-0.5  (varies per point)

  Fitness = −mean(CD × weights) − Σ(lambda_i × normalized_violation_i)
  with the objective term active when all 6 CL targets are solved.
  Higher reward = lower weighted drag = better HPA efficiency.

Key design insight: high-lift sections (CL 1.4–1.6) require significant camber.
A well-cambered upper surface with controlled aft loading minimizes drag across the range.

## Previous Designs
{design_history}

Note: Images show representative airfoil shape at a solved operating condition.
Generate a NEW design to advance the population toward lower weighted drag.
"""

DESIGN_ENTRY = """Design {idx}  [Reward: {reward:.4f}  Rank: {rank}]
  upper_weights: {upper_weights}
  lower_weights: {lower_weights}
  leading_edge_weight: {leading_edge_weight:.4f}
  weighted_CD_mean: {weighted_cd:.6f}
  CDs @ CL[0.8→1.6]: {cds}
  CMs @ CL[0.8→1.6]: {cms}
  confs            : {confs}
  alphas           : {alphas}
"""

RESPONSE_FORMAT = """
You MUST respond with ONLY a valid JSON object (no markdown, no schema definitions).
Do NOT include $schema, title, description, type, properties, or required fields.
TE_thickness must NOT be included — it is fixed at 0.
ALL floats MUST have exactly 8 decimal places (e.g., 0.98521047 not 0.9852).

Example response format:
{example_json}
"""


def format_context(context: List[Dict], scratchpad: str = "") -> str:
    if scratchpad and scratchpad.strip():
        scratchpad_block = (
            "## Parameter-Geometry Knowledge (from previous reflections)\n"
            + scratchpad.strip() + "\n\n"
        )
    else:
        scratchpad_block = ""

    if not context:
        return CONTEXT_FORMAT.format(
            design_history="No previous designs available.",
            action_scratchpad=scratchpad_block,
        )

    history_lines = []
    for i, item in enumerate(context):
        p = item.get("params", {})
        m = item.get("metrics", {})

        uw = p.get("upper_weights", [])
        lw = p.get("lower_weights", [])
        uw_str = "[" + ", ".join(f"{v:.4f}" for v in uw) + "]"
        lw_str = "[" + ", ".join(f"{v:.4f}" for v in lw) + "]"

        cds = m.get("CDs", [])
        cms = m.get("CMs", [])
        confs = m.get("analysis_confidences", [])
        alphas = m.get("alphas", [])

        cds_str = "[" + ", ".join(f"{v:.5f}" for v in cds) + "]" if cds else "n/a"
        cms_str = "[" + ", ".join(f"{v:.4f}" for v in cms) + "]" if cms else "n/a"
        confs_str = "[" + ", ".join(f"{v:.2f}" for v in confs) + "]" if confs else "n/a"
        alphas_str = "[" + ", ".join(f"{v:.2f}" for v in alphas) + "]" if alphas else "n/a"

        entry = DESIGN_ENTRY.format(
            idx=i + 1,
            reward=float(item.get("reward", 0.0)),
            rank=item.get("ranking", "N/A"),
            upper_weights=uw_str,
            lower_weights=lw_str,
            leading_edge_weight=float(p.get("leading_edge_weight", 0.0)),
            weighted_cd=float(0.0 if m.get("weighted_CD_mean") is None else m.get("weighted_CD_mean")),
            cds=cds_str,
            cms=cms_str,
            confs=confs_str,
            alphas=alphas_str,
        )
        history_lines.append(entry)

    return CONTEXT_FORMAT.format(
        design_history="\n".join(history_lines),
        action_scratchpad=scratchpad_block,
    )


def format_response_instructions(example_json: str) -> str:
    return RESPONSE_FORMAT.format(example_json=example_json)
