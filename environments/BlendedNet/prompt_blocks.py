from typing import List, Dict

CONTEXT_FORMAT = """# Blended Wing Body Design Context

## Parameter Space
Each blended-wing-body geometry is defined by 9 planform parameters plus 3 flight conditions:

| Parameter | Type       | Description                          |
|-----------|------------|--------------------------------------|
| B1        | continuous | Span section 1                       |
| B2        | continuous | Span section 2                       |
| B3        | continuous | Span section 3                       |
| C2        | continuous | Chord section 2 (C1 fixed at 1000)   |
| C3        | continuous | Chord section 3                      |
| C4        | continuous | Chord section 4                      |
| S1        | continuous | Sweep section 1                      |
| S2        | continuous | Sweep section 2                      |
| S3        | continuous | Sweep section 3                      |
| Re        | continuous | Reynolds number                      |
| Mach      | continuous | Freestream Mach number               |
| alpha     | continuous | Angle of attack (degrees)            |

## Objective
Maximize the surrogate-predicted lift-to-drag ratio (L/D).

## Previous Designs
{design_history}
"""

DESIGN_ENTRY = """Design {idx}:
  - Planform: B1={B1}, B2={B2}, B3={B3}, C2={C2}, C3={C3}, C4={C4}, S1={S1}, S2={S2}, S3={S3}
  - Flight: Re={Re}, Mach={Mach}, alpha={alpha}
  - Reward: {reward:.4f}
  - Rank: {rank}
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
        entry = DESIGN_ENTRY.format(
            idx=i + 1,
            B1=p.get("B1", "?"), B2=p.get("B2", "?"), B3=p.get("B3", "?"),
            C2=p.get("C2", "?"), C3=p.get("C3", "?"), C4=p.get("C4", "?"),
            S1=p.get("S1", "?"), S2=p.get("S2", "?"), S3=p.get("S3", "?"),
            Re=p.get("Re", "?"), Mach=p.get("Mach", "?"), alpha=p.get("alpha", "?"),
            reward=item.get("reward", 0.0),
            rank=item.get("ranking", "N/A"),
        )
        history_lines.append(entry)

    return CONTEXT_FORMAT.format(design_history="\n".join(history_lines))


def format_response_instructions(example_json: str) -> str:
    return RESPONSE_FORMAT.format(example_json=example_json)
