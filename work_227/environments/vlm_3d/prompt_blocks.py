# Environment-specific prompt blocks for 3D delta wing design.
# Moved as-is from modified_env3d/LLM_Actions/prompts/base_3d.py

from typing import List, Dict

CONTEXT_FORMAT = """# Delta Wing Design Context

## Parameter Space
Each delta wing is defined by the following design parameters:

| Parameter      | Type       | Range / Options                     |
|--------------- |----------- |-------------------------------------|
| le_sweep       | continuous | 45.0 – 80.0 deg                    |
| root_chord_in  | continuous | 10.0 – 50.0 inches                 |
| twist_root     | continuous | -10.0 – 10.0 deg                   |
| twist_tip      | continuous | -10.0 – 10.0 deg (washout < 0)     |
| dihedral       | continuous | -15.0 – 15.0 deg (tips up > 0)     |
| naca_m         | discrete   | 0, 2, 4  (max camber %chord)       |
| naca_p         | discrete   | 0, 4  (camber pos, must be 0 if m=0)|
| naca_t         | integer    | 6 – 24  (thickness %chord)         |

Hardcoded (delta wing identity): taper = 0, tip chord = 0, symmetric wing.

## Objective
Maximize the reward (CL/CDi relative to baseline).

## Previous Designs
{design_history}

Note: If images are provided, they show the Cp distribution and results table for a representative design sampled from the current population (listed as Design 1 for reference). Use these images to understand the mapping between design parameters and resulting aerodynamics, but generate a NEW design to advance the population, not a modification of this one.
"""

DESIGN_ENTRY = """Design {idx}:
  - Parameters: le_sweep={le_sweep}, root_chord_in={root_chord_in}, twist_root={twist_root}, twist_tip={twist_tip}, dihedral={dihedral}, NACA {naca_m}{naca_p}{naca_t:02d}
  - Reward: {reward:.4f}
  - Rank: {rank}
"""

RESPONSE_FORMAT = """
You MUST respond with ONLY a valid JSON object (no markdown, no schema definitions).
Do NOT include $schema, title, description, type, properties, or required fields.

Example response format:
{example_json}
"""


def format_context(context: List[Dict]) -> str:
    if not context:
        return CONTEXT_FORMAT.format(design_history="No previous designs available.")

    history_lines = []
    for i, item in enumerate(context):
        p = item.get('params', {})
        entry = DESIGN_ENTRY.format(
            idx=i + 1,
            le_sweep=p.get('le_sweep', '?'),
            root_chord_in=p.get('root_chord_in', '?'),
            twist_root=p.get('twist_root', '?'),
            twist_tip=p.get('twist_tip', '?'),
            dihedral=p.get('dihedral', '?'),
            naca_m=p.get('naca_m', 0),
            naca_p=p.get('naca_p', 0),
            naca_t=p.get('naca_t', 12),
            reward=item.get('reward', 0.0),
            rank=item.get('ranking', 'N/A'),
        )
        history_lines.append(entry)

    return CONTEXT_FORMAT.format(design_history='\n'.join(history_lines))


def format_response_instructions(example_json: str) -> str:
    return RESPONSE_FORMAT.format(example_json=example_json)
