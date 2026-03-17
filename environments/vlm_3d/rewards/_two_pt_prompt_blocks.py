"""Prompt blocks for the two-point delta wing evaluation (Yiren et al.)."""

from typing import List, Dict

CONTEXT_FORMAT = """# Delta Wing Design Context — Two-Point Evaluation

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
Minimize supersonic drag (CDi at Mach 1.8) while satisfying:
  - CL = 0.1665 at supersonic (Mach 1.8, Re = 80.4e6)
  - CL = 0.6933 at subsonic   (Mach 0.3,  Re = 101.8e6)
  - CM = 0 at both conditions
  - Static margin Kn >= 5% at the subsonic condition

The reward combines -CDi_sup with penalty terms for constraint violations.
A higher (less negative) reward means lower drag and better constraint satisfaction.

## Previous Designs
{design_history}

Note: If images are provided, they show the Cp distribution for a representative design at both supersonic and subsonic conditions. Use these to understand the mapping between design parameters and resulting aerodynamics, but generate a NEW design to advance the population.
"""

DESIGN_ENTRY = """Design {idx}:
  - Parameters: le_sweep={le_sweep}, root_chord_in={root_chord_in}, twist_root={twist_root}, twist_tip={twist_tip}, dihedral={dihedral}, NACA {naca_m}{naca_p}{naca_t:02d}
  - Reward: {reward:.4f}
  - Supersonic: CL={CL_sup:.4f}  CDi={CDi_sup:.5f}  CM={CM_sup:.4f}
  - Subsonic:   CL={CL_sub:.4f}  CDi={CDi_sub:.5f}  CM={CM_sub:.4f}  Kn={Kn:.4f}
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
        m = item.get('metrics', {})
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
            CL_sup=m.get('CL_sup', 0.0),
            CDi_sup=m.get('CDi_sup', 0.0),
            CM_sup=m.get('CM_sup', 0.0),
            CL_sub=m.get('CL_sub', 0.0),
            CDi_sub=m.get('CDi_sub', 0.0),
            CM_sub=m.get('CM_sub', 0.0),
            Kn=m.get('Kn', 0.0),
            rank=item.get('ranking', 'N/A'),
        )
        history_lines.append(entry)

    return CONTEXT_FORMAT.format(design_history='\n'.join(history_lines))


def format_response_instructions(example_json: str) -> str:
    return RESPONSE_FORMAT.format(example_json=example_json)
