# Base prompts and context formatting for airfoil design

from typing import List, Dict

CONTEXT_FORMAT = """# Airfoil Design Context

## Parameter Space
Each airfoil is defined by N control points, where each point has 3 parameters:
- radius_param: Controls radial distance from center. Range: [-1.0, 1.0]
- angle_param: Controls angular offset from default position. Range: [-1.0, 1.0]  
- edgy_param: Controls corner sharpness (0=smooth, 1=sharp). Range: [-1.0, 1.0]

Action vector format: [r0, a0, e0, r1, a1, e1, ..., rN, aN, eN] (flat array of 3*N values)

## Objective
Maximize the reward.

## Previous Designs
{design_history}

Note: If images are provided, they show a representative design sampled from the current population (listed as Design 1 for reference). Use these images to understand the mapping between action parameters and resulting geometry/aerodynamics, but generate a NEW design to advance the population, not a modification of this one.
"""

DESIGN_ENTRY = """Design {idx}:
  - Action: {action_vector}
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
    """Format design history for prompts."""
    if not context:
        return CONTEXT_FORMAT.format(design_history="No previous designs available.")
    
    history_lines = []
    for i, item in enumerate(context):
        entry = DESIGN_ENTRY.format(
            idx=i+1,
            action_vector=item.get('vector', []),
            reward=item.get('reward', 0.0),
            rank=item.get('ranking', 'N/A')
        )
        history_lines.append(entry)
    
    return CONTEXT_FORMAT.format(design_history='\n'.join(history_lines))


def format_response_instructions(example_json: str) -> str:
    """Format response instructions with example."""
    return RESPONSE_FORMAT.format(example_json=example_json)