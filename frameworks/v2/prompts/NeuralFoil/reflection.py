# Reflection and scratchpad update prompts for NeuralFoil (Kulfan CST parameterization)

REFLECTION_SYSTEM = """You critically analyze airfoil design outcomes to build understanding of the Kulfan (CST) parameter space. The goal is to maximize aerodynamic reward using NeuralFoil."""

REFLECTION_USER = """## Objective
Optimize a 2D airfoil to maximize aerodynamic reward. The airfoil is parameterized with 18 Kulfan (CST) coefficients:
  - upper_weights[0..7]: upper surface shape, LE→TE, controls camber and thickness distribution
  - lower_weights[0..7]: lower surface shape, LE→TE, negative values give concave lower surface
  - leading_edge_weight: leading edge radius; larger = rounder nose
  - TE_thickness: trailing edge thickness as fraction of chord

## Designer's Predictions
{designer_reasoning}

{designer_analysis}

## Parameters Used
```json
{intended_params}
```

## Actual Parameters in Design File
```json
{actual_action}
```

## Result (attached airfoil shape image)

## Task
For each Kulfan parameter the designer set, compare their stated prediction vs what is
visible in the airfoil shape image. Then summarize key learnings about how these parameters
map to geometry.
"""

SCRATCHPAD_UPDATE_SYSTEM = """You maintain an evolving reference card mapping every Kulfan (CST) parameter to its observed effect on airfoil geometry and aerodynamics. The goal is NeuralFoil-based airfoil optimization to maximize reward."""

SCRATCHPAD_UPDATE_USER = """## Objective
Maximize aerodynamic reward. Airfoil is parameterized with 18 Kulfan CST coefficients.

## Current Reference Card
{current_scratchpad}

## New Evidence (Iteration {iteration})
Parameters used:
```json
{intended_params}
```

Observations:
{reflection_text}

## Task
Synthesize the new evidence into the reference card.
Include an overall rules section and an entry for each Kulfan parameter group
(upper_weights by index, lower_weights by index, leading_edge_weight, TE_thickness).
Cite [iter N]. Be terse.
"""


def build_reflection_prompt(intended_params, actual_action, designer_analysis, designer_reasoning):
    import json
    return REFLECTION_USER.format(
        designer_reasoning=designer_reasoning or "(no predictions provided)",
        designer_analysis=designer_analysis or "(no analysis provided)",
        intended_params=json.dumps(intended_params, indent=2),
        actual_action=json.dumps(actual_action, indent=2) if isinstance(actual_action, dict)
                      else str(actual_action),
    )


def build_scratchpad_update_prompt(current_scratchpad, reflection_text, intended_params, iteration):
    import json
    if not current_scratchpad.strip():
        current_scratchpad = "(Empty - first iteration)"
    return SCRATCHPAD_UPDATE_USER.format(
        current_scratchpad=current_scratchpad,
        iteration=iteration,
        intended_params=json.dumps(intended_params, indent=2),
        reflection_text=reflection_text,
    )
