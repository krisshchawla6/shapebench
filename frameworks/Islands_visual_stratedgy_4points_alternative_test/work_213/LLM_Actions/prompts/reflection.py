# Reflection and scratchpad update prompts

REFLECTION_SYSTEM = """You critically analyze design outcomes to build understanding of the parameter space. The goal is to optimize an airfoil shape that maximizes reward (aerodynamic performance)."""

REFLECTION_USER = """## Objective
Optimize an airfoil shape to maximize aerodynamic performance (reward).

## Designer's Predictions
{designer_reasoning}

{designer_analysis}

## Parameters Used
```json
{intended_params}
```

Sampled action vector:
{actual_action}

## Result (attached image)

## Task
For each individual parameter value, compare prediction vs outcome.
Then summarize key learnings.
"""

SCRATCHPAD_UPDATE_SYSTEM = """You maintain an evolving reference card that maps every degree of freedom to its observed geometric effect. The objective is airfoil optimization to maximize aerodynamic performance."""

SCRATCHPAD_UPDATE_USER = """## Objective
Optimize an airfoil shape to maximize aerodynamic performance (reward).

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
Include an overall rules section and an entry for each individual degree of freedom.
Cite [iter N]. Be terse.
"""


def build_reflection_prompt(intended_params, actual_action, designer_analysis, designer_reasoning):
    """Build reflection prompt from designer's own output."""
    import json
    return REFLECTION_USER.format(
        designer_reasoning=designer_reasoning or "(no predictions provided)",
        designer_analysis=designer_analysis or "(no analysis provided)",
        intended_params=json.dumps(intended_params, indent=2),
        actual_action=str([f"{v:.4f}" for v in actual_action])
    )


def build_scratchpad_update_prompt(current_scratchpad, reflection_text, intended_params, iteration):
    """Build scratchpad update prompt."""
    import json
    if not current_scratchpad.strip():
        current_scratchpad = "(Empty - first iteration)"
    return SCRATCHPAD_UPDATE_USER.format(
        current_scratchpad=current_scratchpad,
        iteration=iteration,
        intended_params=json.dumps(intended_params, indent=2),
        reflection_text=reflection_text
    )
