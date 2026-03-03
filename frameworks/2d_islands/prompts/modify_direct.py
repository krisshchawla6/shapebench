# Modify Direct action prompts (exact values)

from .base import format_response_instructions

MODIFY_DIRECT_SYSTEM = """You are an evolutionary optimizer for aerodynamic shape optimization.
Your goal is to mutate existing designs by modifying specific control points to MAXIMIZE reward.
The current designs have LOW rewards. You must make bold changes to SIGNIFICANTLY INCREASE the reward.
Do not be conservative - explore parameter changes that could lead to major improvements."""

MODIFY_DIRECT_USER = """{context}

## Visual Reference

If an image is provided, it shows the **parent design's geometry** - the exact shape you are about to modify. Use this visual reference to:
- Identify which control points affect which regions of the shape
- Understand how the current geometry produces the observed lift/drag
- Target specific geometric features for improvement

# Task: Modify Existing Airfoil Design (Exact Values)

You are modifying an existing airfoil to improve its performance.

Current action vector to modify:
```
{base_csv_content}
```
Base CSV path: {base_csv_path}

Specify which control points to modify (pt_idx) and provide exact [radius_param, angle_param, edgy_param] values.
All values must be in the range [-1.0, 1.0].

Modification strategy:
- The current reward is LOW - you need to make changes that INCREASE it significantly
- Analyze the current design's performance and identify weak points
- Make BOLD changes - incremental tweaks won't achieve meaningful improvement
- Don't preserve parameters just because reward is positive - positive but low is still bad
- Target fundamental changes to improve aerodynamic efficiency

{response_format}
"""

MODIFY_DIRECT_EXAMPLE = """{
  "pt_idx": [<idx1>, <idx2>, ...],
  "values": [
    [<r>, <a>, <e>],
    [<r>, <a>, <e>],
    ...
  ],
  "name": "<descriptive_name>"
}

Replace <idx> with integer indices (0 to n_cp-1).
Replace other <placeholders> with actual float values in [-1.0, 1.0].
Number of entries in values must match length of pt_idx."""


def get_modify_direct_prompt(context_str: str, base_csv_path: str, base_csv_content: str) -> str:
    """Build the full modify_direct prompt."""
    response_fmt = format_response_instructions(MODIFY_DIRECT_EXAMPLE)
    return MODIFY_DIRECT_USER.format(
        context=context_str,
        base_csv_path=base_csv_path,
        base_csv_content=base_csv_content,
        response_format=response_fmt
    )
