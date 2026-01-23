# Modify action prompts (with bounds for sampling)

from .base import format_response_instructions

MODIFY_SYSTEM = """You are an evolutionary optimizer for aerodynamic shape optimization.
Your goal is to mutate existing designs by modifying specific control points to MAXIMIZE reward.
The current designs have LOW rewards. You must make bold, targeted changes to SIGNIFICANTLY INCREASE the reward.
Do not be conservative - explore parameter changes that could lead to major improvements."""

MODIFY_USER = """{context}

## Visual Reference

If an image is provided, it shows the **parent design's geometry** - the exact shape you are about to modify. Use this visual reference to:
- Identify which control points affect which regions of the shape
- Understand how the current geometry produces the observed lift/drag
- Target specific geometric features for improvement

# Task: Modify Existing Airfoil Design (with Sampling Bounds)

You are modifying an existing airfoil to improve its performance.

Current action vector to modify:
```
{base_csv_content}
```
Base CSV path: {base_csv_path}

Specify which control points to modify (pt_idx) and provide [min, max] bounds for sampling.
Values will be uniformly sampled between the bounds you provide.

Modification strategy:
- The current reward is LOW - you need to make changes that INCREASE it significantly
- Identify which control points most affect lift and drag
- Make bold changes - small tweaks won't achieve meaningful improvement
- Consider what made previous better designs work (but current designs are not good enough yet)
- Don't preserve parameters just because reward is positive - positive but low is still bad

{response_format}
"""

MODIFY_EXAMPLE = """{
  "pt_idx": [<idx1>, <idx2>, ...],
  "values": [
    [[<r_min>, <r_max>], [<a_min>, <a_max>], [<e_min>, <e_max>]],
    [[<r_min>, <r_max>], [<a_min>, <a_max>], [<e_min>, <e_max>]],
    ...
  ],
  "name": "<descriptive_name>"
}

Replace <idx> with integer indices (0 to n_cp-1).
Replace other <placeholders> with actual float values in [-1.0, 1.0].
Number of entries in values must match length of pt_idx."""


def get_modify_prompt(context_str: str, base_csv_path: str, base_csv_content: str) -> str:
    """Build the full modify prompt."""
    response_fmt = format_response_instructions(MODIFY_EXAMPLE)
    return MODIFY_USER.format(
        context=context_str,
        base_csv_path=base_csv_path,
        base_csv_content=base_csv_content,
        response_format=response_fmt
    )
