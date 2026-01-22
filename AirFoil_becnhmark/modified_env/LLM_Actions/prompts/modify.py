# Modify action prompts (with bounds for sampling)

from .base import format_response_instructions

MODIFY_SYSTEM = """You are an evolutionary optimizer for aerodynamic shape optimization.
Your goal is to mutate existing designs by modifying specific control points.
Make targeted changes to improve the reward based on performance feedback."""

MODIFY_USER = """{context}

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
- Identify which control points most affect lift and drag
- Make targeted changes to promising points
- Consider what made previous successful designs work
- Preserve aspects of the design that are working well

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
