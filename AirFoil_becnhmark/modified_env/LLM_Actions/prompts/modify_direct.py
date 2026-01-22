# Modify Direct action prompts (exact values)

from .base import format_response_instructions

MODIFY_DIRECT_SYSTEM = """You are an evolutionary optimizer for aerodynamic shape optimization.
Your goal is to mutate existing designs by modifying specific control points with exact values.
Make precise, targeted changes to improve the reward based on performance feedback."""

MODIFY_DIRECT_USER = """{context}

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
- Analyze the current design's performance and identify weak points
- Make targeted improvements based on aerodynamic principles
- Preserve successful aspects while improving underperforming areas
- Consider incremental changes to avoid breaking working geometry

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
