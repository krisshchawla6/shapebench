# Generate action prompts (with bounds for sampling)

from .base import format_response_instructions

GENERATE_SYSTEM = """You are an evolutionary optimizer for aerodynamic shape optimization.
Your goal is to generate diverse, novel designs by specifying parameter RANGES for sampling.
Improve the reward based on previous results. Explore the design space creatively."""

GENERATE_USER = """{context}

# Task: Generate New Airfoil Design (with Sampling Bounds)

Create a new airfoil by specifying [min, max] bounds for each parameter.
Values will be uniformly sampled between the bounds you provide.

Design considerations:
- Asymmetric shapes often generate more lift
- Sharp trailing edges can reduce drag  
- Smooth leading edges improve flow attachment
- Analyze patterns from high-performing previous designs

{response_format}
"""

GENERATE_EXAMPLE = """{
  "n_cp": 4,
  "n_sp": 10,
  "params": [
    [[<r0_min>, <r0_max>], [<a0_min>, <a0_max>], [<e0_min>, <e0_max>]],
    [[<r1_min>, <r1_max>], [<a1_min>, <a1_max>], [<e1_min>, <e1_max>]],
    [[<r2_min>, <r2_max>], [<a2_min>, <a2_max>], [<e2_min>, <e2_max>]],
    [[<r3_min>, <r3_max>], [<a3_min>, <a3_max>], [<e3_min>, <e3_max>]]
  ],
  "name": "<descriptive_name>"
}

Replace all <placeholders> with actual float values in [-1.0, 1.0]."""


def get_generate_prompt(context_str: str) -> str:
    """Build the full generate prompt."""
    response_fmt = format_response_instructions(GENERATE_EXAMPLE)
    return GENERATE_USER.format(context=context_str, response_format=response_fmt)
