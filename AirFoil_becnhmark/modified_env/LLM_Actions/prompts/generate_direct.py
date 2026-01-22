# Generate Direct action prompts (exact values)

from .base import format_response_instructions

GENERATE_DIRECT_SYSTEM = """You are an evolutionary optimizer for aerodynamic shape optimization.
Your goal is to generate diverse, novel designs by specifying EXACT parameter values.
Improve the reward based on previous results. Explore the design space creatively."""

GENERATE_DIRECT_USER = """{context}

# Task: Generate New Airfoil Design (Exact Values)

Create a new airfoil by specifying exact [radius_param, angle_param, edgy_param] for each control point.
All values must be in the range [-1.0, 1.0].

Design considerations:
- Asymmetric shapes often generate more lift
- Sharp trailing edges can reduce drag  
- Smooth leading edges improve flow attachment
- Larger radius values create more pronounced deformations
- Analyze patterns from high-performing previous designs

{response_format}
"""

GENERATE_DIRECT_EXAMPLE = """{
  "n_cp": 4,
  "n_sp": 10,
  "params": [
    [<r0>, <a0>, <e0>],
    [<r1>, <a1>, <e1>],
    [<r2>, <a2>, <e2>],
    [<r3>, <a3>, <e3>]
  ],
  "name": "<descriptive_name>"
}

Replace all <placeholders> with actual float values in [-1.0, 1.0]."""


def get_generate_direct_prompt(context_str: str) -> str:
    """Build the full generate_direct prompt."""
    response_fmt = format_response_instructions(GENERATE_DIRECT_EXAMPLE)
    return GENERATE_DIRECT_USER.format(context=context_str, response_format=response_fmt)
