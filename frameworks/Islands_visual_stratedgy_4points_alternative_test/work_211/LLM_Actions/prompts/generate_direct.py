# Generate Direct action prompts (exact values)
# Restructured following Shinka's best practices

from typing import Optional, Tuple
from .base import format_response_instructions

# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

GENERATE_DIRECT_SYSTEM = """You are an expert aerodynamicist and optimization specialist with deep knowledge of airfoil design.
Your task is to analyze previous airfoil designs and propose a design that MAXIMIZES the reward.

Key capabilities:
- Understanding of aerodynamic principles (lift, drag, flow separation)
- Shape optimization intuition (leading edge, trailing edge, camber, thickness)
- Parameter space exploration strategies
- Trade-off analysis between competing objectives

You will specify EXACT parameter values. Be bold - major changes may be needed."""


# =============================================================================
# PROMPT VARIANTS
# =============================================================================

GENERATE_DIRECT_STRATEGY_NEW_DESIGN = """Your goal: MAXIMIZE the reward by exploring the FULL parameter space.

Previous designs show what HAS been tried. Your task is to explore DIFFERENT regions.

**Critical**: The parameter space is [-1.0, 1.0] for each value. Don't just stay in the same region as previous designs - explore the full range!"""

# List of all strategy variants
GENERATE_DIRECT_STRATEGIES = [
    GENERATE_DIRECT_STRATEGY_NEW_DESIGN,
]

GENERATE_DIRECT_STRATEGY_NAMES = [
    "new_design",
]


# =============================================================================
# MAIN USER PROMPT TEMPLATE
# =============================================================================

GENERATE_DIRECT_USER = """# Context: Airfoil Design Optimization

{context}

## Task

{strategy}

First, analyze the provided information:

1. **Image Analysis**: If images are provided, carefully analyze the parent design's geometry and flow fields (pressure, velocity). Identify aerodynamic features such as flow separation, pressure gradients, and wake patterns.

2. **Design Reasoning**: Based on the context and design database, think step-by-step about what design features need to be adjusted to improve the reward. Consider what worked well in high-performing designs and what issues need to be addressed.

Then, propose a COMPLETELY NEW airfoil design with exact parameter values for each control point.

**CRITICAL: You MUST use exactly 8 control points (n_cp=8). Do not use any other number.**

## Output Format

{response_format}

**Response Format:**
Provide your response in the following structure:

## Analysis
[Your analysis of the parent design's images and flow characteristics]

## Reasoning
[Step-by-step reasoning about what design features to adjust and why]

## Design Parameters
[Your JSON with the design parameters as shown in the example above]

Critical: All parameter values must be in the range [-1.0, 1.0]."""


# =============================================================================
# EXAMPLE FORMAT
# =============================================================================

GENERATE_DIRECT_EXAMPLE = """{
  "n_cp": 8,
  "n_sp": 10,
  "params": [
    [<r0>, <a0>, <e0>],
    [<r1>, <a1>, <e1>],
    [<r2>, <a2>, <e2>],
    [<r3>, <a3>, <e3>],
    [<r4>, <a4>, <e4>],
    [<r5>, <a5>, <e5>],
    [<r6>, <a6>, <e6>],
    [<r7>, <a7>, <e7>]
  ],
  "name": "<descriptive_name>"
}

Replace all <placeholders> with actual float values in [-1.0, 1.0].
Each inner array is [radius_param, angle_param, edgy_param] for that control point."""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_generate_direct_prompt(context_str: str, strategy_idx: Optional[int] = None) -> str:
    """Build the full generate_direct prompt with optional strategy variant.
    
    Args:
        context_str: Formatted context with design history
        strategy_idx: Optional strategy index (0-2). If None, uses default (0).
    
    Returns:
        Complete formatted user prompt
    """
    # Select strategy (default to precise optimization)
    if strategy_idx is None:
        strategy_idx = 0
    strategy_idx = strategy_idx % len(GENERATE_DIRECT_STRATEGIES)
    strategy = GENERATE_DIRECT_STRATEGIES[strategy_idx]
    
    response_fmt = format_response_instructions(GENERATE_DIRECT_EXAMPLE)
    
    return GENERATE_DIRECT_USER.format(
        context=context_str,
        strategy=strategy,
        response_format=response_fmt
    )


def get_generate_direct_system(strategy_idx: Optional[int] = None) -> str:
    """Get system prompt for generate_direct action."""
    return GENERATE_DIRECT_SYSTEM


def sample_direct_strategy() -> Tuple[int, str]:
    """Randomly sample a strategy variant for generate_direct."""
    import random
    idx = random.randint(0, len(GENERATE_DIRECT_STRATEGIES) - 1)
    return idx, GENERATE_DIRECT_STRATEGY_NAMES[idx]
