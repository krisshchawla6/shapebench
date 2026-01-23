# Generate action prompts (with bounds for sampling)
# Restructured following Shinka's best practices

from typing import List, Dict, Optional, Tuple
from .base import format_response_instructions

# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

GENERATE_SYSTEM_BASE = """You are an expert aerodynamicist designing airfoils.
Your goal: MAXIMIZE the reward. Higher is always better.
Analyze previous designs and their feedback, then propose completely new designs.
You will specify parameter RANGES [min, max] for sampling."""


# =============================================================================
# PROMPT VARIANTS (following Shinka's multi-variant approach)
# =============================================================================

GENERATE_STRATEGY_NEW_DESIGN = """Maximize reward by proposing parameter ranges for sampling.

Explore the full parameter space [-1.0, 1.0]. Try different parameter combinations."""

# List of all strategy variants
GENERATE_STRATEGIES = [
    GENERATE_STRATEGY_NEW_DESIGN,
]

GENERATE_STRATEGY_NAMES = [
    "new_design",
]


# =============================================================================
# MAIN USER PROMPT TEMPLATE
# =============================================================================

GENERATE_USER = """# Context: Airfoil Design Optimization

{context}

## Visual Reference

If an image is provided, it shows the **parent design's geometry**. This is NOT a design to modify, but a visual reference to help you understand how the parameter arrays map to physical shapes. Use this to infer how changes in parameters affect the geometry.

## Task

{strategy}

Propose a COMPLETELY NEW airfoil design with parameter ranges [min, max] for sampling each control point.

## Output Format

{response_format}

Critical: All parameter values must be in the range [-1.0, 1.0]."""


# =============================================================================
# EXAMPLE FORMAT
# =============================================================================

GENERATE_EXAMPLE = """{
  "n_cp": 4,
  "n_sp": 10,
  "params": [
    [[<r0_min>, <r0_max>], [<a0_min>, <a0_max>], [<e0_min>, <e0_max>]],
    [[<r1_min>, <r1_max>], [<a1_min>, <a1_max>], [<e1_min>, <e1_max>]],
    [[<r2_min>, <r2_max>], [<a2_min>, <a2_max>], [<e2_min>, <e2_max>]],
    [[<r3_min>, <r3_max>], [<a3_min>, <a3_max>], [<e3_min>, <e3_max>]]
  ],
  "name": "<descriptive_strategy_name>"
}

Replace all <placeholders> with actual float values in [-1.0, 1.0].
Bounds format: [min, max] where min <= max."""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_generate_prompt(context_str: str, strategy_idx: Optional[int] = None) -> str:
    """Build the full generate prompt with optional strategy variant.
    
    Args:
        context_str: Formatted context with design history
        strategy_idx: Optional strategy index (0-4). If None, uses default (0).
    
    Returns:
        Complete formatted user prompt
    """
    # Select strategy (default to incremental improvement)
    if strategy_idx is None:
        strategy_idx = 0
    strategy_idx = strategy_idx % len(GENERATE_STRATEGIES)
    strategy = GENERATE_STRATEGIES[strategy_idx]
    
    response_fmt = format_response_instructions(GENERATE_EXAMPLE)
    
    return GENERATE_USER.format(
        context=context_str,
        strategy=strategy,
        response_format=response_fmt
    )


def get_generate_system(strategy_idx: Optional[int] = None) -> str:
    """Get system prompt for generate action.
    
    Args:
        strategy_idx: Optional strategy index for logging purposes
        
    Returns:
        System prompt string
    """
    return GENERATE_SYSTEM_BASE


def sample_strategy() -> Tuple[int, str]:
    """Randomly sample a strategy variant.
    
    Returns:
        Tuple of (strategy_index, strategy_name)
    """
    import random
    idx = random.randint(0, len(GENERATE_STRATEGIES) - 1)
    return idx, GENERATE_STRATEGY_NAMES[idx]


# =============================================================================
# LEGACY SUPPORT
# =============================================================================

# Keep old names for backwards compatibility
GENERATE_SYSTEM = GENERATE_SYSTEM_BASE
