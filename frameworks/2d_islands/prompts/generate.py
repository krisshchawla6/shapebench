# Generate action prompts (with bounds for sampling)

from .base import format_response_instructions

GENERATE_SYSTEM = """You are an evolutionary optimizer for aerodynamic shape optimization.
Your goal is to generate diverse, novel designs by specifying parameter RANGES for sampling.
Improve the reward based on previous results. Explore the design space creatively."""

# =============================================================================
# STRATEGY VARIANTS (placeholder for multiple exploration strategies)
# =============================================================================

GENERATE_STRATEGY_PLACEHOLDER = """[Strategy guidance will be inserted here during prompt generation]"""

# Strategy variant examples (to be filled in):
# - Exploit best designs: Focus on parameter ranges near highest-reward designs
# - Diversify: Target unexplored regions of parameter space including extreme bounds
# - hybrid: explore combinations of differnt designs paramters from the population
# - Novel Design: Generate a completely new design from scratch
# - no startedgy: current way 

GENERATE_STRATEGIES = [
    # Strategy 1: Exploit best designs
    """**Exploitation Strategy**: Analyze the design database to identify the top 3-5 highest-reward designs. 
    Study their parameter patterns and geometric characteristics from the image analysis. 
    Your new design should specify parameter ranges that CENTER around or slightly vary from these successful parameter values. 
    Focus on refining what already works rather than exploring radically new territory.""",
    
    # Strategy 2: Diversify and explore extremes
    """**Exploration Strategy**: Review the design database to identify UNDEREXPLORED regions of the parameter space. 
    Look for parameter combinations or ranges that have rarely been tested. 
    Your new design should push toward boundary regions (near -1.0 or 1.0) or explore parameter combinations 
    distinct from the existing population. Use the image analysis and deisgn database to understand what geometries are missing.""",
    
    # Strategy 3: Hybrid - combine successful designs
    """**Hybrid Strategy**: Examine the design database and identify 2-3 designs with complementary strengths 
    (e.g., one with good lift, another with low drag). Analyze their parameter vectors and geometric features from the images. 
    Your new design should blend parameter characteristics from these designs, creating ranges that interpolate 
    or combine their successful traits into a novel hybrid configuration.""",
    
    # Strategy 4: Novel from scratch
    """**Novel Design Strategy**: Ignore specific parameter values from the database. Instead, use the database only to understand 
    general trends (what geometric features correlate with high/low rewards) and use the image analysis to learn parameter-geometry mappings. 
    Generate a completely fresh design with parameter ranges chosen based on aerodynamic first principles and your judgment 
    as an evolutionary optimizer, not by mimicking existing designs.""",
    
    # Strategy 5: No specific strategy (baseline)
    ""  # Empty string - no strategy guidance, just the base prompt
]

GENERATE_STRATEGY_NAMES = [
    "exploit",
    "diversify", 
    "hybrid",
    "novel",
    "no_strategy"
]

# =============================================================================
# MAIN USER PROMPT TEMPLATE
# =============================================================================

GENERATE_USER = """{context}

# Task: Generate New Airfoil Design (with Sampling Bounds)

First, analyze the provided information:

1. **Image Analysis**: If images are provided, they show a representative design sampled from the population (listed as Design 1 above for reference). Analyze this design critically to understand current population patterns, NOT to modify this specific design.

Reference Design 1's action vector and reward to learn parameter-geometry mappings.

Critically and relentlessly assess the design shown:
- What geometric flaws exist? (e.g., poor leading edge, blunt trailing edge, wrong curvature)
- What flow problems are visible? (separation, high drag wake, poor pressure recovery)
- Which action parameters led to these geometric features?
- List 3-4 key design/geometric changes needed to improve designs in the population

{strategy_block}

2. **Design Reasoning**: Based on the design database and population state, think step-by-step about what NEW design should be explored to advance the population. **Follow the Strategy Focus guidance above** - this determines HOW you should approach the design. Consider unexplored geometric variations and different aerodynamic strategies. Focus on diversity and bold exploration, not incremental tweaking of existing designs.

In your reasoning, provide 4-6 bullet points that:
- **Apply the Strategy Focus approach described above**
- Draw insights from the population database (what patterns lead to high/low rewards?)
- Incorporate observations from the image analysis (what aerodynamic phenomena are present?)
- Identify specific geometric parameters or ranges to explore and WHY
- Apply your judgment as an evolutionary optimizer (what will advance the search?)

Then, create a new airfoil by specifying precise [min, max] bounds for each parameter.
Values will be uniformly sampled between the bounds you provide.

Design considerations: The reasoning above should be applied to the design parameters.

{response_format}

**Response Format:**
Provide your response in the following structure:

## Analysis
[Critically assess Design 1's geometry and flow fields. Reference its action vector and reward. Identify geometric flaws, flow problems, parameter-geometry mappings, and list 3-4 key changes needed]

## Reasoning
[4-6 bullet points with design insights: combine population database patterns, image analysis observations, the Strategy Focus guidance above, and your judgment as an evolutionary optimizer to justify the new design strategy. Explicitly explain how your design follows the Strategy Focus.]

## Design Parameters
[Your JSON with the design parameters as shown in the example above]
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

Replace all <placeholders> with very precise float values in [-1.0, 1.0]."""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_generate_prompt(context_str: str, strategy_idx=None) -> str:
    """Build the full generate prompt with optional strategy variant.
    
    Args:
        context_str: Formatted context with design history
        strategy_idx: Optional strategy index. If None or no strategies defined, 
                     uses empty strategy block.
    
    Returns:
        Complete formatted user prompt
    """
    # Build strategy block
    if GENERATE_STRATEGIES and strategy_idx is not None:
        strategy_idx = strategy_idx % len(GENERATE_STRATEGIES)
        strategy_text = GENERATE_STRATEGIES[strategy_idx]
        strategy_block = f"\n**Strategy Focus**: {strategy_text}\n"
    else:
        # No strategy variants defined yet - use empty block
        strategy_block = ""
    
    response_fmt = format_response_instructions(GENERATE_EXAMPLE)
    
    return GENERATE_USER.format(
        context=context_str,
        strategy_block=strategy_block,
        response_format=response_fmt
    )


def get_generate_system(strategy_idx=None) -> str:
    """Get system prompt for generate action.
    
    Args:
        strategy_idx: Optional strategy index for logging purposes
        
    Returns:
        System prompt string
    """
    return GENERATE_SYSTEM


def sample_strategy():
    """Randomly sample a strategy variant.
    
    Returns:
        Tuple of (strategy_index, strategy_name)
        If no strategies defined, returns (None, "no_strategy")
    """
    if not GENERATE_STRATEGIES:
        return None, "no_strategy"
    
    import random
    idx = random.randint(0, len(GENERATE_STRATEGIES) - 1)
    return idx, GENERATE_STRATEGY_NAMES[idx]