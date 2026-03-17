# Generate action prompts (with direct values)

from .base import format_response_instructions

GENERATE_SYSTEM = """You are an evolutionary optimizer for aerodynamic shape optimization.
Your goal is to generate diverse, novel designs by specifying exact parameter VALUES.
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
    """**Exploitation Strategy**: Analyze the design database to identify the highest-reward designs. 
    Study their parameter patterns and geometric characteristics from the action vectors. 
    Your new design should specify parameter values that aim to maximze and significantly improve the reward from these parameter values. 
    Focus on refining what already works rather than exploring radically new territory.""",
    
    # Strategy 2: Diversify and explore extremes
    """**Exploration Strategy**: Review the design database to identify UNDEREXPLORED regions of the parameter space. 
    Look for parameter combinations that have rarely been tested. 
    Your new design should push toward boundary regions (near -1.0 or 1.0) or explore parameter combinations 
    distinct from the existing population. Use the image analysis and design database to understand what geometries are missing.""",
    
    # Strategy 3: Hybrid - combine successful designs
    """**Hybrid Strategy**: Examine the design database and identify 2-3 designs with complementary strengths 
    (e.g., one with good lift, another with low drag). Analyze their parameter vectors and geometric features from the images. 
    Your new design should blend parameter characteristics from these designs, creating values that interpolate 
    or combine their successful traits into a novel hybrid configuration.""",
    
    # Strategy 4: Novel from scratch
    """**Novel Design Strategy**: Ignore specific parameter values from the database. Instead, use the database only to understand 
    general trends (what geometric features correlate with high/low rewards) and use the image analysis to learn parameter-geometry mappings. 
    Generate a completely fresh design with parameter values chosen based on aerodynamic first principles and your judgment 
    as an evolutionary optimizer, not by mimicking existing designs.""",
    
    # Strategy 5: Visual intuition - image-driven engineering design
    """**Visual Intuition Strategy**: Focus primarily on the provided images. Study the shape geometry, 
    simulation fields, and any visual outputs closely. Apply your knowledge of engineering design principles 
    to critically assess whether the design visually makes sense. 
    Identify specific geometric deficiencies visible in the images — does the shape look structurally sound? 
    Are there obvious flaws like asymmetries, excessive thickness, sharp discontinuities, or poor curvature? 
    Do the simulation results (flow fields, pressure, stress, etc.) reveal problems like separation, 
    high gradients, or inefficient geometry? Design a new shape that directly addresses these visual flaws. 
    Let the images and engineering intuition drive your parameter choices, 
    not statistical patterns from the database. Think like an engineer critically reviewing design results.""",
]

GENERATE_STRATEGY_NAMES = [
    "exploit",
    "diversify", 
    "hybrid",
    "novel",
    "visual_intuition"
]

# =============================================================================
# MAIN USER PROMPT TEMPLATE
# =============================================================================

GENERATE_USER = """{context}

# Task: Generate New Airfoil Design (with Direct Values)

First, analyze the provided information:

1. **Image Analysis**: If images are provided, they show a representative design sampled from the population (listed as Design 1 above for reference). You MUST first describe what you actually see in the images before analyzing action vectors:

- In 3-4 bullet points, describe the **geometry** you observe: What does the shape look like? Is it smooth or jagged? Symmetric or asymmetric? Thick or thin? Are there any obvious geometric flaws visible (discontinuities, pinch points, blunt edges, odd proportions)?
- In 3-4 bullet points, describe the **flow field / simulation results** you observe: What do the pressure, velocity, or other field plots show? Where are high/low regions? Is there visible flow separation, recirculation, wake asymmetry, or pressure spikes?

Only AFTER describing what you see in the images, reference Design 1's action vector and reward to learn parameter-geometry mappings.

Critically and relentlessly assess the design shown:
- Which visible geometric features are problematic and why?
- Which action parameters likely caused the geometric flaws you see?
- List 3-4 key design/geometric changes needed to improve designs in the population

{strategy_block}

2. **Design Reasoning**: Based on the design database and population state, think step-by-step about what NEW design should be explored to advance the population. **Follow the Strategy Focus guidance above** - this determines HOW you should approach the design. Consider unexplored geometric variations and different aerodynamic strategies. Focus on diversity and bold exploration, not incremental tweaking of existing designs based on aerodynamic principles.

In your reasoning, provide 4-6 bullet points that:
- **Apply the Strategy Focus approach described above**
- Draw insights from the population database (what patterns lead to high/low rewards?)
- Incorporate observations from the image analysis (what aerodynamic phenomena are present?)
- Identify specific geometric parameters to explore and WHY
- Apply your judgment as an evolutionary optimizer (what will advance the search?)

Then, create a new airfoil by specifying precise values for each parameter. Use 4+ decimals. n_cp=4 n_sp=20.

Design considerations: The reasoning above should be applied to the design parameters to evolve the population.

{response_format}

**Response Format:**
Provide your response in the following structure:

## Analysis
[If images provided: first describe the geometry (3-4 bullets on what you see) and the flow fields (3-4 bullets on what you see). Then reference Design 1's action vector and reward. Identify which visible flaws map to which parameters, and list 3-4 key changes needed]

## Reasoning
[4-6 key bullet points with design insights: combine population database patterns, image analysis observations, the Strategy Focus guidance above, and your judgment as an evolutionary optimizer to justify the new design strategy. Explicitly explain how your design follows the Strategy Focus.]

## Per-Parameter Predictions
For EACH parameter you set, state your PREDICTION of what it will mechanically do:
- "I am setting param[i][j] to VALUE because I predict it will CAUSE [specific geometric effect]"
- Be specific about the predicted physical outcome, not just your goal

## Design Parameters
[Your JSON with the design parameters as shown in the example above]
"""

GENERATE_format = """{
  "n_cp": 4,
  "n_sp": 20,
  "params": [
    [<r0>, <a0>, <e0>],
    [<r1>, <a1>, <e1>],
    [<r2>, <a2>, <e2>],
    [<r3>, <a3>, <e3>]
  ],
  "name": "<descriptive_name>"
}

Replace all <placeholders> with actual float values in [-1.0, 1.0].
Each inner array is [radius_param, angle_param, edgy_param] for that control point."""


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
    
    response_fmt = format_response_instructions(GENERATE_format)
    
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