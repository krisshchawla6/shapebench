from .base import format_response_instructions

GENERATE_SYSTEM = """You are an evolutionary optimizer for delta wing aerodynamic design.
Your goal is to generate diverse, novel designs by specifying exact parameter VALUES.
Improve the reward based on previous results. Explore the design space creatively.
ALL floats MUST have exactly 8 decimal places (e.g., 0.98521047 not 0.9852)."""

GENERATE_STRATEGIES = [
    """**Exploitation Strategy**: Analyze the design database to identify the highest-reward designs.
    Study their parameter patterns and aerodynamic characteristics.
    Your new design should specify parameter values that aim to maximize and significantly improve the reward from these parameter values.
    Focus on refining what already works rather than exploring radically new territory.""",

    """**Exploration Strategy**: Review the design database to identify UNDEREXPLORED regions of the parameter space.
    Look for parameter combinations that have rarely been tested.
    Your new design should push toward boundary regions or explore parameter combinations
    distinct from the existing population. Use the image analysis and design database to understand what geometries are missing.""",

    """**Hybrid Strategy**: Examine the design database and identify 2-3 designs with complementary strengths
    (e.g., one with high lift, another with low drag). Analyze their parameters and aerodynamic features from the images.
    Your new design should blend parameter characteristics from these designs, creating values that interpolate
    or combine their successful traits into a novel hybrid configuration.""",

    """**Novel Design Strategy**: Ignore specific parameter values from the database. Instead, use the database only to understand
    general trends (what geometric features correlate with high/low rewards) and use the image analysis to learn parameter-aerodynamic mappings.
    Generate a completely fresh design with parameter values chosen based on aerodynamic first principles and your judgment
    as an evolutionary optimizer, not by mimicking existing designs.""",

    "",  # No specific strategy (baseline)
]

GENERATE_STRATEGY_NAMES = [
    "exploit",
    "diversify",
    "hybrid",
    "novel",
    "no_strategy",
]

GENERATE_USER = """{context}

# Task: Generate New Delta Wing Design (with Direct Values)

First, analyze the provided information:

1. **Image Analysis**: If images are provided, they show the Cp distribution and corrected aerodynamic coefficients for a representative design sampled from the population (listed as Design 1 above for reference). Analyze this design critically to understand current population patterns, NOT to modify this specific design.

Reference Design 1's parameters and reward to learn parameter-aerodynamic mappings.

Critically assess the design shown:
- What aerodynamic issues exist? (e.g., poor lift distribution, high induced drag, early vortex breakdown)
- How does the sweep, twist, camber, and thickness affect performance?
- Which parameters most influence the reward (CL/CDi)?
- List 3-4 key design changes needed to improve designs in the population

{strategy_block}

2. **Design Reasoning**: Based on the design database and population state, think step-by-step about what NEW design should be explored to advance the population. **Follow the Strategy Focus guidance above** - this determines HOW you should approach the design. Consider unexplored geometric variations and different aerodynamic strategies. Focus on diversity and bold exploration, not incremental tweaking.

In your reasoning, provide 4-6 bullet points that:
- **Apply the Strategy Focus approach described above**
- Draw insights from the population database (what patterns lead to high/low rewards?)
- Incorporate observations from the image analysis (what aerodynamic phenomena are present?)
- Identify specific design parameters to explore and WHY
- Apply your judgment as an evolutionary optimizer (what will advance the search?)

Then, create a new delta wing by specifying precise values for each parameter.

Design considerations:
- Higher sweep → stronger vortex lift but higher induced drag
- Twist distribution (root vs tip) controls load distribution and stall behavior
- Camber (naca_m, naca_p) adds lift but shifts pitching moment
- Thickness (naca_t) affects wave drag and structural depth
- Dihedral affects lateral stability
- Root chord scales the entire wing

{response_format}

**Response Format:**
Provide your response in the following structure:

## Analysis
[Critically assess the representative design's Cp distribution and aerodynamic performance. Reference its parameters and reward. Identify issues and list 3-4 key changes needed]

## Reasoning
[4-6 bullet points with design insights: combine population database patterns, image analysis observations, the Strategy Focus guidance above, and your judgment as an evolutionary optimizer to justify the new design strategy]

## Design Parameters
[Your JSON with the design parameters as shown in the example above]
"""

GENERATE_FORMAT = """{
  "le_sweep": <float 45.00000000-80.00000000>,
  "root_chord_in": <float 10.00000000-50.00000000>,
  "twist_root": <float -10.00000000 to 10.00000000>,
  "twist_tip": <float -10.00000000 to 10.00000000>,
  "dihedral": <float -15.00000000 to 15.00000000>,
  "naca_m": <int: 0, 2, or 4>,
  "naca_p": <int: 0 or 4 (must be 0 when naca_m=0)>,
  "naca_t": <int 6-24>,
  "name": "<descriptive_name>"
}

Replace all <placeholders> with actual values respecting the stated ranges."""


def get_generate_prompt(context_str: str, strategy_idx=None) -> str:
    if GENERATE_STRATEGIES and strategy_idx is not None:
        strategy_idx = strategy_idx % len(GENERATE_STRATEGIES)
        strategy_text = GENERATE_STRATEGIES[strategy_idx]
        strategy_block = f"\n**Strategy Focus**: {strategy_text}\n" if strategy_text else ""
    else:
        strategy_block = ""

    response_fmt = format_response_instructions(GENERATE_FORMAT)
    return GENERATE_USER.format(
        context=context_str,
        strategy_block=strategy_block,
        response_format=response_fmt,
    )


def get_generate_system(strategy_idx=None) -> str:
    return GENERATE_SYSTEM


def sample_strategy():
    if not GENERATE_STRATEGIES:
        return None, "no_strategy"
    import random
    idx = random.randint(0, len(GENERATE_STRATEGIES) - 1)
    return idx, GENERATE_STRATEGY_NAMES[idx]
