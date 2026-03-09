from .base import format_response_instructions

GENERATE_SYSTEM = """You are an evolutionary optimizer for blended-wing-body (BWB) aerodynamic design.
Your goal is to generate diverse, novel BWB planform designs by specifying exact parameter VALUES.
Improve the reward (lift-to-drag ratio L/D) based on previous results. Explore the design space creatively."""

GENERATE_STRATEGIES = [
    """**Exploitation Strategy**: Analyze the design database to identify the highest-reward designs.
    Study their parameter patterns and aerodynamic characteristics.
    Your new design should specify parameter values that aim to maximize and significantly improve L/D.
    Focus on refining what already works rather than exploring radically new territory.""",

    """**Exploration Strategy**: Review the design database to identify UNDEREXPLORED regions of the parameter space.
    Look for parameter combinations that have rarely been tested.
    Your new design should push toward boundary regions or explore parameter combinations
    distinct from the existing population. Use the image analysis and design database to understand what geometries are missing.""",

    """**Hybrid Strategy**: Examine the design database and identify 2-3 designs with complementary strengths
    (e.g., one with favorable pressure distribution, another with low skin friction).
    Analyze their parameters and aerodynamic features from the Cp/Cfx images.
    Your new design should blend parameter characteristics from these designs, creating values that interpolate
    or combine their successful traits into a novel hybrid configuration.""",

    """**Novel Design Strategy**: Ignore specific parameter values from the database. Instead, use the database only to understand
    general trends (what geometric features correlate with high/low L/D) and use the image analysis to learn parameter-aerodynamic mappings.
    Generate a completely fresh design with parameter values chosen based on aerodynamic first principles and your judgment
    as an evolutionary optimizer, not by mimicking existing designs.""",

    "",
]

GENERATE_STRATEGY_NAMES = [
    "exploit",
    "diversify",
    "hybrid",
    "novel",
    "no_strategy",
]

GENERATE_USER = """{context}

# Task: Generate New Blended-Wing-Body Design (with Direct Values)

First, analyze the provided information:

1. **Image Analysis**: If images are provided, they show the Cp (pressure coefficient) and Cfx (skin friction) distributions for a representative design sampled from the population (listed as Design 1 above for reference). Analyze this design critically to understand current population patterns, NOT to modify this specific design.

Reference Design 1's parameters and reward to learn parameter-aerodynamic mappings.

Critically assess the design shown:
- What aerodynamic issues exist? (e.g., adverse pressure gradients, flow separation, high skin friction regions)
- How do the span sections (B1-B3), chord sections (C2-C4), and sweep angles (S1-S3) affect performance?
- Which parameters most influence the L/D reward?
- List 3-4 key design changes needed to improve designs in the population

{strategy_block}

2. **Design Reasoning**: Based on the design database and population state, think step-by-step about what NEW design should be explored to advance the population. **Follow the Strategy Focus guidance above** — this determines HOW you should approach the design. Consider unexplored geometric variations and different aerodynamic strategies. Focus on diversity and bold exploration, not incremental tweaking.

In your reasoning, provide 4-6 bullet points that:
- **Apply the Strategy Focus approach described above**
- Draw insights from the population database (what patterns lead to high/low L/D?)
- Incorporate observations from the image analysis (what aerodynamic phenomena are present?)
- Identify specific design parameters to explore and WHY
- Apply your judgment as an evolutionary optimizer (what will advance the search?)

Then, create a new BWB planform by specifying precise values for each parameter.

Design considerations (all chord/span in mm, C1 fixed at 1000 mm):
- B1-B3 (span sections): control planform shape and aspect ratio; larger spans improve lift but increase wetted area
- C2-C4 (chord sections): control chord distribution from root to tip; larger chords add area but affect pressure distribution
- S1-S3 (sweep angles in degrees): higher sweep reduces wave drag but can promote spanwise flow and tip stall
- The ratio C2/C1 controls the transition from centre-body to inner wing
- C4/C1 is the tip chord ratio — too small causes tip stall, too large adds drag
- B3/C1 is the dominant span contributor — it strongly controls aspect ratio and induced drag

{response_format}

**Response Format:**
Provide your response in the following structure:

## Analysis
[Critically assess the representative design's Cp and Cfx distributions. Reference its parameters and reward. Identify issues and list 3-4 key changes needed]

## Reasoning
[4-6 bullet points with design insights: combine population database patterns, image analysis observations, the Strategy Focus guidance above, and your judgment as an evolutionary optimizer to justify the new design strategy]

## Design Parameters
[Your JSON with the design parameters as shown in the example above]
"""

GENERATE_FORMAT = """{{
  "B1": <float 100.0-200.0>,
  "B2": <float 50.0-200.0>,
  "B3": <float 200.0-700.0>,
  "C2": <float 550.0-850.0>,
  "C3": <float 180.0-280.0>,
  "C4": <float 60.0-90.0>,
  "S1": <float 40.0-60.0>,
  "S2": <float 40.0-60.0>,
  "S3": <float 24.0-40.0>,
  "name": "<descriptive_name>"
}}

Replace all <placeholders> with actual values respecting the stated ranges.
All chord/span values are in mm (C1 is fixed at 1000 mm). Sweep angles are in degrees."""


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
