from .base import format_response_instructions

GENERATE_SYSTEM = """You are an evolutionary optimizer for 2D airfoil aerodynamic design using NeuralFoil.
Your goal is to generate diverse, high-performing airfoil shapes by specifying Kulfan (CST) parameter values.
Optimize toward the objective (e.g., maximize L/D, maximize CL, minimize CD) based on previous results.
Each airfoil is defined by 18 parameters: 8 upper-surface CST weights, 8 lower-surface CST weights,
one leading-edge weight, and trailing-edge thickness.
ALL floats MUST have exactly 8 decimal places (e.g., 0.98521047 not 0.9852)."""

GENERATE_STRATEGIES = [
    """**Exploitation Strategy**: Identify the highest-reward designs in the database.
    Study their CST coefficient patterns — which surface is more cambered, how the thickness is distributed,
    where suction peaks form. Propose a new design that refines these patterns to further improve the reward.
    Make targeted incremental changes rather than broad exploration.""",

    """**Exploration Strategy**: Examine the database to find UNDEREXPLORED regions of the CST space.
    Look for surface shapes not yet tested: highly reflexed trailing edges, aggressive LE radii,
    unconventional upper/lower weight distributions.
    Push toward parameter regions that are distinct from the existing population.""",

    """**Hybrid Strategy**: Identify 2-3 designs with complementary aerodynamic strengths
    (e.g., one with high CL, another with very low CD or good CM).
    Analyze their CST patterns and how each achieves its strength.
    Blend their parameter characteristics to create a design capturing multiple virtues.""",

    """**Novel Design Strategy**: Ignore specific CST values from the database.
    Instead, use aerodynamic first principles: desired pressure distribution, boundary layer behavior,
    laminar flow extent, and separation resistance.
    Map these physical goals to CST parameters and generate a fresh design from scratch.""",

    "",
]

GENERATE_STRATEGY_NAMES = [
    "exploit", "diversify", "hybrid", "novel", "no_strategy",
]

GENERATE_USER = """{context}

# Task: Generate New NeuralFoil Airfoil Design

First, analyze the provided information:

1. **Image Analysis**: If images are provided, they show the **airfoil shape** and **Cp (pressure coefficient) distribution** for a representative design. Analyze critically:
   - Where is the suction peak on the upper surface? Is it too sharp (risk of separation) or too diffuse?
   - How does the lower-surface pressure recovery compare to the upper?
   - Is the pressure distribution favorable for laminar flow (gradual recovery) or likely turbulent (steep adverse gradient)?
   - What changes to CST coefficients would shift the suction peak, control thickness, or improve pressure recovery?

{strategy_block}

2. **Design Reasoning**: Based on the design database and population state, think step-by-step about what NEW airfoil to explore. Consider:
   - upper_weights[0..2] control leading-edge curvature and suction peak location
   - upper_weights[3..7] control mid-chord to trailing-edge upper surface shape and camber
   - lower_weights[0..2] control lower LE shape and reflex/concavity near the nose
   - lower_weights[3..7] control lower surface aft camber; negative = concave (reflexed)
   - leading_edge_weight: larger = rounder nose, smaller = sharper (good for high-speed)
   - TE_thickness: small (0.001–0.004) is typical for low-drag; larger aids structural stiffness

Provide 4-6 bullet points that:
   - **Apply the Strategy Focus approach described above**
   - Draw insights from the population database
   - Incorporate observations from the image analysis
   - Identify specific CST coefficients to adjust and WHY (aerodynamic reasoning)
   - Apply your judgment as an evolutionary airfoil optimizer

Then, create a new airfoil design by specifying precise values for each Kulfan parameter.

{response_format}

**Response Format:**
Provide your response in the following structure:

<ANALYSIS>
[Critically assess the representative design's shape and Cp distribution. Reference its CST parameters and reward.
Identify aerodynamic issues (suction peak, pressure recovery, separation risk) and list 3-4 key changes needed.]
</ANALYSIS>

<DESIGN_RATIONALE>
[4-6 bullet points combining database patterns, image analysis, strategy guidance, and aerodynamic judgment]
</DESIGN_RATIONALE>

<DESIGN>
[Your JSON with the Kulfan parameters as shown in the example above]
</DESIGN>
"""

GENERATE_FORMAT = """{
  "upper_weights": [<float -0.30000000 to 0.60000000>, <float -0.30000000 to 0.60000000>, <float -0.30000000 to 0.60000000>, <float -0.30000000 to 0.60000000>, <float -0.30000000 to 0.60000000>, <float -0.30000000 to 0.60000000>, <float -0.30000000 to 0.60000000>, <float -0.30000000 to 0.60000000>],
  "lower_weights": [<float -0.30000000 to 0.30000000>, <float -0.30000000 to 0.30000000>, <float -0.30000000 to 0.30000000>, <float -0.30000000 to 0.30000000>, <float -0.30000000 to 0.30000000>, <float -0.30000000 to 0.30000000>, <float -0.30000000 to 0.30000000>, <float -0.30000000 to 0.30000000>],
  "leading_edge_weight": <float -0.50000000 to 0.50000000>,
  "TE_thickness": <float 0.00000000 to 0.01000000>,
  "name": "<descriptive_name>"
}

Replace all <placeholders> with actual float values respecting the stated ranges.
upper_weights and lower_weights must each be a JSON array of exactly 8 floats.
Values go from leading edge (index 0) to trailing edge (index 7)."""


def get_generate_prompt(context_str: str, strategy_idx=None) -> str:
    if GENERATE_STRATEGIES and strategy_idx is not None:
        strategy_idx = strategy_idx % len(GENERATE_STRATEGIES)
        text = GENERATE_STRATEGIES[strategy_idx]
        strategy_block = f"\n**Strategy Focus**: {text}\n" if text else ""
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
    import random
    if not GENERATE_STRATEGIES:
        return None, "no_strategy"
    idx = random.randint(0, len(GENERATE_STRATEGIES) - 1)
    return idx, GENERATE_STRATEGY_NAMES[idx]
