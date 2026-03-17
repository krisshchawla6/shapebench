from .base import format_response_instructions

GENERATE_SYSTEM = """You are an evolutionary optimizer for DrivAerStar vehicle aerodynamic design.
Your goal is to generate diverse, novel vehicle body configurations by specifying exact FFD parameter values.
Minimize drag coefficient (Cd) based on previous results. Explore the design space creatively.
The vehicle is a DrivAer-class sedan with 20 Free-Form Deformation parameters controlling body shape."""

GENERATE_STRATEGIES = [
    """**Exploitation Strategy**: Analyze the design database to identify the lowest-drag designs.
    Study their parameter patterns and aerodynamic characteristics.
    Your new design should specify parameter values that aim to significantly reduce Cd.
    Focus on refining what already works rather than exploring radically new territory.""",

    """**Exploration Strategy**: Review the design database to identify UNDEREXPLORED regions of the parameter space.
    Look for parameter combinations that have rarely been tested.
    Your new design should push toward boundary regions or explore parameter combinations
    distinct from the existing population.""",

    """**Hybrid Strategy**: Examine the design database and identify 2-3 designs with complementary strengths
    (e.g., one with low pressure drag, another with low skin friction drag).
    Analyze their parameters and aerodynamic features from the pressure/WSS images.
    Your new design should blend parameter characteristics from these designs.""",

    """**Novel Design Strategy**: Ignore specific parameter values from the database. Instead, use the database only to understand
    general trends (what geometric features correlate with high/low drag) and use image analysis to learn parameter-aerodynamic mappings.
    Generate a completely fresh design based on aerodynamic first principles.""",

    "",
]

GENERATE_STRATEGY_NAMES = [
    "exploit", "diversify", "hybrid", "novel", "no_strategy",
]

GENERATE_USER = """{context}

# Task: Generate New DrivAerStar Vehicle Design

First, analyze the provided information:

1. **Image Analysis**: If images are provided, they show the Pressure and Wall Shear Stress (x) distributions for a representative design from the population (listed as Design 1). Analyze critically to understand current population patterns.

Critically assess the design shown:
- Where are the high-pressure (stagnation) and low-pressure (suction) regions?
- Are there regions of high wall shear stress indicating boundary layer effects?
- How do body shape parameters (ramp angle, trunklid, diffusor, greenhouse) affect the flow?
- Which parameters most influence the drag coefficient?
- List 3-4 key design changes needed to reduce drag

{strategy_block}

2. **Design Reasoning**: Based on the design database and population state, think step-by-step about what NEW design should be explored. Consider:
- Front-end shaping (ramp angle, bumper, air intake) affects stagnation pressure and underbody flow
- Roof/greenhouse angle affects roof separation and rear wake
- Rear-end shaping (trunklid angle, diffusor angle) strongly affects base pressure drag
- Overall size/width/length affect frontal area and wetted area
- Windscreen and rear window angles affect attached/separated flow transitions

Provide 4-6 bullet points that:
- **Apply the Strategy Focus approach described above**
- Draw insights from the population database
- Incorporate observations from the image analysis
- Identify specific parameters to adjust and WHY
- Apply your judgment as an evolutionary optimizer

Then, create a new vehicle design by specifying precise values for each parameter.

{response_format}

**Response Format:**
Provide your response in the following structure:

## Analysis
[Critically assess the representative design's pressure and WSS distributions. Reference its parameters and reward. Identify aerodynamic issues and list 3-4 key changes needed]

## Reasoning
[4-6 bullet points combining database patterns, image analysis, strategy guidance, and aerodynamic judgment]

## Design Parameters
[Your JSON with the design parameters as shown in the example above]
"""

GENERATE_FORMAT = """{{
  "car_size": <float 0.80-1.20>,
  "car_width": <float -0.10 to 0.10>,
  "car_len": <float -0.10 to 0.10>,
  "ramp_angle": <float -8.0 to 8.0>,
  "front_bumper_length": <float -0.10 to 0.10>,
  "wind_screen_x": <float -0.05 to 0.05>,
  "wind_screen_z": <float -0.05 to 0.05>,
  "side_mirrors_x": <float -0.05 to 0.05>,
  "side_mirrors_z": <float -0.05 to 0.05>,
  "rear_window_x": <float -0.05 to 0.05>,
  "rear_window_z": <float -0.05 to 0.05>,
  "trunklid_angle": <float -8.0 to 8.0>,
  "trunklid_x": <float -0.05 to 0.05>,
  "trunklid_z": <float -0.05 to 0.05>,
  "diffusor_angle": <float -8.0 to 8.0>,
  "car_green_house_angle": <float -8.0 to 8.0>,
  "car_front_hood_angle": <float -8.0 to 8.0>,
  "car_air_intake_angle": <float -8.0 to 8.0>,
  "tires_diameter": <float -0.013 to 0.013>,
  "tires_width": <float -0.015 to 0.015>,
  "name": "<descriptive_name>"
}}

Replace all <placeholders> with actual values respecting the stated ranges.
Angles are in degrees. Lengths are dimensionless FFD perturbations.
car_size is a scale factor (1.0 = baseline)."""


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
        response_format=response_fmt)


def get_generate_system(strategy_idx=None) -> str:
    return GENERATE_SYSTEM


def sample_strategy():
    import random
    if not GENERATE_STRATEGIES:
        return None, "no_strategy"
    idx = random.randint(0, len(GENERATE_STRATEGIES) - 1)
    return idx, GENERATE_STRATEGY_NAMES[idx]
