from typing import List, Dict

CONTEXT_FORMAT = """# DrivAerStar Vehicle Aerodynamic Design Context

## Parameter Space
Each vehicle design is defined by 20 FFD (Free-Form Deformation) parameters:

| Parameter               | Range          | Description                          |
|------------------------|----------------|--------------------------------------|
| car_size               | 0.80 – 1.20   | Overall scale factor                 |
| car_width              | -0.10 – 0.10  | Width change                         |
| car_len                | -0.10 – 0.10  | Length change                        |
| ramp_angle             | -8.0 – 8.0    | Front ramp angle (deg)               |
| front_bumper_length    | -0.10 – 0.10  | Bumper extension                     |
| wind_screen_x          | -0.05 – 0.05  | Windscreen x-shift                   |
| wind_screen_z          | -0.05 – 0.05  | Windscreen z-shift                   |
| side_mirrors_x         | -0.05 – 0.05  | Side mirrors x-shift                 |
| side_mirrors_z         | -0.05 – 0.05  | Side mirrors z-shift                 |
| rear_window_x          | -0.05 – 0.05  | Rear window x-shift                  |
| rear_window_z          | -0.05 – 0.05  | Rear window z-shift                  |
| trunklid_angle         | -8.0 – 8.0    | Trunklid angle (deg)                 |
| trunklid_x             | -0.05 – 0.05  | Trunklid x-shift                     |
| trunklid_z             | -0.05 – 0.05  | Trunklid z-shift                     |
| diffusor_angle         | -8.0 – 8.0    | Diffusor angle (deg)                 |
| car_green_house_angle  | -8.0 – 8.0    | Greenhouse angle (deg)               |
| car_front_hood_angle   | -8.0 – 8.0    | Front hood angle (deg)               |
| car_air_intake_angle   | -8.0 – 8.0    | Air intake angle (deg)               |
| tires_diameter         | -0.013 – 0.013| Tire diameter change                 |
| tires_width            | -0.015 – 0.015| Tire width change                    |

## Objective
Minimize drag coefficient (Cd). Reward = −Cd (higher is better).

## Previous Designs
{design_history}
"""

DESIGN_ENTRY = """Design {idx}:
  - Body: car_size={car_size}, car_width={car_width}, car_len={car_len}
  - Front: ramp_angle={ramp_angle}, front_bumper_length={front_bumper_length}, wind_screen_x={wind_screen_x}, wind_screen_z={wind_screen_z}
  - Rear: trunklid_angle={trunklid_angle}, trunklid_x={trunklid_x}, trunklid_z={trunklid_z}, diffusor_angle={diffusor_angle}
  - Other: side_mirrors_x={side_mirrors_x}, side_mirrors_z={side_mirrors_z}, rear_window_x={rear_window_x}, rear_window_z={rear_window_z}
  - Angles: car_green_house_angle={car_green_house_angle}, car_front_hood_angle={car_front_hood_angle}, car_air_intake_angle={car_air_intake_angle}
  - Tires: tires_diameter={tires_diameter}, tires_width={tires_width}
  - Reward (−Cd): {reward:.6f}
  - Rank: {rank}
"""

RESPONSE_FORMAT = """
You MUST respond with ONLY a valid JSON object (no markdown, no schema definitions).
Do NOT include $schema, title, description, type, properties, or required fields.
ALL floats MUST have exactly 8 decimal places (e.g., 0.98521047 not 0.9852).

Example response format:
{example_json}
"""


def format_context(context: List[Dict]) -> str:
    if not context:
        return CONTEXT_FORMAT.format(design_history="No previous designs available.")

    lines = []
    for i, item in enumerate(context):
        p = item.get("params", {})
        entry = DESIGN_ENTRY.format(
            idx=i + 1, reward=item.get("reward", 0.0),
            rank=item.get("ranking", "N/A"),
            **{k: p.get(k, "?") for k in [
                "car_size", "car_width", "car_len", "ramp_angle",
                "front_bumper_length", "wind_screen_x", "wind_screen_z",
                "side_mirrors_x", "side_mirrors_z", "rear_window_x",
                "rear_window_z", "trunklid_angle", "trunklid_x", "trunklid_z",
                "diffusor_angle", "car_green_house_angle",
                "car_front_hood_angle", "car_air_intake_angle",
                "tires_diameter", "tires_width",
            ]})
        lines.append(entry)

    return CONTEXT_FORMAT.format(design_history="\n".join(lines))


def format_response_instructions(example_json: str) -> str:
    return RESPONSE_FORMAT.format(example_json=example_json)
