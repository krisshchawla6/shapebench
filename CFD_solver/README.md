# Dynamic Flow360 External Flow Sweep

## Usage

Run the sweep template with custom surface mesh and parameters:

```bash
python sweep_template.py <mesh_path> <params_json_path>
```

### Example
```bash
python sweep_template.py "../../surface_meshes/barracuda.stl" "example_params.json"
```

## Files

- `sweep_template.py` - Main dynamic script that accepts mesh path and JSON parameters
- `params_template.json` - JSON schema template with type validation for AI agents
- `example_params.json` - Example parameter configuration
- `README.md` - This usage guide

## Parameter Schema

The JSON parameters file must follow the schema defined in `params_template.json`. Key constraints:

- **velocity_magnitude**: 0.1-1000.0 m/s
- **alpha**: -45° to +45° (angle of attack)
- **beta**: -45° to +45° (sideslip angle)
- **sweep_schema.variable**: Must be "alpha", "beta", or "velocity_magnitude"
- **sweep_schema.values**: Array of 2-20 numeric values
- **sweep_schema.unit**: "fl.u.deg", "fl.u.m/fl.u.s", or "fl.u.dimensionless"

These constraints act as railguards for AI agents generating parameters.
