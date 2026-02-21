# Delta Wing VLM + VortexNet Scripts

## Files

| File | Purpose |
|------|---------|
| `generateDeltawing 1.py` | Core SUAVE geometry + VLM routines (author: Yiren Shen) |
| `VLM.py` | Patched VLM solver (replaces SUAVE's built-in) |
| `generate_design.py` | Wrapper: design JSON + sim CLI args → VLM → results + PNG |
| `generate_design_corrected.py` | Wrapper: design JSON + sim CLI args → VLM + VortexNet → results + PNG |
| `design_params.template.json` | Template with all design parameters and metadata |

## Design Parameters (JSON)

| Parameter | Type | Unit | Default | Range / Notes |
|-----------|------|------|---------|---------------|
| `design_name` | string | — | — | Output subfolder name |
| `le_sweep` | float | deg | — | 45–80 (trained: 55, 65, 75) |
| `root_chord_in` | float | inches | 25.734 | Physical root chord |
| `twist_root` | float | deg | 0.0 | Geometric twist at root |
| `twist_tip` | float | deg | 0.0 | Geometric twist at tip (washout < 0) |
| `dihedral` | float | deg | 0.0 | Dihedral angle (tips up > 0) |
| `naca.m` | int | — | — | Max camber %chord: 0, 2, 4 |
| `naca.p` | int | — | — | Camber position (tenths): 0, 4 |
| `naca.t` | int | — | — | Max thickness %chord: 6–24 |
| `naca.chord_length` | float | — | 1.0 | Airfoil coordinate scale |

## Simulation Conditions (CLI args)

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--aoa` | yes | — | Angle of attack (deg) |
| `--mach` | yes | — | Freestream Mach number |
| `--re` | no | 3e6 | Reynolds number (corrected script only) |

## Usage

```bash
# VLM only
python generate_design.py design.json --aoa 10 --mach 0.3

# VLM + VortexNet correction
python generate_design_corrected.py design.json --aoa 10 --mach 0.3 --re 3e6
```

## Hardcoded (delta wing identity)

Taper = 0, tip chord = 0 m, symmetric = true.

## VLM.py Installation

Copy `VLM.py` to both:
1. `MF-VortexNet/scripts/VLM.py`
2. `<conda-env>/lib/python3.10/site-packages/SUAVE/Methods/Aerodynamics/Common/Fidelity_Zero/Lift/VLM.py`
