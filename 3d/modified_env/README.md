# CFD Simulation Environment (cloned + modified from https://doi.org/10.1016/j.jcp.2020.110080)

**Important Note**
    -temporary value for local development (not accurate convergance) (revert back on HPC nodes for actual runs)
    final_time = 10.0  # Reduced for testing (original: 2.0*(xmax-xmin)=90)
 
Simplified interface for running 2D CFD simulations on parametric shapes using FEniCS.

## Installation

```bash
# Create conda environment with FEniCS
conda create -n fenics_env -c conda-forge python=3.8 fenics=2019.1.0 mshr -y
conda activate fenics_env

# Install dependencies
pip install -r requirements.txt
```

## Files

- `run_case.py` - Wrapper script for running simulations
- `parametered_env.py` - Environment configuration/parameters
- `environment.py` - RL environment class
- `fenics_solver.py` - FEniCS CFD solver (Navier-Stokes IPCS)
- `shapes_utils.py` - Shape generation via Bezier curves
- `meshes_utils.py` - Mesh I/O utilities

## Quick Start

### Option 1: Run from CSV

```bash
python run_case.py actions.csv
```

### Option 2: Python API

```python
import numpy as np
from run_case import env, run_action, run_from_csv

# Define action: 4 control points × 3 params (radius, angle, edge)
# Values in range [-1, 1]
action = np.array([
    [ 0.5,  0.0,  0.3],  # Point 0
    [-0.2,  0.1,  0.5],  # Point 1
    [ 0.3, -0.1,  0.4],  # Point 2
    [-0.4,  0.2,  0.6],  # Point 3
]).flatten()

# Run single action
env.reset()
next_state, terminal, reward = run_action(action)

print(f"Drag: {env.drag[-1]:.4f}")
print(f"Lift: {env.lift[-1]:.4f}")
print(f"Reward (L/D): {reward:.4f}")
```

### Option 3: Run from CSV file

```python
from run_case import run_from_csv

# CSV format: each row is one action (12 comma-separated values)
results = run_from_csv('actions.csv')
```

## Action Format

Each action has 12 values (4 control points × 3 parameters):

| Parameter | Range | Description |
|-----------|-------|-------------|
| radius | [-1, 1] | Deformation magnitude |
| angle | [-1, 1] | Angular offset |
| edge | [-1, 1] | Edge sharpness (0=smooth, 1=sharp) |

**CSV example:**
```
0.5,0.0,0.3,-0.2,0.1,0.5,0.3,-0.1,0.4,-0.4,0.2,0.6
```

## Configuration (parametered_env.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `final_time` | 10.0 | Simulation time (90 for production) |
| `reynolds` | 100.0 | Reynolds number |
| `cfl` | 0.5 | CFL condition |
| `max_deformation` | 3.0 | Max shape deformation |
| `shape_h` | 1.0 | Boundary mesh size |
| `domain_h` | 0.8 | Domain mesh size |
| `cell_limit` | 50000 | Max mesh triangles |

## Output Structure

```
save/
├── csv/       # Shape control point coordinates
├── png/       # Shape visualizations
├── xml/       # Computational meshes
├── sol/       # Flow field images (u, v, p)
├── drag_lift  # Drag/lift values per iteration
└── reward_penalization  # Reward values
```

## Running Tests

```bash
cd modified_env
conda activate fenics_env
python tests/test_airfoil_run/test_4point_airfoil.py
```

## Notes
- Linux/WSL only (uses Unix commands)
- FEniCS 2019.1.0 legacy (stable)
- IPCS scheme for incompressible Navier-Stokes
Local runtime:
- ~2.5 min per action with `final_time=10`
- ~22 min per action with `final_time=90` (production)