# BlendedNet Surrogate Environment

Transolver-based surrogate predicting surface aerodynamic loads (Cp, Cfx, Cfz) on parameterized blended-wing-body (BWB) geometries. Meshes are generated on-the-fly from planform parameters via OpenVSP.

## How it works

```
design.json (9 geom + 3 flight params)
        |
        v
  OpenVSP (model.vsp3)  -->  STL surface mesh (mm -> m)
        |
        v
  PyVista: extract points + normals, subsample 8192 pts
        |
        v
  Transolver: pos (N,3) + fx (N,15) --> Cp, Cfx, Cfz (N,3)
        |
        v
  Post-process: map to full mesh, render Cp/Cfx images, compute L/D
        |
        v
  save/results.json + save/fields.npz + save/sol/*.png
```

The model takes per-point features `fx` = 9 planform params + 3 flight conditions (log10(Re), Mach, alpha) + 3 surface normals, tiled and concatenated for each of the 8192 sampled surface points. OpenVSP generates the mesh from `data/model.vsp3` by setting the Planform user parameters, then exporting STL. The STL is converted from mm to meters to match the original CFD dataset convention.

## Input JSON

```json
{
    "B1": 130.0, "B2": 80.0, "B3": 450.0,
    "C2": 700.0, "C3": 220.0, "C4": 75.0,
    "S1": 50.0, "S2": 48.0, "S3": 30.0,
    "Re": 1e7, "Mach": 0.3, "alpha": 5.0
}
```

| Key | Description | Units / Range |
|-----|-------------|---------------|
| B1, B2, B3 | Span sections | mm (B1: 100-200, B2: 50-200, B3: 200-700) |
| C2, C3, C4 | Chord sections (C1 fixed at 1000) | mm (C2: 550-850, C3: 180-280, C4: 60-90) |
| S1, S2, S3 | Sweep angles | deg (S1: 40-60, S2: 40-60, S3: 24-40) |
| Re | Reynolds number | raw value, log10 applied internally |
| Mach | Freestream Mach | - |
| alpha | Angle of attack | degrees |

## Output structure

```
case_dir/
  save/
    results.json       CL, CD, L/D, Cp/Cfx/Cfz means
    fields.npz         pos (8192,3), Cp/Cfx/Cfz (8192,)
    sol/
      Cp_iso.png       Pressure coefficient, isometric view
      Cp_top.png       Pressure coefficient, top-down view
      Cfx_iso.png      Skin friction x, isometric view
      Cfx_top.png      Skin friction x, top-down view
```

## Usage (ShapeEvolve framework)

```python
from environments.BlendedNet import BlendedNetEnvironment

env = BlendedNetEnvironment(mach=0.3, re=1e7, alpha=5.0)
reward, results = env.simulate("design.json", "case_dir/")
```

## Files

```
BlendedNet/
  environment.py           BaseEnvironment implementation
  mesh_generator.py        OpenVSP mesh generation wrapper
  _vsp_generate_stl.py     Subprocess script (runs under Python 3.10)
  prompt_blocks.py         LLM prompt templates
  postprocessing_blended_body.py   Standalone post-processing
  model/
    transolver_best.pt     Trained weights
    norm_stats.pt          Input/output normalization stats
    Transolver.py          Model definition
    layers/                Attention + MLP modules
  data/
    model.vsp3             OpenVSP parameterized BWB model
    geom_params_*.ini      Reference parameter ranges
    case_data_*.dat        Reference CFD integrated loads
```

## Dependencies

`torch`, `numpy`, `pyvista`, `scipy`, `timm`, `einops`

OpenVSP 3.48+ installed via `.deb`, with Python bindings in a `openvsp310` conda env (Python 3.10).
