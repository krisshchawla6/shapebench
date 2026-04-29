"""DrivAer_Star-specific prompt guidance."""

DRIVAER_CONTEXT = """
Environment: DrivAer_Star
Objective context: minimize drag coefficient Cd (reward = -Cd).

Surrogate integration facts in this repository:
- Inputs are deformed VTK mesh-derived cell features [cx, cy, cz, area, nx, ny, nz].
- Model predicts pressure and wall shear components (x/y/z).
- Drag decomposition should satisfy:
  drag ~= drag_pressure + drag_shear

Important caveats to reason about:
- base_vtk style (E/F/N) should align with norm_stats file family.
- Missing rendered flow images reduces confidence for visual flow diagnosis.
"""

DRIVAER_FAILURE_INTERPRETATION = """
Typical failure patterns to consider (only when supported by evidence):
- style/norm mismatch risk causing surrogate-domain inconsistency,
- boundary-collapsed parameter proposals (many params near hard limits),
- ground-clearance / floor-clearance collapse visible in final-silhouette or geometry evidence
  (e.g., tail-floor droop toward/into the ground plane; "unreal car underbody"),
  which can manifest as weird local flow/pressure tendencies near the floor,
- aggressive coupled body scaling and angle deformations,
- decomposition inconsistency in integrated force bookkeeping,
- implausible Cd/lift magnitudes.

If you detect ground/floor-clearance collapse:
1) Select the closest failure mechanism from the allowed catalogs, preferring:
   - GEOMETRY_DEFORMATION_EXCESSIVE when the deformation appears unrealistic, otherwise
   - MESH_QUALITY_RISK, otherwise
   - OTHER.
2) Recommend mitigations using the allowed mitigation catalog only, preferring:
   - TIGHTEN_PARAMETER_BOUNDS and/or
   - ADD_GEOMETRY_REGULARIZATION
   to represent enforcing an absolute minimum floor clearance (ground clearance) constraint.
"""

