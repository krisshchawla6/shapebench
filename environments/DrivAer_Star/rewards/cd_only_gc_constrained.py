import json
import os

import numpy as np

from environments.base_reward import BaseReward

FAIL_REWARD = -5.0

# Cache base mesh point arrays keyed by absolute vtk path.
# apply_ffd is pure numpy so we only need the points, not the full pv object.
_BASE_PTS_CACHE = {}


def _get_base_pts(vtk_path: str) -> np.ndarray:
    abs_path = os.path.abspath(vtk_path)
    if abs_path not in _BASE_PTS_CACHE:
        import pyvista as pv
        mesh = pv.read(abs_path)
        _BASE_PTS_CACHE[abs_path] = np.array(mesh.points, dtype=np.float64)
    return _BASE_PTS_CACHE[abs_path]


def _compute_gc(params: dict, vtk_path: str) -> float:
    """Floor ground clearance relative to the deformed tire contact level (metres).

    Measures the minimum z of the underbody floor panels (centre of car,
    above the tire height band) minus the minimum z of the deformed mesh
    (tire contact level).  A value < gc_min indicates ground clearance collapse.

    Recommended by diagnostics.json (recommended_next_tests[1]) after finding
    GEOMETRY_DEFORMATION_EXCESSIVE on iter_517_o2 (Cd=0.065, 13/20 params at
    bounds, diffusor_angle=-8, car_size=0.8).
    """
    from environments.DrivAer_Star.mesh_generator import apply_ffd

    base_pts = _get_base_pts(vtk_path)
    d = apply_ffd(base_pts.copy(), params)
    dz = d[:, 2]
    dy = d[:, 1]

    dz_min = float(dz.min())
    dz_len = float(dz.max()) - dz_min
    dy_min = float(dy.min())
    dy_len = float(dy.max()) - dy_min

    if dz_len < 1e-9 or dy_len < 1e-9:
        return 0.0

    # Floor region: above tire height band (7% of total height) and below 30%,
    # laterally centred (inner 60% of width — excludes wheel arch regions).
    floor_mask = (
        (dz > dz_min + 0.07 * dz_len) &
        (dz < dz_min + 0.30 * dz_len) &
        (dy > dy_min + 0.20 * dy_len) &
        (dy < dy_min + 0.80 * dy_len)
    )
    if not floor_mask.any():
        return 0.0

    return float(dz[floor_mask].min() - dz_min)


class CdOnlyGcConstrainedReward(BaseReward):
    """Minimise Cd with a soft ground clearance constraint.

    reward = -Cd - lam_gc * max(0, gc_min - ground_clearance)

    ground_clearance = minimum z of underbody floor panels relative to the
    deformed tire contact level (see _compute_gc).

    In the feasible region (ground_clearance >= gc_min) reward = -Cd, so the
    objective is identical to cd_only when the constraint is satisfied.

    Default gc_min=0.07 m was chosen from mesh analysis: the worst surrogate-
    exploiting design (iter_517_o2, Cd=0.065) achieves 0.074 m, just above
    threshold; more extreme collapse is prevented by the penalty.
    """

    def __init__(self, rho=1.25, u=40.0, area_ref=2.37,
                 lam_gc=1.0, gc_min=0.07,
                 base_vtk=None, **kwargs):
        self.rho = rho
        self.u = u
        self.area_ref = area_ref
        self.lam_gc = lam_gc
        self.gc_min = gc_min
        self.base_vtk = base_vtk

    @staticmethod
    def add_args(parser):
        parser.add_argument('--rho', type=float, default=1.25)
        parser.add_argument('--u', type=float, default=40.0)
        parser.add_argument('--area_ref', type=float, default=2.37)
        parser.add_argument('--lam_gc', type=float, default=1.0,
                            help='Penalty coefficient for ground clearance violation')
        parser.add_argument('--gc_min', type=float, default=0.07,
                            help='Minimum floor ground clearance (m) relative to '
                                 'tire contact level; default 0.07 m')

    def evaluate(self, run_sim, design_path: str, case_dir: str) -> tuple:
        try:
            r = run_sim(design_path, case_dir)

            q = 0.5 * self.rho * self.u ** 2
            denom = q * self.area_ref if q * self.area_ref > 1e-12 else 1.0
            Cd = r['drag'] / denom

            # Ground clearance — use vtk_path from design JSON if present,
            # otherwise fall back to the base_vtk passed at construction.
            params = r['params']
            vtk_path = params.get('vtk_path') or self.base_vtk
            if vtk_path and os.path.exists(vtk_path):
                gc = _compute_gc(params, vtk_path)
            else:
                gc = self.gc_min  # unknown geometry: no penalty

            gc_violation = max(0.0, self.gc_min - gc)
            penalty = self.lam_gc * gc_violation
            reward = -Cd - penalty

            results_dict = {
                "design": params,
                "drag": r["drag"], "drag_pressure": r["drag_p"], "drag_shear": r["drag_w"],
                "lift": r["lift"], "lift_pressure": r["lift_p"], "lift_shear": r["lift_w"],
                "Cd": Cd,
                "ground_clearance": gc,
                "gc_min": self.gc_min,
                "gc_violation": gc_violation,
                "gc_penalty": penalty,
                "reward": reward,
                "rho": self.rho, "u": self.u, "area_ref": self.area_ref,
                "lam_gc": self.lam_gc,
            }
            with open(os.path.join(r["save_dir"], "results.json"), "w") as f:
                json.dump(results_dict, f, indent=2)

            feasible = gc_violation == 0.0
            if feasible:
                feedback = ""
            else:
                feedback = (
                    f"Ground clearance too low: gc={gc:.4f} m < gc_min={self.gc_min:.3f} m  "
                    f"(penalty={penalty:.4f}).  "
                    f"Reduce diffusor_angle magnitude, car_size, or trunklid_z."
                )

            return float(reward), {
                "metrics": {
                    "drag": r["drag"], "Cd": Cd,
                    "lift": r["lift"],
                    "ground_clearance": gc,
                    "gc_violation": gc_violation,
                    "drag_pressure": r["drag_p"], "drag_shear": r["drag_w"],
                    "reward": reward,
                },
                "images": r["images"],
                "feedback": feedback,
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[sim] FAILED: {e}")
            return float(FAIL_REWARD), {
                "metrics": {"reward": FAIL_REWARD},
                "images": [],
                "feedback": "Simulation failed.",
            }
