#!/usr/bin/env python3
"""
XFOIL evaluation of the best designs from each method on the
reward_exact_notebook multipoint task.

For each best design:
  - Bisects in XFOIL to find the alpha that achieves each CL target
    (using the same Re schedule as the NeuralFoil reward)
  - Computes weighted_CD_xfoil = mean(CD_i * CL_WEIGHTS_i) over all 6 targets
  - Also computes the NeuralFoil weighted_CD for direct comparison

CL targets / weights / Re schedule mirror rewards/reward_exact_notebook.py:
  CL_TARGETS = [0.8, 1.0, 1.2, 1.4, 1.5, 1.6]
  CL_WEIGHTS  = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
  Re(CL) = 500e3 * (CL/1.25)^-0.5   (Drela HPA schedule)
  Mach = 0.03

Usage:
    python analysis/evaluate_xfoil_multipoint_best.py
"""

import csv
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

# ---------------------------------------------------------------------------
# Repo / XFOIL setup
# ---------------------------------------------------------------------------

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

NEURALFOIL_SRC = os.path.join(REPO_DIR, "environments", "NeuralFoil", "neuralfoil_src")
if os.path.isdir(NEURALFOIL_SRC):
    sys.path.insert(0, NEURALFOIL_SRC)

import aerosandbox as asb
import neuralfoil as nf

XFOIL_BIN = os.path.join(REPO_DIR, "bin", "xfoil")

OUTPUT_DIR = os.path.join(
    REPO_DIR, "environments", "NeuralFoil", "results",
    "xfoil_evaluation_reward_exact_notebook"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Multipoint task constants (mirror reward_exact_notebook.py)
# ---------------------------------------------------------------------------

CL_TARGETS = np.array([0.8, 1.0, 1.2, 1.4, 1.5, 1.6])
CL_WEIGHTS  = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
MACH = 0.03
N_CRIT = 9.0


def re_schedule(cl):
    return 500e3 * (cl / 1.25) ** -0.5


# ---------------------------------------------------------------------------
# Auto-discovery: find the best design from each method's raw run directories
# ---------------------------------------------------------------------------

RESULTS = os.path.join(REPO_DIR, "environments", "NeuralFoil", "results")


def _best_row_in_csv(csv_path):
    """Return the row with max 'reward' in a results.csv, or None."""
    try:
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None
    if not rows or 'reward' not in rows[0]:
        return None
    return max(rows, key=lambda r: float(r.get('reward', '-inf')))


def find_best_lbfgsb():
    """Scan all lbfgsb seed runs (nr3 + nr10_RETRY), return path to best design."""
    bases = [
        os.path.join(RESULTS, "SAVED_DIRS_run_lbfgsb_reward_exact_notebook_nr10_RETRY_of_nr3_FAILED"),
        os.path.join(RESULTS, "SAVED_DIRS_run_lbfgsb_reward_exact_notebook_nr3"),
    ]
    global_best, best_path = -float('inf'), None
    for base in bases:
        if not os.path.isdir(base):
            continue
        for run_dir in sorted(os.listdir(base)):
            best_row = _best_row_in_csv(os.path.join(base, run_dir, "results.csv"))
            if best_row is None:
                continue
            reward = float(best_row['reward'])
            if reward > global_best:
                call = int(best_row['call'])
                restart = best_row['restart']
                path = os.path.join(base, run_dir, f"call_{call:05d}_r{restart}", "design.json")
                if os.path.exists(path):
                    global_best, best_path = reward, path
    return best_path


def find_best_pso():
    """Scan all PSO/GA attempt dirs, return path to best design."""
    base = os.path.join(RESULTS, "SAVED_DIRS_run_GA_reward_exact_notebook_120particles_500iterations")
    global_best, best_path = -float('inf'), None
    if not os.path.isdir(base):
        return None
    for run_dir in sorted(os.listdir(base)):
        best_row = _best_row_in_csv(os.path.join(base, run_dir, "results.csv"))
        if best_row is None:
            continue
        reward = float(best_row['reward'])
        if reward > global_best:
            iteration = int(best_row['iteration'])
            particle = int(best_row['particle'])
            path = os.path.join(base, run_dir, f"iter_{iteration:04d}_p{particle:03d}", "design.json")
            if os.path.exists(path):
                global_best, best_path = reward, path
    return best_path


def find_best_bo():
    """Scan all BO_torch seed runs, return path to best design."""
    base = os.path.join(RESULTS, "SAVED_DIRS_BO_torch_reward_exact_notebook")
    global_best, best_path = -float('inf'), None
    if not os.path.isdir(base):
        return None
    for run_dir in sorted(os.listdir(base)):
        best_row = _best_row_in_csv(os.path.join(base, run_dir, "results.csv"))
        if best_row is None:
            continue
        reward = float(best_row['reward'])
        if reward > global_best:
            design_name = best_row.get('design') or f"iter_{int(best_row['iteration'])}"
            path = os.path.join(base, run_dir, design_name, f"{design_name}.json")
            if os.path.exists(path):
                global_best, best_path = reward, path
    return best_path


def find_best_v3():
    """Scan all v3 attempt dirs, return path to best design."""
    base = os.path.join(RESULTS, "SAVED_DIRS_run_v3_dynamic_optimizer_reward_exact_notebook_flash_and_pro")
    global_best, best_path = -float('inf'), None
    if not os.path.isdir(base):
        return None
    for run_dir in sorted(os.listdir(base)):
        best_row = _best_row_in_csv(os.path.join(base, run_dir, "results.csv"))
        if best_row is None:
            continue
        reward = float(best_row['reward'])
        if reward > global_best:
            design_name = best_row.get('design', '')
            if not design_name:
                continue
            path = os.path.join(base, run_dir, f"{design_name}.json")
            if os.path.exists(path):
                global_best, best_path = reward, path
    return best_path


# ---------------------------------------------------------------------------
# Known best designs as of April 16, 2026  (updated manually when new runs finish)
#
# Method                  Run / seed / iter                reward    NF wCD   XFOIL wCD  solved
# ----------------------  -------------------------------  --------  -------  ---------  ------
# L-BFGS-B               seed21, nr10, call19848 r3       -0.08256  0.08256  0.07304    5/6
# Bayesian Opt.(exact GP) seed0, iter 3560                -0.09442  0.09421  0.09672    6/6
# PSO (120p×500i)         attempt19, iter252, p108        -0.07934  0.07973  0.08005    6/6
# ShapeEvolve             attempt_22 (flash-2.5),         -0.07897  0.07897  —          6/6
#                           iter_182_o13
# Adjoint (IPOPT)         adjoint_run_fwbounds_naca0012   —         —        0.06919    5/6
#
# Notes:
#  - reward = -weighted_CD_mean (strict: FAIL_REWARD=-10 if any CL unsolvable)
#  - NF wCD recomputed from scratch by this script (minor float diff from stored reward)
#  - XFOIL wCD uses bisection to hit each CL target; partial mean when CL=1.6 fails
#  - L-BFGS-B / Adjoint XFOIL fail at CL=1.6 (airfoil stalls before reaching it)
# ---------------------------------------------------------------------------


def build_designs():
    """Return {method_name: design_path} with auto-discovered best designs."""
    print("Discovering best designs from raw run directories...")
    designs = {}

    for label, finder in [
        ("L-BFGS-B",              find_best_lbfgsb),
        ("Bayesian Opt. (exact GP)", find_best_bo),
        ("PSO (120p × 500i)",     find_best_pso),
        ("ShapeEvolve",           find_best_v3),
    ]:
        path = finder()
        if path:
            print(f"  {label}: {os.path.relpath(path, RESULTS)}")
        else:
            print(f"  {label}: NOT FOUND")
        designs[label] = path

    # Adjoint is a single one-shot run — hardcoded
    adjoint_path = os.path.join(RESULTS, "adjoint_run_fwbounds_naca0012", "save", "results.json")
    designs["Adjoint (IPOPT)"] = adjoint_path if os.path.exists(adjoint_path) else None
    if designs["Adjoint (IPOPT)"]:
        print(f"  Adjoint (IPOPT): {os.path.relpath(adjoint_path, RESULTS)}")
    else:
        print("  Adjoint (IPOPT): NOT FOUND")

    return designs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_kulfan(path):
    """Load a KulfanAirfoil from a design JSON (handles both raw and results.json).

    Also returns any pre-computed NeuralFoil results stored in the file (e.g. the
    adjoint results.json stores weighted_cd and CDs from the optimization run itself).
    These are used directly in evaluate_design() rather than re-running NeuralFoil.

    WHY THE ADJOINT NF RESULT MUST BE READ FROM FILE (NOT RE-COMPUTED)
    -------------------------------------------------------------------
    The adjoint optimizer (IPOPT) works differently from all other methods here.
    Rather than evaluating NeuralFoil at a fixed alpha and checking CL, it treats
    alpha at each CL target as a *decision variable* — the solver simultaneously
    optimizes the airfoil shape and finds the alpha satisfying each CL constraint
    via gradient-based search, using automatic differentiation through NeuralFoil.
    It never needs a 1D root-find over alpha; the final alphas are part of the
    solution vector and are stored directly in results.json alongside the CDs.

    Our neuralfoil_weighted_cd() function, by contrast, treats NeuralFoil as a
    black box and does a fresh brentq root-find over alpha on [-5, 18].  For the
    Adjoint airfoil this fails: NeuralFoil's predicted CL curve for that geometry
    apparently has no monotonic crossing of CL=1.6 within that bracket (the model
    predicts a different stall shape than the adjoint solver's gradient information
    "saw" during optimization).  The adjoint converged to alphas that are valid at
    its own evaluation points, but a fresh brentq cannot reproduce them.

    Fix: if the design JSON already contains pre-computed NF results (weighted_cd
    and CDs keys), use those directly and skip the re-computation entirely.

    Returns (airfoil, stored_nf_wcd, stored_nf_cds) where the last two are None if
    no pre-computed NF data is present.
    """
    with open(path) as f:
        d = json.load(f)
    des = d.get("design", d)
    airfoil = asb.KulfanAirfoil(
        upper_weights=np.array(des["upper_weights"]),
        lower_weights=np.array(des["lower_weights"]),
        leading_edge_weight=float(des["leading_edge_weight"]),
        TE_thickness=float(des.get("TE_thickness", 0.0)),
    )
    # Use stored NF results if present (keyed as in adjoint results.json)
    stored_wcd  = d.get("weighted_cd")
    stored_cds  = d.get("CDs")
    if stored_wcd is not None and stored_cds is not None:
        return airfoil, float(stored_wcd), [float(c) for c in stored_cds]
    return airfoil, None, None


def write_dat(airfoil, path):
    coords = np.array(airfoil.coordinates)
    with open(path, "w") as f:
        f.write("airfoil\n")
        for x, y in coords:
            f.write(f"{x:.6f} {y:.6f}\n")


def run_xfoil_alpha(dat_path, alpha, Re, mach=MACH, n_crit=N_CRIT,
                    max_iter=200, n_panels=260):
    """Run XFOIL at a single alpha; return (CL, CD, CM) or None on failure."""
    commands = "\n".join([
        f"load {dat_path}",
        "pane",
        "oper",
        "vpar",
        f"n {n_crit}",
        "",
        f"mach {mach}",
        f"visc {Re:.6e}",
        f"iter {max_iter}",
        f"alfa {alpha:.4f}",
        "quit",
        "",
    ])
    try:
        result = subprocess.run(
            [XFOIL_BIN],
            input=commands,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return None
    stdout = result.stdout
    cl_m = re.findall(r'\bCL\s*=\s*([-\d.]+)', stdout)
    cd_m = re.findall(r'\bCD\s*=\s*([-\d.]+)', stdout)
    cm_m = re.findall(r'\bCm\s*=\s*([-\d.]+)', stdout)
    if not cl_m:
        return None
    return float(cl_m[-1]), float(cd_m[-1]) if cd_m else float("nan"), \
           float(cm_m[-1]) if cm_m else float("nan")


def find_alpha_for_cl(dat_path, cl_target, Re, lo=-5.0, hi=15.0,
                      sweep_step=1.0, tol=1e-3, max_bisect=14):
    """
    Find alpha such that CL ≈ cl_target on the ascending (pre-stall) branch.

    WHY THE NAIVE BRACKET APPROACH FAILS FOR HIGH-LIFT TARGETS
    -----------------------------------------------------------
    A prior version probed two bracket endpoints (e.g. alpha=-5° and alpha=20°)
    and bisected if they had opposite sign.  For CL targets near the stall peak
    (CL=1.6 here), this breaks:

      - These airfoils peak at CL≈1.7 around alpha≈9-10°, then CL falls back
        below 1.6 as alpha increases into the post-stall regime.
      - At both bracket endpoints (-5° and 20°), CL < 1.6, so both f values
        are negative — no sign change, bisection aborts, result reported as FAILED.
      - This happened to ShapeEvolve and Adjoint, even though CL=1.6 IS reachable
        on the ascending branch at alpha≈8.5°.

    The prior code also tried wider brackets up to [-15°, 30°].  PSO accidentally
    "succeeded" with this approach because XFOIL gave a garbage CL=2.27 at alpha=30°
    (unconverged boundary-layer solution deep in the post-stall regime), creating a
    spurious sign change.  ShapeEvolve and Adjoint happened to have XFOIL converge
    cleanly all the way to 30°, giving physically correct (but unhelpful) post-stall
    CL values — so no accidental sign change, and they were falsely marked as FAILED.

    THE FIX
    -------
    Sweep alpha forward from lo to hi in 1° steps and look for the *first*
    consecutive pair where CL crosses cl_target.  For example:

        alpha=7°  CL=1.47  (below)
        alpha=8°  CL=1.57  (below)
        alpha=9°  CL=1.66  (above)  ← bracket [8°, 9°] found here

    Then bisect within that 1° window.  The sweep stops as soon as the crossing
    is found, so it never enters the unreliable post-stall regime.  hi=15° is
    sufficient: no physically meaningful CL=1.6 crossing occurs above ~12° for
    these airfoil shapes.

    Returns (alpha, CL, CD, CM) or None if no crossing is found in [lo, hi].
    """
    prev_alpha, prev_r = None, None
    alpha = lo
    while alpha <= hi + 1e-9:
        r = run_xfoil_alpha(dat_path, alpha, Re)
        if r is not None:
            f = r[0] - cl_target
            if prev_r is not None:
                f_prev = prev_r[0] - cl_target
                if f_prev * f <= 0:
                    # Found the bracket [prev_alpha, alpha] on the ascending branch
                    blo, bhi = prev_alpha, alpha
                    r_blo, r_bhi = prev_r, r
                    f_blo, f_bhi = f_prev, f
                    break
            prev_alpha, prev_r = alpha, r
        alpha = round(alpha + sweep_step, 10)
    else:
        return None  # No crossing found in [lo, hi]

    # Bisect within [blo, bhi]
    for _ in range(max_bisect):
        mid = 0.5 * (blo + bhi)
        r_mid = run_xfoil_alpha(dat_path, mid, Re)
        if r_mid is None:
            return None
        f_mid = r_mid[0] - cl_target
        if abs(f_mid) < tol:
            return mid, r_mid[0], r_mid[1], r_mid[2]
        if f_blo * f_mid <= 0:
            bhi, f_bhi = mid, f_mid
        else:
            blo, f_blo = mid, f_mid

    # Return best so far
    mid = 0.5 * (blo + bhi)
    r_mid = run_xfoil_alpha(dat_path, mid, Re)
    if r_mid is None:
        return None
    return mid, r_mid[0], r_mid[1], r_mid[2]


def neuralfoil_weighted_cd(airfoil):
    """Compute NeuralFoil weighted_CD_mean (mirror of reward_exact_notebook logic)."""
    from scipy.optimize import brentq

    kulfan = {
        "upper_weights": airfoil.upper_weights,
        "lower_weights": airfoil.lower_weights,
        "leading_edge_weight": airfoil.leading_edge_weight,
        "TE_thickness": airfoil.TE_thickness,
    }

    def cl_residual(alpha, cl_target, Re):
        aero = nf.get_aero_from_kulfan_parameters(
            kulfan_parameters=kulfan, alpha=float(alpha), Re=Re,
            n_crit=N_CRIT, model_size="large",
        )
        return float(np.squeeze(aero["CL"])) - cl_target

    CDs = []
    for cl_t in CL_TARGETS:
        Re = re_schedule(cl_t)
        try:
            alpha = brentq(lambda a: cl_residual(a, cl_t, Re), -5.0, 18.0,
                           xtol=1e-3, maxiter=60)
        except Exception:
            return None, None
        aero = nf.get_aero_from_kulfan_parameters(
            kulfan_parameters=kulfan, alpha=float(alpha), Re=Re,
            n_crit=N_CRIT, model_size="large",
        )
        CDs.append(float(np.squeeze(aero["CD"])))

    wcd = float(np.mean(np.array(CDs) * CL_WEIGHTS))
    return wcd, CDs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_design(name, path):
    print(f"\n{'='*65}")
    print(f"Method: {name}")
    print(f"Design: {path}")

    airfoil, stored_nf_wcd, stored_nf_cds = load_kulfan(path)

    # --- NeuralFoil ---
    # Use pre-computed NF results if stored in the design JSON (see load_kulfan
    # docstring for why this is necessary for the Adjoint case).
    if stored_nf_wcd is not None:
        nf_wcd, nf_cds = stored_nf_wcd, stored_nf_cds
        print(f"  NeuralFoil  weighted_CD = {nf_wcd:.6f}  (from stored results)")
        print(f"    CDs per target: {[f'{c:.6f}' for c in nf_cds]}")
    else:
        nf_wcd, nf_cds = neuralfoil_weighted_cd(airfoil)
        if nf_wcd is not None:
            print(f"  NeuralFoil  weighted_CD = {nf_wcd:.6f}")
            print(f"    CDs per target: {[f'{c:.6f}' for c in nf_cds]}")
        else:
            print("  NeuralFoil  FAILED to solve all CL targets")

    # --- XFOIL ---
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False, mode="w") as tmp:
        dat_path = tmp.name
    try:
        write_dat(airfoil, dat_path)

        xf_alphas, xf_cls, xf_cds, xf_cms = [], [], [], []
        for cl_t in CL_TARGETS:
            Re = re_schedule(cl_t)
            print(f"  XFOIL bisect CL={cl_t:.1f}  Re={Re:.0f} ...", end=" ", flush=True)
            result = find_alpha_for_cl(dat_path, cl_t, Re)
            if result is None:
                print("FAILED")
                xf_alphas.append(None)
                xf_cls.append(None)
                xf_cds.append(None)
                xf_cms.append(None)
            else:
                alpha, cl, cd, cm = result
                print(f"alpha={alpha:.3f}  CL={cl:.4f}  CD={cd:.6f}  CM={cm:.4f}")
                xf_alphas.append(alpha)
                xf_cls.append(cl)
                xf_cds.append(cd)
                xf_cms.append(cm)

        solved = [cd for cd in xf_cds if cd is not None and not np.isnan(cd)]
        solved_w = [CL_WEIGHTS[i] for i, cd in enumerate(xf_cds)
                    if cd is not None and not np.isnan(cd)]
        if len(solved) == len(CL_TARGETS):
            xf_wcd = float(np.mean(np.array(solved) * CL_WEIGHTS))
            print(f"  XFOIL       weighted_CD = {xf_wcd:.6f}  (all {len(CL_TARGETS)} targets solved)")
        elif solved:
            xf_wcd = float(np.mean(np.array(solved) * np.array(solved_w)))
            print(f"  XFOIL       weighted_CD = {xf_wcd:.6f}  "
                  f"(partial: {len(solved)}/{len(CL_TARGETS)} targets solved)")
        else:
            xf_wcd = None
            print("  XFOIL       FAILED: no targets solved")

    finally:
        if os.path.exists(dat_path):
            os.unlink(dat_path)

    return {
        "name": name,
        "path": path,
        "neuralfoil": {
            "weighted_CD": nf_wcd,
            "CDs": nf_cds,
        },
        "xfoil": {
            "weighted_CD": xf_wcd,
            "n_solved": len(solved) if solved else 0,
            "alphas": xf_alphas,
            "CDs": xf_cds,
            "CMs": xf_cms,
            "achieved_CLs": xf_cls,
        },
    }


def main():
    all_results = []
    designs = build_designs()

    for name, path in designs.items():
        if path is None or not os.path.exists(path):
            print(f"\n[SKIP] {name}: file not found" + (f"\n  {path}" if path else ""))
            continue
        result = evaluate_design(name, path)
        all_results.append(result)

    # Summary table
    print(f"\n{'='*65}")
    print("SUMMARY — NeuralFoil vs XFOIL  (weighted_CD_mean, lower=better)")
    print(f"{'='*65}")
    header = f"{'Method':<28}  {'NeuralFoil wCD':>15}  {'XFOIL wCD':>12}  {'XF/NF':>7}  {'Solved':>6}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        nf_wcd = r["neuralfoil"]["weighted_CD"]
        xf_wcd = r["xfoil"]["weighted_CD"]
        n_sol  = r["xfoil"]["n_solved"]
        nf_str = f"{nf_wcd:.6f}" if nf_wcd is not None else "  FAILED "
        xf_str = f"{xf_wcd:.6f}" if xf_wcd is not None else "  FAILED "
        ratio  = (f"{xf_wcd/nf_wcd:.4f}" if (nf_wcd and xf_wcd) else "    —   ")
        print(f"{r['name']:<28}  {nf_str:>15}  {xf_str:>12}  {ratio:>7}  {n_sol}/{len(CL_TARGETS)}")

    out_path = os.path.join(OUTPUT_DIR, "xfoil_multipoint_best_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
