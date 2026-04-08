"""
Direct XFOIL evaluation of best airfoil designs found by normalized-penalty
optimization, compared against the original adjoint result and NACA0012.

Calls XFOIL via subprocess (bypasses AeroSandbox wrapper which crashes on
this system due to a display/Fortran I/O issue).
"""

import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, REPO_DIR)

NEURALFOIL_SRC = os.path.join(REPO_DIR, "environments", "NeuralFoil", "neuralfoil_src")
if os.path.isdir(NEURALFOIL_SRC):
    sys.path.insert(0, NEURALFOIL_SRC)

import aerosandbox as asb

OUTPUT_DIR = os.path.join(REPO_DIR, "environments", "NeuralFoil", "results",
                           "xfoil_validation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use xfoil from shared scratch storage so compute nodes can find it
# (compute nodes may not have it installed system-wide)
XFOIL_BIN = os.path.join(REPO_DIR, "bin", "xfoil")


def load_kulfan(path):
    d = json.load(open(path))
    des = d.get("design", d)
    return asb.KulfanAirfoil(
        upper_weights=np.array(des["upper_weights"]),
        lower_weights=np.array(des["lower_weights"]),
        leading_edge_weight=float(des["leading_edge_weight"]),
        TE_thickness=float(des.get("TE_thickness", 0.0)),
    )


def write_airfoil_dat(airfoil, path):
    """Write Kulfan airfoil coordinates to a plain dat file for XFOIL."""
    coords = np.array(airfoil.coordinates)
    with open(path, "w") as f:
        f.write("airfoil\n")
        for x, y in coords:
            f.write(f"{x:.6f} {y:.6f}\n")


def run_xfoil(dat_path, alpha=5.0, Re=1e7, mach=0.2, n_crit=9.0,
              max_iter=200, n_panels=260):
    """Run XFOIL on the given airfoil dat file and return CL, CD, CM."""
    with tempfile.TemporaryDirectory() as tmpdir:
        commands = "\n".join([
            f"load {dat_path}",
            f"pane",
            f"oper",
            f"vpar",
            f"n {n_crit}",
            "",                      # exit vpar
            f"mach {mach}",          # mach must be set before visc
            f"visc {Re:.6e}",
            f"iter {max_iter}",
            f"alfa {alpha}",
            "quit",
            "",
        ])

        try:
            result = subprocess.run(
                [XFOIL_BIN],
                input=commands,
                capture_output=True,
                text=True,
                timeout=120,
            )
            stdout = result.stdout
        except subprocess.TimeoutExpired:
            return None, "XFOIL timeout"

        # Parse final converged result directly from stdout.
        # Convergence lines look like:
        #   "       a =  5.000      CL =  0.5800"
        #   "      Cm = -0.0002     CD =  0.00617 ..."
        # Transition lines: "Side 1  free  transition at x/c =  0.0510"
        cl_match  = re.findall(r'\bCL\s*=\s*([-\d.]+)', stdout)
        cd_match  = re.findall(r'\bCD\s*=\s*([-\d.]+)', stdout)
        cm_match  = re.findall(r'\bCm\s*=\s*([-\d.]+)', stdout)
        xtr_match = re.findall(r'Side \d.*?x/c\s*=\s*([\d.]+)', stdout)
        if not cl_match:
            return None, "No CL in stdout. Tail:\n" + "\n".join(stdout.splitlines()[-20:])
        CL = float(cl_match[-1])
        CD = float(cd_match[-1]) if cd_match else float("nan")
        CM = float(cm_match[-1]) if cm_match else float("nan")
        xtr_u = float(xtr_match[-2]) if len(xtr_match) >= 2 else float("nan")
        xtr_l = float(xtr_match[-1]) if len(xtr_match) >= 1 else float("nan")
        return {
            "alpha": float(alpha),
            "CL": CL,
            "CD": CD,
            "CM": CM,
            "L_D": CL / CD if CD > 1e-9 else float("nan"),
            "xtr_upper": xtr_u,
            "xtr_lower": xtr_l,
        }, None


DESIGNS = {
    "NACA0012 (baseline)": None,  # special case
    # "Adjoint IPOPT from NACA0012 (NeuralFoil L/D=191.9)":
    #     os.path.join(REPO_DIR, "environments/NeuralFoil/results/ld_adjoint_run/save/results.json"),
    "Adjoint IPOPT from NACA0012 (NeuralFoil L/D=191.9)":
        os.path.join(REPO_DIR, "environments/NeuralFoil/results/ld_adjoint_run_NACA0012_original/save/results.json"),
    "GA att5 normalized λ=500 (NeuralFoil L/D=351.9)":
        os.path.join(REPO_DIR, "environments/NeuralFoil/adjoint/best_GA_att5.json"),
    "L-BFGS-B s3 normalized λ=500 (NeuralFoil L/D=351.9)":
        os.path.join(REPO_DIR, "environments/NeuralFoil/adjoint/best_lbfgsb_s3.json"),
    "L-BFGS-B s0 normalized λ=500 (NeuralFoil L/D=351.9→IPOPT 300.7)":
        os.path.join(REPO_DIR, "environments/NeuralFoil/adjoint/best_lbfgsb_s0.json"),
    "IPOPT CM≥−0.130 from GA att5 (NeuralFoil L/D=351.6)":
        os.path.join(REPO_DIR, "environments/NeuralFoil/results/ld_adjoint_cm130_from_GA_att5_orig_bounds/ld_adjoint_cm130_GA_att5.json"),
    "GA conf95 att5 (NeuralFoil L/D=314.4)":
        os.path.join(REPO_DIR, "environments/NeuralFoil/adjoint/best_GA_conf95_att5.json"),
    "lbfgsb conf95 s4 (NeuralFoil L/D=303.4)":
        os.path.join(REPO_DIR, "environments/NeuralFoil/adjoint/best_lbfgsb_conf95_s4.json"),
    "IPOPT CM≥−0.130 from GA conf95 att5 (NeuralFoil L/D=351.3)":
        os.path.join(REPO_DIR, "environments/NeuralFoil/results/ld_adjoint_cm130_from_GA_conf95_att5_orig_bounds/ld_adjoint_cm130_GA_conf95_att5.json"),
    "IPOPT framework bounds from GA att5 (NeuralFoil L/D=365.9)":
        os.path.join(REPO_DIR, "environments/NeuralFoil/results/ld_adjoint_cm130_from_GA_att5/ld_adjoint_cm130_GA_att5.json"),
}

ALPHA  = 5.0
RE     = 1e7
MACH   = 0.2
N_CRIT = 9.0

summary = []

for name, path in DESIGNS.items():
    print(f"\n{'='*60}")
    print(f"Design: {name}")

    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False, mode="w") as tmp:
        dat_path = tmp.name

    try:
        if path is None:
            af = asb.KulfanAirfoil("naca0012")
        else:
            af = load_kulfan(path)

        # Also get NeuralFoil prediction for comparison
        aero_nf = af.get_aero_from_neuralfoil(
            alpha=ALPHA, Re=RE, mach=MACH, model_size="large")
        ld_nf   = float(aero_nf["CL"]) / float(aero_nf["CD"])
        conf_nf = float(aero_nf["analysis_confidence"])
        cm_nf   = float(aero_nf["CM"])
        print(f"  NeuralFoil:  L/D={ld_nf:.3f}, CM={cm_nf:.4f}, conf={conf_nf:.4f}")

        write_airfoil_dat(af, dat_path)
        result, err = run_xfoil(dat_path, alpha=ALPHA, Re=RE, mach=MACH, n_crit=N_CRIT)

        if result is None:
            print(f"  XFOIL FAILED: {err}")
            xfoil_entry = {"error": err}
        else:
            print(f"  XFOIL:       L/D={result['L_D']:.3f}, CL={result['CL']:.4f}, "
                  f"CD={result['CD']:.6f}, CM={result['CM']:.4f}, "
                  f"xtr_u={result['xtr_upper']:.3f}, xtr_l={result['xtr_lower']:.3f}")
            xfoil_entry = result

        summary.append({
            "name": name,
            "neuralfoil": {"L_D": ld_nf, "CM": cm_nf, "conf": conf_nf},
            "xfoil": xfoil_entry,
        })

    finally:
        if os.path.exists(dat_path):
            os.unlink(dat_path)

print(f"\n{'='*60}")
print("SUMMARY — NeuralFoil vs XFOIL")
print(f"{'='*60}")
print(f"{'Design':<45} {'NF L/D':>8} {'XF L/D':>8} {'XF CM':>8} {'XF conf(xtr)':>12}")
for s in summary:
    nf = s["neuralfoil"]
    xf = s["xfoil"]
    if "error" in xf:
        xf_ld = "FAILED"
        xf_cm = "-"
        xf_xtr = "-"
    else:
        xf_ld  = f"{xf['L_D']:.1f}"
        xf_cm  = f"{xf['CM']:.4f}"
        xf_xtr = f"u={xf['xtr_upper']:.2f}/l={xf['xtr_lower']:.2f}"
    short = s["name"][:45]
    print(f"{short:<45} {nf['L_D']:>8.1f} {xf_ld:>8} {xf_cm:>8} {xf_xtr:>12}")

with open(os.path.join(OUTPUT_DIR, "xfoil_validation_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to: {OUTPUT_DIR}/xfoil_validation_summary.json")
