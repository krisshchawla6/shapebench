"""
Batch XFOIL validation of best designs from scale-up runs:
  GA attempts 6-25  (20 runs)
  lbfgsb seeds 5-39 (35 runs)

For each run, finds the best design by fitness_total from results.csv,
loads the design.json from the corresponding call directory, and evaluates
with XFOIL at alpha=5 deg, Re=1e7, Mach=0.2, N_crit=9.0.

All 55 evaluations run in parallel via multiprocessing.Pool.

Results saved to:
  environments/NeuralFoil/results/xfoil_validation_scale_up/summary.json
"""

import csv
import json
import multiprocessing
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

RESULTS_BASE = os.path.join(REPO_DIR, "environments", "NeuralFoil", "results")
OUTPUT_DIR   = os.path.join(RESULTS_BASE, "xfoil_validation_scale_up")
XFOIL_BIN    = os.path.join(REPO_DIR, "bin", "xfoil")

ALPHA  = 5.0
RE     = 1e7
MACH   = 0.2
N_CRIT = 9.0

os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_best_design_path(run_dir, run_type):
    """Return (fitness, design_json_path) for the best row in results.csv."""
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(csv_path):
        return None, None
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None
    best = max(rows, key=lambda r: float(r["fitness_total"]))
    fitness = float(best["fitness_total"])

    if run_type == "GA":
        call_dir = os.path.join(
            run_dir,
            f"iter_{int(best['iteration']):04d}_p{int(best['particle']):03d}",
        )
    else:  # lbfgsb
        call_dir = os.path.join(
            run_dir,
            f"call_{int(best['call']):05d}_r{int(best['restart'])}",
        )

    design_path = os.path.join(call_dir, "design.json")
    if not os.path.exists(design_path):
        return fitness, None
    return fitness, design_path


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
    coords = np.array(airfoil.coordinates)
    with open(path, "w") as f:
        f.write("airfoil\n")
        for x, y in coords:
            f.write(f"{x:.6f} {y:.6f}\n")


def run_xfoil(dat_path):
    commands = "\n".join([
        f"load {dat_path}",
        "pane",
        "oper",
        "vpar",
        f"n {N_CRIT}",
        "",
        f"mach {MACH}",
        f"visc {RE:.6e}",
        "iter 200",
        f"alfa {ALPHA}",
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

    cl_match  = re.findall(r'\bCL\s*=\s*([-\d.]+)', stdout)
    cd_match  = re.findall(r'\bCD\s*=\s*([-\d.]+)', stdout)
    cm_match  = re.findall(r'\bCm\s*=\s*([-\d.]+)', stdout)
    xtr_match = re.findall(r'Side \d.*?x/c\s*=\s*([\d.]+)', stdout)
    if not cl_match:
        return None, "No CL in stdout"
    CL = float(cl_match[-1])
    CD = float(cd_match[-1]) if cd_match else float("nan")
    CM = float(cm_match[-1]) if cm_match else float("nan")
    xtr_u = float(xtr_match[-2]) if len(xtr_match) >= 2 else float("nan")
    xtr_l = float(xtr_match[-1]) if len(xtr_match) >= 1 else float("nan")
    return {
        "CL": CL, "CD": CD, "CM": CM,
        "L_D": CL / CD if CD > 1e-9 else float("nan"),
        "xtr_upper": xtr_u, "xtr_lower": xtr_l,
    }, None


def evaluate_one(args):
    """Worker function: evaluate one design. Returns result dict."""
    label, run_type, idx, run_dir = args

    if not os.path.isdir(run_dir):
        return {"label": label, "run_type": run_type, "id": idx,
                "error": "run dir not found"}

    fitness, design_path = find_best_design_path(run_dir, run_type)
    if design_path is None:
        return {"label": label, "run_type": run_type, "id": idx,
                "error": "design not found", "nf_fitness": fitness}

    try:
        af = load_kulfan(design_path)
    except Exception as e:
        return {"label": label, "run_type": run_type, "id": idx,
                "error": str(e), "nf_fitness": fitness}

    aero_nf = af.get_aero_from_neuralfoil(alpha=ALPHA, Re=RE, mach=MACH, model_size="large")
    nf_ld   = float(aero_nf["CL"]) / float(aero_nf["CD"])
    nf_cm   = float(aero_nf["CM"])
    nf_conf = float(aero_nf["analysis_confidence"])

    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False, mode="w") as tmp:
        dat_path = tmp.name
    try:
        write_airfoil_dat(af, dat_path)
        xf, err = run_xfoil(dat_path)
    finally:
        if os.path.exists(dat_path):
            os.unlink(dat_path)

    if xf is None:
        print(f"  {label}: XFOIL FAILED: {err}", flush=True)
        return {"label": label, "run_type": run_type, "id": idx,
                "nf_fitness": fitness, "nf_ld": nf_ld, "nf_cm": nf_cm,
                "nf_conf": nf_conf, "xfoil_error": err}

    cm_ok = "YES" if xf["CM"] >= -0.133 else "NO"
    print(f"  {label}: NF={nf_ld:.1f} XF={xf['L_D']:.1f} CM={xf['CM']:.4f} {cm_ok}",
          flush=True)
    return {"label": label, "run_type": run_type, "id": idx,
            "nf_fitness": fitness, "nf_ld": nf_ld, "nf_cm": nf_cm,
            "nf_conf": nf_conf, "xfoil": xf}


def main():
    runs = []
    for att in range(6, 26):
        d = os.path.join(
            RESULTS_BASE,
            f"run_GA_ld_ratio_constrained_m02_re1e7_normalized_120particles_500iterations_attempt_{att}_AWS",
        )
        runs.append((f"GA_att{att}", "GA", att, d))

    for seed in range(5, 40):
        d = os.path.join(
            RESULTS_BASE,
            f"run_lbfgsb_ld_ratio_constrained_m02_re1e7_normalized_seed{seed}_nr3",
        )
        runs.append((f"lbfgsb_s{seed}", "lbfgsb", seed, d))

    n_workers = min(len(runs), multiprocessing.cpu_count())
    print(f"Evaluating {len(runs)} designs using {n_workers} workers...", flush=True)

    with multiprocessing.Pool(processes=n_workers) as pool:
        summary = pool.map(evaluate_one, runs)

    # Sort by XFOIL L/D descending
    valid = [s for s in summary if "xfoil" in s]
    valid.sort(key=lambda s: -s["xfoil"]["L_D"])

    print("\n" + "=" * 65)
    print("TOP 10 BY XFOIL L/D")
    print("=" * 65)
    print(f"{'Label':<16} {'NF L/D':>8} {'XF L/D':>8} {'XF CM':>8} {'xtr_u':>6} {'CM ok?':>6}")
    for s in valid[:10]:
        cm_ok = "YES" if s["xfoil"]["CM"] >= -0.133 else "NO"
        print(f"{s['label']:<16} {s['nf_ld']:>8.1f} {s['xfoil']['L_D']:>8.1f} "
              f"{s['xfoil']['CM']:>8.4f} {s['xfoil']['xtr_upper']:>6.3f} {cm_ok:>6}")

    print(f"\nFailed: {[s['label'] for s in summary if 'xfoil' not in s and 'error' not in s or 'xfoil_error' in s]}")

    out_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
