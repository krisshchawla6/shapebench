#!/usr/bin/env python3
"""
Stage 2 (IPOPT) + Stage 3 (XFOIL) pipeline for BO_torch seeds 5-9.

For each seed, loads the best design from the raw BO run, warm-starts IPOPT
with CM>=-0.125 / conf>=0.85, then validates with XFOIL.

Outputs:
  environments/NeuralFoil/results/SAVED_DIRS_ld_adjoint_ld_ratio_constrained/ld_adjoint_cm125_conf85_from_BO_torch_seed{s}/
  (loads from: environments/NeuralFoil/results/SAVED_DIRS_run_BO_torch_ld_ratio_constrained_m02_re1e7_normalized/)
  environments/NeuralFoil/results/xfoil_validation_BO_torch_best_cm125_conf85/summary.json
    (extended with seeds 5-9 appended after existing seeds 0-3 entries)
"""

import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

NEURALFOIL_SRC = os.path.join(REPO_DIR, "environments", "NeuralFoil", "neuralfoil_src")
if os.path.isdir(NEURALFOIL_SRC):
    sys.path.insert(0, NEURALFOIL_SRC)

import aerosandbox as asb
from environments.NeuralFoil.adjoint.optimize_ld_cm125_conf85 import (
    run_ld_adjoint_optimization_cm125_conf85,
)

RESULTS_BASE = os.path.join(REPO_DIR, "environments", "NeuralFoil", "results")
XFOIL_BIN = os.path.join(REPO_DIR, "bin", "xfoil")
XFOIL_OUT_DIR = os.path.join(RESULTS_BASE, "xfoil_validation_BO_torch_best_cm125_conf85")
SUMMARY_PATH = os.path.join(XFOIL_OUT_DIR, "summary.json")

ALPHA = 5.0
RE = 1e7
MACH = 0.2
N_CRIT = 9.0

SEEDS = [5, 6, 7, 8, 9]

# Row index (0-based, excluding header) where best reward was first achieved.
BEST_ITERS = {5: 1923, 6: 1686, 7: 4863, 8: 1329, 9: 2978}


def load_design(seed):
    run_dir = os.path.join(
        RESULTS_BASE,
        "SAVED_DIRS_run_BO_torch_ld_ratio_constrained_m02_re1e7_normalized",
        f"run_BO_torch_ld_ratio_constrained_m02_re1e7_normalized_seed{seed}_n6000",
    )
    save_path = os.path.join(run_dir, f"iter_{BEST_ITERS[seed]:04d}", "save", "results.json")
    with open(save_path) as f:
        d = json.load(f)
    print(f"  seed{seed}: raw BO best L_D={d['L_D']:.4f}  reward={d['reward']:.4f}")
    return d["design"]


def write_airfoil_dat(airfoil, path):
    coords = np.array(airfoil.coordinates)
    with open(path, "w") as f:
        f.write("airfoil\n")
        for x, y in coords:
            f.write(f"{x:.6f} {y:.6f}\n")


def run_xfoil(airfoil):
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False, mode="w") as tmp:
        dat_path = tmp.name
    try:
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
        write_airfoil_dat(airfoil, dat_path)
        result = subprocess.run(
            [XFOIL_BIN], input=commands, capture_output=True, text=True, timeout=120
        )
        stdout = result.stdout
    finally:
        if os.path.exists(dat_path):
            os.unlink(dat_path)

    cl_match  = re.findall(r'\bCL\s*=\s*([-\d.]+)', stdout)
    cd_match  = re.findall(r'\bCD\s*=\s*([-\d.]+)', stdout)
    cm_match  = re.findall(r'\bCm\s*=\s*([-\d.]+)', stdout)
    xtr_match = re.findall(r'Side \d.*?x/c\s*=\s*([\d.]+)', stdout)
    if not cl_match:
        raise RuntimeError(f"XFOIL failed — no CL in stdout:\n{stdout[-1000:]}")
    CL = float(cl_match[-1])
    CD = float(cd_match[-1]) if cd_match else float("nan")
    CM = float(cm_match[-1]) if cm_match else float("nan")
    xtr_u = float(xtr_match[-2]) if len(xtr_match) >= 2 else float("nan")
    return {"CL": CL, "CD": CD, "CM": CM, "L_D": CL / CD, "xtr_upper": xtr_u}


def main():
    os.makedirs(XFOIL_OUT_DIR, exist_ok=True)

    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH) as f:
            summary = json.load(f)
        existing_labels = {e["label"] for e in summary}
        print(f"Loaded existing summary with {len(summary)} entries: {existing_labels}")
    else:
        summary = []
        existing_labels = set()

    for seed in SEEDS:
        label = f"BO_torch_seed{seed}"
        if label in existing_labels:
            print(f"\n[seed{seed}] Already in summary — skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"[seed{seed}] Stage 2: IPOPT CM>=-0.125 conf>=0.85")
        print(f"{'='*60}")

        design = load_design(seed)
        name = f"ld_adjoint_cm125_conf85_from_BO_torch_seed{seed}"
        output_dir = os.path.join(RESULTS_BASE, "SAVED_DIRS_ld_adjoint_ld_ratio_constrained", name)

        ipopt_result = run_ld_adjoint_optimization_cm125_conf85(
            initial_kulfan=design,
            alpha=ALPHA,
            re=RE,
            mach=MACH,
            model_size="large",
            output_dir=output_dir,
            name=name,
        )
        print(f"  IPOPT: NF L/D={ipopt_result['L_D']:.4f}  CM={ipopt_result['CM']:.5f}"
              f"  conf={ipopt_result['analysis_confidence']:.4f}"
              f"  solver_ok={ipopt_result['solver_success']}  n_iters={ipopt_result['n_iters']}")

        print(f"[seed{seed}] Stage 3: XFOIL validation")
        refined_af = asb.KulfanAirfoil(
            upper_weights=np.array(ipopt_result["upper_weights"]),
            lower_weights=np.array(ipopt_result["lower_weights"]),
            leading_edge_weight=float(ipopt_result["leading_edge_weight"]),
            TE_thickness=0.0,
        )
        xf = run_xfoil(refined_af)
        print(f"  XFOIL: L/D={xf['L_D']:.4f}  CM={xf['CM']:.4f}  xtr_u={xf['xtr_upper']:.4f}"
              f"  CM_ok={'YES' if xf['CM'] >= -0.133 else 'NO'}")

        entry = {
            "label": label,
            "nf_ld": ipopt_result["L_D"],
            "nf_cm": ipopt_result["CM"],
            "nf_conf": ipopt_result["analysis_confidence"],
            "xfoil": xf,
        }
        summary.append(entry)
        existing_labels.add(label)

        with open(SUMMARY_PATH, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  → summary.json updated ({len(summary)} entries)")

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"{'Label':<22} {'NF L/D':>8} {'XF L/D':>8} {'XF CM':>8} {'xtr_u':>6} {'CM ok?':>6}")
    for e in summary:
        if "xfoil" in e:
            cm_ok = "YES" if e["xfoil"]["CM"] >= -0.133 else "NO"
            print(f"{e['label']:<22} {e['nf_ld']:>8.3f} {e['xfoil']['L_D']:>8.3f}"
                  f" {e['xfoil']['CM']:>8.4f} {e['xfoil']['xtr_upper']:>6.3f} {cm_ok:>6}")

    print(f"\nResults saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
