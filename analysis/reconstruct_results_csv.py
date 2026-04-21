"""Reconstruct results.csv from per-evaluation save/results.json files.

For runs preempted after the results.csv was overwritten on restart, this
script reads the per-evaluation results.json stored in each design directory
and reconstructs the full results.csv.

Supported run layouts
---------------------
  L-BFGS-B (BlendedNet):
    call_XXXXX_rY/save/results.json  →  call=XXXXX, restart=Y
    CSV cols: call, restart, reward, best_reward, Cp_mean, Cfx_mean, L_D

  BO / PSO (DrivAer_Star):
    iter_XXXX/save/results.json  →  iteration=XXXX
    CSV cols: iteration, particle, sample, design, reward, best_reward,
              drag, Cd, lift

Design directories that have no save/results.json (preempted mid-call) are
filled with reward = NaN and best_reward = the previous running best, so the
convergence curve (which uses minimum.accumulate on best_reward) is correct.

Usage
-----
    cd /scratch/ShapeEvolve
    source venv/bin/activate
    python analysis/reconstruct_results_csv.py <run_dir> [<run_dir> ...]

A backup of the existing results.csv is written to results.csv.bak before
any overwrite.
"""

import csv
import json
import math
import os
import re
import shutil
import sys


# ── L-BFGS-B reconstruction ───────────────────────────────────────────────────

_LBFGSB_RE = re.compile(r"^call_(\d+)_r(\d+)$")


def _lbfgsb_call_dirs(run_dir):
    """Return sorted list of (call_int, restart_int, dir_path) tuples."""
    entries = []
    for name in os.listdir(run_dir):
        m = _LBFGSB_RE.match(name)
        if m:
            entries.append((int(m.group(1)), int(m.group(2)),
                            os.path.join(run_dir, name)))
    entries.sort(key=lambda x: x[0])
    return entries


def reconstruct_lbfgsb(run_dir):
    entries = _lbfgsb_call_dirs(run_dir)
    if not entries:
        print(f"  [SKIP] no call_* dirs found in {run_dir}")
        return

    rows = []
    best = math.inf
    missing = 0
    for call, restart, ddir in entries:
        rj = os.path.join(ddir, "save", "results.json")
        if os.path.exists(rj):
            with open(rj) as f:
                d = json.load(f)
            reward = float(d["reward"])
            best = min(best, -reward)   # best_reward is most-negative reward
        else:
            reward = float("nan")
            missing += 1
        best_reward = -best if not math.isinf(best) else float("nan")
        rows.append((call, restart, reward, best_reward))

    csv_path = os.path.join(run_dir, "results.csv")
    _backup(csv_path)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["call", "restart", "reward", "best_reward",
                    "Cp_mean", "Cfx_mean", "L_D"])
        for call, restart, reward, best_reward in rows:
            w.writerow([
                call, restart,
                "" if math.isnan(reward) else f"{reward:.6f}",
                "" if math.isnan(best_reward) else f"{best_reward:.6f}",
                "0.000000", "0.000000", "0.0000",
            ])

    print(f"  lbfgsb  {os.path.basename(run_dir):50s}  "
          f"{len(rows):5d} rows  {missing:3d} missing (no save/)")


# ── BO / PSO reconstruction ───────────────────────────────────────────────────

_BO_RE = re.compile(r"^iter_(\d+)$")


def _bo_iter_dirs(run_dir):
    entries = []
    for name in os.listdir(run_dir):
        m = _BO_RE.match(name)
        if m:
            entries.append((int(m.group(1)), os.path.join(run_dir, name)))
    entries.sort(key=lambda x: x[0])
    return entries


def reconstruct_bo(run_dir):
    entries = _bo_iter_dirs(run_dir)
    if not entries:
        print(f"  [SKIP] no iter_* dirs found in {run_dir}")
        return

    rows = []
    best = math.inf
    missing = 0
    for iteration, ddir in entries:
        rj = os.path.join(ddir, "save", "results.json")
        if os.path.exists(rj):
            with open(rj) as f:
                d = json.load(f)
            reward = float(d["reward"])
            drag   = float(d.get("drag", float("nan")))
            cd     = float(d.get("Cd",   float("nan")))
            lift   = float(d.get("lift", float("nan")))
            best = min(best, -reward)
        else:
            reward = drag = cd = lift = float("nan")
            missing += 1
        best_reward = -best if not math.isinf(best) else float("nan")
        design_name = f"iter_{iteration:04d}"
        rows.append((iteration, reward, best_reward, drag, cd, lift,
                     design_name))

    csv_path = os.path.join(run_dir, "results.csv")
    _backup(csv_path)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "particle", "sample", "design",
                    "reward", "best_reward", "drag", "Cd", "lift"])
        for iteration, reward, best_reward, drag, cd, lift, dname in rows:
            def fmt(v):
                return "" if math.isnan(v) else f"{v:.6f}"
            w.writerow([iteration, 0, dname, dname,
                        fmt(reward), fmt(best_reward),
                        fmt(drag), fmt(cd), fmt(lift)])

    print(f"  bo/pso  {os.path.basename(run_dir):50s}  "
          f"{len(rows):5d} rows  {missing:3d} missing (no save/)")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _backup(csv_path):
    if os.path.exists(csv_path):
        bak = csv_path + ".bak"
        shutil.copy2(csv_path, bak)


def _detect_type(run_dir):
    names = os.listdir(run_dir)
    if any(_LBFGSB_RE.match(n) for n in names):
        return "lbfgsb"
    if any(_BO_RE.match(n) for n in names):
        return "bo"
    return None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python analysis/reconstruct_results_csv.py <run_dir> ...")
        sys.exit(1)

    for run_dir in sys.argv[1:]:
        if not os.path.isdir(run_dir):
            print(f"  [WARN] not a directory: {run_dir}")
            continue
        kind = _detect_type(run_dir)
        if kind == "lbfgsb":
            reconstruct_lbfgsb(run_dir)
        elif kind == "bo":
            reconstruct_bo(run_dir)
        else:
            print(f"  [SKIP] unrecognised layout: {run_dir}")


if __name__ == "__main__":
    main()
