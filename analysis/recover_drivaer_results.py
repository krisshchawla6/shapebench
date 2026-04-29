#!/usr/bin/env python3
"""
Reconstruct results_recovered.csv for DrivAer_Star runs (all reward variants).

Per-design results.json files survive SLURM preemption restarts; only
results.csv is overwritten.  This script rebuilds a trajectory CSV from
those saved files.

Columns reconstructed:
  iteration    — from design directory name
  sample       — from design directory name
  design       — directory name (e.g. iter_0_s0, iter_0000)
  reward       — from save/results.json
  best_reward  — running maximum of reward (sorted by iteration, sample)
  sample_type  — v3 only: llm_center (_s suffix) or llm_optimizer (_o suffix)
  drag         — from save/results.json
  Cd           — from save/results.json
  lift         — from save/results.json
  Cl           — from save/results.json (downforce_efficiency runs)
  violation    — from save/results.json (cd_cl_constrained runs)
  particle     — BO only: always 0 (single-seed BO runs)

NOT recoverable:
  island       — v3 only: search island index (not stored in any per-design file)

Output: results_recovered.csv written next to results.csv in each run dir.
"""

import csv
import json
import re
import sys
from pathlib import Path


# ── Directory name parsers ────────────────────────────────────────────────────

# v3: iter_{iteration}_s{sample}  or  iter_{iteration}_o{sample}
_V3_RE = re.compile(r"^iter_(\d+)_([so])(\d+)$")
# BO: iter_{iteration}  (zero-padded, e.g. iter_0000)
_BO_RE = re.compile(r"^iter_(\d+)$")


def _parse_v3(name: str):
    m = _V3_RE.match(name)
    if not m:
        return None
    iteration = int(m.group(1))
    sample_type = "llm_center" if m.group(2) == "s" else "llm_optimizer"
    sample = int(m.group(3))
    return dict(iteration=iteration, sample=sample, sample_type=sample_type)


def _parse_bo(name: str):
    m = _BO_RE.match(name)
    if not m:
        return None
    return dict(iteration=int(m.group(1)), sample=0, particle=0)


# ── Core recovery ─────────────────────────────────────────────────────────────

def recover_run(run_dir: Path, mode: str) -> int:
    """
    Scan run_dir for iter_* design directories, read save/results.json from
    each, and write results_recovered.csv.  Returns number of rows written.
    """
    assert mode in ("v3", "bo")
    parse = _parse_v3 if mode == "v3" else _parse_bo

    rows = []
    for entry in sorted(run_dir.iterdir()):
        if not entry.is_dir():
            continue
        meta = parse(entry.name)
        if meta is None:
            continue
        rj = entry / "save" / "results.json"
        if not rj.exists():
            continue
        with rj.open() as f:
            d = json.load(f)
        row = {
            "iteration":   meta["iteration"],
            "sample":      meta["sample"],
            "design":      entry.name,
            "reward":      d.get("reward", ""),
            "best_reward": "",          # filled below
            "drag":        d.get("drag", ""),
            "Cd":          d.get("Cd", ""),
            "lift":        d.get("lift", ""),
            "violation":   d.get("violation", ""),
        }
        if mode == "v3":
            row["sample_type"] = meta["sample_type"]
            row["island"] = ""          # not recoverable
        else:
            row["particle"] = meta["particle"]
        rows.append(row)

    if not rows:
        print(f"  [warn] no designs found in {run_dir.name}", file=sys.stderr)
        return 0

    # Sort by (iteration, sample) — matches original CSV order
    rows.sort(key=lambda r: (r["iteration"], r["sample"]))

    # Running best_reward
    best = float("-inf")
    for r in rows:
        try:
            v = float(r["reward"])
            if v > best:
                best = v
        except (ValueError, TypeError):
            pass
        r["best_reward"] = f"{best:.6f}" if best > float("-inf") else ""

    # Column order matches original results.csv as closely as possible
    if mode == "v3":
        fieldnames = ["iteration", "sample", "design", "reward", "best_reward",
                      "sample_type", "drag", "Cd", "lift", "violation", "island"]
    else:
        fieldnames = ["iteration", "particle", "sample", "design", "reward",
                      "best_reward", "drag", "Cd", "lift", "violation"]

    out = run_dir / "results_recovered.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {run_dir.name}: {len(rows)} rows → {out.name}")
    return len(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    base = Path("environments/DrivAer_Star/results")

    total = 0

    for reward_variant in ["cd_cl_constrained", "downforce_efficiency"]:
        print(f"\n=== {reward_variant} — v3 runs ===")
        for body in ["E", "F", "N"]:
            for att in range(1, 11):
                run_dir = base / (
                    f"run_v3_dynamic_optimizer_{reward_variant}_drivaer_star_vtk_"
                    f"{body}_attempt_{att}_flash_2_5_n6000"
                )
                if run_dir.exists():
                    total += recover_run(run_dir, mode="v3")

        print(f"\n=== {reward_variant} — BO runs ===")
        for body in ["E", "F", "N"]:
            for seed in range(10):
                run_dir = base / (
                    f"run_BO_torch_{reward_variant}_vtk_{body}_seed{seed}_n1000"
                )
                if run_dir.exists():
                    total += recover_run(run_dir, mode="bo")

    print(f"\nTotal rows recovered: {total}")


if __name__ == "__main__":
    main()
