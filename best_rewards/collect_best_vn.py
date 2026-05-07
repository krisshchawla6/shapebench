"""
Collect the best design per method per reward for VortexNet.

All run dirs live flat in results/ with the naming pattern:
  run_{method}_vortexnet_{reward}_seed{N}_{config}/

Methods (5): BO_torch, GA, cmaes, lbfgsb, v3_dynamic_optimizer
Rewards (10): ld_ratio, max_kn, multi_cl_min_cd_sub, single_pt_ld,
              sub_max_ld_cl, sub_max_ld_cm, sub_min_cd_cl,
              sub_min_cd_cl_cm, sub_min_cd_kn, two_pt_multi
Seeds: 7 per method per reward (seed0-seed6)

CSV design path conventions:
  BO_torch / cmaes  : design col -> {design}/{design}.json  (subdir)
  v3                : design col -> {design}.json            (flat)
  GA                : iteration+particle -> iter_{i:04d}_p{p:03d}/design.json
  lbfgsb            : call+restart -> call_{c:05d}_r{r}/design.json

Writes:
  best_rewards/VortexNet/{reward}/{method}/
    best_design.json
    best_result.json
"""

import json
import re
import shutil
from pathlib import Path
from collections import defaultdict

import pandas as pd

RESULTS_ROOT = Path("/scratch/ShapeEvolve/environments/vortexnet/results")
OUTPUT_ROOT = Path("/scratch/ShapeEvolve/best_rewards/VortexNet")

METRIC_COLS = ["CL", "CDi", "CM", "L_D"]

# Pattern: run_{method}_vortexnet_{reward}_seed{N}_{config}
DIR_RE = re.compile(r"^run_(.+?)_vortexnet_(.+?)_seed\d+_")


def resolve_design_path(run_dir: Path, row: pd.Series) -> Path:
    if "design" in row.index and pd.notna(row["design"]):
        design = str(row["design"])
        flat = run_dir / f"{design}.json"
        if flat.exists():
            return flat
        return run_dir / design / f"{design}.json"
    if "call" in row.index:
        call = int(row["call"])
        restart = int(row["restart"])
        return run_dir / f"call_{call:05d}_r{restart}" / "design.json"
    it = int(row["iteration"])
    pt = int(row["particle"])
    return run_dir / f"iter_{it:04d}_p{pt:03d}" / "design.json"


def collect_best(reward: str, method: str, run_dirs: list[Path]) -> None:
    best_reward = float("-inf")
    best_row = None
    best_run_dir = None

    for run_dir in run_dirs:
        csv_path = run_dir / "results.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if df.empty or "reward" not in df.columns:
            continue
        idx = df["reward"].idxmax()
        val = df.loc[idx, "reward"]
        if val > best_reward:
            best_reward = val
            best_row = df.loc[idx]
            best_run_dir = run_dir

    if best_row is None:
        print(f"  [SKIP] no valid rows for {reward}/{method}")
        return

    design_file = resolve_design_path(best_run_dir, best_row)
    if not design_file.exists():
        print(f"  [WARN] design file not found: {design_file}")
        return

    out_dir = OUTPUT_ROOT / reward / method
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(design_file, out_dir / "best_design.json")

    result: dict = {
        "reward": float(best_row["reward"]),
        "source": {
            "run_dir": str(best_run_dir),
            "design_file": str(design_file),
        },
    }
    for col in METRIC_COLS:
        if col in best_row.index:
            result[col] = float(best_row[col])

    (out_dir / "best_result.json").write_text(json.dumps(result, indent=2))
    print(f"  [{reward}/{method}] reward={best_reward:.4f}  src={best_run_dir.name}")


def main() -> None:
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)

    for d in sorted(RESULTS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        m = DIR_RE.match(d.name)
        if m:
            method, reward = m.group(1), m.group(2)
            groups[(reward, method)].append(d)

    for (reward, method), run_dirs in sorted(groups.items()):
        collect_best(reward, method, run_dirs)

    print("\nDone.")


if __name__ == "__main__":
    main()
