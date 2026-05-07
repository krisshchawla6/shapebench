"""
Collect the best design per method per reward for DrivaerStar.

Methods:
  BO_torch              run_BO_torch_{reward}_seed{N}_n{M}
  v3_dynamic_optimizer  run_v3_dynamic_optimizer_{reward_ds}_attempt_{N}_{cfg}
                        (reward contains '_drivaer_star' which is stripped)
  drivaer_star_3d_islands  run_drivaer_star_3d_islands[_800]
                        (treated as reward='cd_only_vtk_E', 2 seeds)

Design path:
  BO_torch / islands: {design}/{design}.json  (subdir) or flat {design}.json
  v3:                 flat {design}.json

Writes:
  best_rewards/DrivaerStar/{reward}/{method}/
    best_design.json
    best_result.json
"""

import json
import re
import shutil
from pathlib import Path
from collections import defaultdict

import pandas as pd

RESULTS_ROOT = Path("/scratch/ShapeEvolve/environments/DrivAer_Star/results")
OUTPUT_ROOT = Path("/scratch/ShapeEvolve/best_rewards/DrivaerStar")

METRIC_COLS = ["Cd", "drag", "lift"]

BO_RE = re.compile(r"^run_BO_torch_(.+?)_seed\d+_n\d+$")
V3_RE = re.compile(r"^run_v3_dynamic_optimizer_(.+?)_attempt_\d+_")
ISLANDS_RE = re.compile(r"^run_drivaer_star_3d_islands")


def normalize_v3_reward(raw: str) -> str:
    """Strip '_drivaer_star' from v3 reward name to match BO_torch naming."""
    return raw.replace("_drivaer_star", "")


def resolve_design_path(run_dir: Path, row: pd.Series) -> Path:
    design = str(row["design"])
    flat = run_dir / f"{design}.json"
    if flat.exists():
        return flat
    return run_dir / design / f"{design}.json"


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
    print(f"  [{reward}/{method}] reward={best_reward:.6f}  src={best_run_dir.name}")


def main() -> None:
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    islands_dirs: list[Path] = []

    for d in sorted(RESULTS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if "_FAILED" in name:
            continue

        m = BO_RE.match(name)
        if m:
            groups[("BO_torch", m.group(1))].append(d)
            continue

        m = V3_RE.match(name)
        if m:
            reward = normalize_v3_reward(m.group(1))
            groups[("v3_dynamic_optimizer", reward)].append(d)
            continue

        if ISLANDS_RE.match(name):
            islands_dirs.append(d)

    # Write BO_torch and v3 results
    for (method, reward), run_dirs in sorted(groups.items()):
        collect_best(reward, method, run_dirs)

    # Write islands as a single method+reward group
    if islands_dirs:
        collect_best("cd_only_vtk_E", "drivaer_star_3d_islands", islands_dirs)

    print("\nDone.")


if __name__ == "__main__":
    main()
