"""Collect best design per method per reward for the BlendedNet environment."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

RESULTS_ROOT = Path("/scratch/ShapeEvolve/environments/BlendedNet/results")
OUT_ROOT = Path("/scratch/ShapeEvolve/best_rewards/BlendedNet")

METRIC_COLS = ["Cp_mean", "Cfx_mean", "L_D"]

# Known method prefixes — longest first to avoid prefix ambiguity (GA vs GA_parallel)
METHODS = ["v3_flash2_5", "GA_parallel", "BO_torch", "cmaes", "lbfgsb", "GA"]

# Known rewards — derived from v3 dirs (most canonical names); longest first
REWARDS = sorted([
    "multipoint_mach_cd",
    "multipoint_mach_range",
    "range_optimization",
    "shapebench_5_max_LD_total_drag",
    "shapebench_5_max_LD_warmstart_cornerA",
    "shapebench_5_total_drag_constrained",
    "shapebench_5_max_LD",
    "shapebench_5_total_drag",
    "shapebench_5",
    "shapebench_case5",
    "shapebench_case6",
    "static_margin_constrained",
], key=len, reverse=True)

# Skip dirs containing these substrings
SKIP_PATTERNS = ["_crashed_backup", "_FAILED"]


def parse_dir(name: str) -> tuple[str, str] | None:
    """Return (method, reward) from run dir name, or None if unrecognised."""
    if not name.startswith("run_"):
        return None
    for pat in SKIP_PATTERNS:
        if pat in name:
            return None
    rest = name[4:]  # strip 'run_'
    for method in METHODS:
        prefix = method + "_"
        if not rest.startswith(prefix):
            continue
        tail = rest[len(prefix):]  # e.g. 'shapebench_5_seed0_n500'
        for reward in REWARDS:
            if tail == reward or tail.startswith(reward + "_"):
                return method, reward
    return None


def resolve_design_path(run_dir: Path, row: pd.Series) -> Path:
    """Map a results.csv row to the design JSON path."""
    if "design" in row.index and pd.notna(row.get("design")):
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
    best_design: dict | None = None
    best_result: dict | None = None

    for run_dir in run_dirs:
        csv_path = run_dir / "results.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if df.empty or "reward" not in df.columns:
            continue
        df = df[pd.to_numeric(df["reward"], errors="coerce").notna()]
        if df.empty:
            continue
        idx = df["reward"].idxmax()
        row = df.loc[idx]
        r = float(row["reward"])
        if r <= best_reward:
            continue

        design_path = resolve_design_path(run_dir, row)
        if not design_path.exists():
            print(f"  [WARN] design file not found: {design_path}")
            continue

        best_reward = r
        best_design = json.loads(design_path.read_text())
        result: dict = {"reward": r, "source_dir": str(run_dir)}
        for col in METRIC_COLS:
            if col in row.index and pd.notna(row[col]):
                result[col] = float(row[col])
        best_result = result

    if best_design is None:
        print(f"  [WARN] no valid design found for {method}/{reward}")
        return

    out_dir = OUT_ROOT / reward / method
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "best_design.json").write_text(json.dumps(best_design, indent=2))
    (out_dir / "best_result.json").write_text(json.dumps(best_result, indent=2))
    print(f"  {reward}/{method}: reward={best_reward:.4f}")


def main() -> None:
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)

    for d in sorted(RESULTS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        parsed = parse_dir(d.name)
        if parsed:
            method, reward = parsed
            groups[(reward, method)].append(d)
        else:
            if d.name.startswith("run_"):
                print(f"  [SKIP] unrecognised: {d.name}")

    for (reward, method), run_dirs in sorted(groups.items()):
        collect_best(reward, method, run_dirs)

    print("\nDone.")


if __name__ == "__main__":
    main()
