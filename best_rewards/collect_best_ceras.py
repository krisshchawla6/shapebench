"""Collect best design per method per reward for the CERAS environment."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

RESULTS_ROOT = Path("/scratch/ShapeEvolve/environments/CERAS/results")
OUT_ROOT = Path("/scratch/ShapeEvolve/best_rewards/Ceras")

METRIC_COLS = ["fuel_mass_kg", "static_margin"]

# Reward values that indicate a failed/penalty evaluation — never treat as best
FAILURE_REWARDS = {0.0, -10.0, -100000.0}

# openevolve/shinka: design files are UUIDs with no individual JSON; use
# best_design.json from the seed that achieved the best valid CSV reward.
FLAT_BEST_METHODS = {"openevolve", "shinka"}

# Known methods (longest prefix first to avoid GA matching before GAp)
METHODS = ["openevolve", "shinka", "lbfgsb", "cmaes", "GAp", "BO", "v3"]
# Only fuel_mass has results across all methods; other rewards only BO/GAp/cmaes/lbfgsb
REWARDS = ["fuel_mass"]


def parse_dir(name: str) -> tuple[str, str] | None:
    """Return (method, reward) from run dir name, or None if unrecognised."""
    if not name.startswith("run_"):
        return None
    rest = name[4:]  # strip 'run_'
    for method in METHODS:
        if not (rest.startswith(method + "_") or rest == method):
            continue
        tail = rest[len(method):]  # e.g. '_fuel_mass_s3' or '_fuel_mass'
        if tail == "" or tail.startswith("_"):
            tail = tail.lstrip("_")
        else:
            continue
        for reward in sorted(REWARDS, key=len, reverse=True):
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


def is_valid_reward(r: float) -> bool:
    return r not in FAILURE_REWARDS and r > -100000.0


def collect_flat_best(reward: str, method: str, run_dirs: list[Path]) -> None:
    """openevolve/shinka: read best valid reward from results.csv, use best_design.json."""
    best_score = float("-inf")
    best_design: dict | None = None
    best_result: dict | None = None

    for run_dir in run_dirs:
        bd = run_dir / "best_design.json"
        csv_path = run_dir / "results.csv"
        if not bd.exists() or not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if df.empty or "reward" not in df.columns:
            continue
        df["reward"] = pd.to_numeric(df["reward"], errors="coerce")
        valid = df[df["reward"].apply(lambda x: pd.notna(x) and is_valid_reward(x))]
        if valid.empty:
            continue
        score = float(valid["reward"].max())
        if score > best_score:
            best_score = score
            best_design = json.loads(bd.read_text())
            best_result = {"reward": score, "source_dir": str(run_dir)}

    if best_design is None:
        print(f"  [WARN] no valid data for {method}/{reward}")
        return

    out_dir = OUT_ROOT / reward / method
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "best_design.json").write_text(json.dumps(best_design, indent=2))
    (out_dir / "best_result.json").write_text(json.dumps(best_result, indent=2))
    print(f"  {reward}/{method}: reward={best_score:.4f}")


def collect_best(reward: str, method: str, run_dirs: list[Path]) -> None:
    if method in FLAT_BEST_METHODS:
        collect_flat_best(reward, method, run_dirs)
        return

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
        df["reward"] = pd.to_numeric(df["reward"], errors="coerce")
        valid = df[df["reward"].apply(lambda x: pd.notna(x) and is_valid_reward(x))]
        if valid.empty:
            continue
        idx = valid["reward"].idxmax()
        row = valid.loc[idx]
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
