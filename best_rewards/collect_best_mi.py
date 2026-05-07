"""
Collect the best design per method per reward for Mixed_integer_yiren.

Structure in results/:
  - Named reward subfolders (e.g. ld_ratio_basic/, single_pt_ld/, ...):
      Contains run_{method}_{reward}_attempt_{N}/ dirs.
  - Top-level run dirs for the base 'ld_ratio' reward:
      run_{method_config}_attempt_{N}/ dirs directly under results/.

For each reward x method pair, scans all attempt run dirs, finds the globally
best reward, resolves the design file, and writes:

  best_rewards/Mixed_integer_yiren/{reward}/{method}/
    best_design.json   -- full design parameters
    best_result.json   -- reward/aerodynamic metrics + source provenance
"""

import json
import re
import shutil
from pathlib import Path

import pandas as pd

RESULTS_ROOT = Path(
    "/scratch/ShapeEvolve/environments/Mixed_integer_yiren/results"
)
OUTPUT_ROOT = Path("/scratch/ShapeEvolve/best_rewards/Mixed_integer_yiren")

METRIC_COLS = ["LtoD", "CL", "CDi", "if_cranked", "if_Ttail", "if_canard"]

# These methods store best design at run_dir/best_design.json and use
# results.json["best_score"] rather than results.csv for the reward.
FLAT_BEST_METHODS = {"openevolve_adapter", "shinka_adapter"}


def resolve_design_path(run_dir: Path, row: pd.Series) -> Path:
    """Return path to design JSON based on CSV row format."""
    if "design" in row.index and pd.notna(row["design"]):
        design = str(row["design"])
        # v3 and similar: {design}.json written flat in run_dir
        flat = run_dir / f"{design}.json"
        if flat.exists():
            return flat
        # BO-torch style: {design}/{design}.json subdir
        return run_dir / design / f"{design}.json"
    if "call" in row.index:
        # lbfgsb style
        call = int(row["call"])
        restart = int(row["restart"])
        return run_dir / f"call_{call:05d}_r{restart}" / "design.json"
    # GA style: iteration + particle
    it = int(row["iteration"])
    pt = int(row["particle"])
    return run_dir / f"iter_{it:04d}_p{pt:03d}" / "design.json"


def collect_flat_best(reward: str, method: str, run_dirs: list[Path]) -> None:
    """Handle openevolve/shinka: best design is at run_dir/best_design.json."""
    best_score = float("-inf")
    best_run_dir = None

    for run_dir in run_dirs:
        rj = run_dir / "results.json"
        if not rj.exists():
            continue
        data = json.loads(rj.read_text())
        score = float(data.get("best_score", float("-inf")))
        if score > best_score:
            best_score = score
            best_run_dir = run_dir

    if best_run_dir is None:
        print(f"  [SKIP] no results.json for {reward}/{method}")
        return

    design_file = best_run_dir / "best_design.json"
    if not design_file.exists():
        print(f"  [WARN] best_design.json not found in: {best_run_dir}")
        return

    out_dir = OUTPUT_ROOT / reward / method
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(design_file, out_dir / "best_design.json")

    result = {
        "reward": best_score,
        "source": {
            "run_dir": str(best_run_dir),
            "design_file": str(design_file),
        },
    }
    (out_dir / "best_result.json").write_text(json.dumps(result, indent=2))
    print(f"  [{reward}/{method}] reward={best_score:.4f}  src={best_run_dir.name}")


def collect_best_for_group(
    reward: str, method: str, run_dirs: list[Path]
) -> None:
    """Find best design across all run_dirs (attempts) and write output."""
    # Normalise method name for FLAT_BEST_METHODS check (top-level ld_ratio
    # methods embed the reward name, e.g. openevolve_adapter_ld_ratio_flash25)
    base_method = method.split("_ld_ratio_")[0] if "_ld_ratio_" in method else method
    if base_method in FLAT_BEST_METHODS:
        collect_flat_best(reward, method, run_dirs)
        return

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
    print(
        f"  [{reward}/{method}] reward={best_reward:.4f}"
        f"  src={best_run_dir.name}"
    )


def collect_named_reward(reward_dir: Path) -> None:
    """Process a named reward subdirectory."""
    reward = reward_dir.name
    pattern = re.compile(
        rf"^run_(.+?)_{re.escape(reward)}_attempt_\d+$"
    )
    methods: dict[str, list[Path]] = {}
    for d in sorted(reward_dir.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if m:
            method = m.group(1)
            methods.setdefault(method, []).append(d)

    for method, run_dirs in sorted(methods.items()):
        collect_best_for_group(reward, method, run_dirs)


def collect_toplevel_ld_ratio() -> None:
    """Process top-level run dirs (base ld_ratio reward, various configs)."""
    reward = "ld_ratio"
    pattern = re.compile(r"^run_(.+)_attempt_\d+$")
    methods: dict[str, list[Path]] = {}
    for d in sorted(RESULTS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if m:
            method = m.group(1)
            methods.setdefault(method, []).append(d)

    for method, run_dirs in sorted(methods.items()):
        collect_best_for_group(reward, method, run_dirs)


def main() -> None:
    skip = {"plots"}
    for d in sorted(RESULTS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        if d.name in skip or d.name.startswith("run_"):
            continue
        collect_named_reward(d)

    collect_toplevel_ld_ratio()
    print("\nDone.")


if __name__ == "__main__":
    main()
