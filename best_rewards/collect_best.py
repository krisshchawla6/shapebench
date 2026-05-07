"""
Collect the best design per method per reward for Superwing.

For each of the 30 rewards (sw_001..sw_030) and all methods, scans all seed
runs, finds the globally best reward, then writes:

  best_rewards/Superwing/sw_NNN/<method>/
    best_design.json   -- full design parameters
    best_result.json   -- reward/aerodynamic metrics + source provenance
"""

import json
import glob
import shutil
from pathlib import Path

import pandas as pd

RESULTS_ROOT = Path("/scratch/ShapeEvolve/environments/SuperWing/results")
OUTPUT_ROOT = Path("/scratch/ShapeEvolve/best_rewards/Superwing")

REWARDS = [f"sw_{i:03d}" for i in range(1, 31)]

# ── per-method helpers ────────────────────────────────────────────────────────

def _best_csv(reward: str, method: str):
    """
    Returns (best_row, sample_dir, best_reward_value) by scanning all seed
    results.csv files and taking the row with max 'reward'.
    """
    pattern = str(RESULTS_ROOT / method / f"{reward}_s*" / "results.csv")
    csv_files = sorted(glob.glob(pattern))
    best_val = float("-inf")
    best_row = None
    best_dir = None
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if df.empty or "reward" not in df.columns:
            continue
        valid = df[df["reward"].notna()]
        if valid.empty:
            continue
        idx = valid["reward"].idxmax()
        val = valid.loc[idx, "reward"]
        if val > best_val:
            best_val = val
            best_row = valid.loc[idx]
            best_dir = Path(csv_file).parent
    return best_row, best_dir, best_val


def _best_results_json(reward: str, method: str):
    """
    Returns (best_score, sample_dir) by reading results.json['best_score']
    across all seeds (used for adapter methods).
    """
    pattern = str(RESULTS_ROOT / method / f"{reward}_s*" / "results.json")
    json_files = sorted(glob.glob(pattern))
    best_val = float("-inf")
    best_dir = None
    for jf in json_files:
        data = json.loads(Path(jf).read_text())
        val = data.get("best_score", float("-inf"))
        if val > best_val:
            best_val = val
            best_dir = Path(jf).parent
    return best_val, best_dir


# ── per-method design path resolvers ─────────────────────────────────────────

def collect_bo_torch(reward: str) -> None:
    """columns: iteration, particle, sample, design, reward, best_reward, CL, CD, L_D, CM"""
    method = "BO_torch"
    best_row, best_dir, best_val = _best_csv(reward, method)
    if best_row is None:
        print(f"  [SKIP] {reward}/{method}: no data")
        return
    design_name = best_row["design"]
    design_file = best_dir / design_name / f"{design_name}.json"
    _write_output(reward, method, best_val, best_row, best_dir, design_file)


def collect_lbfgsb(reward: str) -> None:
    """columns: call, restart, reward, best_reward, CL, CD, L_D, CM"""
    method = "lbfgsb"
    best_row, best_dir, best_val = _best_csv(reward, method)
    if best_row is None:
        print(f"  [SKIP] {reward}/{method}: no data")
        return
    call = int(best_row["call"])
    restart = int(best_row["restart"])
    design_file = best_dir / f"call_{call:05d}_r{restart}" / "design.json"
    _write_output(reward, method, best_val, best_row, best_dir, design_file)


def collect_ga_parallel(reward: str) -> None:
    """columns: iteration, particle, reward, gbest_reward, CL, CD, L_D, CM"""
    method = "GA_parallel"
    best_row, best_dir, best_val = _best_csv(reward, method)
    if best_row is None:
        print(f"  [SKIP] {reward}/{method}: no data")
        return
    iteration = int(best_row["iteration"])
    particle = int(best_row["particle"])
    design_file = best_dir / f"iter_{iteration:04d}_p{particle:03d}" / "design.json"
    _write_output(reward, method, best_val, best_row, best_dir, design_file)


def collect_v3(reward: str) -> None:
    """columns: iteration, sample, design, reward, best_reward, sample_type, CL, CD, L_D, CM, island"""
    method = "v3_dynamic_optimizer"
    best_row, best_dir, best_val = _best_csv(reward, method)
    if best_row is None:
        print(f"  [SKIP] {reward}/{method}: no data")
        return
    design_name = best_row["design"]
    design_file = best_dir / f"{design_name}.json"
    _write_output(reward, method, best_val, best_row, best_dir, design_file)


def collect_adapter(reward: str, method: str) -> None:
    """
    openevolve_adapter / shinka_adapter: best_design.json already exists at
    sample level; reward from results.json['best_score'].
    No CL/CD/L_D/CM in CSV for these methods.
    """
    best_val, best_dir = _best_results_json(reward, method)
    if best_dir is None:
        print(f"  [SKIP] {reward}/{method}: no data")
        return
    design_file = best_dir / "best_design.json"
    _write_output(reward, method, best_val, None, best_dir, design_file)


# ── shared writer ─────────────────────────────────────────────────────────────

def _write_output(
    reward: str,
    method: str,
    best_val: float,
    best_row,          # pd.Series or None
    best_dir: Path,
    design_file: Path,
) -> None:
    if not design_file.exists():
        print(f"  [WARN] {reward}/{method}: design file not found: {design_file}")
        return

    out_dir = OUTPUT_ROOT / reward / method
    out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(design_file, out_dir / "best_design.json")

    def _get(col):
        if best_row is not None and col in best_row.index:
            v = best_row[col]
            return float(v) if pd.notna(v) else None
        return None

    result = {
        "reward": best_val,
        "CL":  _get("CL"),
        "CD":  _get("CD"),
        "L_D": _get("L_D"),
        "CM":  _get("CM"),
        "source": {
            "sample_dir":  str(best_dir),
            "design_file": str(design_file),
        },
    }
    (out_dir / "best_result.json").write_text(json.dumps(result, indent=2))
    print(f"  [{reward}/{method}] reward={best_val:.4f}  src={best_dir.name}")


# ── dispatch table ────────────────────────────────────────────────────────────

COLLECTORS = {
    "BO_torch":           collect_bo_torch,
    "lbfgsb":             collect_lbfgsb,
    "GA_parallel":        collect_ga_parallel,
    "v3_dynamic_optimizer": collect_v3,
    "openevolve_adapter": lambda r: collect_adapter(r, "openevolve_adapter"),
    "shinka_adapter":     lambda r: collect_adapter(r, "shinka_adapter"),
}


def main() -> None:
    for reward in REWARDS:
        for method, fn in COLLECTORS.items():
            fn(reward)
    print("\nDone.")


if __name__ == "__main__":
    main()
