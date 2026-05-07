"""
Collect best design per reward per method for NeuralFoil.

Output structure:
  best_rewards/NeuralFoil/{reward}/
    {method}/best_design.json  +  best_result.json
    analysis plots (copied from environments/NeuralFoil/analysis/)

Rewards:
  reward_exact_notebook
  ld_ratio_constrained_m02_re1e7_normalized

Methods:
  BO_torch | v3_dynamic_optimizer | GA | lbfgsb
"""

import json
import shutil
from pathlib import Path

import pandas as pd

RESULTS_ROOT  = Path("/scratch/ShapeEvolve/environments/NeuralFoil/results")
ANALYSIS_ROOT = Path("/scratch/ShapeEvolve/environments/NeuralFoil/analysis")
OUTPUT_ROOT   = Path("/scratch/ShapeEvolve/best_rewards/NeuralFoil")

# ── design path helper ────────────────────────────────────────────────────────

def resolve_design(run_dir: Path, row: pd.Series) -> Path:
    if "design" in row.index and pd.notna(row.get("design")):
        d = str(row["design"])
        flat = run_dir / f"{d}.json"
        if flat.exists():
            return flat
        return run_dir / d / f"{d}.json"   # BO_torch subdir style
    if "call" in row.index:
        call = int(row["call"]); restart = int(row["restart"])
        return run_dir / f"call_{call:05d}_r{restart}" / "design.json"
    it = int(row["iteration"]); pt = int(row["particle"])
    return run_dir / f"iter_{it:04d}_p{pt:03d}" / "design.json"


def metric(row, col):
    if row is not None and col in row.index:
        v = row[col]
        return float(v) if pd.notna(v) else None
    return None


# ── scan a list of run dirs → best (row, run_dir) ────────────────────────────

def best_across(run_dirs: list[Path]):
    best_val = float("-inf"); best_row = None; best_dir = None
    for rd in run_dirs:
        csv = rd / "results.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        valid = df[df["reward"].notna() & (df["reward"] > -999)]
        if valid.empty:
            continue
        idx = valid["reward"].idxmax()
        val = valid.loc[idx, "reward"]
        if val > best_val:
            best_val = val; best_row = valid.loc[idx]; best_dir = rd
    return best_row, best_dir, best_val


def write_output(reward: str, method: str, best_val: float,
                 best_row, best_dir: Path) -> None:
    if best_dir is None:
        print(f"  [SKIP] {reward}/{method}: no data")
        return
    design_file = resolve_design(best_dir, best_row)
    if not design_file.exists():
        print(f"  [WARN] {reward}/{method}: design not found {design_file}")
        return
    out = OUTPUT_ROOT / reward / method
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(design_file, out / "best_design.json")
    result = {
        "reward": best_val,
        "CL":  metric(best_row, "CL"),
        "CD":  metric(best_row, "CD"),
        "L_D": metric(best_row, "L_D"),
        "CM":  metric(best_row, "CM"),
        "source": {"run_dir": str(best_dir), "design_file": str(design_file)},
    }
    (out / "best_result.json").write_text(json.dumps(result, indent=2))
    print(f"  [{reward}/{method}] reward={best_val:.4f}  src={best_dir.name}")


# ── enumerate run dirs from SAVED_DIRS ───────────────────────────────────────

def from_saved(saved_dirs_glob: str) -> list[Path]:
    """Return all run sub-dirs inside matching SAVED_DIRS_* folders."""
    runs: list[Path] = []
    for sd in sorted(RESULTS_ROOT.glob(saved_dirs_glob)):
        if not sd.is_dir():
            continue
        for rd in sorted(sd.iterdir()):
            if rd.is_dir() and (rd / "results.csv").exists():
                runs.append(rd)
    return runs


# ── reward definitions ────────────────────────────────────────────────────────

REWARDS = {
    "reward_exact_notebook": {
        "BO_torch": (
            from_saved("SAVED_DIRS_BO_torch_reward_exact_notebook")
        ),
        "v3_dynamic_optimizer": (
            from_saved("SAVED_DIRS_run_v3_dynamic_optimizer_reward_exact_notebook*")
        ),
        "GA": (
            from_saved("SAVED_DIRS_run_GA_reward_exact_notebook*")
        ),
        "lbfgsb": (
            from_saved("SAVED_DIRS_run_lbfgsb_reward_exact_notebook*")
        ),
    },
    "ld_ratio_constrained_m02_re1e7_normalized": {
        "BO_torch": (
            from_saved("SAVED_DIRS_run_BO_torch_ld_ratio_constrained_m02_re1e7_normalized")
        ),
        "v3_dynamic_optimizer": (
            from_saved("SAVED_DIRS_run_v3_dynamic_optimizer_ld_ratio_constrained_m02_re1e7_normalized")
        ),
    },
}

# ── analysis plot mapping ─────────────────────────────────────────────────────

PLOT_MAP = {
    "reward_exact_notebook": {
        "files": [
            "comparison_graph.png",
            "ablation_comparison.png",
            "ablation_comparison_academic.png",
            "ablation_comparison_academic.pdf",
            "lhs_sweep_analysis.png",
        ],
        "src": ANALYSIS_ROOT,
    },
    "ld_ratio_constrained_m02_re1e7_normalized": {
        "files": None,   # copy all pngs from l_d_onepoint/
        "src": ANALYSIS_ROOT / "l_d_onepoint",
    },
}


def copy_plots(reward: str) -> None:
    info = PLOT_MAP.get(reward)
    if info is None:
        return
    src_dir: Path = info["src"]
    if not src_dir.exists():
        return
    dst = OUTPUT_ROOT / reward
    dst.mkdir(parents=True, exist_ok=True)
    files = info["files"]
    if files is None:
        files = [f.name for f in src_dir.iterdir()
                 if f.suffix in {".png", ".pdf"} and f.is_file()]
    copied = 0
    for fname in files:
        src = src_dir / fname
        if src.exists():
            shutil.copy2(src, dst / fname)
            copied += 1
    if copied:
        print(f"  [plots] {reward}: {copied} files")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    for reward, methods in REWARDS.items():
        for method, run_dirs in methods.items():
            best_row, best_dir, best_val = best_across(run_dirs)
            write_output(reward, method, best_val, best_row, best_dir)
        copy_plots(reward)
    print("\nDone.")


if __name__ == "__main__":
    main()
