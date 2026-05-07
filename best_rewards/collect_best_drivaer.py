"""
Collect best design per reward per method for DrivAer_Star.

Output structure:
  best_rewards/DrivaerStar/{reward}/
    {method}/best_design.json  +  best_result.json
    convergence / design-panel plots (copied from analysis_plots_* dirs)

Reward folders  =  (reward_type, vtk_variant), e.g. cd_only_vtk_E
Methods         =  BO_torch | v3_dynamic_optimizer | GA | lbfgsb
"""

import json
import re
import shutil
from pathlib import Path

import pandas as pd

RESULTS_ROOT = Path("/scratch/ShapeEvolve/environments/DrivAer_Star/results")
OUTPUT_ROOT  = Path("/scratch/ShapeEvolve/best_rewards/DrivaerStar")

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


# ── write output files ────────────────────────────────────────────────────────

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
        "Cd":   metric(best_row, "Cd"),
        "drag": metric(best_row, "drag"),
        "lift": metric(best_row, "lift"),
        "source": {"run_dir": str(best_dir), "design_file": str(design_file)},
    }
    (out / "best_result.json").write_text(json.dumps(result, indent=2))
    print(f"  [{reward}/{method}] reward={best_val:.4f}  src={best_dir.name}")


# ── enumerate run dirs by reward × method ────────────────────────────────────

def get_run_dirs(pattern: str) -> list[Path]:
    """Glob for run dirs directly under RESULTS_ROOT or one level inside SAVED_DIRS."""
    direct = list(RESULTS_ROOT.glob(pattern))
    nested = list(RESULTS_ROOT.glob(f"SAVED_DIRS_*/{pattern}"))
    return [p for p in direct + nested if p.is_dir()]


# (reward_key, base_reward, vtk_tag)
REWARD_DEFS = [
    ("cd_only_vtk_E",               "cd_only",               "E"),
    ("cd_only_vtk_F",               "cd_only",               "F"),
    ("cd_only_vtk_N",               "cd_only",               "N"),
    ("cd_cl_constrained_vtk_E",     "cd_cl_constrained",     "E"),
    ("cd_cl_constrained_vtk_F",     "cd_cl_constrained",     "F"),
    ("cd_cl_constrained_vtk_N",     "cd_cl_constrained",     "N"),
    ("downforce_efficiency_vtk_E",  "downforce_efficiency",  "E"),
    ("downforce_efficiency_vtk_F",  "downforce_efficiency",  "F"),
    ("downforce_efficiency_vtk_N",  "downforce_efficiency",  "N"),
    ("cd_only_tightened_bounds_vtk_E", "cd_only_tightened_bounds", "E"),
]


def vtk_tag(vtk: str, always_explicit: bool = False) -> str:
    """
    Return glob fragment for vtk variant.
    cd_only uses no tag for vtk_E (it's the implicit default).
    cd_cl_constrained and downforce_efficiency always have an explicit _vtk_E tag.
    """
    if vtk != "E" or always_explicit:
        return f"_vtk_{vtk}"
    return ""


# Rewards where vtk_E is always written explicitly in the dir name
ALWAYS_EXPLICIT_VTK = {"cd_cl_constrained", "downforce_efficiency", "cd_only_tightened_bounds"}


def collect_reward(reward_key: str, base_reward: str, vtk: str) -> None:
    explicit = base_reward in ALWAYS_EXPLICIT_VTK
    vtk_part = vtk_tag(vtk, always_explicit=explicit)

    # ── BO_torch ──────────────────────────────────────────────────────────────
    if base_reward == "cd_only_tightened_bounds":
        bo_dirs = (
            get_run_dirs(f"run_BO_torch_cd_only_tight_bounds{vtk_part}_seed*")
            + get_run_dirs(f"run_BO_torch_cd_only_super_tight_bounds{vtk_part}_seed*")
        )
    else:
        bo_dirs = get_run_dirs(f"run_BO_torch_{base_reward}{vtk_part}_seed*")
    row, rd, val = best_across(bo_dirs)
    write_output(reward_key, "BO_torch", val, row, rd)

    # ── v3_dynamic_optimizer ─────────────────────────────────────────────────
    if base_reward == "cd_only_tightened_bounds":
        v3_dirs = (
            get_run_dirs(f"run_v3_dynamic_optimizer_cd_only_tight_bounds_drivaer_star{vtk_part}_attempt*")
            + get_run_dirs(f"run_v3_dynamic_optimizer_cd_only_super_tight_bounds_drivaer_star{vtk_part}_attempt*")
        )
    else:
        v3_dirs = get_run_dirs(
            f"run_v3_dynamic_optimizer_{base_reward}_drivaer_star{vtk_part}_attempt*"
        )
        # cd_only vtk_E: some runs also omit the vtk tag entirely
        if vtk == "E" and not explicit:
            v3_dirs += get_run_dirs(
                f"run_v3_dynamic_optimizer_{base_reward}_drivaer_star_attempt*"
            )
    row, rd, val = best_across(v3_dirs)
    write_output(reward_key, "v3_dynamic_optimizer", val, row, rd)

    # ── GA ───────────────────────────────────────────────────────────────────
    if vtk == "E" and base_reward not in {"cd_only_tightened_bounds"}:
        ga_dirs = get_run_dirs(f"run_GA_{base_reward}_*attempt*")
    else:
        ga_dirs = []
    row, rd, val = best_across(ga_dirs)
    write_output(reward_key, "GA", val, row, rd)

    # ── lbfgsb ───────────────────────────────────────────────────────────────
    if vtk == "E" and base_reward not in {"cd_only_tightened_bounds"}:
        lb_dirs = get_run_dirs(f"run_lbfgsb_{base_reward}_seed*")
    else:
        lb_dirs = []
    row, rd, val = best_across(lb_dirs)
    write_output(reward_key, "lbfgsb", val, row, rd)


# ── copy analysis plots ───────────────────────────────────────────────────────

ANALYSIS_MAP = {
    "cd_only":               RESULTS_ROOT / "analysis_plots_cd_only",
    "cd_cl_constrained":     RESULTS_ROOT / "analysis_plots_cd_cl_constrained",
    "downforce_efficiency":  RESULTS_ROOT / "analysis_plots_downforce_efficiency",
    "cd_only_tightened_bounds": RESULTS_ROOT / "analysis_plots_cd_only_tightened_bounds",
}

VTK_TAG_RE = {
    "E": re.compile(r"vtk_E|tightened_bounds_vtk_E", re.IGNORECASE),
    "F": re.compile(r"vtk_F", re.IGNORECASE),
    "N": re.compile(r"vtk_N", re.IGNORECASE),
}


def copy_plots(reward_key: str, base_reward: str, vtk: str) -> None:
    src_dir = ANALYSIS_MAP.get(base_reward)
    if src_dir is None or not src_dir.exists():
        return
    dst = OUTPUT_ROOT / reward_key
    dst.mkdir(parents=True, exist_ok=True)
    pattern = VTK_TAG_RE[vtk]
    copied = 0
    for f in src_dir.iterdir():
        if f.suffix in {".png", ".pdf"} and (
            pattern.search(f.name) or not re.search(r"vtk_[EFN]", f.name)
        ):
            shutil.copy2(f, dst / f.name)
            copied += 1
    if copied:
        print(f"  [plots] {reward_key}: {copied} files from {src_dir.name}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    for reward_key, base_reward, vtk in REWARD_DEFS:
        collect_reward(reward_key, base_reward, vtk)
        copy_plots(reward_key, base_reward, vtk)
    print("\nDone.")


if __name__ == "__main__":
    main()
