"""
Populate best_rewards/NeuralFoil/ from benchmark_700 results.

Structure created:
  best_rewards/NeuralFoil/{reward}/
    reward.py                   ← copy of reward module
    {method}/
      best_design.json          ← overall best across all seeds
      best_result.json
      seed{N}/
        best_design.json        ← per-seed best
        best_result.json

Methods: lbfgsb, cmaes, GA, BO_torch, v3_dynamic_optimizer,
         shinka_adapter, openevolve_adapter
"""

import csv, json, shutil
from pathlib import Path
import pandas as pd

BENCH   = Path("/scratch/ShapeEvolve/environments/NeuralFoil/results/benchmark_700")
REWARDS_SRC = Path("/scratch/ShapeEvolve/environments/NeuralFoil/rewards")
OUTPUT  = Path("/scratch/ShapeEvolve/best_rewards/NeuralFoil")
METHODS = [
    "lbfgsb", "cmaes", "GA", "BO_torch",
    "v3_dynamic_optimizer", "shinka_adapter", "openevolve_adapter",
]
REWARDS = [
    "constrained_ld", "glider_endurance", "ld_ratio",
    "ld_ratio_alpha10_normalized", "ld_ratio_alpha3_normalized",
    "ld_ratio_alpha7_normalized", "ld_ratio_alpha8_normalized",
    "ld_ratio_constrained_m02_re1e7", "ld_ratio_constrained_m02_re1e7_normalized",
    "ld_ratio_constrained_m02_re1e7_normalized_conf95",
    "ld_ratio_m015_normalized", "ld_ratio_m025_normalized",
    "ld_ratio_relaxed_normalized", "low_re_multipoint",
    "max_cl", "max_cl_constrained", "min_cd_cl_floor", "min_cd_cl_target",
    "multipoint_alpha_avg_ld", "multipoint_cl_avg_cd", "multipoint_hpa",
    "multipoint_re_robustness", "range_proxy_subsonic",
    "reward_exact_notebook", "reward_hpa_endurance_weighted",
    "reward_hpa_high_cl", "reward_hpa_low_cl", "reward_hpa_mid_cl_unequal",
    "reward_hpa_strict", "weighted_cl_avg_cd",
]
RESULT_COLS = ["CL", "CD", "L_D", "CM", "total_violation", "feasible",
               "n_solved", "fitness_objective", "fitness_penalty", "fitness_total"]


def _safe_float(row, col):
    try:
        v = row.get(col, "")
        return float(v) if v not in ("", None) else None
    except (ValueError, TypeError):
        return None


def resolve_design(seed_dir: Path, row: dict, method: str) -> Path | None:
    """Return path to the design JSON for a given results.csv row."""
    if method == "lbfgsb":
        call = int(row["call"]); restart = int(row["restart"])
        return seed_dir / f"call_{call:05d}_r{restart}" / "design.json"

    if method == "cmaes":
        it = int(row["iteration"])
        name = f"iter_{it:05d}"
        return seed_dir / name / f"{name}.json"

    if method == "GA":
        it = int(row["iteration"]); pt = int(row["particle"])
        return seed_dir / f"iter_{it:04d}_p{pt:03d}" / "design.json"

    if method == "BO_torch":
        it = int(row["iteration"])
        name = f"iter_{it:04d}"
        return seed_dir / name / f"{name}.json"

    if method == "v3_dynamic_optimizer":
        design = str(row.get("design", "")).strip()
        return seed_dir / f"{design}.json"

    if method in ("shinka_adapter", "openevolve_adapter"):
        # best_design.json at seed root IS the best design for this seed
        return seed_dir / "best_design.json"

    return None


def read_best_from_seed(seed_dir: Path, method: str):
    """
    Read results.csv and return (best_row_dict, best_reward_float).
    Uses pandas for fast vectorised max on large files (lbfgsb can have 200k rows).
    For shinka/OE, best_design is pre-stored at seed root.
    """
    csv_path = seed_dir / "results.csv"
    if not csv_path.exists():
        return None, None

    df = pd.read_csv(csv_path)
    if df.empty:
        return None, None

    if method in ("shinka_adapter", "openevolve_adapter"):
        best_val = df["reward"].max()
        best_row = df.loc[df["reward"].idxmax()].to_dict()
        return best_row, float(best_val)

    rew_col = "reward"
    valid = df[df[rew_col].notna()]
    if valid.empty:
        return None, None
    idx = valid[rew_col].idxmax()
    best_val = float(valid.loc[idx, rew_col])
    best_row = valid.loc[idx].to_dict()
    return best_row, best_val


def collect_seed(reward: str, method: str, seed: int) -> dict | None:
    """
    Return a result dict for this seed, or None if no data.
    Writes best_design.json + best_result.json to output.
    """
    seed_dir = BENCH / method / reward / f"seed{seed}"
    if not seed_dir.exists():
        return None

    best_row, best_val = read_best_from_seed(seed_dir, method)
    if best_row is None or best_val is None or best_val <= float("-inf"):
        return None

    design_path = resolve_design(seed_dir, best_row, method)
    if design_path is None or not design_path.exists():
        # Try fallback: seed-level best_design.json (written by some frameworks)
        fallback = seed_dir / "best_design.json"
        if fallback.exists():
            design_path = fallback
        else:
            return None

    out_dir = OUTPUT / reward / method / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(design_path, out_dir / "best_design.json")

    result = {
        "reward": best_val,
        "method": method,
        "seed": seed,
        **{c: _safe_float(best_row, c) for c in RESULT_COLS},
        "source": {
            "seed_dir": str(seed_dir),
            "design_file": str(design_path),
        },
    }
    (out_dir / "best_result.json").write_text(json.dumps(result, indent=2))
    return result


def collect_method(reward: str, method: str):
    seed_results = []
    for seed in range(5):
        r = collect_seed(reward, method, seed)
        if r:
            seed_results.append(r)

    if not seed_results:
        print(f"    [{method}] no data")
        return

    # Overall best across seeds
    overall = max(seed_results, key=lambda x: x["reward"])
    seed_dir = BENCH / method / reward / f"seed{overall['seed']}"
    design_path = Path(overall["source"]["design_file"])

    out_dir = OUTPUT / reward / method
    out_dir.mkdir(parents=True, exist_ok=True)
    if design_path.exists():
        shutil.copy2(design_path, out_dir / "best_design.json")

    summary = {
        "best_reward":  overall["reward"],
        "best_seed":    overall["seed"],
        "mean_reward":  sum(r["reward"] for r in seed_results) / len(seed_results),
        "n_seeds":      len(seed_results),
        "seeds": [{"seed": r["seed"], "reward": r["reward"]} for r in seed_results],
        **{c: overall.get(c) for c in RESULT_COLS},
        "source": overall["source"],
    }
    (out_dir / "best_result.json").write_text(json.dumps(summary, indent=2))

    rewards_str = ", ".join(f"s{r['seed']}:{r['reward']:.3f}" for r in seed_results)
    print(f"    [{method}] best={overall['reward']:.4f}  seeds=[{rewards_str}]")


def copy_reward_py(reward: str):
    out_dir = OUTPUT / reward
    out_dir.mkdir(parents=True, exist_ok=True)
    src = REWARDS_SRC / f"{reward}.py"
    if src.exists():
        shutil.copy2(src, out_dir / "reward.py")


def main():
    import sys
    for reward in REWARDS:
        print(f"\n=== {reward} ===", flush=True)
        copy_reward_py(reward)
        for method in METHODS:
            collect_method(reward, method)
            sys.stdout.flush()
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
