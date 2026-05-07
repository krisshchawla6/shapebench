"""
Ranking analysis for the ShapeEvolve benchmark.
Produces statistics demonstrating optimizer rank instability across
shape categories and problem formulations.
"""
from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/scratch/ShapeEvolve/best_rewards")
OUT = ROOT / "analysis"
OUT.mkdir(exist_ok=True)

# ── method family mapping (many naming variants → canonical name) ─────────────
FAMILY = {
    "BO": "BO", "BO_torch": "BO", "BO_torch_approx": "BO",
    "BO_torch_approx_ld_ratio_500calls": "BO",
    "GA": "GA", "GAp": "GA", "GA_parallel": "GA",
    "GA_ld_ratio_20p_20i": "GA", "GA_ld_ratio_30p_100i": "GA",
    "cmaes": "CMA-ES",
    "lbfgsb": "L-BFGS-B", "lbfgsb_ld_ratio_2r40i": "L-BFGS-B",
    "lbfgsb_ld_ratio_nr3_maxiter200": "L-BFGS-B",
    "v3": "LLM (v3)", "v3_dynamic_optimizer": "LLM (v3)",
    "v3_flash2_5": "LLM (v3)", "v3_dynamic_optimizer_ld_ratio_flash25": "LLM (v3)",
    "openevolve": "OpenEvolve", "openevolve_adapter": "OpenEvolve",
    "openevolve_adapter_ld_ratio_flash25": "OpenEvolve",
    "shinka": "Shinka", "shinka_adapter": "Shinka",
    "shinka_adapter_ld_ratio_flash25": "Shinka",
    "drivaer_star_3d_islands": "Islands (GA)",
}


# ── 1. Load all data ──────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    rows = []
    for env_dir in sorted(ROOT.iterdir()):
        if not env_dir.is_dir() or not (env_dir / "..").exists():
            continue
        if any(env_dir.name.endswith(ext) for ext in [".py", ".txt", ".csv", ".json"]):
            continue
        if env_dir.name == "analysis":
            continue
        for reward_dir in sorted(env_dir.iterdir()):
            if not reward_dir.is_dir():
                continue
            for method_dir in sorted(reward_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                for fname in ["best_result.json", "best_reward.json"]:
                    rj = method_dir / fname
                    if rj.exists():
                        data = json.loads(rj.read_text())
                        val = data.get("reward") or data.get("best_reward")
                        if val is not None:
                            method = method_dir.name
                            rows.append({
                                "env": env_dir.name,
                                "problem": f"{env_dir.name}/{reward_dir.name}",
                                "reward": reward_dir.name,
                                "method": method,
                                "family": FAMILY.get(method, method),
                                "value": float(val),
                            })
                        break
    return pd.DataFrame(rows)


# ── 2. Normalize scores per problem → [0, 1] ──────────────────────────────────
def add_normalized(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["norm"] = np.nan
    for problem, grp in df.groupby("problem"):
        lo, hi = grp["value"].min(), grp["value"].max()
        if hi > lo:
            df.loc[grp.index, "norm"] = (grp["value"] - lo) / (hi - lo)
        else:
            df.loc[grp.index, "norm"] = 1.0
    return df


# ── 3. Compute per-problem ranks (1 = best) ────────────────────────────────────
def add_ranks(df: pd.DataFrame, group_col: str = "method") -> pd.DataFrame:
    df = df.copy()
    df["rank"] = np.nan
    for problem, grp in df.groupby("problem"):
        ranked = grp.sort_values("value", ascending=False).reset_index()
        for pos, (_, row) in enumerate(ranked.iterrows(), start=1):
            df.loc[row["index"], "rank"] = pos
    return df


def build_pivot(df: pd.DataFrame, value_col: str, index_col: str = "method") -> pd.DataFrame:
    return df.pivot_table(index=index_col, columns="problem",
                          values=value_col, aggfunc="first")


# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    df = load_data()
    df = add_normalized(df)
    df = add_ranks(df)

    print(f"\n{'='*70}")
    print("SHAPEEVOLVE BENCHMARK — OPTIMIZER RANKING ANALYSIS")
    print(f"{'='*70}")
    print(f"  Environments : {df['env'].nunique()}")
    print(f"  Problems     : {df['problem'].nunique()}  (env × reward)")
    print(f"  Method variants : {df['method'].nunique()}")
    print(f"  Method families : {df['family'].nunique()}")
    print(f"  Total cases  : {len(df)}")

    # ── Family-level aggregation ──────────────────────────────────────────────
    # For each (problem, family) keep the best value seen across variants
    fam_df = (df.groupby(["problem", "env", "reward", "family"])["value"]
                .max().reset_index())
    fam_df = add_normalized(fam_df.rename(columns={"family": "method"}))
    fam_df = add_ranks(fam_df)
    fam_df = fam_df.rename(columns={"method": "family"})

    # ── Table 1: Average rank & win-rate per family ───────────────────────────
    summary = []
    for fam, grp in fam_df.groupby("family"):
        problems = grp["problem"].nunique()
        avg_rank = grp["rank"].mean()
        avg_norm = grp["norm"].mean()
        wins = (grp["rank"] == 1).sum()
        top3 = (grp["rank"] <= 3).sum()
        rank_std = grp["rank"].std()
        summary.append({
            "Family": fam,
            "#Problems": problems,
            "Avg Rank": round(avg_rank, 2),
            "Rank Std": round(rank_std, 2),
            "Win Rate": f"{wins}/{problems}",
            "Top-3 Rate": f"{top3}/{problems}",
            "Avg Norm Score": round(avg_norm, 3),
        })
    tbl1 = pd.DataFrame(summary).sort_values("Avg Rank")
    print(f"\n{'─'*70}")
    print("TABLE 1 — Family-level summary (ranked by avg rank across all problems)")
    print(f"{'─'*70}")
    print(tbl1.to_string(index=False))
    tbl1.to_csv(OUT / "table1_family_summary.csv", index=False)

    # ── Table 2: Per-environment average rank per family ─────────────────────
    env_rank = (fam_df.groupby(["env", "family"])["rank"]
                      .mean().round(2).reset_index()
                      .pivot(index="family", columns="env", values="rank"))
    env_rank["Overall Avg"] = env_rank.mean(axis=1).round(2)
    env_rank = env_rank.sort_values("Overall Avg")
    print(f"\n{'─'*70}")
    print("TABLE 2 — Avg rank per family per environment (lower = better)")
    print(f"{'─'*70}")
    print(env_rank.to_string())
    env_rank.to_csv(OUT / "table2_env_rank_matrix.csv")

    # ── Table 3: Rank instability — spread of ranks per family ───────────────
    instability = []
    for fam, grp in fam_df.groupby("family"):
        ranks = grp["rank"].dropna()
        instability.append({
            "Family": fam,
            "Min Rank": int(ranks.min()),
            "Max Rank": int(ranks.max()),
            "Rank Range": int(ranks.max() - ranks.min()),
            "Rank Std": round(ranks.std(), 2),
            "#Problems": len(ranks),
            "Times #1": int((ranks == 1).sum()),
            "Times Last": int((ranks == ranks.max()).sum()),
        })
    tbl3 = pd.DataFrame(instability).sort_values("Rank Std", ascending=False)
    print(f"\n{'─'*70}")
    print("TABLE 3 — Rank instability per family (sorted by rank std, desc)")
    print(f"{'─'*70}")
    print(tbl3.to_string(index=False))
    tbl3.to_csv(OUT / "table3_rank_instability.csv", index=False)

    # ── Table 4: Pairwise win matrix (family A beats B on how many problems) ──
    families = sorted(fam_df["family"].unique())
    win_matrix = pd.DataFrame(0, index=families, columns=families)
    for problem, grp in fam_df.groupby("problem"):
        grp = grp.set_index("family")
        for f1, f2 in combinations(families, 2):
            if f1 in grp.index and f2 in grp.index:
                v1, v2 = grp.loc[f1, "value"], grp.loc[f2, "value"]
                if v1 > v2:
                    win_matrix.loc[f1, f2] += 1
                elif v2 > v1:
                    win_matrix.loc[f2, f1] += 1
    win_matrix["Total wins"] = win_matrix.sum(axis=1)
    win_matrix = win_matrix.sort_values("Total wins", ascending=False)
    print(f"\n{'─'*70}")
    print("TABLE 4 — Pairwise wins (row beats column on N problems)")
    print(f"{'─'*70}")
    print(win_matrix.to_string())
    win_matrix.to_csv(OUT / "table4_pairwise_wins.csv")

    # ── Table 5: Spearman rank correlations between environments ─────────────
    # For each env, build a family → avg_norm_score vector; correlate pairs
    env_vectors: dict[str, pd.Series] = {}
    for env, grp in fam_df.groupby("env"):
        env_vectors[env] = grp.groupby("family")["norm"].mean()

    envs = sorted(env_vectors.keys())
    corr_matrix = pd.DataFrame(np.nan, index=envs, columns=envs)
    for e1, e2 in combinations(envs, 2):
        shared = env_vectors[e1].index.intersection(env_vectors[e2].index)
        if len(shared) >= 3:
            r, p = stats.spearmanr(env_vectors[e1][shared], env_vectors[e2][shared])
            corr_matrix.loc[e1, e2] = round(r, 3)
            corr_matrix.loc[e2, e1] = round(r, 3)
    for e in envs:
        corr_matrix.loc[e, e] = 1.0
    print(f"\n{'─'*70}")
    print("TABLE 5 — Spearman rank correlation of family scores between environments")
    print("  (close to 0 or negative = rankings change across environments)")
    print(f"{'─'*70}")
    print(corr_matrix.to_string())
    corr_matrix.to_csv(OUT / "table5_spearman_corr.csv")

    # ── Stat 6: Within-problem rank volatility ───────────────────────────────
    # For each problem, what is the spread between best and worst method?
    # High spread = big performance gap, rankings really matter
    gaps = []
    for problem, grp in fam_df.groupby("problem"):
        env, reward = problem.split("/", 1)
        n_methods = len(grp)
        best_val = grp["value"].max()
        worst_val = grp["value"].min()
        norm_gap = grp["norm"].max() - grp["norm"].min()  # always 1 if ≥2 methods
        best_fam = grp.loc[grp["value"].idxmax(), "family"]
        worst_fam = grp.loc[grp["value"].idxmin(), "family"]
        gaps.append({
            "Problem": problem,
            "Env": env,
            "#Methods": n_methods,
            "Best Family": best_fam,
            "Worst Family": worst_fam,
            "Best Value": round(best_val, 4),
            "Worst Value": round(worst_val, 4),
        })
    tbl6 = pd.DataFrame(gaps).sort_values("#Methods", ascending=False)
    print(f"\n{'─'*70}")
    print("TABLE 6 — Per-problem winner and loser (shows rank volatility)")
    print(f"{'─'*70}")
    print(tbl6.to_string(index=False))
    tbl6.to_csv(OUT / "table6_per_problem_winners.csv", index=False)

    # ── Stat 7: How often does each family rank #1 per environment? ──────────
    win_by_env = (fam_df[fam_df["rank"] == 1]
                  .groupby(["env", "family"])
                  .size()
                  .unstack(fill_value=0))
    # add total problems per env for reference
    total_per_env = fam_df.groupby("env")["problem"].nunique()
    win_by_env.loc["Total problems"] = total_per_env
    print(f"\n{'─'*70}")
    print("TABLE 7 — #1 finishes per family per environment (columns=env)")
    print(f"{'─'*70}")
    print(win_by_env.to_string())
    win_by_env.to_csv(OUT / "table7_wins_by_env.csv")

    # ── Key headline numbers for paper ───────────────────────────────────────
    print(f"\n{'═'*70}")
    print("KEY NUMBERS FOR PAPER")
    print(f"{'═'*70}")

    # No single winner
    best_wins = tbl1.sort_values("Avg Rank").iloc[0]
    max_wins_row = (fam_df[fam_df["rank"] == 1]
                    .groupby("family").size().reset_index(name="wins")
                    .sort_values("wins", ascending=False).iloc[0])
    print(f"\n• Best avg-rank family : {best_wins['Family']} (avg rank {best_wins['Avg Rank']})")
    print(f"• Most #1 finishes     : {max_wins_row['family']} ({max_wins_row['wins']} problems)")
    total_problems = fam_df["problem"].nunique()
    print(f"• Total problems       : {total_problems}")
    print(f"• No single method wins all problems — max win rate: "
          f"{max_wins_row['wins']}/{total_problems} "
          f"({100*max_wins_row['wins']/total_problems:.1f}%)")

    # Rank reversal: pairs of environments where rankings flip
    reversals = 0
    pairs_checked = 0
    for e1, e2 in combinations(envs, 2):
        shared = env_vectors[e1].index.intersection(env_vectors[e2].index)
        if len(shared) >= 3:
            pairs_checked += 1
            r = corr_matrix.loc[e1, e2]
            if not np.isnan(r) and r < 0.3:
                reversals += 1
    print(f"\n• Env pairs with Spearman ρ < 0.3 (weak/no agreement): "
          f"{reversals}/{pairs_checked}")

    avg_corr = corr_matrix.replace(1.0, np.nan).stack().mean()
    print(f"• Mean cross-env Spearman ρ : {avg_corr:.3f}")

    # Rank instability headline
    most_volatile = tbl3.iloc[0]
    print(f"\n• Most volatile family : {most_volatile['Family']} "
          f"(ranks #{int(most_volatile['Min Rank'])}–#{int(most_volatile['Max Rank'])} "
          f"across problems, σ={most_volatile['Rank Std']})")

    print(f"\nAll tables saved to: {OUT}")
    print()


if __name__ == "__main__":
    main()
