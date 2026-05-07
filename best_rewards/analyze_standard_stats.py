"""
Standard benchmark statistics for academic paper.
Computes Friedman test, Nemenyi post-hoc, critical differences,
performance profiles, and regret metrics.
"""
from __future__ import annotations

import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon

warnings.filterwarnings("ignore")

ROOT = Path("/scratch/ShapeEvolve/best_rewards")
OUT = ROOT / "analysis"
OUT.mkdir(exist_ok=True)

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


def load_data() -> pd.DataFrame:
    rows = []
    for env_dir in sorted(ROOT.iterdir()):
        if not env_dir.is_dir() or env_dir.name in ("analysis",):
            continue
        if any(env_dir.name.endswith(e) for e in (".py", ".txt", ".csv", ".json")):
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


def family_best(df: pd.DataFrame) -> pd.DataFrame:
    """Best value per (problem, family)."""
    return (df.groupby(["problem", "env", "reward", "family"])["value"]
              .max().reset_index().rename(columns={"family": "method"}))


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["norm"] = np.nan
    for problem, grp in df.groupby("problem"):
        lo, hi = grp["value"].min(), grp["value"].max()
        df.loc[grp.index, "norm"] = (grp["value"] - lo) / (hi - lo) if hi > lo else 1.0
    return df


def rank_within_problem(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rank"] = np.nan
    for problem, grp in df.groupby("problem"):
        order = grp["value"].rank(ascending=False, method="min")
        df.loc[grp.index, "rank"] = order
    return df


# ── Friedman + Nemenyi critical difference ────────────────────────────────────
def nemenyi_cd(n_methods: int, n_problems: int, alpha: float = 0.05) -> float:
    """Critical difference for Nemenyi post-hoc (two-tailed)."""
    # q_alpha values from Demšar 2006, Table 5 (alpha=0.05)
    q = {2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728,
         6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    q_a = q.get(n_methods, 3.164)
    return q_a * np.sqrt(n_methods * (n_methods + 1) / (6 * n_problems))


def friedman_test(rank_pivot: pd.DataFrame) -> tuple[float, float]:
    """Run Friedman test on a methods × problems rank matrix."""
    data = [rank_pivot[col].dropna().values for col in rank_pivot.columns]
    # Only use problems where all methods have results
    complete = rank_pivot.dropna(axis=1)
    if complete.shape[1] < 3:
        return np.nan, np.nan
    groups = [complete.loc[m].values for m in complete.index]
    stat, p = friedmanchisquare(*groups)
    return stat, p


# ── Performance profile (Dolan & Moré 2002) ──────────────────────────────────
def performance_profile(norm_pivot: pd.DataFrame, tau_max: float = 5.0,
                         n_points: int = 200) -> pd.DataFrame:
    """
    Fraction of problems where method achieves norm_score >= best * (1/tau).
    Here we use: rho(tau) = fraction of problems where norm >= 1 - (tau-1)/tau_max
    Simplified: for each tau in [0,1], fraction of problems where method's
    normalized score >= (1 - tau). tau=0 means exactly optimal, tau=1 means any.
    """
    methods = norm_pivot.index.tolist()
    taus = np.linspace(0, 1, n_points)
    profile = {}
    for m in methods:
        scores = norm_pivot.loc[m].dropna().values
        profile[m] = [np.mean(scores >= (1 - t)) for t in taus]
    return pd.DataFrame(profile, index=taus)


# ── Wilcoxon signed-rank pairwise ─────────────────────────────────────────────
def pairwise_wilcoxon(norm_pivot: pd.DataFrame) -> pd.DataFrame:
    methods = norm_pivot.index.tolist()
    results = []
    for m1, m2 in combinations(methods, 2):
        shared = norm_pivot.loc[[m1, m2]].dropna(axis=1)
        if shared.shape[1] < 5:
            continue
        v1, v2 = shared.loc[m1].values, shared.loc[m2].values
        if np.all(v1 == v2):
            continue
        try:
            stat, p = wilcoxon(v1, v2)
            winner = m1 if v1.mean() > v2.mean() else m2
            results.append({"Method A": m1, "Method B": m2,
                             "W-stat": round(stat, 1), "p-value": round(p, 4),
                             "Significant (p<0.05)": p < 0.05,
                             "Better method": winner})
        except Exception:
            pass
    return pd.DataFrame(results).sort_values("p-value")


# ── Regret / optimality gap ───────────────────────────────────────────────────
def regret_stats(norm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimality gap = 1 - norm_score (0 = optimal, 1 = worst).
    Reports mean, median, 90th-pct gap per method.
    """
    rows = []
    for method, grp in norm_df.groupby("method"):
        gaps = 1.0 - grp["norm"].dropna()
        rows.append({
            "Family": method,
            "N problems": len(gaps),
            "Mean gap": round(gaps.mean(), 3),
            "Median gap": round(gaps.median(), 3),
            "90th-pct gap": round(gaps.quantile(0.9), 3),
            "% within 5% of best": round(100 * (gaps <= 0.05).mean(), 1),
            "% within 10% of best": round(100 * (gaps <= 0.10).mean(), 1),
        })
    return pd.DataFrame(rows).sort_values("Mean gap")


# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    df = load_data()
    fdf = normalize(rank_within_problem(family_best(df)))

    # Pivot: methods × problems (normalized scores)
    norm_pivot = fdf.pivot_table(index="method", columns="problem",
                                  values="norm", aggfunc="first")
    rank_pivot = fdf.pivot_table(index="method", columns="problem",
                                  values="rank", aggfunc="first")

    methods = norm_pivot.index.tolist()
    n_methods = len(methods)
    n_problems = norm_pivot.shape[1]

    print(f"\n{'='*70}")
    print("STANDARD BENCHMARK STATISTICS")
    print(f"{'='*70}")
    print(f"  Methods: {n_methods}  |  Problems: {n_problems}")

    # ── 1. Average rank (Demšar 2006) ─────────────────────────────────────────
    avg_ranks = rank_pivot.mean(axis=1).sort_values()
    print(f"\n{'─'*70}")
    print("1. AVERAGE RANK (Demšar 2006) — lower is better")
    print(f"{'─'*70}")
    for m, r in avg_ranks.items():
        n = rank_pivot.loc[m].notna().sum()
        print(f"  {m:<22} avg rank = {r:.3f}  (over {n} problems)")
    avg_ranks.round(3).to_csv(OUT / "stat1_avg_ranks.csv", header=["avg_rank"])

    # ── 2. Friedman test — use largest complete subset ────────────────────────
    # Find the largest subset of methods with >= 10 jointly-covered problems
    CORE = ["BO", "GA", "L-BFGS-B", "LLM (v3)", "OpenEvolve", "Shinka"]
    core_present = [m for m in CORE if m in rank_pivot.index]
    complete = rank_pivot.loc[core_present].dropna(axis=1)
    n_complete = complete.shape[1]
    n_core = len(core_present)
    groups = [complete.loc[m].values for m in core_present]
    stat_f, p_f = friedmanchisquare(*groups)
    print(f"\n{'─'*70}")
    print("2. FRIEDMAN TEST (are rankings significantly different?)")
    print(f"{'─'*70}")
    print(f"  Methods used: {core_present}")
    print(f"  χ²_F = {stat_f:.3f},  p = {p_f:.2e},  df = {n_core - 1}")
    print(f"  Problems with all {n_core} methods present: {n_complete}")
    sig = "YES — rankings differ significantly" if p_f < 0.05 else "NO"
    print(f"  Significant at α=0.05: {sig}")

    # ── 3. Nemenyi critical difference ────────────────────────────────────────
    cd = nemenyi_cd(n_core, n_complete)
    # recompute avg ranks on the complete subset for Nemenyi
    avg_ranks_core = complete.rank(axis=0, ascending=True).mean(axis=1).sort_values()
    print(f"\n{'─'*70}")
    print("3. NEMENYI POST-HOC — Critical Difference (α=0.05)")
    print(f"{'─'*70}")
    print(f"  CD = {cd:.3f}  (avg rank diff > {cd:.3f} → significantly different)")
    print(f"\n  Avg ranks on complete subset ({n_complete} problems):")
    for m, r in avg_ranks_core.items():
        print(f"    {m:<22} {r:.3f}")
    print(f"\n  Pairs significantly different (|Δavg_rank| > CD={cd:.3f}):")
    sig_pairs = []
    for m1, m2 in combinations(avg_ranks_core.index, 2):
        delta = abs(avg_ranks_core[m1] - avg_ranks_core[m2])
        if delta > cd:
            sig_pairs.append((m1, m2, round(avg_ranks[m1], 3),
                               round(avg_ranks[m2], 3), round(delta, 3)))
            print(f"    {m1} vs {m2}: Δ={delta:.3f}")
    if not sig_pairs:
        print("    (none — insufficient complete problems for significance)")
    pd.DataFrame(sig_pairs, columns=["Method A","Method B","Rank A","Rank B","Delta"]
                 ).to_csv(OUT / "stat3_nemenyi_sig_pairs.csv", index=False)

    # ── 4. Wilcoxon signed-rank pairwise ─────────────────────────────────────
    wdf = pairwise_wilcoxon(norm_pivot)
    print(f"\n{'─'*70}")
    print("4. WILCOXON SIGNED-RANK TESTS (pairwise, on normalized scores)")
    print(f"{'─'*70}")
    sig_w = wdf[wdf["Significant (p<0.05)"] == True]
    print(f"  {len(wdf)} pairs tested, {len(sig_w)} significant at α=0.05:")
    print(wdf.to_string(index=False))
    wdf.to_csv(OUT / "stat4_wilcoxon.csv", index=False)

    # ── 5. Optimality gap / regret ────────────────────────────────────────────
    rdf = regret_stats(fdf)
    print(f"\n{'─'*70}")
    print("5. OPTIMALITY GAP (1 - normalized score; 0 = best possible)")
    print(f"{'─'*70}")
    print(rdf.to_string(index=False))
    rdf.to_csv(OUT / "stat5_optimality_gap.csv", index=False)

    # ── 6. Performance profile AUC ────────────────────────────────────────────
    pp = performance_profile(norm_pivot)
    auc = pp.mean(axis=0).sort_values(ascending=False)  # higher AUC = better
    print(f"\n{'─'*70}")
    print("6. PERFORMANCE PROFILE AUC (Dolan & Moré 2002)")
    print("   AUC of ρ(τ): fraction of problems within τ of optimal (higher=better)")
    print(f"{'─'*70}")
    for m, a in auc.items():
        print(f"  {m:<22}  AUC = {a:.4f}")
    auc.round(4).to_csv(OUT / "stat6_performance_profile_auc.csv", header=["AUC"])
    pp.to_csv(OUT / "stat6_performance_profile_data.csv")

    # ── 7. Scott-Knott-style grouping via avg rank gaps ───────────────────────
    print(f"\n{'─'*70}")
    print("7. RANK GROUPING (methods within CD of each other = statistically tied)")
    print(f"{'─'*70}")
    sorted_methods = avg_ranks.index.tolist()
    groups_sk = []
    current_group = [sorted_methods[0]]
    for i in range(1, len(sorted_methods)):
        if avg_ranks[sorted_methods[i]] - avg_ranks[current_group[0]] <= cd:
            current_group.append(sorted_methods[i])
        else:
            groups_sk.append(current_group)
            current_group = [sorted_methods[i]]
    groups_sk.append(current_group)
    for gi, grp in enumerate(groups_sk, 1):
        print(f"  Group {gi}: {', '.join(grp)}  "
              f"(avg ranks {avg_ranks[grp].min():.2f}–{avg_ranks[grp].max():.2f})")

    # ── 8. Cross-environment Spearman matrix + summary ───────────────────────
    env_vectors: dict[str, pd.Series] = {}
    for env, grp in fdf.groupby("env"):
        env_vectors[env] = grp.groupby("method")["norm"].mean()
    envs = sorted(env_vectors)
    corrs = []
    for e1, e2 in combinations(envs, 2):
        shared = env_vectors[e1].index.intersection(env_vectors[e2].index)
        if len(shared) >= 3:
            r, p = stats.spearmanr(env_vectors[e1][shared], env_vectors[e2][shared])
            corrs.append({"Env A": e1, "Env B": e2,
                          "Spearman ρ": round(r, 3), "p-value": round(p, 4),
                          "Significant": p < 0.05})
    corr_df = pd.DataFrame(corrs).sort_values("Spearman ρ")
    print(f"\n{'─'*70}")
    print("8. CROSS-ENVIRONMENT SPEARMAN RANK CORRELATIONS")
    print(f"{'─'*70}")
    print(corr_df.to_string(index=False))
    mean_r = corr_df["Spearman ρ"].mean()
    neg = (corr_df["Spearman ρ"] < 0).sum()
    weak = (corr_df["Spearman ρ"].abs() < 0.3).sum()
    print(f"\n  Mean ρ = {mean_r:.3f}  |  Negative ρ: {neg}/{len(corr_df)}"
          f"  |  |ρ| < 0.3: {weak}/{len(corr_df)}")
    corr_df.to_csv(OUT / "stat8_cross_env_spearman.csv", index=False)

    # ── Headline numbers ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("HEADLINE NUMBERS")
    print(f"{'='*70}")
    print(f"\n  • Friedman χ²_F = {stat_f:.1f},  p = {p_f:.2e}  → rankings are"
          f" {'SIGNIFICANTLY' if p_f<0.05 else 'NOT'} different (α=0.05, {n_core} methods, {n_complete} problems)")
    print(f"  • Nemenyi CD = {cd:.3f}  ({len(sig_pairs)} method pairs significantly differ)")
    best_m = avg_ranks.index[0]
    worst_m = avg_ranks.index[-1]
    print(f"  • Best avg rank: {best_m} ({avg_ranks[best_m]:.2f})  |  "
          f"Worst: {worst_m} ({avg_ranks[worst_m]:.2f})")
    best_gap = rdf.sort_values("Mean gap").iloc[0]
    worst_gap = rdf.sort_values("Mean gap").iloc[-1]
    print(f"  • Mean optimality gap: {best_gap['Family']} = {best_gap['Mean gap']} (best)  "
          f"|  {worst_gap['Family']} = {worst_gap['Mean gap']} (worst)")
    print(f"  • Mean cross-env ρ = {mean_r:.3f}  |  "
          f"{weak}/{len(corr_df)} env pairs |ρ|<0.3  |  {neg} negative")
    print(f"\n  All outputs → {OUT}")


if __name__ == "__main__":
    main()
