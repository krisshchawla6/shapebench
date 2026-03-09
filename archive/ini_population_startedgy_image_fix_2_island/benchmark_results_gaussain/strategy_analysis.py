#!/usr/bin/env python3
"""Analyze strategy performance: % improvement and average reward improvement over parent."""

import json
import os
import re
from collections import defaultdict
import statistics

BASE = os.path.dirname(os.path.abspath(__file__))

# Load lineage
with open(os.path.join(BASE, "lineage.json")) as f:
    lineage = json.load(f)

# Build lookup: id -> {reward, parent_id, island}
by_id = {entry["id"]: entry for entry in lineage}

# Load strategies per design
strategies = {}
for entry in lineage:
    strat_file = os.path.join(BASE, f"design_{entry['id']}", "context", "strategy.txt")
    if os.path.exists(strat_file):
        with open(strat_file) as f:
            txt = f.read().strip()
            m = re.match(r"Strategy:\s+(\w+)\s+\(idx=(\d+)\)", txt)
            if m:
                strategies[entry["id"]] = {"name": m.group(1), "idx": int(m.group(2))}

# Compute improvements per strategy (only for designs with a parent)
strat_data = defaultdict(lambda: {
    "count": 0,
    "improvements": [],       # reward - parent_reward (absolute delta)
    "pct_improvements": [],   # (reward - parent_reward) / |parent_reward| * 100
    "rewards": [],
    "improved_count": 0,      # how many improved over parent
    "parent_rewards": [],
    "failed_count": 0,        # designs with reward <= -5.0 (penalty)
    "best_reward": float('-inf'),  # best reward produced by this strategy
    "best_design_id": None,
    "best_delta": float('-inf'),   # greatest single improvement delta
    "best_delta_design_id": None,
})

initial_designs = []

for entry in lineage:
    did = entry["id"]
    reward = entry["reward"]
    parent_id = entry["parent_id"]

    if did not in strategies:
        continue

    strat_name = strategies[did]["name"]

    if parent_id is None:
        initial_designs.append({"id": did, "reward": reward, "strategy": strat_name})
        continue

    parent_reward = by_id[parent_id]["reward"]
    delta = reward - parent_reward
    pct = (delta / abs(parent_reward) * 100) if abs(parent_reward) > 1e-10 else float('nan')

    d = strat_data[strat_name]
    d["count"] += 1
    d["improvements"].append(delta)
    d["rewards"].append(reward)
    d["parent_rewards"].append(parent_reward)
    if reward > parent_reward:
        d["improved_count"] += 1
    if reward <= -5.0:
        d["failed_count"] += 1
    if reward > d["best_reward"]:
        d["best_reward"] = reward
        d["best_design_id"] = did
    if delta > d["best_delta"]:
        d["best_delta"] = delta
        d["best_delta_design_id"] = did
    # Only include pct if parent reward is meaningful (not a penalty itself)
    if abs(parent_reward) > 1e-10:
        d["pct_improvements"].append(pct)

# Print results
print("=" * 90)
print("STRATEGY PERFORMANCE ANALYSIS (child vs parent)")
print("=" * 90)
print(f"Total designs: {len(lineage)}, Initial population (no parent): {len(initial_designs)}, With parent: {len(lineage) - len(initial_designs)}")
print()

# Sort by average reward improvement descending
sorted_strats = sorted(strat_data.items(), key=lambda x: statistics.mean(x[1]["improvements"]) if x[1]["improvements"] else 0, reverse=True)

for strat_name, d in sorted_strats:
    n = d["count"]
    if n == 0:
        continue

    avg_delta = statistics.mean(d["improvements"])
    med_delta = statistics.median(d["improvements"])
    std_delta = statistics.stdev(d["improvements"]) if n > 1 else 0.0
    avg_reward = statistics.mean(d["rewards"])
    improve_rate = d["improved_count"] / n * 100
    fail_rate = d["failed_count"] / n * 100

    # Filter out nan for pct
    valid_pcts = [p for p in d["pct_improvements"] if p == p]  # filter nan
    avg_pct = statistics.mean(valid_pcts) if valid_pcts else float('nan')
    med_pct = statistics.median(valid_pcts) if valid_pcts else float('nan')

    print(f"--- {strat_name.upper()} (n={n}) ---")
    print(f"  % designs that improved over parent:  {improve_rate:.1f}%  ({d['improved_count']}/{n})")
    print(f"  % designs that failed (reward<=-5):   {fail_rate:.1f}%  ({d['failed_count']}/{n})")
    print(f"  Avg reward delta (child - parent):    {avg_delta:+.4f}")
    print(f"  Median reward delta:                  {med_delta:+.4f}")
    print(f"  Std dev of reward delta:              {std_delta:.4f}")
    print(f"  Avg % improvement over parent:        {avg_pct:+.1f}%")
    print(f"  Median % improvement over parent:     {med_pct:+.1f}%")
    print(f"  Avg absolute reward:                  {avg_reward:.4f}")
    print(f"  Best reward produced:                 {d['best_reward']:.4f}  (design_{d['best_design_id']})")
    print()

# Detailed breakdown: top improvements and worst regressions per strategy
print("=" * 90)
print("TOP 3 IMPROVEMENTS & WORST 3 REGRESSIONS PER STRATEGY")
print("=" * 90)

for strat_name, d in sorted_strats:
    if d["count"] == 0:
        continue

    # Collect (design_id, delta, reward, parent_reward) tuples
    entries_for_strat = []
    for entry in lineage:
        did = entry["id"]
        if did not in strategies or strategies[did]["name"] != strat_name:
            continue
        if entry["parent_id"] is None:
            continue
        parent_reward = by_id[entry["parent_id"]]["reward"]
        delta = entry["reward"] - parent_reward
        entries_for_strat.append((did, delta, entry["reward"], parent_reward, entry["parent_id"]))

    entries_for_strat.sort(key=lambda x: x[1], reverse=True)

    print(f"\n--- {strat_name.upper()} ---")
    print(f"  Best improvements:")
    for did, delta, rew, prew, pid in entries_for_strat[:3]:
        print(f"    design_{did}: {prew:.4f} -> {rew:.4f}  (delta={delta:+.4f}, parent=design_{pid})")
    print(f"  Worst regressions:")
    for did, delta, rew, prew, pid in entries_for_strat[-3:]:
        print(f"    design_{did}: {prew:.4f} -> {rew:.4f}  (delta={delta:+.4f}, parent=design_{pid})")

# Summary comparison table
print("\n" + "=" * 90)
print("SUMMARY COMPARISON TABLE")
print("=" * 90)
header = f"{'Strategy':<20} {'Count':>5} {'Improve%':>9} {'AvgDelta':>10} {'MedDelta':>10} {'BestDelta':>10} {'Avg%Impr':>10} {'AvgReward':>10} {'BestReward':>11} {'BestDesign':>12} {'FailRate':>9}"
print(header)
print("-" * len(header))

for strat_name, d in sorted_strats:
    n = d["count"]
    if n == 0:
        continue
    avg_delta = statistics.mean(d["improvements"])
    med_delta = statistics.median(d["improvements"])
    avg_reward = statistics.mean(d["rewards"])
    improve_rate = d["improved_count"] / n * 100
    fail_rate = d["failed_count"] / n * 100
    valid_pcts = [p for p in d["pct_improvements"] if p == p]
    avg_pct = statistics.mean(valid_pcts) if valid_pcts else float('nan')
    best_rew = d["best_reward"]
    best_id = d["best_design_id"]
    best_delta = d["best_delta"]

    print(f"{strat_name:<20} {n:>5} {improve_rate:>8.1f}% {avg_delta:>+10.4f} {med_delta:>+10.4f} {best_delta:>+10.4f} {avg_pct:>+9.1f}% {avg_reward:>10.4f} {best_rew:>11.4f} {'design_'+str(best_id):>12} {fail_rate:>8.1f}%")

# Initial population breakdown
print("\n" + "=" * 90)
print("INITIAL POPULATION (no parent - baseline seeds)")
print("=" * 90)
for d in sorted(initial_designs, key=lambda x: x["id"]):
    print(f"  design_{d['id']}: reward={d['reward']:.4f}, strategy={d['strategy']}, island={by_id[d['id']]['island']}")
