#!/usr/bin/env python3
"""
Test the diversity scoring function with real benchmark data.
"""

import os
import sys
import json
import numpy as np

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diversity import compute_diversity_score, format_diversity_report, compute_population_diversity


def load_designs_from_results(results_json_path: str):
    """Load design vectors from a results.json file."""
    with open(results_json_path, 'r') as f:
        data = json.load(f)
    
    designs = []
    for item in data:
        csv_path = item['csv_path']
        if os.path.exists(csv_path):
            vector = np.loadtxt(csv_path, delimiter=',')
            if vector.ndim > 1:
                vector = vector.flatten()
            designs.append({
                'vector': vector.tolist(),
                'reward': item['reward'],
                'ranking': item['rank']
            })
    
    return designs


def analyze_diversity_in_benchmark(results_json_path: str):
    """Analyze diversity of designs in a benchmark run."""
    designs = load_designs_from_results(results_json_path)
    
    if len(designs) < 2:
        print("Not enough designs to analyze diversity")
        return
    
    print(f"Analyzing {len(designs)} designs from: {results_json_path}")
    print("=" * 80)
    
    # Compute pairwise diversity
    diversity_matrix = []
    for i, design in enumerate(designs):
        # Compare to all previous designs
        context = designs[:i]
        if context:
            metrics = compute_diversity_score(design['vector'], context)
            diversity_matrix.append({
                'design_idx': i,
                'reward': design['reward'],
                **metrics
            })
    
    if not diversity_matrix:
        print("No diversity comparisons available")
        return
    
    # Print summary statistics
    min_dists = [d['min_distance'] for d in diversity_matrix]
    mean_dists = [d['mean_distance'] for d in diversity_matrix]
    div_scores = [d['diversity_score'] for d in diversity_matrix]
    
    print("\nOverall Diversity Statistics:")
    print("-" * 80)
    print(f"Average minimum distance (novelty): {np.mean(min_dists):.4f}")
    print(f"Average mean distance: {np.mean(mean_dists):.4f}")
    print(f"Average diversity score: {np.mean(div_scores):.4f}")
    print(f"Min diversity score: {np.min(div_scores):.4f} (least novel design)")
    print(f"Max diversity score: {np.max(div_scores):.4f} (most novel design)")
    
    # Identify duplicates or near-duplicates
    near_duplicates = [d for d in diversity_matrix if d['diversity_score'] < 0.05]
    if near_duplicates:
        print(f"\n⚠️  Found {len(near_duplicates)} near-duplicate designs:")
        for dup in near_duplicates:
            print(f"   Design {dup['design_idx']}: diversity={dup['diversity_score']:.4f}, reward={dup['reward']:.6f}")
    
    # Show per-design diversity
    print("\nPer-Design Diversity:")
    print("-" * 80)
    for d in diversity_matrix[:10]:  # Show first 10
        print(f"Design {d['design_idx']} (reward={d['reward']:.6f}):")
        print(f"  Min dist: {d['min_distance']:.4f}, Diversity: {d['diversity_score']:.4f}")


def test_diversity_function():
    """Test the diversity function with synthetic data."""
    print("\n" + "=" * 80)
    print("TEST 1: Identical designs (should have 0 diversity)")
    print("=" * 80)
    
    context = [
        {'vector': [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 'reward': 0.1}
    ]
    new_design = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    
    metrics = compute_diversity_score(new_design, context)
    print(format_diversity_report(metrics))
    
    print("\n" + "=" * 80)
    print("TEST 2: Completely different designs (should have high diversity)")
    print("=" * 80)
    
    context = [
        {'vector': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'reward': 0.1}
    ]
    new_design = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    
    metrics = compute_diversity_score(new_design, context)
    print(format_diversity_report(metrics))
    
    print("\n" + "=" * 80)
    print("TEST 3: Slightly modified design (should have low-moderate diversity)")
    print("=" * 80)
    
    context = [
        {'vector': [0.5, 0.0, 0.0, 0.5, 0.0, 0.0], 'reward': 0.1}
    ]
    new_design = [0.55, 0.05, 0.02, 0.48, 0.01, -0.02]
    
    metrics = compute_diversity_score(new_design, context)
    print(format_diversity_report(metrics))


def main():
    print("=" * 80)
    print("DIVERSITY FUNCTION TESTS")
    print("=" * 80)
    
    # Test with synthetic data
    test_diversity_function()
    
    # Test with real benchmark data
    benchmark_dirs = [
        '/scratch/LLM_Evolve/AirFoil_becnhmark/benchmark_results_generate_direct',
        '/scratch/LLM_Evolve/AirFoil_becnhmark/benchmark_results_modify',
    ]
    
    for bench_dir in benchmark_dirs:
        results_file = os.path.join(bench_dir, 'results.json')
        if os.path.exists(results_file):
            print("\n" + "=" * 80)
            print(f"ANALYZING: {os.path.basename(bench_dir)}")
            print("=" * 80)
            analyze_diversity_in_benchmark(results_file)


if __name__ == '__main__':
    main()
