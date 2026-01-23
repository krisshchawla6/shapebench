#!/usr/bin/env python3
"""
Debug script to identify the source of low diversity in LLM-generated designs.
Compares generation with and without context to understand the issue.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diversity import compute_diversity_score, format_diversity_report
from LLM_agent import run_llm_action

def test_generation_diversity(action: str, n_samples: int = 10, use_context: bool = True):
    """Test diversity of generated designs."""
    
    print(f"\n{'='*80}")
    print(f"Testing {action} action")
    print(f"Context: {'WITH' if use_context else 'WITHOUT'}")
    print(f"Samples: {n_samples}")
    print(f"{'='*80}\n")
    
    # Setup context
    if use_context:
        baseline_path = '/scratch/LLM_Evolve/AirFoil_becnhmark/modified_env/baseline_action.csv'
        if os.path.exists(baseline_path):
            baseline_vec = np.loadtxt(baseline_path, delimiter=',')
            if baseline_vec.ndim > 1:
                baseline_vec = baseline_vec.flatten()
            context = [{
                'vector': baseline_vec.tolist(),
                'reward': 0.0264,
                'ranking': 0,
                'images': [],
                'drag': -4.328319,
                'lift': 0.114442
            }]
        else:
            context = [{
                'vector': [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                'reward': 0.0264,
                'ranking': 0,
                'images': [],
                'drag': -4.328319,
                'lift': 0.114442
            }]
    else:
        context = []
    
    output_dir = f'/scratch/LLM_Evolve/AirFoil_becnhmark/modified_env/LLM_Actions/debug_{action}_{"with" if use_context else "no"}_context'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate designs
    generated_vectors = []
    generation_details = []
    
    for i in range(n_samples):
        print(f"\n--- Sample {i+1}/{n_samples} ---")
        
        try:
            csv_path = run_llm_action(
                action=action,
                context=context,
                output_dir=output_dir,
                name=f'test_{i}',
                temperature=1.0,
                skip_vis=True
            )
            
            if csv_path and os.path.exists(csv_path):
                vec = np.loadtxt(csv_path, delimiter=',')
                if vec.ndim > 1:
                    vec = vec.flatten()
                
                # Compute diversity relative to all previous samples
                diversity_metrics = compute_diversity_score(
                    vec.tolist(),
                    [{'vector': v.tolist()} for v in generated_vectors]
                )
                
                generated_vectors.append(vec)
                generation_details.append({
                    'sample_idx': i,
                    'vector': vec.tolist(),
                    'csv_path': csv_path,
                    'diversity_score': diversity_metrics['diversity_score'],
                    'min_distance': diversity_metrics['min_distance']
                })
                
                print(f"✓ Generated: {os.path.basename(csv_path)}")
                if i > 0:
                    print(f"  Diversity: {diversity_metrics['diversity_score']:.4f} (min dist: {diversity_metrics['min_distance']:.4f})")
            else:
                print(f"✗ Generation failed")
                
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Analyze results
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}\n")
    
    if not generated_vectors:
        print("No designs generated!")
        return
    
    # Convert to numpy array for analysis
    vectors_array = np.array(generated_vectors)
    
    # Count unique designs
    unique_vectors = np.unique(vectors_array, axis=0)
    n_unique = len(unique_vectors)
    n_duplicates = len(vectors_array) - n_unique
    
    print(f"Total designs: {len(vectors_array)}")
    print(f"Unique designs: {n_unique} ({n_unique/len(vectors_array)*100:.1f}%)")
    print(f"Duplicates: {n_duplicates} ({n_duplicates/len(vectors_array)*100:.1f}%)")
    
    # Parameter statistics
    print(f"\nParameter Statistics:")
    print(f"  Mean: {vectors_array.mean():.4f}")
    print(f"  Std:  {vectors_array.std():.4f}")
    print(f"  Min:  {vectors_array.min():.4f}")
    print(f"  Max:  {vectors_array.max():.4f}")
    
    # Diversity evolution
    diversity_scores = [d['diversity_score'] for d in generation_details[1:]]  # Skip first (no comparison)
    if diversity_scores:
        print(f"\nDiversity Evolution:")
        print(f"  Mean diversity: {np.mean(diversity_scores):.4f}")
        print(f"  Std diversity:  {np.std(diversity_scores):.4f}")
        print(f"  Min diversity:  {np.min(diversity_scores):.4f}")
        print(f"  Max diversity:  {np.max(diversity_scores):.4f}")
        
        # Count near-duplicates
        near_dupes = sum(1 for d in diversity_scores if d < 0.05)
        print(f"  Near-duplicates (diversity < 0.05): {near_dupes}/{len(diversity_scores)}")
    
    # Show which samples are duplicates
    print(f"\nDuplicate Analysis:")
    for i in range(len(generated_vectors)):
        matches = []
        for j in range(i):
            if np.allclose(generated_vectors[i], generated_vectors[j], atol=1e-6):
                matches.append(j)
        
        if matches:
            print(f"  Sample {i} is duplicate of: {matches}")
    
    # Show actual parameter values for first few designs
    print(f"\nFirst 5 Parameter Vectors:")
    for i in range(min(5, len(generated_vectors))):
        vec = generated_vectors[i]
        print(f"  Sample {i}: {vec[:6]}... (showing first 6 params)")
    
    # Save detailed results
    results_file = os.path.join(output_dir, 'debug_analysis.json')
    with open(results_file, 'w') as f:
        json.dump({
            'action': action,
            'use_context': use_context,
            'n_samples': n_samples,
            'n_unique': int(n_unique),
            'n_duplicates': int(n_duplicates),
            'parameter_stats': {
                'mean': float(vectors_array.mean()),
                'std': float(vectors_array.std()),
                'min': float(vectors_array.min()),
                'max': float(vectors_array.max())
            },
            'diversity_stats': {
                'mean': float(np.mean(diversity_scores)) if diversity_scores else 0.0,
                'std': float(np.std(diversity_scores)) if diversity_scores else 0.0,
                'min': float(np.min(diversity_scores)) if diversity_scores else 0.0,
                'max': float(np.max(diversity_scores)) if diversity_scores else 0.0
            },
            'generation_details': generation_details
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    return {
        'n_unique': n_unique,
        'n_total': len(vectors_array),
        'diversity_scores': diversity_scores
    }


def compare_with_without_context(action: str = 'generate_direct', n_samples: int = 10):
    """Compare diversity with and without context."""
    
    print("\n" + "="*80)
    print(f"COMPARATIVE ANALYSIS: {action}")
    print("="*80)
    
    # Test with context
    print("\n" + "="*80)
    print("PART 1: WITH CONTEXT")
    print("="*80)
    results_with = test_generation_diversity(action, n_samples, use_context=True)
    
    # Test without context
    print("\n" + "="*80)
    print("PART 2: WITHOUT CONTEXT")
    print("="*80)
    results_without = test_generation_diversity(action, n_samples, use_context=False)
    
    # Compare
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    if results_with and results_without:
        print(f"\nWith Context:")
        print(f"  Unique: {results_with['n_unique']}/{results_with['n_total']} ({results_with['n_unique']/results_with['n_total']*100:.1f}%)")
        if results_with['diversity_scores']:
            print(f"  Mean diversity: {np.mean(results_with['diversity_scores']):.4f}")
        
        print(f"\nWithout Context:")
        print(f"  Unique: {results_without['n_unique']}/{results_without['n_total']} ({results_without['n_unique']/results_without['n_total']*100:.1f}%)")
        if results_without['diversity_scores']:
            print(f"  Mean diversity: {np.mean(results_without['diversity_scores']):.4f}")
        
        print(f"\nConclusion:")
        if results_with['n_unique'] > results_without['n_unique']:
            print(f"  ✓ Context HELPS diversity (+{results_with['n_unique'] - results_without['n_unique']} more unique designs)")
        elif results_with['n_unique'] < results_without['n_unique']:
            print(f"  ✗ Context HURTS diversity (-{results_without['n_unique'] - results_with['n_unique']} unique designs)")
        else:
            print(f"  = Context has NO EFFECT on diversity")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug diversity issues')
    parser.add_argument('--action', type=str, default='generate_direct',
                        choices=['generate', 'generate_direct', 'modify', 'modify_direct'],
                        help='Action to test')
    parser.add_argument('--n-samples', type=int, default=10,
                        help='Number of samples to generate')
    parser.add_argument('--compare', action='store_true',
                        help='Compare with and without context')
    parser.add_argument('--no-context', action='store_true',
                        help='Test without context')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_with_without_context(args.action, args.n_samples)
    else:
        test_generation_diversity(
            args.action, 
            args.n_samples, 
            use_context=not args.no_context
        )


if __name__ == '__main__':
    main()
