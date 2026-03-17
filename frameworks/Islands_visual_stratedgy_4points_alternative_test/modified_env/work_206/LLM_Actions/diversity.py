"""
Diversity scoring functions for measuring design novelty in parameter space.

These functions help identify duplicate designs and measure exploration diversity,
but should NOT be used to adjust reward values - diversity is tracked separately.
"""

from typing import List, Dict
import numpy as np


def compute_diversity_score(new_design: List[float], context: List[Dict]) -> Dict[str, float]:
    """Compute diversity metrics for a new design relative to existing designs.
    
    Args:
        new_design: Parameter vector for the new design (list of floats)
        context: List of existing design dictionaries with 'vector' key
        
    Returns:
        Dictionary with diversity metrics:
            - min_distance: Minimum L2 distance to any existing design (novelty)
            - mean_distance: Average L2 distance to all existing designs
            - max_distance: Maximum L2 distance to any existing design
            - diversity_score: Normalized score in [0, 1] where higher = more diverse
            - n_comparisons: Number of designs compared against
    """
    if not context or not new_design:
        return {
            'min_distance': 0.0,
            'mean_distance': 0.0,
            'max_distance': 0.0,
            'diversity_score': 1.0,  # Completely novel if no context
            'n_comparisons': 0
        }
    
    new_vec = np.array(new_design)
    distances = []
    
    for item in context:
        existing_vec = np.array(item.get('vector', []))
        
        # Skip if vectors have different lengths
        if len(existing_vec) != len(new_vec):
            continue
        
        # Compute L2 (Euclidean) distance
        l2_dist = np.linalg.norm(new_vec - existing_vec)
        distances.append(l2_dist)
    
    if not distances:
        return {
            'min_distance': 0.0,
            'mean_distance': 0.0,
            'max_distance': 0.0,
            'diversity_score': 1.0,
            'n_comparisons': 0
        }
    
    distances = np.array(distances)
    min_dist = float(np.min(distances))
    mean_dist = float(np.mean(distances))
    max_dist = float(np.max(distances))
    
    # Compute diversity score:
    # - Based on minimum distance (novelty = how far from nearest neighbor)
    # - Normalized by expected maximum distance in parameter space
    # - Parameters range [-1, 1], so max possible L2 for n params is sqrt(n * (2^2)) = 2*sqrt(n)
    n_params = len(new_vec)
    max_possible_dist = 2.0 * np.sqrt(n_params)
    
    # Diversity score: min_distance normalized by max possible
    # 0 = identical to existing design, 1 = maximally different
    diversity_score = min(1.0, min_dist / max_possible_dist)
    
    return {
        'min_distance': min_dist,
        'mean_distance': mean_dist,
        'max_distance': max_dist,
        'diversity_score': diversity_score,
        'n_comparisons': len(distances)
    }


def format_diversity_report(diversity_metrics: Dict[str, float]) -> str:
    """Format diversity metrics into a human-readable string.
    
    Args:
        diversity_metrics: Output from compute_diversity_score()
        
    Returns:
        Formatted string describing the diversity
    """
    min_dist = diversity_metrics['min_distance']
    mean_dist = diversity_metrics['mean_distance']
    div_score = diversity_metrics['diversity_score']
    n_comp = diversity_metrics.get('n_comparisons', 0)
    
    # Interpret diversity score
    if div_score > 0.7:
        interpretation = "HIGHLY NOVEL - very different from existing designs"
    elif div_score > 0.4:
        interpretation = "MODERATELY NOVEL - reasonably different"
    elif div_score > 0.15:
        interpretation = "SOMEWHAT NOVEL - minor variations"
    elif div_score > 0.05:
        interpretation = "LOW NOVELTY - similar to existing designs"
    else:
        interpretation = "DUPLICATE or NEAR-DUPLICATE - almost identical to existing design"
    
    return f"""Diversity Analysis (compared to {n_comp} existing designs):
  - Nearest neighbor distance: {min_dist:.4f}
  - Average distance: {mean_dist:.4f}
  - Diversity score: {div_score:.4f} ({interpretation})"""


def is_duplicate(new_design: List[float], context: List[Dict], threshold: float = 0.05) -> bool:
    """Check if a new design is a duplicate of an existing design.
    
    Args:
        new_design: Parameter vector for the new design
        context: List of existing design dictionaries
        threshold: Diversity score below which design is considered duplicate (default: 0.05)
        
    Returns:
        True if design is a duplicate, False otherwise
    """
    metrics = compute_diversity_score(new_design, context)
    return metrics['diversity_score'] < threshold


def compute_population_diversity(designs: List[Dict]) -> Dict[str, float]:
    """Compute overall diversity metrics for a population of designs.
    
    Args:
        designs: List of design dictionaries with 'vector' key
        
    Returns:
        Dictionary with population-level metrics:
            - mean_pairwise_distance: Average distance between all pairs
            - min_pairwise_distance: Minimum distance (closest pair)
            - max_pairwise_distance: Maximum distance (furthest pair)
            - population_diversity: Overall diversity score [0, 1]
    """
    if len(designs) < 2:
        return {
            'mean_pairwise_distance': 0.0,
            'min_pairwise_distance': 0.0,
            'max_pairwise_distance': 0.0,
            'population_diversity': 0.0,
            'n_pairs': 0
        }
    
    # Extract vectors
    vectors = []
    for d in designs:
        vec = np.array(d.get('vector', []))
        if len(vec) > 0:
            vectors.append(vec)
    
    if len(vectors) < 2:
        return {
            'mean_pairwise_distance': 0.0,
            'min_pairwise_distance': 0.0,
            'max_pairwise_distance': 0.0,
            'population_diversity': 0.0,
            'n_pairs': 0
        }
    
    # Compute all pairwise distances
    pairwise_dists = []
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            if len(vectors[i]) == len(vectors[j]):
                dist = np.linalg.norm(vectors[i] - vectors[j])
                pairwise_dists.append(dist)
    
    if not pairwise_dists:
        return {
            'mean_pairwise_distance': 0.0,
            'min_pairwise_distance': 0.0,
            'max_pairwise_distance': 0.0,
            'population_diversity': 0.0,
            'n_pairs': 0
        }
    
    pairwise_dists = np.array(pairwise_dists)
    mean_dist = float(np.mean(pairwise_dists))
    min_dist = float(np.min(pairwise_dists))
    max_dist = float(np.max(pairwise_dists))
    
    # Normalize by max possible distance
    n_params = len(vectors[0])
    max_possible_dist = 2.0 * np.sqrt(n_params)
    population_diversity = min(1.0, mean_dist / max_possible_dist)
    
    return {
        'mean_pairwise_distance': mean_dist,
        'min_pairwise_distance': min_dist,
        'max_pairwise_distance': max_dist,
        'population_diversity': population_diversity,
        'n_pairs': len(pairwise_dists)
    }
