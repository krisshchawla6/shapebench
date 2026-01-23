# Diversity Scoring Module Documentation

## Overview
Created a separate `diversity.py` module to measure how different designs are from each other in parameter space. This helps identify duplicate designs and track exploration, **WITHOUT modifying reward values**. Diversity and performance are tracked independently.

## Location
`/scratch/LLM_Evolve/AirFoil_becnhmark/modified_env/LLM_Actions/diversity.py`

## Functions

### `compute_diversity_score(new_design, context)`
Computes diversity metrics for a new design relative to existing designs.

**Parameters:**
- `new_design`: List of floats representing the parameter vector
- `context`: List of design dictionaries with 'vector' key

**Returns:**
Dictionary with:
- `min_distance`: Minimum L2 distance to any existing design (novelty metric)
- `mean_distance`: Average L2 distance to all existing designs
- `max_distance`: Maximum L2 distance to any existing design
- `diversity_score`: Normalized score in [0, 1] where higher = more diverse
- `n_comparisons`: Number of existing designs compared against

**Diversity Score Interpretation:**
- `1.0`: Maximally different (complete opposite in parameter space)
- `0.7+`: Highly novel
- `0.4-0.7`: Moderately novel
- `0.15-0.4`: Somewhat novel (minor variations)
- `0.05-0.15`: Low novelty (similar to existing)
- `<0.05`: Duplicate or near-duplicate

### `format_diversity_report(diversity_metrics)`
Formats diversity metrics into a human-readable string.

**Parameters:**
- `diversity_metrics`: Output dictionary from `compute_diversity_score()`

**Returns:**
- Formatted string with interpretation

### `is_duplicate(new_design, context, threshold=0.05)`
Quick check if a design is too similar to existing designs.

**Parameters:**
- `new_design`: List of floats representing the parameter vector
- `context`: List of design dictionaries
- `threshold`: Diversity score below which design is considered duplicate (default: 0.05)

**Returns:**
- `True` if duplicate, `False` otherwise

### `compute_population_diversity(designs)`
Compute overall diversity metrics for a population of designs.

**Parameters:**
- `designs`: List of design dictionaries with 'vector' key

**Returns:**
Dictionary with:
- `mean_pairwise_distance`: Average distance between all pairs
- `min_pairwise_distance`: Closest pair distance
- `max_pairwise_distance`: Furthest pair distance
- `population_diversity`: Overall diversity score [0, 1]
- `n_pairs`: Number of pairs compared

## Usage Example

```python
from diversity import (
    compute_diversity_score, 
    format_diversity_report,
    is_duplicate,
    compute_population_diversity
)

# Context with existing designs
context = [
    {'vector': [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 'reward': 0.1},
    {'vector': [0.8, 0.1, 0.0, 0.9, 0.1, 0.0], 'reward': 0.15}
]

# New design to evaluate
new_design = [0.9, 0.05, 0.02, 0.95, 0.05, 0.01]

# Compute diversity
metrics = compute_diversity_score(new_design, context)
print(format_diversity_report(metrics))
```

## Benchmark Analysis Results

### generate_direct (Slurm run)
- **Total designs**: 6
- **Average diversity score**: 0.0358 (VERY LOW!)
- **Near-duplicates**: 4 out of 6 designs (66.7%)
- **Most novel design**: diversity = 0.1541

### modify (Slurm run)
- **Total designs**: 6
- **Average diversity score**: 0.0478 (VERY LOW!)
- **Near-duplicates**: 4 out of 6 designs (66.7%)
- **Most novel design**: diversity = 0.2391

## Key Findings

1. **Critical Problem Identified**: Both `generate_direct` and `modify` actions are generating near-duplicate designs, explaining why the agent gets stuck.

2. **Root Causes**:
   - LLM has low temperature (deterministic) and tends to converge
   - No explicit diversity incentive in the prompts
   - Context may not provide enough variety

3. **Solutions to Implement**:
   - Add diversity penalty/reward in design selection
   - Include diversity score in LLM feedback
   - Increase temperature for more stochastic sampling
   - Add explicit diversity instructions to prompts
   - Filter out near-duplicates before evaluation

## Important: Diversity vs Reward

**Diversity and reward are tracked separately and should NOT be combined:**
- **Reward**: Measures aerodynamic performance (lift, drag, etc.)
- **Diversity**: Measures exploration and novelty in parameter space

**Why keep them separate:**
1. Performance and exploration serve different purposes
2. Adding diversity to reward can bias optimization away from high-performance regions
3. Allows independent analysis of exploration vs exploitation
4. Enables post-hoc diversity analysis without affecting benchmarks

## Integration Points

### For Logging and Analysis (✅ Recommended)
### For Logging and Analysis (✅ Recommended)
```python
from diversity import compute_diversity_score, format_diversity_report

# After LLM generates design, compute diversity for logging
metrics = compute_diversity_score(new_params, context)
print(f"Diversity: {metrics['diversity_score']:.4f}")

# Save diversity metrics separately (don't modify reward!)
design_metadata = {
    'params': new_params,
    'reward': reward,  # Keep reward separate
    'diversity': metrics['diversity_score']  # Track diversity independently
}
```

### For Duplicate Detection (✅ Recommended)
```python
from diversity import is_duplicate

# Check before evaluating expensive CFD simulation
if is_duplicate(new_params, context, threshold=0.05):
    print("⚠️  Design is near-duplicate, skipping CFD evaluation")
    # Either skip or retry generation
```

### For Population Analysis (✅ Recommended)
```python
from diversity import compute_population_diversity

# Analyze entire benchmark run
pop_metrics = compute_population_diversity(all_designs)
print(f"Population diversity: {pop_metrics['population_diversity']:.4f}")
```

### DON'T Do This (❌ Not Recommended)
```python
# ❌ BAD: Don't adjust reward with diversity
adjusted_reward = base_reward + diversity_score * 0.1  # NO!

# ❌ BAD: Don't use diversity for design selection
best_design = max(designs, key=lambda d: d['reward'] + d['diversity'])  # NO!
```

## Test Script
Run `/scratch/LLM_Evolve/AirFoil_becnhmark/modified_env/LLM_Actions/test_diversity.py` to:
- Test diversity function with synthetic data
- Analyze diversity in benchmark results
- Identify duplicate designs

## Files Created/Modified
- `modified_env/LLM_Actions/diversity.py`: **New standalone module** (diversity functions)
- `modified_env/LLM_Actions/test_diversity.py`: Test and analysis script
- `prompts/base.py`: Removed diversity functions (kept prompts clean)
- `prompts/__init__.py`: Removed diversity exports
- `DIVERSITY_FUNCTION.md`: This documentation

## Next Steps
1. ✅ Created diversity module (separate from rewards)
2. ✅ Tested on benchmark data
3. ✅ Identified low diversity as major issue (66% duplicates)
4. 🔲 Add diversity logging to LLM_agent.py (for analysis only)
5. 🔲 Implement duplicate detection to skip redundant CFD runs
6. 🔲 Add population diversity metrics to benchmark reports
7. 🔲 Investigate why LLM generates duplicates (temperature? prompts?)
8. 🔲 Consider diversity-aware prompt engineering (without changing rewards)
