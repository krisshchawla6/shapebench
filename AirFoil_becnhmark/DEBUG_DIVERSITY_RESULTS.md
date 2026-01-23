# Diversity Debugging Results

## Summary
Successfully identified the root cause of low diversity in LLM-generated designs through controlled testing.

## Test Setup
- **Action tested**: `generate_direct`
- **Samples per test**: 10 designs
- **Comparison**: WITH context vs WITHOUT context
- **Temperature**: 1.0 (same for both)

## Key Findings

### 1. Context Reduces Diversity
| Metric | With Context | Without Context |
|--------|--------------|-----------------|
| Mean diversity | 0.096 | 0.115 |
| Near-duplicates (< 0.05) | 4/9 (44%) | 1/9 (11%) |
| Parameter mean | 0.31 | 0.04 |
| Parameter std | 0.41 | 0.42 |
| Parameter range | [0.6, 0.9] | [-0.9, 0.8] |

### 2. Parameter Clustering
**WITH Context:**
- Parameters heavily biased towards [0.7, 0.9] range
- Mean = 0.31 (positive bias)
- Most values in positive range
- 44% of designs are near-duplicates

**WITHOUT Context:**
- Parameters spread across full [-1, 1] space
- Mean = 0.04 (centered)
- Better exploration of parameter space
- Only 11% near-duplicates

### 3. Root Cause
The LLM is **over-exploiting** the context information:
1. Context shows baseline design with certain parameter values
2. LLM infers these are "good" values
3. LLM generates new designs in similar parameter regions
4. Result: Clustering around baseline, low diversity

## Visual Evidence

**Sample parameter vectors (first 6 values):**

WITH context:
```
Sample 0: [0.7, 0.3, 0.1, 0.7, -0.3, 0.1]
Sample 1: [0.75, 0.5, 0.2, 0.85, -0.3, 0.1]
Sample 2: [0.7, 0.3, 0.1, 0.9, 0.5, 0.2]
Sample 3: [0.8, 0.2, 0.1, 0.9, 0.5, 0.3]
```
→ **Notice**: All start with 0.7-0.8 in first parameter!

WITHOUT context:
```
Sample 0: [0.2, 0.3, 0.1, 0.7, 0.6, -0.2]
Sample 1: [-0.1, 0.2, 0.0, 0.3, 0.8, 0.1]
Sample 2: [0.2, 0.3, -0.1, 0.7, 0.5, 0.2]
Sample 3: [0.2, 0.3, 0.0, 0.7, 0.5, 0.2]
```
→ **Much more variety** in parameter values!

## Implications for Benchmark

This explains why the Slurm benchmarks had:
- 66% duplicates in `generate_direct`
- 66% duplicates in `modify`
- Agent getting "stuck" on similar designs
- Poor exploration of design space

## Possible Solutions

### 1. Increase Temperature (✅ Easy)
```python
run_llm_action(temperature=1.5)  # Higher = more random
```

### 2. Diverse Context Sampling (✅ Recommended)
Instead of just parent + nearby designs, sample from across the database:
- High performers
- Low performers  
- Random samples
- Diverse parameter regions

### 3. Add Diversity Instructions to Prompts (✅ Recommended)
```python
"Generate designs that explore DIFFERENT parameter regions.
Avoid clustering around previous values."
```

### 4. Novelty-Based Rejection (✅ Recommended)
```python
if diversity_score < 0.1:
    print("Design too similar, retrying...")
    # regenerate with higher temperature or modified prompt
```

### 5. Parameter Space Partitioning (Advanced)
Explicitly guide LLM to explore different regions:
- "Generate design with parameters mostly in [-1, 0]"
- "Generate design with parameters mostly in [0, 1]"
- Rotate through regions

## Files Created
- `debug_diversity.py`: Debug script for testing with/without context
- `debug_generate_direct_with_context/`: Results with context
- `debug_generate_direct_no_context/`: Results without context
- `debug_output.log`: Full test log

## Usage
```bash
# Test a single action
python debug_diversity.py --action generate_direct --n-samples 10

# Compare with/without context
python debug_diversity.py --action generate_direct --n-samples 10 --compare

# Test without context only
python debug_diversity.py --action generate_direct --n-samples 10 --no-context
```

## Next Steps
1. ✅ Root cause identified (context causes clustering)
2. 🔲 Implement diverse context sampling
3. 🔲 Add diversity instructions to prompts
4. 🔲 Test with higher temperature
5. 🔲 Implement novelty-based rejection
6. 🔲 Rerun benchmark with improvements
