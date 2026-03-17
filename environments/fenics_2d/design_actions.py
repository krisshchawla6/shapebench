import os, sys
import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from shapes_utils import Shape # Not needed for action manipulation directly

def save_action_csv(path, action_array):
    """Save action array as a flat, comma-separated CSV."""
    # Ensure it's a flat array
    action_flat = np.array(action_array).flatten()
    # Save as a single row
    np.savetxt(path, [action_flat], delimiter=',', fmt='%.6f')
    return path

def linear_sample(b):
    """Uniform sampling between [lo, hi]."""
    return np.random.uniform(b[0], b[1]) if isinstance(b, (list, tuple)) else b

def gaussian_sampling_vectors(vectors, std_scale=0.04):
    """Sample from Gaussian centered at mean of provided vectors."""
    vecs = np.array(vectors)
    mean = np.mean(vecs, axis=0)
    std = np.std(vecs, axis=0) + std_scale # Add base noise
    return np.random.normal(mean, std)

def gaussian_sampling(mean, std=0.04, bounds=(-1.0, 1.0)):
    """Sample a Gaussian action vector around the provided mean.

    mean: Flat or nested array of mean values (one per action parameter).
    std: Fixed standard deviation for all parameters.
    bounds: Tuple of (min, max) bounds for each action value.
    """
    mean_arr = np.array(mean, dtype=float).flatten()
    std_arr = np.ones(len(mean_arr)) * float(std)
    sampled = np.random.normal(mean_arr, std_arr)
    return np.clip(sampled, bounds[0], bounds[1])


def generate(n_cp, n_sp, params, out_dir='./output', name='shape'):
    """
    Generate action vector with linear sampling.
    params: List of [val0, val1, val2] per control point.
    """
    os.makedirs(out_dir, exist_ok=True)
    # params is list of N points, each has 3 values (or bounds)
    actions = []
    for p in params:
        # p is [v0, v1, v2] where each can be value or bounds
        pt_action = [linear_sample(p[0]), linear_sample(p[1]), linear_sample(p[2])]
        actions.extend(pt_action)
    
    return save_action_csv(f"{out_dir}/{name}_0.csv", actions)

def generate_direct(n_cp, n_sp, params, out_dir='./output', name='shape'):
    """
    Pure direct generation - LLM provides exact values.
    params: List of [val0, val1, val2] per control point.
    """
    os.makedirs(out_dir, exist_ok=True)
    actions = []
    for p in params:
        # p is [v0, v1, v2]
        actions.extend(p)
    return save_action_csv(f"{out_dir}/{name}_0.csv", actions)

def modify(base_csv, pt_idx, values, out_dir='./output', name=None):
    """
    Modify an existing action CSV.
    base_csv: Path to baseline action CSV.
    values: List of [val0, val1, val2] for each index in pt_idx.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Load base action
    # Assuming base_csv is a flat comma-separated file
    try:
        base_action = np.loadtxt(base_csv, delimiter=',')
        if base_action.ndim == 2:
            base_action = base_action[0] # Handle if it was saved with multiple rows
    except Exception as e:
        print(f"Error loading base CSV {base_csv}: {e}")
        return None

    # Reshape to (N, 3) for easier indexing
    # We assume the length is divisible by 3
    n_pts = len(base_action) // 3
    action_matrix = base_action.reshape((n_pts, 3))

    for i, idx in enumerate(pt_idx):
        if idx >= n_pts:
            continue # Skip invalid indices
        
        v = values[i] # [v0, v1, v2] (can be bounds)
        
        # Update values
        action_matrix[idx, 0] = linear_sample(v[0])
        action_matrix[idx, 1] = linear_sample(v[1])
        action_matrix[idx, 2] = linear_sample(v[2])
    
    # Determine output name
    if name is None:
        # Try to infer index from filename or default to next
        import re
        match = re.search(r'_(\d+)\.csv$', base_csv)
        idx_num = int(match.group(1)) + 1 if match else 1
        base_name = os.path.basename(base_csv).split('_')[0]
        name = f"{base_name}_{idx_num}"
    else:
        # If name is provided, use it directly (append .csv if needed, but save_action_csv assumes full path or we construct it)
        # The logic below matches original: f"{out_dir}/{name or s.name}_{s.index+1}.csv"
        # We'll just use the provided name logic
        pass

    # Reconstruct name logic to match original's intent but adapted
    final_name = f"{name}.csv" if not name.endswith('.csv') else name
    
    return save_action_csv(f"{out_dir}/{final_name}", action_matrix.flatten())

def modify_direct(base_csv, pt_idx, values, out_dir='./output', name=None):
    """Pure direct modification of action CSV."""
    # Logic is same as modify but values are exact
    # We can reuse modify since linear_sample handles scalar values too
    return modify(base_csv, pt_idx, values, out_dir, name)

def generate_from_vectors(n_cp, n_sp, vectors, out_dir='./output', name='gauss_gen'):
    """
    Generate action by Gaussian sampling from prior action vectors.
    vectors: List of action arrays (flat or shaped).
    """
    # vectors should be a list of lists/arrays
    # We assume each vector in 'vectors' is the full action array
    vec = gaussian_sampling_vectors(vectors)
    return save_action_csv(f"{out_dir}/{name}_0.csv", vec)


def gaussain(n_cp, n_sp, params, out_dir='./output', name='gaussain'):
    """
    Generate action vector by Gaussian sampling from LLM-proposed mean values.
    params: List of [val0, val1, val2] per control point (mean values).
    """
    os.makedirs(out_dir, exist_ok=True)
    mean = np.array(params).flatten()
    vec = gaussian_sampling(mean)
    return save_action_csv(f"{out_dir}/{name}_0.csv", vec)

def run_action(action, **kwargs):
    return globals()[action](**kwargs)

if __name__ == "__main__":
    # Example 1: Linear sample
    # 3 values per point: [radius_param, angle_param, edgy_param]
    b = [[(0.0, 0.1), (-0.1, 0.1), 0.0]]*4
    print(f"Linear Gen: {run_action('generate', n_cp=4, n_sp=10, params=b, name='lin')}")
    
    # Example 2: Gaussian sampling
    # Priors are flat action arrays (size 12 for 4 points)
    priors = [
        np.zeros(12),
        np.ones(12) * 0.1
    ]
    sampled_vec = run_action('generate_from_vectors', n_cp=4, n_sp=10, vectors=priors)
    print(f"Gaussian Sampled file: {sampled_vec}")
