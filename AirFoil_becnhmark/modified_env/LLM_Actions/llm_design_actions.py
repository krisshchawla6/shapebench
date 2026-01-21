import os, sys
import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shapes_utils import Shape

def save_csv(path, n_cp, n_sp, cp, r, e):
    with open(path, 'w') as f:
        f.write(f"{n_cp} {n_sp}\n")
        [f.write(f"{x}\n") for x in r]
        [f.write(f"{x}\n") for x in e]
        [f.write(f"{p[0]} {p[1]}\n") for p in cp]
    return path

def linear_sample(b):
    """Uniform sampling between [lo, hi]."""
    return np.random.uniform(b[0], b[1]) if isinstance(b, (list, tuple)) else b

def gaussian_sampling(vectors, std_scale=0.1):
    """Sample from Gaussian centered at mean of provided vectors."""
    vecs = np.array(vectors)
    mean = np.mean(vecs, axis=0)
    std = np.std(vecs, axis=0) + std_scale # Add base noise
    return np.random.normal(mean, std)

def generate(n_cp, n_sp, params, out_dir='./output', name='shape'):
    """Generate with linear sampling support (params can be values or [lo, hi] bounds)."""
    os.makedirs(out_dir, exist_ok=True)
    cp, r, e = np.zeros((n_cp, 2)), np.zeros(n_cp), np.zeros(n_cp)
    for i, p in enumerate(params):
        cp[i], r[i], e[i] = [linear_sample(p[0]), linear_sample(p[1])], linear_sample(p[2]), linear_sample(p[3])
    return save_csv(f"{out_dir}/{name}_0.csv", n_cp, n_sp, cp, r, e)

def generate_direct(n_cp, n_sp, params, out_dir='./output', name='shape'):
    """Pure direct generation - LLM provides exact values, no sampling."""
    os.makedirs(out_dir, exist_ok=True)
    cp, r, e = np.zeros((n_cp, 2)), np.zeros(n_cp), np.zeros(n_cp)
    for i, p in enumerate(params):
        cp[i], r[i], e[i] = [p[0], p[1]], p[2], p[3]
    return save_csv(f"{out_dir}/{name}_0.csv", n_cp, n_sp, cp, r, e)

def modify(base_csv, pt_idx, values, out_dir='./output', name=None):
    """Modify with linear sampling support (values can be direct or [lo, hi] bounds)."""
    os.makedirs(out_dir, exist_ok=True)
    s = Shape(); s.read_csv(base_csv)
    for i, idx in enumerate(pt_idx):
        v = values[i]
        # value format: [x, y, radius, edgy]
        s.control_pts[idx] = [linear_sample(v[0]), linear_sample(v[1])]
        s.radius[idx] = linear_sample(v[2])
        s.edgy[idx] = linear_sample(v[3])
    return save_csv(f"{out_dir}/{name or s.name}_{s.index+1}.csv", s.n_control_pts, s.n_sampling_pts, s.control_pts, s.radius, s.edgy)

def modify_direct(base_csv, pt_idx, values, out_dir='./output', name=None):
    """Pure direct modification - LLM provides exact values, no sampling."""
    os.makedirs(out_dir, exist_ok=True)
    s = Shape(); s.read_csv(base_csv)
    for i, idx in enumerate(pt_idx):
        v = values[i]
        # value format: [x, y, radius, edgy]
        s.control_pts[idx] = [v[0], v[1]]
        s.radius[idx] = v[2]
        s.edgy[idx] = v[3]
    return save_csv(f"{out_dir}/{name or s.name}_{s.index+1}.csv", s.n_control_pts, s.n_sampling_pts, s.control_pts, s.radius, s.edgy)

def generate_from_vectors(n_cp, n_sp, vectors, out_dir='./output', name='gauss_gen'):
    """Generate shape by Gaussian sampling from a set of prior vectors."""
    vec = gaussian_sampling(vectors)
    vec = vec.reshape((n_cp, 4))
    return generate(n_cp, n_sp, vec.tolist(), out_dir=out_dir, name=name)

def run_action(action, **kwargs):
    return globals()[action](**kwargs)

if __name__ == "__main__":
    # Example 1: Linear sample (LLM provided bounds)
    b = [[(0.8,1.2), (-0.1,0.1), 0.5, 0.5]]*4
    print(f"Linear Gen: {run_action('generate', n_cp=4, n_sp=10, params=b, name='lin')}")
    
    # Example 2: Gaussian sampling from priors (set of vectors)
    priors = [[1.0, 0.0, 0.5, 0.5], [1.1, 0.1, 0.4, 0.6], [0.9, -0.1, 0.6, 0.4]]
    sampled_vec = run_action('gaussian_sampling', vectors=priors)
    print(f"Gaussian Sampled Vector: {sampled_vec}")
