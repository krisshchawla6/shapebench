import numpy as np

def powerlaw_sample_parent_and_inspiration(database, n_inspiration, alpha=3.0):
    n_items = len(database)
    if n_items == 0:
        return None, []
    ranks = np.arange(1, n_items + 1)
    probabilities = ranks ** (-alpha)
    probabilities /= probabilities.sum()
    n_needed = min(1 + n_inspiration, n_items)
    indices = np.random.choice(n_items, size=n_needed, replace=False, p=probabilities)
    parent = database[indices[0]]
    inspirations = [database[i] for i in indices[1:]]
    return parent, inspirations
