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


def powerlaw_sample_parent_from_island(database, island_idx, alpha=3.0):
    mask = np.array([int(entry[4]) == island_idx for entry in database])
    island_db = database[mask]
    if len(island_db) == 0:
        return None
    ranks = np.arange(1, len(island_db) + 1)
    probabilities = ranks ** (-alpha)
    probabilities /= probabilities.sum()
    idx = np.random.choice(len(island_db), p=probabilities)
    return island_db[idx]


def sample_inspirations_from_island(database, island_idx, parent_csv, n_inspiration,
                                     elite_ratio=0.3):
    if n_inspiration <= 0:
        return []
    mask = np.array([int(entry[4]) == island_idx for entry in database])
    island_db = database[mask]
    if len(island_db) == 0:
        return []

    pool = [entry for entry in island_db if entry[0] != parent_csv]
    if not pool:
        return []

    inspirations = [pool[0]]
    used = {pool[0][0]}
    if len(inspirations) >= n_inspiration:
        return inspirations[:n_inspiration]

    num_elites = max(0, int(n_inspiration * elite_ratio))
    for entry in pool[1:]:
        if len(inspirations) >= n_inspiration or num_elites <= 0:
            break
        if entry[0] not in used:
            inspirations.append(entry)
            used.add(entry[0])
            num_elites -= 1
    if len(inspirations) >= n_inspiration:
        return inspirations[:n_inspiration]

    remaining = [e for e in pool if e[0] not in used]
    if remaining:
        needed = n_inspiration - len(inspirations)
        n_random = min(needed, len(remaining))
        for idx in np.random.choice(len(remaining), size=n_random, replace=False):
            inspirations.append(remaining[idx])

    return inspirations[:n_inspiration]
