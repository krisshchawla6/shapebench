import numpy as np


def powerlaw_sample_parent_and_inspiration(database, n_inspiration, alpha=3.0):
    """Rank-based powerlaw selection. Higher alpha = more exploitation."""
    n_items = len(database)
    if n_items == 0:
        return None, []

    ranks = np.arange(1, n_items + 1)
    probabilities = ranks ** (-alpha)
    probabilities = probabilities / probabilities.sum()

    n_needed = min(1 + n_inspiration, n_items)
    indices = np.random.choice(n_items, size=n_needed, replace=False, p=probabilities)

    parent = database[indices[0]]
    inspirations = [database[i] for i in indices[1:]]
    return parent, inspirations


def powerlaw_sample_parent_from_island(database, island_idx, alpha=3.0):
    """Sample a single parent from an island using powerlaw rank-based selection."""
    mask = np.array([int(entry[4]) == island_idx for entry in database], dtype=bool)
    island_db = database[mask]
    if len(island_db) == 0:
        return None
    ranks = np.arange(1, len(island_db) + 1)
    probabilities = ranks ** (-alpha)
    probabilities = probabilities / probabilities.sum()
    idx = np.random.choice(len(island_db), p=probabilities)
    return island_db[idx]


def sample_inspirations_from_island(database, island_idx, parent_csv, n_inspiration, elite_ratio=0.3):
    """Sample inspirations from an island:
    1. Best design on island (always)
    2. Top elites
    3. Random fills remaining slots
    All excluding parent."""
    if n_inspiration <= 0:
        return []
    mask = np.array([int(entry[4]) == island_idx for entry in database], dtype=bool)
    island_db = database[mask]
    if len(island_db) == 0:
        return []

    pool = [entry for entry in island_db if entry[0] != parent_csv]
    if not pool:
        return []

    inspirations = []
    used_csvs = set()

    inspirations.append(pool[0])
    used_csvs.add(pool[0][0])
    if len(inspirations) >= n_inspiration:
        return inspirations[:n_inspiration]

    num_elites = max(0, int(n_inspiration * elite_ratio))
    for entry in pool[1:]:
        if len(inspirations) >= n_inspiration or num_elites <= 0:
            break
        if entry[0] not in used_csvs:
            inspirations.append(entry)
            used_csvs.add(entry[0])
            num_elites -= 1
    if len(inspirations) >= n_inspiration:
        return inspirations[:n_inspiration]

    remaining_pool = [e for e in pool if e[0] not in used_csvs]
    if remaining_pool:
        needed = n_inspiration - len(inspirations)
        n_random = min(needed, len(remaining_pool))
        random_indices = np.random.choice(len(remaining_pool), size=n_random, replace=False)
        for idx in random_indices:
            inspirations.append(remaining_pool[idx])

    return inspirations[:n_inspiration]
