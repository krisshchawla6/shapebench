import numpy as np


def perform_migration(database, num_islands, migration_rate):
    """Move-based elitist migration between islands."""
    if num_islands < 2 or migration_rate <= 0:
        return database
    total_migrated = 0
    for source_idx in range(num_islands):
        mask = np.array([int(entry[4]) == source_idx for entry in database])
        island_indices = np.where(mask)[0]
        if len(island_indices) <= 1:
            continue
        best_idx = island_indices[0]
        eligible = [idx for idx in island_indices if idx != best_idx]
        if not eligible:
            continue
        num_migrants = max(1, int(len(island_indices) * migration_rate))
        num_migrants = min(num_migrants, len(eligible))
        migrants = np.random.choice(eligible, size=num_migrants, replace=False)
        dest_islands = [i for i in range(num_islands) if i != source_idx]
        for migrant_idx in migrants:
            database[migrant_idx, 4] = np.random.choice(dest_islands)
            total_migrated += 1
    if total_migrated > 0:
        print(f"  Migration: moved {total_migrated} designs between islands")
    return database
