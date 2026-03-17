import numpy as np


def update_database(database, x, reward, results, island_idx=0):
    """Database columns: [design_path, rank, reward, results, island_idx]"""
    entry = np.array([[x, 0, reward, results, island_idx]], dtype=object)
    if len(database) == 0:
        return entry
    database = np.append(database, entry, axis=0)
    indices = np.argsort(database[:, 2].astype(float))[::-1]
    database = database[indices]
    for i in range(len(database)):
        database[i, 1] = i
    return database


def empty_database():
    return np.array([], dtype=object).reshape(0, 5)
