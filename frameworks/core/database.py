import json
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


def _to_serializable(obj):
    """Recursively convert numpy scalars/arrays to plain Python types."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_database(database, path):
    """Persist the in-memory database to a JSON file."""
    entries = []
    for row in database:
        entries.append({
            'path':    str(row[0]) if row[0] is not None else None,
            'rank':    int(row[1]),
            'reward':  float(row[2]),
            'results': _to_serializable(row[3]) if isinstance(row[3], dict) else {},
            'island':  int(row[4]),
        })
    with open(path, 'w') as f:
        json.dump(entries, f)


def load_database(path):
    """Load a JSON database file back to a numpy object array."""
    with open(path) as f:
        entries = json.load(f)
    if not entries:
        return empty_database()
    rows = []
    for e in entries:
        rows.append([e['path'], e['rank'], e['reward'], e['results'], e['island']])
    db = np.array(rows, dtype=object)
    return db
