"""FrameworkSampler -- StateSampler adapter for ShapeEvolve framework sampling methods.

Wraps any ShapeEvolve framework's existing sampling (powerlaw, island-based, etc.)
behind discover's StateSampler interface so the training loop can use either PUCT
or the framework's native sampling.

The framework's sampling module is passed in directly -- nothing is reimplemented.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Callable, Optional

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "discover"))

from ttt_discover.tinker_utils.sampler import StateSampler
from ttt_discover.tinker_utils.state import State

from .shapeevolve_state import ShapeEvolveState, params_to_construction

logger = logging.getLogger(__name__)


class FrameworkSampler(StateSampler):
    """Adapts a ShapeEvolve framework's sampling into discover's StateSampler.

    Maintains an internal archive of ShapeEvolveStates (mirroring the framework's
    numpy database), and delegates sample selection to the framework's sampling
    functions (e.g., powerlaw_sample_parent_and_inspiration).

    Args:
        fw_sampling: framework sampling module (has powerlaw_sample_parent_and_inspiration, etc.)
        fw_design_actions: framework design_actions module (for BOUNDS, CONTINUOUS_KEYS)
        env_type: the Environment class (for create_initial_state)
        problem_type: environment name string
        log_path: directory for persistence
        alpha: power-law exponent for rank-based selection
        n_inspirations: number of inspiration states to sample alongside parent
        num_islands: number of islands (1 = single population)
    """

    def __init__(
        self,
        fw_sampling: Any,
        env_type: type,
        problem_type: str,
        log_path: str,
        alpha: float = 3.0,
        n_inspirations: int = 2,
        num_islands: int = 1,
    ):
        self._fw_sampling = fw_sampling
        self._env_type = env_type
        self._problem_type = problem_type
        self._log_path = log_path
        self._alpha = alpha
        self._n_inspirations = n_inspirations
        self._num_islands = num_islands

        self._lock = threading.Lock()
        self._archive: list = []
        self._visit_counts: dict = {}

        self._file_path = os.path.join(log_path, "framework_sampler.json")
        self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(self._file_path):
            try:
                with open(self._file_path) as f:
                    data = json.load(f)

                self._archive = [ShapeEvolveState(**s) for s in data.get("states", [])]
                self._visit_counts = data.get("visit_counts", {})
                logger.info(f"Loaded {len(self._archive)} states from {self._file_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load sampler state: {e}")

        initial = self._env_type.create_initial_state(self._problem_type)
        self._archive = [initial]

    def _to_database_row(self, state: ShapeEvolveState):
        """Convert a ShapeEvolveState to the numpy row format the framework expects:
        [json_path, rank, reward, results_dict, island_idx]
        """
        params = state.design_params

        return np.array([
            state.design_path or "",
            0,
            state.value or 0.0,
            {
                "images": state.image_paths,
                "feedback": state.observation or "",
                "metrics": {},
            },
            0,
        ], dtype=object)

    def _build_database(self) -> np.ndarray:
        """Build the framework-style numpy database from the archive."""
        if not self._archive:
            return np.empty((0, 5), dtype=object)

        ranked = sorted(
            self._archive,
            key=lambda s: s.value if s.value is not None else float("-inf"),
            reverse=True,
        )

        rows = []
        for rank, state in enumerate(ranked):
            row = self._to_database_row(state)
            row[1] = rank
            rows.append(row)

        return np.array(rows, dtype=object)

    def sample_states(self, num_states: int) -> list[State]:
        with self._lock:
            valid = [s for s in self._archive if s.value is not None and s.value > -5.0]

            if not valid:
                return ([self._archive[0]] * num_states) if self._archive else []

            database = self._build_database()
            sampled = []

            for _ in range(num_states):
                if hasattr(self._fw_sampling, "powerlaw_sample_parent_and_inspiration"):
                    parent, _inspirations = self._fw_sampling.powerlaw_sample_parent_and_inspiration(
                        database, self._n_inspirations, alpha=self._alpha
                    )

                    if parent is not None:
                        state = self._db_row_to_state(parent)

                        if state:
                            sampled.append(state)
                            self._visit_counts[state.id] = self._visit_counts.get(state.id, 0) + 1
                            continue

                idx = np.random.choice(len(valid))
                sampled.append(valid[idx])
                self._visit_counts[valid[idx].id] = self._visit_counts.get(valid[idx].id, 0) + 1

            return sampled

    def _db_row_to_state(self, db_row) -> Optional[ShapeEvolveState]:
        """Find the archive state matching a database row."""
        design_path = db_row[0]
        reward = float(db_row[2])
        for s in self._archive:
            if s.design_path == design_path:
                return s
            if s.value is not None and abs(s.value - reward) < 1e-8:
                return s
        return None

    def update_states(
        self,
        states: list[State],
        parent_states: list[State],
        save: bool = True,
        step: int | None = None,
    ):
        with self._lock:
            for child, parent in zip(states, parent_states):
                self._set_parent_info(child, parent)
                if child not in self._archive:
                    self._archive.append(child)
            if save:
                self._save()

    def flush(self, step: int | None = None):
        with self._lock:
            self._save()

    def record_failed_rollout(self, parent: State):
        with self._lock:
            self._visit_counts[parent.id] = self._visit_counts.get(parent.id, 0) + 1

    def _save(self):
        os.makedirs(os.path.dirname(self._file_path), exist_ok=True)

        data = {
            "states": [s.to_dict() for s in self._archive if hasattr(s, "to_dict")],
            "visit_counts": self._visit_counts,
        }

        tmp = self._file_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)

        os.replace(tmp, self._file_path)

    def get_sample_stats(self) -> dict:
        with self._lock:
            values = [s.value for s in self._archive if s.value is not None]

            return {
                "sampler/archive_size": len(self._archive),
                "sampler/best_value": max(values) if values else 0.0,
                "sampler/mean_value": float(np.mean(values)) if values else 0.0,
                "sampler/total_visits": sum(self._visit_counts.values()),
            }
