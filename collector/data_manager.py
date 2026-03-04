"""
collector/data_manager.py — Dataset CRUD for gesture sample files.

Handles per-gesture .npy file creation, deletion, updating, and
exposes summary statistics used by the collector UI.
"""

import os
import numpy as np


class DataManager:
    """
    Manages the flat data/ directory of per-gesture NumPy arrays.

    Layout:  data/<LABEL>.npy   shape (N, 63)  float32
    """

    def __init__(self, data_dir: str = "data"):
        self._dir = data_dir
        os.makedirs(self._dir, exist_ok=True)

    # ── Read ──────────────────────────────────────────────────────────────────

    def sample_count(self, label: str) -> int:
        """Number of saved samples for a label, or 0 if none."""
        path = self._path(label)
        if not os.path.exists(path):
            return 0
        return len(np.load(path))

    def summary(self, labels: list[str]) -> dict[str, int]:
        """{ label: count } for every label in the provided list."""
        return {lbl: self.sample_count(lbl) for lbl in labels}

    def has_data(self, label: str) -> bool:
        return os.path.exists(self._path(label))

    def locked_labels(self, labels: list[str], target: int) -> list[str]:
        """Labels whose sample count meets or exceeds target."""
        return [lbl for lbl in labels if self.sample_count(lbl) >= target]

    def pending_labels(self, labels: list[str], target: int) -> list[str]:
        """Labels that still need samples."""
        return [lbl for lbl in labels if self.sample_count(lbl) < target]

    # ── Write ─────────────────────────────────────────────────────────────────

    def save(self, label: str, samples: list[np.ndarray]) -> None:
        """Overwrite the .npy file for label with the provided samples."""
        arr = np.array(samples, dtype=np.float32)
        np.save(self._path(label), arr)

    def delete(self, label: str) -> None:
        """Remove the .npy file for label if it exists."""
        path = self._path(label)
        if os.path.exists(path):
            os.remove(path)

    def delete_all(self, labels: list[str]) -> None:
        """Wipe all .npy files for every label in the list."""
        for lbl in labels:
            self.delete(lbl)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _path(self, label: str) -> str:
        return os.path.join(self._dir, f"{label}.npy")