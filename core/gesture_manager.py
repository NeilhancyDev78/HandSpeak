"""
core/gesture_manager.py — Gesture registry built from gesture_config.json.

Single source of truth for label ordering, outputs, and types.
Label order is alphabetically sorted and must match training order exactly.
"""

import json
import os


class GestureManager:
    """Parses and exposes gesture metadata from config."""

    def __init__(self, config_path: str = "config/gesture_config.json"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Gesture config not found: {config_path}")

        with open(config_path, "r") as f:
            raw = json.load(f)

        self._gestures: dict = raw["gestures"]

        # Stable sorted order — must be consistent across train / infer
        self._names: list[str] = sorted(self._gestures.keys())

    # ── Label access ──────────────────────────────────────────────────────────

    @property
    def gesture_names(self) -> list[str]:
        """Sorted label list. Index position == model output neuron."""
        return self._names

    @property
    def num_classes(self) -> int:
        return len(self._names)

    def get_output(self, label: str) -> str:
        """What the gesture produces — a character, ' ', 'DELETE', 'SPEAK'."""
        return self._gestures.get(label, {}).get("output", "")

    def get_type(self, label: str) -> str:
        """'letter' or 'command'."""
        return self._gestures.get(label, {}).get("type", "letter")

    def is_command(self, label: str) -> bool:
        return self.get_type(label) == "command"

    def label_at(self, index: int) -> str:
        return self._names[index]

    def index_of(self, label: str) -> int:
        return self._names.index(label)