"""
core/gesture_engine.py — Temporal voting engine for stable gesture commits.

Sits between raw per-frame predictions and the text buffer.
Implements:
  • Majority-vote window        — smooths jitter over N frames
  • Minimum confidence gate     — rejects low-certainty predictions
  • Hold requirement            — gesture must be stable for K frames
  • Break detection             — hand must leave / change before repeat
  • Commit cooldown             — enforces minimum gap between commits
"""

import time
import numpy as np
from collections import deque


class GestureEngine:
    """
    Stateful temporal filter that converts a stream of per-frame
    (label, confidence) pairs into discrete, debounced commit events.
    """

    def __init__(self, gesture_manager, config: dict):
        self._gm   = gesture_manager
        rec        = config["recognition"]

        self._vote_window    = deque(maxlen=rec["vote_window_frames"])
        self._vote_threshold = rec["vote_threshold_pct"]
        self._min_confidence = rec["min_confidence"]
        self._hold_required  = rec["hold_frames_required"]
        self._break_timeout  = rec["break_timeout_ms"]  / 1000.0
        self._cooldown       = rec["commit_cooldown_ms"] / 1000.0
        self._repeat_break   = rec["repeat_break_ms"]   / 1000.0

        self._hold_count        = 0
        self._hold_candidate    = None
        self._last_commit_time  = 0.0
        self._last_committed    = None
        self._last_hand_time    = 0.0   # last time a hand was present

        # Keras model — loaded separately via load_model()
        self._model = None

    # ── Model management ─────────────────────────────────────────────────────

    def load_model(self, path: str) -> None:
        import tensorflow as tf
        self._model = tf.keras.models.load_model(path)

    def reload_model(self, path: str) -> None:
        self.load_model(path)

    # ── Per-frame update ──────────────────────────────────────────────────────

    def update(self, tracker_result: dict | None) -> dict | None:
        """
        Feed one tracker result and return a commit dict if a gesture
        should be registered this frame, else None.

        Commit dict: { "label": str, "confidence": float }
        """
        if tracker_result is None or not tracker_result.get("found"):
            self._on_no_hand()
            return None

        self._last_hand_time = time.time()

        features = tracker_result.get("features")
        if features is None or self._model is None:
            return None

        label, confidence = self._classify(features)

        if confidence < self._min_confidence:
            self._reset_hold()
            return None

        self._vote_window.append(label)
        dominant, vote_pct = self._dominant_vote()

        if dominant is None or vote_pct < self._vote_threshold:
            self._reset_hold()
            return None

        # Hold accumulation
        if dominant == self._hold_candidate:
            self._hold_count += 1
        else:
            self._hold_candidate = dominant
            self._hold_count     = 1

        if self._hold_count < self._hold_required:
            return None

        return self._try_commit(dominant, confidence)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _classify(self, features: np.ndarray) -> tuple[str, float]:
        pred  = self._model.predict(features[np.newaxis, :], verbose=0)[0]
        idx   = int(np.argmax(pred))
        return self._gm.gesture_names[idx], float(pred[idx])

    def _dominant_vote(self) -> tuple[str | None, float]:
        if not self._vote_window:
            return None, 0.0
        counts: dict[str, int] = {}
        for v in self._vote_window:
            counts[v] = counts.get(v, 0) + 1
        top = max(counts, key=counts.get)
        return top, counts[top] / len(self._vote_window)

    def _try_commit(self, label: str, confidence: float) -> dict | None:
        now = time.time()

        cooldown_ok = (now - self._last_commit_time) >= self._cooldown

        if label == self._last_committed:
            # Repeated letter — require a break interval since last hand loss
            hand_was_absent = (now - self._last_hand_time) >= self._repeat_break
            if not hand_was_absent:
                return None

        if not cooldown_ok:
            return None

        self._last_commit_time = now
        self._last_committed   = label
        self._reset_hold()
        self._vote_window.clear()
        return {"label": label, "confidence": confidence}

    def _reset_hold(self) -> None:
        self._hold_count     = 0
        self._hold_candidate = None

    def _on_no_hand(self) -> None:
        self._reset_hold()
        self._vote_window.clear()

    # ── Read-only state for UI ────────────────────────────────────────────────

    @property
    def cooldown_progress(self) -> float:
        """0.0 → 1.0; 1.0 means cooldown fully elapsed."""
        elapsed = time.time() - self._last_commit_time
        return min(elapsed / self._cooldown, 1.0)

    @property
    def hold_progress(self) -> float:
        """0.0 → 1.0 fraction of hold frames accumulated."""
        if self._hold_required == 0:
            return 1.0
        return min(self._hold_count / self._hold_required, 1.0)