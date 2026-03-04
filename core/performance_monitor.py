"""
core/performance_monitor.py — FPS tracking and adaptive inference throttle.
"""

import time
from collections import deque


class PerformanceMonitor:
    """
    Tracks rolling FPS and adjusts how often inference runs
    to keep the render loop smooth on lower-end hardware.
    """

    def __init__(self, config: dict):
        perf = config.get("performance", {})
        self._target      = perf.get("target_fps",      30)
        self._history_len = perf.get("fps_history_len", 30)
        self._max_skip    = perf.get("max_infer_skip",   4)

        self._times:     deque = deque(maxlen=self._history_len)
        self._last_tick: float = time.time()
        self.infer_every: int  = 1

    # ── Per-frame call ────────────────────────────────────────────────────────

    def tick(self) -> None:
        now = time.time()
        self._times.append(now - self._last_tick)
        self._last_tick = now
        self._adapt()

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        return 1.0 / (sum(self._times) / len(self._times))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _adapt(self) -> None:
        fps = self.fps
        if fps < 20 and self.infer_every < self._max_skip:
            self.infer_every += 1
        elif fps > 26 and self.infer_every > 1:
            self.infer_every -= 1