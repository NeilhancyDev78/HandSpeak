"""
core/hand_tracker.py — MediaPipe HandLandmarker wrapper with threaded capture.

Runs landmark detection in a background thread so the main render loop
is never blocked by inference latency.
"""

import threading
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


class HandTracker:
    """
    Threaded MediaPipe hand landmark detector.

    Detection runs on a dedicated thread; the main thread reads the
    latest result via `last_result` without blocking.
    """

    def __init__(self, config: dict):
        inf   = config["inference"]
        self._res_w, self._res_h = inf["infer_resolution"]

        base_options = mp_python.BaseOptions(
            model_asset_path=inf["landmark_model"]
        )
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=inf["detection_confidence"],
            min_hand_presence_confidence=inf["presence_confidence"],
            min_tracking_confidence=inf["tracking_confidence"],
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

        self._lock       = threading.Lock()
        self._raw_frame  = None
        self._last_result: dict | None = None
        self._running    = False
        self._thread     = threading.Thread(target=self._loop, daemon=True)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)

    # ── Frame feed ────────────────────────────────────────────────────────────

    def feed(self, frame_bgr) -> None:
        """Push a new BGR frame to be processed on the tracker thread."""
        with self._lock:
            self._raw_frame = frame_bgr

    # ── Result access ─────────────────────────────────────────────────────────

    @property
    def last_result(self) -> dict | None:
        """Latest detection result — safe to read from any thread."""
        with self._lock:
            return self._last_result

    # ── Internal loop ─────────────────────────────────────────────────────────

    def _loop(self) -> None:
        import cv2
        while self._running:
            with self._lock:
                frame = self._raw_frame

            if frame is None:
                continue

            small = cv2.resize(frame, (self._res_w, self._res_h))
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            detection = self._landmarker.detect(mp_img)

            if detection.hand_landmarks:
                lms = detection.hand_landmarks[0]
                result = {
                    "landmarks": lms,
                    "features":  self._normalise(lms),
                    "found":     True,
                }
            else:
                result = {"landmarks": None, "features": None, "found": False}

            with self._lock:
                self._last_result = result

    # ── Feature extraction ────────────────────────────────────────────────────

    @staticmethod
    def _normalise(landmarks) -> np.ndarray:
        """
        21 NormalizedLandmark objects → (63,) float32.
        Subtracts wrist origin, scales to unit range — view-invariant.
        """
        pts = np.array(
            [[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32
        )
        pts -= pts[0]
        scale = np.max(np.abs(pts))
        if scale > 0:
            pts /= scale
        return pts.flatten()