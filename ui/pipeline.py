"""
ui/pipeline.py — Main video loop. Orchestrates camera, tracker,
engine, buffer, TTS, overlay, and mouse input.
"""

import cv2
import time


class VideoPipeline:
    """
    Owns the OpenCV window and drives the per-frame update cycle.

    Call run() to block until the user quits.
    """

    def __init__(self, hand_tracker, gesture_engine, gesture_manager,
                 text_buffer, tts_engine, performance_monitor,
                 overlay, config: dict):
        self._tracker  = hand_tracker
        self._engine   = gesture_engine
        self._gm       = gesture_manager
        self._buf      = text_buffer
        self._tts      = tts_engine
        self._perf     = performance_monitor
        self._overlay  = overlay

        cam = config["camera"]
        self._cap = cv2.VideoCapture(cam["index"])
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,    cam["buffer"])
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,   cam["width"])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  cam["height"])

        ui = config["ui"]
        self._title    = ui["window_title"]
        self._disp_w   = cam["width"]
        self._disp_h   = cam["height"]

        self._mouse_x  = 0
        self._mouse_y  = 0
        self._frame_n  = 0

        # Latest tracker result cached for the engine
        self._last_tracker_result = None

        # Last engine state dict forwarded to the overlay
        self._engine_state: dict = {
            "label":             None,
            "confidence":        0.0,
            "cooldown_progress": 1.0,
            "hold_progress":     0.0,
        }

    # ── Public entry ──────────────────────────────────────────────────────────

    def run(self) -> None:
        cv2.namedWindow(self._title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._title, self._disp_w, self._disp_h)
        cv2.setMouseCallback(self._title, self._on_mouse)

        self._tracker.start()

        try:
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                self._frame_n += 1

                # Feed tracker every N frames based on perf
                if self._frame_n % self._perf.infer_every == 0:
                    self._tracker.feed(frame)

                tracker_result = self._tracker.last_result
                commit         = self._engine.update(tracker_result)

                # Handle commit
                if commit:
                    output     = self._gm.get_output(commit["label"])
                    should_tts = self._buf.apply(output)
                    if should_tts and self._buf.text:
                        self._tts.speak(self._buf.text)

                # Build engine state snapshot for overlay
                self._engine_state = {
                    "label":             (tracker_result or {}).get("label")
                                          if tracker_result and tracker_result.get("found")
                                          else None,
                    "confidence":        commit["confidence"] if commit else
                                         self._engine_state.get("confidence", 0.0),
                    "cooldown_progress": self._engine.cooldown_progress,
                    "hold_progress":     self._engine.hold_progress,
                }

                # Update button hover states
                for btn in self._overlay.buttons:
                    btn.hovered = btn.hit(self._mouse_x, self._mouse_y)

                # Render
                canvas = cv2.resize(frame, (self._disp_w, self._disp_h))
                canvas = self._overlay.draw(
                    canvas,
                    tracker_result,
                    self._engine_state,
                    self._buf.display_text,
                    self._perf.fps,
                    committed=bool(commit),
                )

                cv2.imshow(self._title, canvas)
                self._perf.tick()

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                self._handle_key(key)

        finally:
            self._tracker.stop()
            self._cap.release()
            cv2.destroyAllWindows()

    # ── Input handlers ────────────────────────────────────────────────────────

    def _handle_key(self, key: int) -> None:
        if key == ord("d"):
            self._do_delete()
        elif key == ord("s"):
            self._do_speak()
        elif key == ord("c"):
            self._do_clear()

    def _on_mouse(self, event, x, y, flags, param) -> None:
        self._mouse_x = x
        self._mouse_y = y
        if event == cv2.EVENT_LBUTTONDOWN:
            if self._overlay.btn_delete.hit(x, y):
                self._do_delete()
            elif self._overlay.btn_speak.hit(x, y):
                self._do_speak()
            elif self._overlay.btn_clear.hit(x, y):
                self._do_clear()

    def _do_delete(self) -> None:
        self._buf.apply("DELETE")
        self._overlay.btn_delete.flash()

    def _do_speak(self) -> None:
        if self._buf.text:
            self._tts.speak(self._buf.text)
        self._overlay.btn_speak.flash()

    def _do_clear(self) -> None:
        self._buf.clear()
        self._overlay.btn_clear.flash()