"""
core/tts_engine.py — Non-blocking TTS via per-utterance daemon threads.

Spawns a fresh pyttsx3 engine per call to work around the Windows
SAPI5 / COM event-loop exhaustion bug that silently fails on reuse.
"""

import threading


class TTSEngine:
    """Thread-safe, non-blocking text-to-speech dispatcher."""

    def __init__(self, config: dict):
        tts = config.get("tts", {})
        self._rate   = tts.get("rate",   160)
        self._volume = tts.get("volume", 1.0)
        self._lock   = threading.Lock()

    # ── Public ────────────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Fire-and-forget. Returns immediately; speech runs on daemon thread."""
        if not text.strip():
            return
        t = threading.Thread(target=self._say, args=(text,), daemon=True)
        t.start()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _say(self, text: str) -> None:
        with self._lock:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate",   self._rate)
                engine.setProperty("volume", self._volume)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as exc:
                print(f"[TTS] Error: {exc}")