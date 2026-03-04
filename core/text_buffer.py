"""
core/text_buffer.py — Smart text assembly with word-aware delete.
"""


class TextBuffer:
    """
    Manages the growing transcript string.

    Supports character append, word-aware delete, space, clear,
    and returns a TTS trigger signal on SPEAK.
    """

    MAX_LENGTH = 512

    def __init__(self):
        self._text = ""

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def text(self) -> str:
        return self._text

    @property
    def display_text(self) -> str:
        """Text with a blinking-cursor pipe appended."""
        return self._text + "|"

    def apply(self, gesture_output: str) -> bool:
        """
        Process a committed gesture output string.

        Returns True if TTS should be triggered, False otherwise.
        """
        match gesture_output:
            case "SPEAK":
                return bool(self._text.strip())
            case "DELETE":
                self._delete_last()
            case _:
                if len(self._text) < self.MAX_LENGTH:
                    self._text += gesture_output
        return False

    def clear(self) -> None:
        self._text = ""

    # ── Internal ──────────────────────────────────────────────────────────────

    def _delete_last(self) -> None:
        """Remove last character (simple — one keystroke per commit)."""
        self._text = self._text[:-1]