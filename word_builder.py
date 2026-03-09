"""
word_builder.py
---------------
Accumulates predicted letters into words and sentences.

Logic
-----
• A letter is "confirmed" only after it is held stable for
  HOLD_FRAMES consecutive frames – prevents jitter from creating noise.
• A short pause (no hand detected for PAUSE_FRAMES frames) acts as a
  word/space delimiter.
• The caller can also trigger word completion explicitly.
"""

from collections import deque
from typing import Optional


# ── Tunable constants ─────────────────────────────────────────────────────────
HOLD_FRAMES  = 20   # frames a prediction must be stable before accepted
PAUSE_FRAMES = 30   # frames of silence → word boundary
MIN_CONFIDENCE = 0.70  # predictions below this threshold are ignored


class WordBuilder:
    """
    Stateful letter → word → sentence accumulator.

    Call update() once per frame with the current prediction.
    Call on_word_complete() callback to be notified when a word finishes.
    """

    def __init__(self, on_word_ready=None):
        """
        Parameters
        ----------
        on_word_ready : callable(word: str) | None
            Called whenever a completed word is available.
        """
        self._on_word_ready = on_word_ready

        # Current stable prediction tracking
        self._last_letter:  Optional[str] = None
        self._hold_counter: int = 0

        # Letter accumulation
        self._current_word: list[str] = []
        self._sentence: list[str] = []

        # Pause detection
        self._no_hand_counter: int = 0

        # Per-session history (for display)
        self._last_added_letter: Optional[str] = None

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, letter: Optional[str], confidence: float = 1.0) -> dict:
        """
        Process one frame's prediction.

        Parameters
        ----------
        letter     : predicted letter ("A"–"Z") or None (no hand)
        confidence : classifier confidence, 0–1

        Returns
        -------
        state dict with keys:
            current_word  – letters collected so far (str)
            sentence      – completed words joined by spaces (str)
            last_added    – the most recently appended letter (str | None)
            just_added    – True if a letter was added this frame (bool)
        """
        just_added = False

        if letter is None or confidence < MIN_CONFIDENCE:
            # No valid detection
            self._no_hand_counter += 1
            self._hold_counter = 0
            self._last_letter  = None

            if self._no_hand_counter >= PAUSE_FRAMES:
                self._flush_word()

        else:
            self._no_hand_counter = 0

            if letter == self._last_letter:
                self._hold_counter += 1
            else:
                self._last_letter  = letter
                self._hold_counter = 1

            if self._hold_counter == HOLD_FRAMES:
                # Letter confirmed – append if different from last
                if not self._current_word or self._current_word[-1] != letter:
                    self._current_word.append(letter)
                    self._last_added_letter = letter
                    just_added = True

        return {
            "current_word": "".join(self._current_word),
            "sentence":     " ".join(self._sentence),
            "last_added":   self._last_added_letter,
            "just_added":   just_added,
        }

    # ------------------------------------------------------------------
    # Manual controls
    # ------------------------------------------------------------------

    def complete_word(self):
        """Force the current buffer to be treated as a finished word."""
        self._flush_word()

    def backspace(self):
        """Remove the last appended letter."""
        if self._current_word:
            self._current_word.pop()

    def clear(self):
        """Reset everything."""
        self._current_word.clear()
        self._sentence.clear()
        self._last_added_letter = None
        self._hold_counter = 0
        self._no_hand_counter = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_word(self) -> str:
        return "".join(self._current_word)

    @property
    def sentence(self) -> str:
        return " ".join(self._sentence)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _flush_word(self):
        """Move current_word → sentence and notify callback."""
        if not self._current_word:
            return
        word = "".join(self._current_word)
        self._sentence.append(word)
        self._current_word.clear()
        self._no_hand_counter = 0
        print(f"[WordBuilder] Word complete: '{word}'")
        if self._on_word_ready:
            self._on_word_ready(word)
