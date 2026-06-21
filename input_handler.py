"""
input_handler.py  -  BrailleAI Module 2: Perkins Keyboard Chord Input
=====================================================================
PC-runnable reference port of the ESP32-S3 firmware's keyboard input task
(Section III of the project outline).

A Perkins-style brailler has 6 dot keys (plus space). The user presses one or
more dot keys *together* to form a single Braille cell ("chord"). The firmware
debounces the keys and, once an 80 ms chord window elapses with no further
presses, decodes the accumulated dot set into a character.

This class is a small, deterministic state machine implementing exactly that:

    ci = ChordInput(on_char=print)
    for d in (1, 2, 5): ci.key_down(d, t)   # dots for 'h'
    ci.tick(t + ci.window_s)                 # window elapsed -> emit 'h'

It reuses braille_core.decode_chord so the dot tables stay in one place. The
two Grade-1 prefix cells are handled here (stateful), because they affect the
*next* cell rather than producing a character of their own:

    CAPITAL_SIGN  (decode -> '^')  -> next letter is uppercased
    NUMBER_SIGN   (decode -> '#')  -> following a-j cells become digits 1-0,
                                      until a space or a non-digit letter.
"""
from __future__ import annotations

import time
from typing import Callable, Iterable, Optional, Set

from braille_core import decode_chord, _DIGIT_LETTER

# Reverse of braille_core._DIGIT_LETTER: braille letter -> digit char.
# In number mode the cell for 'a'..'j' is read as '1'..'0'.
_LETTER_TO_DIGIT = {letter: digit for digit, letter in _DIGIT_LETTER.items()}

# Default chord window (seconds) - matches the 80 ms firmware debounce window.
DEFAULT_WINDOW_S = 0.08

# Tolerance so a tick exactly at the window boundary isn't lost to float error
# (e.g. (0.1 + 0.08) - 0.1 == 0.07999999999999999 < 0.08).
_EPS = 1e-6


class ChordInput:
    """Accumulates simultaneous dot-key presses into decoded text."""

    def __init__(self,
                 on_char: Optional[Callable[[str], None]] = None,
                 window_s: float = DEFAULT_WINDOW_S) -> None:
        self.on_char = on_char
        self.window_s = window_s
        self.text: str = ""

        # chord-in-progress state
        self._active: bool = False
        self._dots: Set[int] = set()
        self._chord_start: float = 0.0

        # Grade-1 prefix state (affects the next cell)
        self._caps_pending: bool = False
        self._number_mode: bool = False

    # -- low-level key events -------------------------------------------
    def key_down(self, dot: int, t: Optional[float] = None) -> None:
        """Register that dot key `dot` (1..6) went down at time `t`."""
        if dot not in (1, 2, 3, 4, 5, 6):
            return
        now = time.time() if t is None else t
        # If an earlier chord is still open and its window has elapsed, close it
        # first so distinct chords at distinct times never merge.
        if self._active and (now - self._chord_start) >= (self.window_s - _EPS):
            self._finalize()
        if not self._active:
            self._active = True
            self._dots = set()
            self._chord_start = now
        self._dots.add(dot)

    def tick(self, t: Optional[float] = None) -> None:
        """Advance the clock; finalize the chord once the window has elapsed."""
        now = time.time() if t is None else t
        if self._active and (now - self._chord_start) >= (self.window_s - _EPS):
            self._finalize()

    # -- explicit controls ----------------------------------------------
    def space(self) -> None:
        """Insert a space and reset the number-mode latch (like a real space key)."""
        # flush any chord still being held before the space
        if self._active:
            self._finalize()
        self._number_mode = False
        self._caps_pending = False
        self._emit(" ")

    def reset(self) -> None:
        """Clear all buffered text and latched state."""
        self.text = ""
        self._active = False
        self._dots = set()
        self._caps_pending = False
        self._number_mode = False

    def feed_chord(self, dots: Iterable[int]) -> Optional[str]:
        """Convenience: decode a complete chord immediately (no timing)."""
        self._active = True
        self._dots = set(dots)
        return self._finalize()

    # -- internals ------------------------------------------------------
    def _finalize(self) -> Optional[str]:
        dots = frozenset(self._dots)
        self._active = False
        self._dots = set()
        if not dots:
            return None
        sym = decode_chord(dots)
        return self._handle(sym)

    def _handle(self, sym: Optional[str]) -> Optional[str]:
        if sym is None:
            return None                      # unknown chord -> ignored
        if sym == "^":                       # capital sign
            self._caps_pending = True
            return None
        if sym == "#":                       # number sign
            self._number_mode = True
            return None
        if sym == " ":
            self.space()
            return " "

        ch = sym
        if self._number_mode:
            if ch in _LETTER_TO_DIGIT:       # a-j -> 1-0
                ch = _LETTER_TO_DIGIT[ch]
            else:
                self._number_mode = False    # any other letter ends number run
        if self._caps_pending:
            ch = ch.upper()
            self._caps_pending = False

        self._emit(ch)
        return ch

    def _emit(self, ch: str) -> None:
        self.text += ch
        if self.on_char is not None:
            self.on_char(ch)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from braille_core import char_to_dots, NUMBER_SIGN, CAPITAL_SIGN

    emitted = []
    ci = ChordInput(on_char=emitted.append)

    def chord(dots, t):
        for d in dots:
            ci.key_down(d, t)
        ci.tick(t + ci.window_s)

    t = 0.0
    chord(CAPITAL_SIGN, t); t += 0.1
    chord(char_to_dots("h"), t); t += 0.1
    chord(char_to_dots("i"), t); t += 0.1
    ci.space()
    chord(NUMBER_SIGN, t); t += 0.1
    chord(char_to_dots("4"), t); t += 0.1
    chord(char_to_dots("2"), t); t += 0.1

    print("text   :", repr(ci.text))
    print("emitted:", "".join(emitted))
    assert ci.text == "Hi 42", ci.text
    print("input_handler self-test OK")
