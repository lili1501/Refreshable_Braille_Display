"""
braille_core.py  -  BrailleAI Module 1: The Braille Engine
============================================================
Pure-Python reference implementation of the on-device translation logic
described in Section III (Firmware & Software Stack) of the project outline.

This mirrors what the ESP32-S3 C++ firmware does, so we can unit-test the
logic on a PC before porting:

    ASCII  <->  6-dot Braille pattern  <->  (left, right) half-cell states (0-7)
                                       ->   servo PWM angles
    Perkins keyboard chord (set of dots) -> ASCII
    emotion tone -> tactile Braille prefix symbol

Dot numbering for a 6-dot cell (standard Braille):
        1  4
        2  5
        3  6
A "pattern" here is a frozenset of the raised dot numbers, e.g. {1,2} == 'b'.
"""

from __future__ import annotations
from typing import Dict, FrozenSet, Iterable, Tuple, Optional

# ---------------------------------------------------------------------------
# 1. Grade-1 Braille lookup table  (character -> set of raised dots)
# ---------------------------------------------------------------------------
# Letters a-z follow the classic Braille pattern.
_LETTER_DOTS: Dict[str, Tuple[int, ...]] = {
    "a": (1,),        "b": (1, 2),     "c": (1, 4),      "d": (1, 4, 5),
    "e": (1, 5),      "f": (1, 2, 4),  "g": (1, 2, 4, 5), "h": (1, 2, 5),
    "i": (2, 4),      "j": (2, 4, 5),  "k": (1, 3),      "l": (1, 2, 3),
    "m": (1, 3, 4),   "n": (1, 3, 4, 5), "o": (1, 3, 5), "p": (1, 2, 3, 4),
    "q": (1, 2, 3, 4, 5), "r": (1, 2, 3, 5), "s": (2, 3, 4), "t": (2, 3, 4, 5),
    "u": (1, 3, 6),   "v": (1, 2, 3, 6), "w": (2, 4, 5, 6), "x": (1, 3, 4, 6),
    "y": (1, 3, 4, 5, 6), "z": (1, 3, 5, 6),
}

# Punctuation / common symbols (Grade 1 subset)
_PUNCT_DOTS: Dict[str, Tuple[int, ...]] = {
    " ": (),          ",": (2,),       ";": (2, 3),      ":": (2, 5),
    ".": (2, 5, 6),   "!": (2, 3, 5),  "?": (2, 3, 6),   "'": (3,),
    "-": (3, 6),      "(": (1, 2, 6),  ")": (3, 4, 5),
}

# Digits use the number sign (dots 3,4,5,6) followed by letters a-j.
# For single-cell representation we expose a-j mapping for 1-9,0.
_DIGIT_LETTER = {"1": "a", "2": "b", "3": "c", "4": "d", "5": "e",
                 "6": "f", "7": "g", "8": "h", "9": "i", "0": "j"}

NUMBER_SIGN = frozenset((3, 4, 5, 6))   # prefix that precedes digits
CAPITAL_SIGN = frozenset((6,))          # prefix that precedes a capital letter

# Build the master char -> frozenset(dots) table
ASCII_TO_DOTS: Dict[str, FrozenSet[int]] = {}
for _c, _d in {**_LETTER_DOTS, **_PUNCT_DOTS}.items():
    ASCII_TO_DOTS[_c] = frozenset(_d)

# Reverse lookup: frozenset(dots) -> char  (used by chord decoder)
DOTS_TO_ASCII: Dict[FrozenSet[int], str] = {v: k for k, v in ASCII_TO_DOTS.items()}


# ---------------------------------------------------------------------------
# 2. ASCII -> Braille dot pattern
# ---------------------------------------------------------------------------
def char_to_dots(ch: str) -> FrozenSet[int]:
    """Return the set of raised dots for a single character (lowercased)."""
    if ch in _DIGIT_LETTER:                       # digit -> its letter pattern
        return ASCII_TO_DOTS[_DIGIT_LETTER[ch]]
    return ASCII_TO_DOTS.get(ch.lower(), frozenset())


def text_to_cells(text: str) -> list[FrozenSet[int]]:
    """
    Convert a text string to a list of Braille cells (dot sets).
    Inserts the number sign before runs of digits and a capital sign
    before uppercase letters - matching Grade-1 transcription rules.
    """
    cells: list[FrozenSet[int]] = []
    in_number = False
    for ch in text:
        if ch.isdigit():
            if not in_number:
                cells.append(NUMBER_SIGN)
                in_number = True
            cells.append(char_to_dots(ch))
            continue
        in_number = False
        if ch.isupper():
            cells.append(CAPITAL_SIGN)
        cells.append(char_to_dots(ch))
    return cells


# ---------------------------------------------------------------------------
# 3. 6-dot pattern  ->  left / right half-cell states (0-7)
# ---------------------------------------------------------------------------
# Mechanics (Section II): each 6-dot cell is driven by TWO servos, one per
# column (left = dots 1,2,3 ; right = dots 4,5,6). Each servo slider has 8
# bump profiles -> a 3-bit state 0..7 encoding which of its 3 dots are raised.
#
# Bit layout for a half-cell state:
#   bit0 -> top dot   (1 or 4)
#   bit1 -> middle    (2 or 5)
#   bit2 -> bottom    (3 or 6)
LEFT_DOTS = (1, 2, 3)
RIGHT_DOTS = (4, 5, 6)


def dots_to_halfcell_states(dots: Iterable[int]) -> Tuple[int, int]:
    """Map a dot set to (left_state, right_state), each 0..7."""
    d = set(dots)
    left = (1 if 1 in d else 0) | (2 if 2 in d else 0) | (4 if 3 in d else 0)
    right = (1 if 4 in d else 0) | (2 if 5 in d else 0) | (4 if 6 in d else 0)
    return left, right


def halfcell_states_to_dots(left: int, right: int) -> FrozenSet[int]:
    """Inverse of dots_to_halfcell_states (useful for verification)."""
    out = set()
    if left & 1: out.add(1)
    if left & 2: out.add(2)
    if left & 4: out.add(3)
    if right & 1: out.add(4)
    if right & 2: out.add(5)
    if right & 4: out.add(6)
    return frozenset(out)


# ---------------------------------------------------------------------------
# 4. Half-cell state (0-7)  ->  servo PWM angle
# ---------------------------------------------------------------------------
# The MG90S servo sweeps across 8 discrete slider positions. We spread the 8
# states evenly across a usable angular range to keep detents distinct.
SERVO_MIN_ANGLE = 10     # degrees - mechanical safety margin
SERVO_MAX_ANGLE = 170
NUM_STATES = 8


def state_to_servo_angle(state: int) -> float:
    """Map a 0..7 slider state to a servo angle in degrees."""
    if not 0 <= state < NUM_STATES:
        raise ValueError(f"state must be 0..7, got {state}")
    step = (SERVO_MAX_ANGLE - SERVO_MIN_ANGLE) / (NUM_STATES - 1)
    return round(SERVO_MIN_ANGLE + state * step, 1)


def cell_to_servo_angles(dots: Iterable[int]) -> Tuple[float, float]:
    """Full pipeline: dot set -> (left_servo_angle, right_servo_angle)."""
    left, right = dots_to_halfcell_states(dots)
    return state_to_servo_angle(left), state_to_servo_angle(right)


# ---------------------------------------------------------------------------
# 5. Perkins chord decoding  (set of pressed dot keys -> ASCII)
# ---------------------------------------------------------------------------
# The keyboard reports which of the 6 dot keys were pressed within the 80ms
# chord window (Section III). We decode that dot set back to a character.
def decode_chord(pressed_dots: Iterable[int]) -> Optional[str]:
    """
    Decode a simultaneously-pressed set of dot keys into a character.
    Returns None if the chord matches no known Grade-1 cell.
    """
    key = frozenset(pressed_dots)
    if key == NUMBER_SIGN:
        return "#"        # signals "next cells are numbers" to the caller
    if key == CAPITAL_SIGN:
        return "^"        # signals "next letter is capital"
    return DOTS_TO_ASCII.get(key)


# ---------------------------------------------------------------------------
# 6. Emotion -> tactile Braille prefix symbol  (Mode 1)
# ---------------------------------------------------------------------------
# The LLM classifies tone; we prepend a distinct single-cell prefix so the
# DeafBlind user feels the emotion before reading the summarized text.
EMOTION_PREFIX_DOTS: Dict[str, FrozenSet[int]] = {
    "neutral":  frozenset(),            # blank cell
    "happy":    frozenset((2, 6)),
    "sad":      frozenset((3, 5)),
    "angry":    frozenset((1, 2, 3, 6)),
    "fearful":  frozenset((1, 4, 6)),
    "disgust":  frozenset((1, 4, 5, 6)),  # distinct pattern for disgust
    "question": frozenset((2, 3, 6)),   # same as '?'
    "urgent":   frozenset((1, 2, 3, 4, 5, 6)),  # full cell = strong alert
}


def emotion_to_prefix(emotion: str) -> FrozenSet[int]:
    """Return the tactile prefix cell for a detected emotion tone."""
    return EMOTION_PREFIX_DOTS.get(emotion.lower(), frozenset())


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = "Hi 7!"
    print(f"Text: {sample!r}")
    cells = text_to_cells(sample)
    for i, c in enumerate(cells):
        l, r = dots_to_halfcell_states(c)
        la, ra = state_to_servo_angle(l), state_to_servo_angle(r)
        print(f"  cell[{i}] dots={sorted(c)!s:<15} states=({l},{r}) "
              f"angles=({la}, {ra})")

    # round-trip chord decode for the alphabet
    bad = [ch for ch in "abcdefghijklmnopqrstuvwxyz"
           if decode_chord(char_to_dots(ch)) != ch]
    print("Chord round-trip OK" if not bad else f"Chord FAILED: {bad}")

    print("Emotion 'happy' prefix dots:", sorted(emotion_to_prefix("happy")))