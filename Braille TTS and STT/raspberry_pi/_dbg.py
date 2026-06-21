"""_dbg.py - scratch trace used while debugging the chord state machine."""
from input_handler import ChordInput
from braille_core import char_to_dots, NUMBER_SIGN, CAPITAL_SIGN

emitted = []
ci = ChordInput(on_char=emitted.append)

def chord(dots, t):
    for d in dots:
        ci.key_down(d, t)
    ci.tick(t + ci.window_s)
    print("after", sorted(dots), "buf=", repr(ci.text))

t = 0.0
chord(CAPITAL_SIGN, t); t += 0.1
chord(char_to_dots("h"), t); t += 0.1
chord(char_to_dots("i"), t); t += 0.1
ci.space()
chord(NUMBER_SIGN, t); t += 0.1
chord(char_to_dots("4"), t); t += 0.1
chord(char_to_dots("2"), t); t += 0.1
print("FINAL", repr(ci.text))   # expect: Hi 42