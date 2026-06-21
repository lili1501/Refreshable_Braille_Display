#pragma once
#include <Arduino.h>

void initBrailleKeyboard();

// Call this in loop(). Returns:
//   'a'-'z'  = detected Braille letter
//   ' '      = space
//   '\n'     = enter/send
//   '\b'     = backspace
//   'M'      = mode switch
//   0        = no input yet (chord still in progress or idle)
char readBrailleChord();

// Get the raw dot pattern of the last detected chord (bits 0-5 = dots 1-6)
uint8_t getLastChordPattern();
