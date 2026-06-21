#pragma once
#include <Arduino.h>

void initBrailleDisplay();
void clearDisplay();

// Display a single character on the cell
void displayChar(char c);

// Display a string scrolling through one character at a time
void displayBrailleString(const char* text, int scrollIndex);

// Display a raw 6-bit dot pattern directly
// pattern bits: bit 0 = dot 1, bit 1 = dot 2, ..., bit 5 = dot 6
void displayPattern(uint8_t pattern);

// Show a "correct" animation (flash all dots up and down)
void displayCorrect();

// Set just one dot up or down (for testing/calibration)
void setDot(int dotNumber, bool up);

// Convert ASCII to 6-bit Braille pattern
uint8_t charToBraillePattern(char c);

// Get servo calibration angles
int getDownAngle();
int getUpAngle();