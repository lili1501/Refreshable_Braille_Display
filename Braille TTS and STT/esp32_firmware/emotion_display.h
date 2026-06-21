#pragma once
#include <Arduino.h>

// ============================================================
// Emotion link + tactile rendering (BrailleAI Mode 1)
// ------------------------------------------------------------
// The Raspberry Pi runs emotion_inference.py (AVEmotionNet) and
// streams one line per prediction over UART:
//
//     EMOTION:<label>:<confidence>\n      e.g. "EMOTION:happy:0.62\n"
//
// labels: angry | disgust | fearful | happy | neutral | sad
//
// This module opens that UART (pins/baud from config.h), caches the
// latest emotion, flashes a single-cell tactile PREFIX for it, and
// renders a finished sentence across the 6-servo Braille cell.
//
// It uses the display API in braille_output.h, so call
// initBrailleDisplay() (in setup) before rendering.
// ============================================================

// Open the UART from the Pi. Call once in setup().
void initEmotionLink();

// Non-blocking: read any pending serial bytes, parse complete
// "EMOTION:..." lines, and flash the prefix cell when one arrives.
// Call every loop() iteration.
void pollEmotion();

// Latest emotion label received from the Pi ("neutral" until first line).
String currentEmotion();
float  currentEmotionConfidence();

// Flash the tactile single-cell prefix for an emotion label
// (held for EMOTION_DISPLAY_MS), then clear.
void showEmotionPrefix(const String& label);

// Scroll a finished line across the single cell, one char at a
// time (SCROLL_DELAY_MS between chars), then blank the cell.
void renderTextOnDisplay(const String& text);
