#pragma once
#include <Arduino.h>

// ============================================================
// AI sentence correction / completion (Claude agent)
// ------------------------------------------------------------
// Takes the raw Grade 2 output — which may have hanging or
// mis-typed contractions, e.g. "ye, i nee help" — and asks an
// LLM to fix typos and complete words from sentence context,
// returning natural English: "Yes, I need help."
//
// Requires WIFI_SSID / WIFI_PASS and CLAUDE_API_KEY to be set
// in config.h. If networking or the API call fails, the raw
// text is returned unchanged so the keyboard still works offline.
// ============================================================

// Connect to Wi-Fi. Safe to call repeatedly; returns true once
// connected. Blocks up to timeoutMs while attempting to join.
bool initCorrector(uint32_t timeoutMs = 8000);

// True if Wi-Fi is currently connected.
bool correctorOnline();

// Send `raw` to the agent and return the corrected sentence.
// On any failure, returns `raw` unchanged.
String correctSentence(const String& raw);
