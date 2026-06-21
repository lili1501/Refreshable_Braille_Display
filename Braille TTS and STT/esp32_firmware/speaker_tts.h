#pragma once
#include <Arduino.h>

// ============================================================
// Text-to-Speech playback (Deepgram Aura -> MAX98357A amp)
// ------------------------------------------------------------
// Sends a phrase (e.g. the corrected sentence from the Claude
// agent) to Deepgram /v1/speak, requests raw LINEAR16 PCM
// (container=none) at TTS_SAMPLE_RATE, and streams the audio
// straight to the MAX98357A class-D amp over I2S Port 1 — no
// base64, JSON, or MP3 decoding needed.
//
// Requires WIFI_SSID / WIFI_PASS and a valid DEEPGRAM_API_KEY
// in config.h. Wi-Fi is shared with the Claude corrector; call
// initCorrector() (or speak(), which connects on demand) first.
// ============================================================

// Configure I2S Port 1 for the speaker. Call once in setup().
void initSpeaker();

// Synthesize `text` via Deepgram and play it on the speaker.
// Blocks until playback finishes. Returns true if audio played,
// false on empty text, no Wi-Fi, or an API/parse error.
bool speak(const String& text);
