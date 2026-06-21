#pragma once
#include <Arduino.h>

// ============================================================
// Speech-to-Text (INMP441 mic -> Deepgram /v1/listen)
// ------------------------------------------------------------
// Records a few seconds of audio from the INMP441 I2S microphone
// (Port 0) into a PSRAM buffer, POSTs the raw LINEAR16 PCM to
// Deepgram's prerecorded transcription endpoint, and returns the
// recognized text.
//
// Requires WIFI_SSID / WIFI_PASS and a valid DEEPGRAM_API_KEY in
// config.h. Wi-Fi is shared with the Claude corrector; call
// initCorrector() (or recordAndTranscribe(), which connects on
// demand) first.
// ============================================================

// Configure I2S Port 0 for the INMP441 mic. Call once in setup().
void initMic();

// Record RECORD_SECONDS of audio, transcribe it via Deepgram, and
// return the transcript. Blocks for the whole record + upload time.
// Returns "" on capture/network/parse failure.
String recordAndTranscribe();
