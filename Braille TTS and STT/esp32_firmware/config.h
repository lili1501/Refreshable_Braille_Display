#pragma once

// ============================================================
// BrailleAI — Configuration
// ============================================================
// Secrets (Wi-Fi + API keys) are NOT stored here. They live in
// secrets.h, which is git-ignored. Copy secrets.example.h to
// secrets.h and fill in your own values before building.
#include "secrets.h"

// ----- Device Modes -----
enum DeviceMode {
    MODE_CONVERSATION = 0,
    MODE_TUTOR = 1,
    MODE_ASSISTANT = 2,
    MODE_COUNT = 3
};

// ----- Braille Keyboard Pins -----
// IMPORTANT: GPIO 33-37 are reserved by octal PSRAM (N16R8 variant).
// All pins below are safe on ESP32-S3 N16R8.
#define KEY_DOT1    1
#define KEY_DOT2    2
#define KEY_DOT3    38
#define KEY_DOT4    39
#define KEY_DOT5    40
#define KEY_DOT6    41
#define KEY_SPACE   42
#define KEY_ENTER   45
#define KEY_BKSP    47
#define KEY_MODE    48

#define NUM_DOT_KEYS  6
#define NUM_ALL_KEYS  10
#define CHORD_TIMEOUT_MS  80   // ms window for simultaneous key detection

// ----- Servo Pins (Braille Display) -----
#define SERVO_DOT1   9    // top-left
#define SERVO_DOT2   10   // middle-left
#define SERVO_DOT3   11   // bottom-left
#define SERVO_DOT4   12   // top-right
#define SERVO_DOT5   17   // middle-right
#define SERVO_DOT6   18   // bottom-right
#define NUM_SERVOS   6
#define NUM_CELLS    1
#define NUM_DOTS     6

#define DOT_DOWN_ANGLE   0
#define DOT_UP_ANGLE     90
#define SERVO_SETTLE_MS  200

// ----- I2S Microphone (INMP441) — Port 0 -----
#define MIC_SCK    4
#define MIC_WS     5
#define MIC_SD     6
#define MIC_SAMPLE_RATE  16000
#define MIC_BITS         16
#define MIC_CHANNEL      I2S_CHANNEL_FMT_ONLY_LEFT
#define RECORD_SECONDS   4
#define RECORD_BUFFER_SIZE (MIC_SAMPLE_RATE * RECORD_SECONDS * 2)

// ----- I2S Speaker (MAX98357A) — Port 1 -----
#define SPK_BCLK   7
#define SPK_LRC    8
#define SPK_DIN    16
#define SPK_SD_PIN 46

// ----- SD Card (SPI) -----
// GPIO 34-37 forbidden by octal PSRAM. Using SPI2 with safe pins.
#define SD_MOSI    13
#define SD_MISO    14
#define SD_SCK     21
#define SD_CS      15
#define LOG_FILE   "/braille_log.jsonl"

// ----- Status LEDs -----
#define LED_MODE   3     // strapping pin, OK as output
#define LED_WIFI   0  // strapping pin — only safe as output

// ----- Emotion link: UART from the Raspberry Pi -----
// The Pi runs emotion_inference.py and sends "EMOTION:<label>:<conf>\n".
// Pick a FREE GPIO for RX — most pins above are already used. On the
// ESP32-S3 (native-USB Serial Monitor), hardware UART0 RX (GPIO44) is
// usually free for this. Verify against your wiring before flashing.
#define EMOTION_RX_PIN  44     // ESP32 RX  <-  Pi TX (e.g. Pi GPIO14)
#define EMOTION_TX_PIN  -1     // Pi only transmits; ESP32 does not reply
#define EMOTION_BAUD    115200
#define EMOTION_UART_NUM 1     // HardwareSerial port index (Serial1)

// ----- Wi-Fi + API Keys -----
// Defined in secrets.h (git-ignored): WIFI_SSID, WIFI_PASS,
// WIT_AI_TOKEN, CLAUDE_API_KEY, GOOGLE_TTS_API_KEY, DEEPGRAM_API_KEY.

// ----- API Endpoints -----
#define WIT_AI_URL     "https://api.wit.ai/speech"
#define CLAUDE_API_URL "https://api.anthropic.com/v1/messages"
#define CLAUDE_MODEL   "claude-sonnet-4-6"

// ----- Deepgram (STT + TTS) -----
// DEEPGRAM_API_KEY is defined in secrets.h
#define DEEPGRAM_TTS_URL  "https://api.deepgram.com/v1/speak"
#define DEEPGRAM_STT_URL  "https://api.deepgram.com/v1/listen"
#define DG_TTS_MODEL      "aura-2-thalia-en"   // Deepgram Aura voice
#define DG_STT_MODEL      "nova-3"

// ----- Text-to-Speech (Deepgram Aura -> MAX98357A) -----
#define TTS_SAMPLE_RATE  16000        // LINEAR16 PCM streamed straight to I2S

// ----- Braille Display Timing -----
#define SCROLL_DELAY_MS     800   // ms between auto-scroll steps
#define EMOTION_DISPLAY_MS  1200  // ms to show emotion prefix before text

// ----- Tutor Settings -----
#define MASTERY_THRESHOLD   4     // score out of 5 to consider a letter "mastered"
#define PROMOTION_ACCURACY  0.8   // 80% accuracy to advance to next level
#define STREAK_BONUS        3     // consecutive correct to trigger encouragement
