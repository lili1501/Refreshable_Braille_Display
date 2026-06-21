// braille_test.ino — BrailleAI integrated firmware
// 6-dot chord keyboard + Grade 2 typing, optional STT/TTS/AI correction,
// servo Braille-cell output, and emotion prefixes streamed from the
// Raspberry Pi (emotion_inference.py) over UART.

#include "config.h"
#include "braille_input.h"
#include "braille_grade2.h"
#include "braille_correct.h"
#include "speaker_tts.h"
#include "mic_stt.h"
#include "braille_output.h"
#include "emotion_display.h"

// ---- Spoken output ----
// true  = on ENTER, speak the finished sentence through the MAX98357A
//         speaker via Deepgram TTS (needs Wi-Fi + DEEPGRAM_API_KEY).
#define ENABLE_TTS true

// ---- Speech input ----
// true  = press MODE to record from the INMP441 mic, transcribe via
//         Deepgram STT, print it, and speak it back (needs Wi-Fi).
#define ENABLE_STT true

// ---- AI correction ----
// true  = on ENTER, send the line to the Claude agent to fix typos and
//         complete hanging contractions (needs Wi-Fi + CLAUDE_API_KEY).
// false = pure offline Grade 2 echo, no network.
#define ENABLE_AI_CORRECTION true

// ---- Diagnostic mode ----
// true  = print raw LOW/HIGH state of all 10 GPIOs (verify wiring)
// false = normal chord decoding test
#define DIAG_MODE false

// ---- Servo Braille display ----
// true = render finished lines + emotion prefix on the 6-servo cell.
#define ENABLE_DISPLAY true

// ---- Emotion prefixes (from the Raspberry Pi over UART) ----
// true = listen for "EMOTION:<label>:<conf>" and flash a tactile prefix.
#define ENABLE_EMOTION true

// ---- Grade 2 sentence buffer ----
// committed = finished words already expanded to text.
// wordCells = raw braille cells of the word currently being typed;
//             they are re-translated as a group so contractions,
//             wordsigns, capital and number signs all work.
String committed = "";
#define MAX_WORD_CELLS 48
uint8_t wordCells[MAX_WORD_CELLS];
int     wordLen = 0;

// Full live text = committed words + the current word expanded so far.
String liveText() {
  return committed + grade2Word(wordCells, wordLen);
}

void pushCell(uint8_t pattern) {
  if (wordLen < MAX_WORD_CELLS) wordCells[wordLen++] = pattern;
}

void commitWord(const char* sep) {
  committed += grade2Word(wordCells, wordLen);
  committed += sep;
  wordLen = 0;
}

// All 10 keys with human-readable labels, in pin order
struct KeyInfo { int pin; const char* name; };
static const KeyInfo allKeys[NUM_ALL_KEYS] = {
  { KEY_DOT1, "DOT1" }, { KEY_DOT2, "DOT2" }, { KEY_DOT3, "DOT3" },
  { KEY_DOT4, "DOT4" }, { KEY_DOT5, "DOT5" }, { KEY_DOT6, "DOT6" },
  { KEY_SPACE, "SPACE" }, { KEY_ENTER, "ENTER" },
  { KEY_BKSP, "BKSP" }, { KEY_MODE, "MODE" }
};

// Prints only the keys currently pressed (LOW). Reprints only on change
// so the Serial Monitor isn't flooded.
void runDiagnostic() {
  static uint16_t lastMask = 0xFFFF;  // force first print
  uint16_t mask = 0;
  for (int i = 0; i < NUM_ALL_KEYS; i++) {
    if (digitalRead(allKeys[i].pin) == LOW) mask |= (1 << i);
  }

  if (mask != lastMask) {
    lastMask = mask;
    Serial.print("Pressed: ");
    if (mask == 0) {
      Serial.print("(none)");
    } else {
      bool first = true;
      for (int i = 0; i < NUM_ALL_KEYS; i++) {
        if (mask & (1 << i)) {
          if (!first) Serial.print(", ");
          Serial.printf("%s(GPIO%d)", allKeys[i].name, allKeys[i].pin);
          first = false;
        }
      }
    }
    Serial.println();
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);              // let USB CDC enumerate

  if (DIAG_MODE) {
    Serial.println("\n=== BrailleAI Keyboard DIAGNOSTIC ===");
    Serial.println("Press each key one at a time. It should name that key.");
    Serial.println("If the wrong name shows, your wiring/pin map is swapped.");
    Serial.println("If nothing shows, check GND connection on that button.\n");
  } else {
    Serial.println("\n=== BrailleAI Keyboard Test (Grade 2) ===");
    Serial.println("Chord the 6 dot keys, then lift all fingers to register a cell.");
    Serial.println("Words are expanded with Grade 2 contractions on SPACE/ENTER.");
    Serial.println("Solo wordsigns (b=but, t=that, y=you...), groupsigns (ch/sh/the/and/ing),");
    Serial.println("capital sign (dot 6) and number sign (dots 3-4-5-6) all supported.");
    Serial.println("SPACE / ENTER / BKSP / MODE keys also active.\n");
  }

  initBrailleKeyboard();

  if (!DIAG_MODE && ENABLE_AI_CORRECTION) {
    initCorrector();   // non-fatal: falls back to raw text if Wi-Fi fails
  }

  if (!DIAG_MODE && ENABLE_TTS) {
    initSpeaker();     // configure I2S Port 1 for the MAX98357A
  }

  if (!DIAG_MODE && ENABLE_STT) {
    initMic();         // configure I2S Port 0 for the INMP441
  }

  if (!DIAG_MODE && ENABLE_DISPLAY) {
    initBrailleDisplay();   // attach the 6 servos for the Braille cell
  }

  if (!DIAG_MODE && ENABLE_EMOTION) {
    initEmotionLink();      // open the UART from the Raspberry Pi
  }
}

void loop() {
  if (DIAG_MODE) {
    runDiagnostic();
    delay(5);
    return;
  }

  if (ENABLE_EMOTION) {
    pollEmotion();   // non-blocking: flashes a tactile prefix when the Pi sends one
  }

  char c = readBrailleChord();

  if (c != 0) {
    switch (c) {
      case '\n': {
        commitWord("");           // flush current word, no trailing space
        Serial.print("\n[ENTER] raw   = \"");
        Serial.print(committed);
        Serial.println("\"");
        String lineOut = committed;
        if (ENABLE_AI_CORRECTION) {
          lineOut = correctSentence(committed);
          Serial.print("[AI]    fixed = \"");
          Serial.print(lineOut);
          Serial.println("\"");
        }
        // lineOut is the finished sentence -> speak it + show it on the cell.
        if (ENABLE_TTS) {
          speak(lineOut);
        }
        if (ENABLE_DISPLAY && lineOut.length() > 0) {
          renderTextOnDisplay(lineOut);
        }
        committed = "";
        break;
      }
      case '\b':
        if (wordLen > 0) {
          wordLen--;              // drop last cell of the word being typed
        } else if (committed.length() > 0) {
          committed.remove(committed.length() - 1);
        }
        Serial.print("[BKSP] -> \"");
        Serial.print(liveText());
        Serial.println("\"");
        break;
      case 'M':
        Serial.println("[MODE] switch pressed");
        if (ENABLE_STT) {
          String heard = recordAndTranscribe();
          Serial.print("[STT]   heard = \"");
          Serial.print(heard);
          Serial.println("\"");
          if (heard.length() > 0 && ENABLE_TTS) {
            speak(heard);   // echo it back through the speaker
          }
          if (heard.length() > 0 && ENABLE_DISPLAY) {
            renderTextOnDisplay(heard);   // also show it on the Braille cell
          }
        }
        break;
      case ' ':
        commitWord(" ");          // expand finished word + add space
        Serial.print("[SPACE] -> \"");
        Serial.print(committed);
        Serial.println("\"");
        break;
      default: {
        // A dot chord. Capture the raw pattern and add it to the word;
        // the whole word is re-translated so Grade 2 contractions apply.
        uint8_t pat = getLastChordPattern();
        pushCell(pat);
        Serial.print("Cell added -> word so far = \"");
        Serial.print(grade2Word(wordCells, wordLen));
        Serial.print("\"   line = \"");
        Serial.print(liveText());
        Serial.println("\"");
        break;
      }
    }
  }

  delay(2);  // light loop pacing
}
