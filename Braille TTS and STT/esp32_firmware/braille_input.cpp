#include "braille_input.h"
#include "config.h"

// Pin arrays
static const int dotPins[NUM_DOT_KEYS] = {
    KEY_DOT1, KEY_DOT2, KEY_DOT3, KEY_DOT4, KEY_DOT5, KEY_DOT6
};

// Grade 1 Braille lookup table
// Index = 6-bit dot pattern (bit 0 = dot 1, bit 5 = dot 6)
// Value = ASCII character, 0 = undefined pattern
static const char brailleToAscii[64] = {
    ' ',  // 0b000000 = blank/space
    'a',  // 0b000001 = dot 1
    'b',  // 0b000011 = dots 1,2  (index 3 but bit pattern is 0b000011 = 3)
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

// Better approach: explicit mapping from dot pattern to char
struct BrailleMapping {
    uint8_t dots;   // 6-bit pattern
    char    ascii;
};

static const BrailleMapping brailleMap[] = {
    { 0b000001, 'a' },   // dot 1
    { 0b000011, 'b' },   // dots 1,2
    { 0b001001, 'c' },   // dots 1,4
    { 0b011001, 'd' },   // dots 1,4,5
    { 0b010001, 'e' },   // dots 1,5
    { 0b001011, 'f' },   // dots 1,2,4
    { 0b011011, 'g' },   // dots 1,2,4,5
    { 0b010011, 'h' },   // dots 1,2,5
    { 0b001010, 'i' },   // dots 2,4
    { 0b011010, 'j' },   // dots 2,4,5
    { 0b000101, 'k' },   // dots 1,3
    { 0b000111, 'l' },   // dots 1,2,3
    { 0b001101, 'm' },   // dots 1,3,4
    { 0b011101, 'n' },   // dots 1,3,4,5
    { 0b010101, 'o' },   // dots 1,3,5
    { 0b001111, 'p' },   // dots 1,2,3,4
    { 0b011111, 'q' },   // dots 1,2,3,4,5
    { 0b010111, 'r' },   // dots 1,2,3,5
    { 0b001110, 's' },   // dots 2,3,4
    { 0b011110, 't' },   // dots 2,3,4,5
    { 0b100101, 'u' },   // dots 1,3,6
    { 0b100111, 'v' },   // dots 1,2,3,6
    { 0b111010, 'w' },   // dots 2,4,5,6
    { 0b101101, 'x' },   // dots 1,3,4,6
    { 0b111101, 'y' },   // dots 1,3,4,5,6
    { 0b110101, 'z' },   // dots 1,3,5,6
};
static const int BRAILLE_MAP_SIZE = sizeof(brailleMap) / sizeof(brailleMap[0]);

// Chord detection state
static bool chordActive = false;
static unsigned long chordStartTime = 0;
static uint8_t currentChord = 0;
static uint8_t lastChordPattern = 0;

// Debounce for special keys
static unsigned long lastSpecialKeyTime = 0;
#define SPECIAL_KEY_DEBOUNCE_MS 200

void initBrailleKeyboard() {
    for (int i = 0; i < NUM_DOT_KEYS; i++) {
        pinMode(dotPins[i], INPUT_PULLUP);
    }
    pinMode(KEY_SPACE, INPUT_PULLUP);
    pinMode(KEY_ENTER, INPUT_PULLUP);
    pinMode(KEY_BKSP,  INPUT_PULLUP);
    pinMode(KEY_MODE,  INPUT_PULLUP);

    Serial.println("[Keyboard] Initialized, 10 keys ready");
}

// Look up a 6-bit dot pattern in the Braille table
static char lookupBraille(uint8_t pattern) {
    for (int i = 0; i < BRAILLE_MAP_SIZE; i++) {
        if (brailleMap[i].dots == pattern) {
            return brailleMap[i].ascii;
        }
    }
    return '?';  // unknown pattern
}

uint8_t getLastChordPattern() {
    return lastChordPattern;
}

char readBrailleChord() {
    unsigned long now = millis();

    // Check special keys first (with debounce)
    if (now - lastSpecialKeyTime > SPECIAL_KEY_DEBOUNCE_MS) {
        if (digitalRead(KEY_SPACE) == LOW) {
            lastSpecialKeyTime = now;
            while (digitalRead(KEY_SPACE) == LOW) delay(5); // wait for release
            return ' ';
        }
        if (digitalRead(KEY_ENTER) == LOW) {
            lastSpecialKeyTime = now;
            while (digitalRead(KEY_ENTER) == LOW) delay(5);
            return '\n';
        }
        if (digitalRead(KEY_BKSP) == LOW) {
            lastSpecialKeyTime = now;
            while (digitalRead(KEY_BKSP) == LOW) delay(5);
            return '\b';
        }
        if (digitalRead(KEY_MODE) == LOW) {
            lastSpecialKeyTime = now;
            while (digitalRead(KEY_MODE) == LOW) delay(5);
            return 'M';
        }
    }

    // Read all 6 dot keys
    uint8_t chord = 0;
    bool anyPressed = false;
    for (int i = 0; i < NUM_DOT_KEYS; i++) {
        if (digitalRead(dotPins[i]) == LOW) {
            chord |= (1 << i);
            anyPressed = true;
        }
    }

    // State machine for chord detection
    if (anyPressed && !chordActive) {
        // New chord starting — a finger just landed
        chordActive = true;
        chordStartTime = now;
        currentChord = chord;
        return 0;
    }

    if (anyPressed && chordActive) {
        // Chord in progress — accumulate any newly pressed keys
        // (fingers don't all land at the same millisecond)
        currentChord |= chord;
        return 0;
    }

    if (!anyPressed && chordActive) {
        // All fingers lifted — chord is complete
        if (now - chordStartTime > 20) { // minimum 20ms to avoid noise
            chordActive = false;
            uint8_t finalChord = currentChord;
            currentChord = 0;
            lastChordPattern = finalChord;

            if (finalChord == 0) return 0; // no dots pressed (noise)

            char c = lookupBraille(finalChord);
            Serial.printf("[Keyboard] Chord: 0b%s%s%s%s%s%s → '%c'\n",
                (finalChord & 0x20) ? "1" : "0",
                (finalChord & 0x10) ? "1" : "0",
                (finalChord & 0x08) ? "1" : "0",
                (finalChord & 0x04) ? "1" : "0",
                (finalChord & 0x02) ? "1" : "0",
                (finalChord & 0x01) ? "1" : "0",
                c);
            return c;
        }
    }

    return 0; // no input yet
}
