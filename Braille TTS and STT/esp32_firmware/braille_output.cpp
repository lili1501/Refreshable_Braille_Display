#include "braille_output.h"
#include "config.h"
#include <ESP32Servo.h>

// One servo per dot
static Servo dotServos[NUM_DOTS];
static const int servoPins[NUM_DOTS] = {
    SERVO_DOT1, SERVO_DOT2, SERVO_DOT3,
    SERVO_DOT4, SERVO_DOT5, SERVO_DOT6
};

// Per-dot calibration — adjust if individual servos need fine-tuning
static int downAngles[NUM_DOTS] = {
    DOT_DOWN_ANGLE, DOT_DOWN_ANGLE, DOT_DOWN_ANGLE,
    DOT_DOWN_ANGLE, DOT_DOWN_ANGLE, DOT_DOWN_ANGLE
};
static int upAngles[NUM_DOTS] = {
    DOT_UP_ANGLE, DOT_UP_ANGLE, DOT_UP_ANGLE,
    DOT_UP_ANGLE, DOT_UP_ANGLE, DOT_UP_ANGLE
};

// Track current state to avoid redundant servo writes
static uint8_t currentPattern = 0xFF;

// Grade 1 Braille: 6-bit pattern per letter
// Bit 0 = dot 1, bit 1 = dot 2, ..., bit 5 = dot 6
static const uint8_t brailleAlpha[26] = {
    0b000001,  // a: dot 1
    0b000011,  // b: dots 1,2
    0b001001,  // c: dots 1,4
    0b011001,  // d: dots 1,4,5
    0b010001,  // e: dots 1,5
    0b001011,  // f: dots 1,2,4
    0b011011,  // g: dots 1,2,4,5
    0b010011,  // h: dots 1,2,5
    0b001010,  // i: dots 2,4
    0b011010,  // j: dots 2,4,5
    0b000101,  // k: dots 1,3
    0b000111,  // l: dots 1,2,3
    0b001101,  // m: dots 1,3,4
    0b011101,  // n: dots 1,3,4,5
    0b010101,  // o: dots 1,3,5
    0b001111,  // p: dots 1,2,3,4
    0b011111,  // q: dots 1,2,3,4,5
    0b010111,  // r: dots 1,2,3,5
    0b001110,  // s: dots 2,3,4
    0b011110,  // t: dots 2,3,4,5
    0b100101,  // u: dots 1,3,6
    0b100111,  // v: dots 1,2,3,6
    0b111010,  // w: dots 2,4,5,6
    0b101101,  // x: dots 1,3,4,6
    0b111101,  // y: dots 1,3,4,5,6
    0b110101,  // z: dots 1,3,5,6
};

uint8_t charToBraillePattern(char c) {
    if (c >= 'a' && c <= 'z') return brailleAlpha[c - 'a'];
    if (c >= 'A' && c <= 'Z') return brailleAlpha[c - 'A'];
    if (c == ' ') return 0x00;
    if (c == '.') return 0b110100;
    if (c == ',') return 0b000010;
    if (c == '?') return 0b100110;
    if (c == '!') return 0b010110;
    return 0x3F;  // unknown = all dots up (visible error)
}

int getDownAngle() { return DOT_DOWN_ANGLE; }
int getUpAngle()   { return DOT_UP_ANGLE; }

void initBrailleDisplay() {
    // Allocate the 4 PWM timers used by ESP32Servo library
    ESP32PWM::allocateTimer(0);
    ESP32PWM::allocateTimer(1);
    ESP32PWM::allocateTimer(2);
    ESP32PWM::allocateTimer(3);

    for (int i = 0; i < NUM_DOTS; i++) {
        dotServos[i].setPeriodHertz(50);  // standard 50Hz for hobby servos
        dotServos[i].attach(servoPins[i], 500, 2500);  // 500-2500us pulse range
    }

    clearDisplay();
    Serial.println("[Display] 6 servos attached, all dots down");
}

void clearDisplay() {
    for (int i = 0; i < NUM_DOTS; i++) {
        dotServos[i].write(downAngles[i]);
    }
    currentPattern = 0;
    delay(SERVO_SETTLE_MS);
}

void displayPattern(uint8_t pattern) {
    if (pattern == currentPattern) return;  // no change, skip

    bool anyChanged = false;
    for (int i = 0; i < NUM_DOTS; i++) {
        bool wasUp = currentPattern & (1 << i);
        bool shouldBeUp = pattern & (1 << i);

        if (wasUp != shouldBeUp) {
            int angle = shouldBeUp ? upAngles[i] : downAngles[i];
            dotServos[i].write(angle);
            anyChanged = true;
        }
    }

    currentPattern = pattern;

    if (anyChanged) {
        delay(SERVO_SETTLE_MS);  // let servos reach their new positions
    }
}

void displayChar(char c) {
    uint8_t pattern = charToBraillePattern(c);
    Serial.printf("[Display] '%c' → 0b%c%c%c%c%c%c (dots: ",
        c,
        (pattern & 0x20) ? '1' : '0',
        (pattern & 0x10) ? '1' : '0',
        (pattern & 0x08) ? '1' : '0',
        (pattern & 0x04) ? '1' : '0',
        (pattern & 0x02) ? '1' : '0',
        (pattern & 0x01) ? '1' : '0'
    );
    bool first = true;
    for (int i = 0; i < 6; i++) {
        if (pattern & (1 << i)) {
            if (!first) Serial.print(",");
            Serial.print(i + 1);
            first = false;
        }
    }
    Serial.println(")");

    displayPattern(pattern);
}

void displayBrailleString(const char* text, int scrollIndex) {
    int len = strlen(text);
    char c = (scrollIndex < len) ? text[scrollIndex] : ' ';
    displayChar(c);
}

void displayCorrect() {
    displayPattern(0x3F);  // all dots up
    delay(300);
    displayPattern(0x00);  // all down
    delay(200);
    displayPattern(0x3F);  // all up again
    delay(300);
    clearDisplay();
}

void setDot(int dotNumber, bool up) {
    if (dotNumber < 1 || dotNumber > NUM_DOTS) return;
    int idx = dotNumber - 1;
    int angle = up ? upAngles[idx] : downAngles[idx];
    dotServos[idx].write(angle);

    // Update pattern tracking
    if (up) currentPattern |= (1 << idx);
    else    currentPattern &= ~(1 << idx);

    delay(SERVO_SETTLE_MS);
}