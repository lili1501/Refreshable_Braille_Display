#include "emotion_display.h"
#include "config.h"
#include "braille_output.h"

// Dedicated hardware UART for the Pi link (separate from USB Serial).
static HardwareSerial EmotionSerial(EMOTION_UART_NUM);

static String  g_line = "";          // incoming line buffer
static String  g_emotion = "neutral";
static float   g_confidence = 0.0f;

// ---- emotion label -> tactile single-cell prefix ----------------------
// The six labels begin with distinct letters (a/d/f/h/n/s), so we use the
// first letter's Braille cell as a learnable prefix. Override here if you
// prefer dedicated patterns.
static char prefixCharFor(const String& label) {
    if (label.length() == 0) return 'n';
    char c = (char)tolower(label[0]);
    if (c >= 'a' && c <= 'z') return c;
    return 'n';
}

void initEmotionLink() {
    EmotionSerial.begin(EMOTION_BAUD, SERIAL_8N1, EMOTION_RX_PIN, EMOTION_TX_PIN);
    Serial.printf("[Emotion] UART%d listening on RX=GPIO%d @ %d baud\n",
                  EMOTION_UART_NUM, EMOTION_RX_PIN, EMOTION_BAUD);
}

String currentEmotion()            { return g_emotion; }
float  currentEmotionConfidence()  { return g_confidence; }

void showEmotionPrefix(const String& label) {
    char p = prefixCharFor(label);
    Serial.printf("[Emotion] prefix '%c' for \"%s\"\n", p, label.c_str());
    displayChar(p);
    delay(EMOTION_DISPLAY_MS);
    clearDisplay();
}

void renderTextOnDisplay(const String& text) {
    for (size_t i = 0; i < text.length(); i++) {
        displayChar(text[i]);
        delay(SCROLL_DELAY_MS);
    }
    clearDisplay();
}

// Parse one complete line: "EMOTION:<label>:<confidence>"
static void parseLine(const String& line) {
    if (!line.startsWith("EMOTION:")) return;
    int a = line.indexOf(':');
    int b = line.indexOf(':', a + 1);
    if (a < 0 || b < 0) return;
    String label = line.substring(a + 1, b);
    String conf  = line.substring(b + 1);
    label.trim();
    g_emotion = label;
    g_confidence = conf.toFloat();
    showEmotionPrefix(g_emotion);   // flash the tactile prefix on arrival
}

void pollEmotion() {
    while (EmotionSerial.available()) {
        char ch = (char)EmotionSerial.read();
        if (ch == '\n' || ch == '\r') {
            if (g_line.length() > 0) {
                parseLine(g_line);
                g_line = "";
            }
        } else {
            g_line += ch;
            if (g_line.length() > 64) g_line = "";  // guard against garbage
        }
    }
}
