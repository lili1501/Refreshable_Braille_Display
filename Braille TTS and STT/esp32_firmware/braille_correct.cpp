#include "braille_correct.h"
#include "config.h"
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>

// Instruction given to the agent. Plain ASCII so it JSON-escapes cleanly.
static const char* SYS_PROMPT =
    "You are a Grade 2 braille typing assistant. The user types quickly with "
    "braille contractions and frequently leaves words hanging, abbreviated, or "
    "with small typos (for example 'ye' for 'yes', 'nee' for 'need'). Using the "
    "context of the whole sentence, correct spelling and complete any unfinished "
    "words into natural, grammatical English. Fix capitalization and punctuation. "
    "Do not add new ideas or change the meaning. Reply with ONLY the corrected "
    "sentence and nothing else.";

static bool wifiReady = false;

bool correctorOnline() {
    return WiFi.status() == WL_CONNECTED;
}

bool initCorrector(uint32_t timeoutMs) {
    if (correctorOnline()) { wifiReady = true; return true; }

    Serial.printf("[AI] Connecting to Wi-Fi \"%s\" ...\n", WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);

    uint32_t start = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - start < timeoutMs) {
        delay(250);
        Serial.print(".");
    }
    Serial.println();

    wifiReady = correctorOnline();
    if (wifiReady) {
        Serial.printf("[AI] Wi-Fi connected: %s\n", WiFi.localIP().toString().c_str());
    } else {
        Serial.println("[AI] Wi-Fi connect FAILED — correction disabled, using raw text");
    }
    return wifiReady;
}

// Escape a string so it is safe inside a JSON double-quoted value.
static String jsonEscape(const String& in) {
    String out;
    out.reserve(in.length() + 8);
    for (size_t i = 0; i < in.length(); i++) {
        char c = in[i];
        switch (c) {
            case '\"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if ((uint8_t)c < 0x20) {
                    char b[7];
                    snprintf(b, sizeof(b), "\\u%04x", (uint8_t)c);
                    out += b;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// Pull the first "text":"..." string value out of Claude's JSON response,
// decoding the common backslash escapes. Avoids a full JSON dependency.
static String extractText(const String& body) {
    int k = body.indexOf("\"text\":");
    if (k < 0) return "";
    int q = body.indexOf('\"', k + 7);
    if (q < 0) return "";
    q++;  // first char of the value

    String out;
    for (int i = q; i < (int)body.length(); i++) {
        char c = body[i];
        if (c == '\\') {
            char n = (i + 1 < (int)body.length()) ? body[i + 1] : 0;
            switch (n) {
                case 'n':  out += '\n'; break;
                case 'r':  out += '\r'; break;
                case 't':  out += '\t'; break;
                case '\"': out += '\"'; break;
                case '\\': out += '\\'; break;
                case '/':  out += '/';  break;
                case 'u':
                    if (i + 5 < (int)body.length()) {
                        char hex[5] = { body[i + 2], body[i + 3], body[i + 4], body[i + 5], 0 };
                        long cp = strtol(hex, nullptr, 16);
                        if (cp < 0x80 && cp > 0) out += (char)cp;  // keep basic ASCII
                        i += 4;
                    }
                    break;
                default: out += n; break;
            }
            i++;  // skip the escaped character
        } else if (c == '\"') {
            break;  // closing quote ends the value
        } else {
            out += c;
        }
    }
    out.trim();
    return out;
}

String correctSentence(const String& raw) {
    if (raw.length() == 0) return raw;
    if (!wifiReady && !initCorrector()) return raw;  // offline -> passthrough

    WiFiClientSecure client;
    client.setInsecure();  // hackathon-friendly; pin Anthropic's cert for production

    HTTPClient https;
    if (!https.begin(client, CLAUDE_API_URL)) {
        Serial.println("[AI] begin() failed, using raw text");
        return raw;
    }
    https.addHeader("content-type", "application/json");
    https.addHeader("x-api-key", CLAUDE_API_KEY);
    https.addHeader("anthropic-version", "2023-06-01");
    https.setTimeout(15000);

    String reqBody = String("{\"model\":\"") + CLAUDE_MODEL +
                     "\",\"max_tokens\":256,\"system\":\"" + jsonEscape(SYS_PROMPT) +
                     "\",\"messages\":[{\"role\":\"user\",\"content\":\"" +
                     jsonEscape(raw) + "\"}]}";

    String result = raw;  // default fallback
    int code = https.POST(reqBody);
    if (code == 200) {
        String resp = https.getString();
        String corrected = extractText(resp);
        if (corrected.length() > 0) result = corrected;
        else Serial.println("[AI] Could not parse response, using raw text");
    } else {
        Serial.printf("[AI] HTTP %d — using raw text\n", code);
        if (code > 0) Serial.println(https.getString());
    }

    https.end();
    return result;
}
