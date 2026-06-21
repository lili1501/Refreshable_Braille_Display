#include "mic_stt.h"
#include "config.h"
#include "braille_correct.h"   // reuse initCorrector()/correctorOnline() Wi-Fi
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include "driver/i2s.h"

#define MIC_I2S_PORT  I2S_NUM_0

// INMP441 is a 24-bit mic that left-justifies its sample in a 32-bit
// I2S frame. We read 32-bit words and shift down to 16-bit PCM. A
// smaller shift = more gain; tune if speech is too quiet/clipping.
#define MIC_GAIN_SHIFT  14

static bool micReady = false;

void initMic() {
    i2s_config_t cfg = {
        .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate          = MIC_SAMPLE_RATE,
        .bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT,   // 24-bit data in 32-bit frame
        .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,   // INMP441 L/R -> GND = left
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count        = 8,
        .dma_buf_len          = 256,
        .use_apll             = false,
        .tx_desc_auto_clear   = false,
        .fixed_mclk           = 0
    };
    if (i2s_driver_install(MIC_I2S_PORT, &cfg, 0, NULL) != ESP_OK) {
        Serial.println("[STT] i2s_driver_install (mic) failed");
        return;
    }

    i2s_pin_config_t pins = {
        .mck_io_num   = I2S_PIN_NO_CHANGE,
        .bck_io_num   = MIC_SCK,
        .ws_io_num    = MIC_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num  = MIC_SD
    };
    i2s_set_pin(MIC_I2S_PORT, &pins);
    i2s_zero_dma_buffer(MIC_I2S_PORT);

    micReady = true;
    Serial.println("[STT] Mic I2S ready (INMP441 on Port 0)");
}

// Pull the first "transcript":"..." string out of Deepgram's JSON,
// decoding the common backslash escapes. Avoids a full JSON parser.
static String extractTranscript(const String& body) {
    int k = body.indexOf("\"transcript\":");
    if (k < 0) return "";
    int q = body.indexOf('\"', k + 13);
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
                default:   out += n;    break;
            }
            i++;  // skip escaped char
        } else if (c == '\"') {
            break;  // closing quote
        } else {
            out += c;
        }
    }
    out.trim();
    return out;
}

String recordAndTranscribe() {
    if (!micReady) { Serial.println("[STT] initMic() not called"); return ""; }

    if (!correctorOnline() && !initCorrector()) {
        Serial.println("[STT] No Wi-Fi — cannot reach Deepgram");
        return "";
    }

    const size_t numSamples = (size_t)MIC_SAMPLE_RATE * RECORD_SECONDS;
    const size_t pcmBytes   = numSamples * sizeof(int16_t);

    // 16-bit PCM buffer lives in PSRAM (N16R8 has 8MB; ~125KB needed for 4s).
    int16_t* pcm = (int16_t*)ps_malloc(pcmBytes);
    if (!pcm) pcm = (int16_t*)malloc(pcmBytes);   // fallback to internal RAM
    if (!pcm) { Serial.println("[STT] PCM buffer alloc failed"); return ""; }

    // Discard the first DMA reads while the mic settles.
    int32_t raw[256];
    size_t bytesRead = 0;
    for (int i = 0; i < 4; i++) {
        i2s_read(MIC_I2S_PORT, raw, sizeof(raw), &bytesRead, portMAX_DELAY);
    }

    Serial.printf("[STT] Recording %d s...\n", RECORD_SECONDS);
    size_t got = 0;
    while (got < numSamples) {
        i2s_read(MIC_I2S_PORT, raw, sizeof(raw), &bytesRead, portMAX_DELAY);
        int n = bytesRead / sizeof(int32_t);
        for (int i = 0; i < n && got < numSamples; i++) {
            pcm[got++] = (int16_t)(raw[i] >> MIC_GAIN_SHIFT);
        }
    }
    Serial.println("[STT] Captured, uploading to Deepgram...");

    WiFiClientSecure client;
    client.setInsecure();   // hackathon-friendly; pin Deepgram's cert for production

    HTTPClient https;
    String url = String(DEEPGRAM_STT_URL) +
                 "?model=" DG_STT_MODEL
                 "&encoding=linear16&channels=1&sample_rate=" + String(MIC_SAMPLE_RATE);

    String transcript = "";
    if (https.begin(client, url)) {
        https.addHeader("Authorization", "Token " DEEPGRAM_API_KEY);
        https.addHeader("Content-Type", "audio/raw");
        https.setTimeout(20000);

        int code = https.POST((uint8_t*)pcm, pcmBytes);
        if (code == 200) {
            String resp = https.getString();
            transcript = extractTranscript(resp);
            if (transcript.length() == 0)
                Serial.println("[STT] Empty transcript (no speech detected?)");
        } else {
            Serial.printf("[STT] HTTP %d\n", code);
            if (code > 0) Serial.println(https.getString());
        }
        https.end();
    } else {
        Serial.println("[STT] begin() failed");
    }

    free(pcm);
    return transcript;
}
