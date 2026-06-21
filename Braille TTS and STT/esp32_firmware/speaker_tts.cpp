#include "speaker_tts.h"
#include "config.h"
#include "braille_correct.h"   // reuse initCorrector()/correctorOnline() Wi-Fi
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include "driver/i2s.h"

#define TTS_I2S_PORT  I2S_NUM_1

static bool speakerReady = false;

// ---- Escape a string so it is safe inside a JSON double-quoted value. ----
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

// ------------------------------------------------------------
// Streaming sink: HTTPClient::writeToStream() feeds the de-chunked
// HTTP body here. Deepgram /v1/speak with container=none returns
// RAW little-endian LINEAR16 PCM (no JSON, no base64), so we just
// reassemble 16-bit samples and push them straight to I2S,
// duplicated to both channels so the mono amp plays at full volume.
// Constant, tiny RAM use regardless of phrase length.
// ------------------------------------------------------------
class I2SAudioSink : public Stream {
public:
    // Stream requires these; they are never used for a write-only sink.
    int available() override { return 0; }
    int read() override { return -1; }
    int peek() override { return -1; }

    size_t write(uint8_t b) override { feed(b); return 1; }
    size_t write(const uint8_t* buf, size_t size) override {
        for (size_t i = 0; i < size; i++) feed(buf[i]);
        return size;
    }

    void finish() { flushFrames(); }
    bool gotAudio() const { return _sampleCount > 0; }

private:
    bool    _haveLow = false;
    uint8_t _lowByte = 0;

    int16_t  _frames[256 * 2];   // 256 stereo frames per I2S write
    int      _frameCount = 0;
    uint32_t _sampleCount = 0;

    // LINEAR16 is little-endian: low byte first, then high byte.
    void feed(uint8_t b) {
        if (!_haveLow) { _lowByte = b; _haveLow = true; }
        else {
            int16_t s = (int16_t)((uint16_t)_lowByte | ((uint16_t)b << 8));
            _haveLow = false;
            pushSample(s);
        }
    }

    void pushSample(int16_t s) {
        _frames[_frameCount * 2]     = s;   // left
        _frames[_frameCount * 2 + 1] = s;   // right
        _frameCount++;
        _sampleCount++;
        if (_frameCount >= 256) flushFrames();
    }

    void flushFrames() {
        if (_frameCount == 0) return;
        size_t written;
        i2s_write(TTS_I2S_PORT, _frames, _frameCount * 2 * sizeof(int16_t),
                  &written, portMAX_DELAY);   // blocks -> real-time pacing
        _frameCount = 0;
    }
};

void initSpeaker() {
    pinMode(SPK_SD_PIN, OUTPUT);
    digitalWrite(SPK_SD_PIN, LOW);   // amp in shutdown until we play (no hiss)

    i2s_config_t cfg = {
        .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
        .sample_rate          = TTS_SAMPLE_RATE,
        .bits_per_sample      = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format       = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count        = 8,
        .dma_buf_len          = 256,
        .use_apll             = false,
        .tx_desc_auto_clear   = true,
        .fixed_mclk           = 0
    };
    if (i2s_driver_install(TTS_I2S_PORT, &cfg, 0, NULL) != ESP_OK) {
        Serial.println("[TTS] i2s_driver_install failed");
        return;
    }

    i2s_pin_config_t pins = {
        .mck_io_num   = I2S_PIN_NO_CHANGE,
        .bck_io_num   = SPK_BCLK,
        .ws_io_num    = SPK_LRC,
        .data_out_num = SPK_DIN,
        .data_in_num  = I2S_PIN_NO_CHANGE
    };
    i2s_set_pin(TTS_I2S_PORT, &pins);
    i2s_zero_dma_buffer(TTS_I2S_PORT);

    speakerReady = true;
    Serial.println("[TTS] Speaker I2S ready (MAX98357A on Port 1)");
}

bool speak(const String& text) {
    if (text.length() == 0) return false;
    if (!speakerReady) { Serial.println("[TTS] initSpeaker() not called"); return false; }

    // Share Wi-Fi with the Claude corrector.
    if (!correctorOnline() && !initCorrector()) {
        Serial.println("[TTS] No Wi-Fi — cannot reach Deepgram");
        return false;
    }

    // container=none + linear16 -> the response body IS raw PCM, ready for I2S.
    String url = String(DEEPGRAM_TTS_URL) +
                 "?model=" DG_TTS_MODEL
                 "&encoding=linear16&container=none&sample_rate=" + String(TTS_SAMPLE_RATE);
    String body = String("{\"text\":\"") + jsonEscape(text) + "\"}";

    Serial.printf("[TTS] Speaking: \"%s\"  (free heap %u, largest block %u)\n",
                  text.c_str(), ESP.getFreeHeap(), ESP.getMaxAllocHeap());

    // A TLS connect needs a large contiguous block; after the Claude
    // request the heap can be fragmented, which shows up as HTTP -1
    // (connection refused). Retry a couple of times with a short pause.
    bool ok = false;
    for (int attempt = 1; attempt <= 3 && !ok; attempt++) {
        WiFiClientSecure client;
        client.setInsecure();   // hackathon-friendly; pin Deepgram's cert for production

        HTTPClient https;
        if (!https.begin(client, url)) {
            Serial.println("[TTS] begin() failed");
            delay(150);
            continue;
        }
        https.addHeader("Authorization", "Token " DEEPGRAM_API_KEY);
        https.addHeader("Content-Type", "application/json");
        https.setConnectTimeout(15000);
        https.setTimeout(15000);

        int code = https.POST(body);
        if (code == 200) {
            digitalWrite(SPK_SD_PIN, HIGH);   // enable amp just before playback
            delay(5);

            I2SAudioSink sink;
            https.writeToStream(&sink);       // de-chunks body -> reassembles -> I2S
            sink.finish();
            ok = sink.gotAudio();

            // Let the DMA buffers drain, then mute the amp to avoid a tail pop.
            delay(60);
            i2s_zero_dma_buffer(TTS_I2S_PORT);
            digitalWrite(SPK_SD_PIN, LOW);

            if (!ok) Serial.println("[TTS] HTTP 200 but no audio returned");
        } else {
            Serial.printf("[TTS] attempt %d/3 failed: HTTP %d (%s), largest block %u\n",
                          attempt, code, https.errorToString(code).c_str(),
                          ESP.getMaxAllocHeap());
            if (code > 0) Serial.println(https.getString());
        }

        https.end();
        if (!ok) delay(200);
    }

    return ok;
}
