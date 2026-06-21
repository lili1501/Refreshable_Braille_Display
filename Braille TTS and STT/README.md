# BrailleAI — Bidirectional Braille AI Communicator

A two-way communication aid for **DeafBlind** users:

- A hearing person **speaks** → the device shows a tactile **emotion prefix** plus
  a short **Braille summary** on a refreshable servo cell.
- The user **types** on a 6-dot **Perkins keyboard** → the text is cleaned by an
  AI agent and **spoken aloud** through a speaker.

The system spans **two devices** (one folder each), validated by a PC simulation.

```
                emotion (UART)                 Wi-Fi (HTTPS)
 Raspberry Pi ───────────────▶ ESP32-S3 ◀───────────────▶ Cloud
 (Linux)      "EMOTION:happy:0.62\n"  │  keyboard·servos      Deepgram (STT/TTS)
  camera+mic                          │  mic·speaker·SD       Claude (correction)
  AVEmotionNet                        ▼                       Apps Script → Sheets
  emotion_inference.py        tactile prefix + Braille text
```

---

## Repository layout

```
.
├── esp32_firmware/                # Arduino sketch — flash to the ESP32-S3
│   ├── esp32_firmware.ino         #   main loop (keyboard, AI, TTS/STT, display, emotion)
│   ├── config.h                   #   pins, modes, endpoints (no secrets)
│   ├── secrets.example.h          #   template → copy to secrets.h and fill in
│   ├── braille_input.cpp/.h       #   10-key Perkins chord reader (80 ms state machine)
│   ├── braille_grade2.cpp/.h      #   Grade 2 contraction expansion
│   ├── braille_correct.cpp/.h     #   Claude typo/grammar correction over Wi-Fi
│   ├── braille_output.cpp/.h      #   6-servo single Braille cell (Grade 1)
│   ├── mic_stt.cpp/.h             #   INMP441 mic  → Deepgram STT
│   ├── speaker_tts.cpp/.h         #   Deepgram Aura → MAX98357A speaker
│   └── emotion_display.cpp/.h     #   UART emotion receiver + tactile prefix/render
│
├── raspberry_pi/                  # Python — runs on Raspberry Pi OS (Linux)
│   ├── emotion_inference.py       #   LIVE service: AVEmotionNet → "EMOTION:…" over UART
│   ├── braille_core.py            #   reference sim: ASCII⇄Braille, chords, prefixes
│   ├── display_driver.py          #   reference sim: refreshable display + paging
│   ├── input_handler.py           #   reference sim: chord state machine
│   ├── ai_backend.py              #   reference sim: STT→LLM→TTS + PiEmotionReceiver
│   ├── cloud_sync.py              #   reference sim: SD/JSON + Google Sheets sync
│   ├── device.py                  #   reference sim: full integrated walkthrough
│   ├── run_tests.py               #   runs every module self-test
│   └── _dbg.py, *.jsonl           #   scratch trace + sample logs (logs git-ignored)
│
├── README.md
└── .gitignore
```

**Which file goes where**
- **`esp32_firmware/` → the ESP32-S3.** Open the folder in the Arduino IDE and flash it.
- **`raspberry_pi/` → the Raspberry Pi (Raspberry Pi OS / Linux).** Run
  `emotion_inference.py` there. The other Python files are the reference simulation
  and run on any PC for testing.

---

## 1. ESP32-S3 firmware — setup & flash

1. **Secrets:** in `esp32_firmware/`, copy the template and fill in your keys:
   - Windows: `copy secrets.example.h secrets.h`
   - Linux/macOS: `cp secrets.example.h secrets.h`
   - Fill `WIFI_SSID/PASS`, `CLAUDE_API_KEY`, `DEEPGRAM_API_KEY` (see Security below).
2. **Arduino IDE:** Board = **ESP32-S3 (N16R8)**; install the **`ESP32Servo`** library.
3. **Feature toggles** at the top of `esp32_firmware.ino`:
   - `ENABLE_TTS`, `ENABLE_STT`, `ENABLE_AI_CORRECTION` — cloud features (need Wi-Fi).
   - `ENABLE_DISPLAY` — drive the 6-servo Braille cell.
   - `ENABLE_EMOTION` — listen for emotion lines from the Pi.
   - `DIAG_MODE` — print raw key states to verify wiring.
4. **Flash** and open the Serial Monitor at **115200** baud.

### Wiring (from `config.h`)
- **Keyboard (10 keys):** `KEY_DOT1..6`, `KEY_SPACE/ENTER/BKSP/MODE`.
- **Braille cell (6 servos):** `SERVO_DOT1..6` (0°=down, 90°=up).
- **Mic (INMP441, I2S0):** `MIC_SCK/WS/SD`. **Speaker (MAX98357A, I2S1):** `SPK_BCLK/LRC/DIN/SD_PIN`.
- **SD (SPI):** `SD_MOSI/MISO/SCK/CS`. **LEDs:** `LED_MODE/LED_WIFI`.
- **Emotion UART from Pi:** Pi **TX → ESP32 `EMOTION_RX_PIN` (GPIO44)**, **GND→GND**, 115200.
  *(Most GPIOs are already used — verify GPIO44/UART0 RX is free on your board, or pick another free pin.)*

---

## 2. Raspberry Pi emotion service — setup & run (Linux)

> Runs on **Raspberry Pi OS (Linux)** — PyTorch/transformers are not available on QNX.

1. Flash **Raspberry Pi OS (64-bit)** to the Pi.
2. Enable the serial port: `sudo raspi-config` → *Interface Options* → *Serial Port*
   → login shell **No**, hardware serial **Yes** (gives you `/dev/serial0`).
3. Install dependencies:
   ```bash
   pip install torch transformers librosa opencv-python numpy pyserial sounddevice gdown
   ```
4. Plug in a **USB camera + mic**. Test predictions without the ESP32 first:
   ```bash
   python emotion_inference.py --no-serial
   ```
5. Wire **Pi TX (GPIO14) → ESP32 RX (GPIO44)** + common GND, then run live:
   ```bash
   python emotion_inference.py --port /dev/serial0
   ```
   The model auto-downloads (`ft_best.pt`) and streams `EMOTION:<label>:<conf>\n`
   at 115200 baud. The ESP32 flashes the matching tactile prefix.

---

## 3. PC reference simulation (optional, for testing logic)

From `raspberry_pi/` on any machine with Python:
```bash
python run_tests.py        # all module self-tests
python device.py           # full integrated walkthrough (mock providers)
python ai_backend.py       # the three AI modes with mock STT/LLM/TTS
```

---

## Emotion labels → tactile prefixes
The model emits `anger | disgust | fear | happy | neutral | sad` (mapped to the
Braille prefix names `angry/disgust/fearful/happy/neutral/sad`). The firmware
flashes a **single-cell prefix** using each label's **first letter** (`a d f h n s`,
all distinct cells) for `EMOTION_DISPLAY_MS`, then shows the text.

## Three device modes
1. **Face-to-Face** — speaker → emotion prefix + Braille summary; user types → spoken.
2. **AI Braille Tutor** — interactive drilling.
3. **AI Assistant** — answers typed questions.

## Grade 1 vs Grade 2
- **Output** (Braille cell, `braille_output`) and the Python reference use **Grade 1**.
- **Input** (`braille_grade2`) supports full **Grade 2** contractions (e.g. `the`,
  `and`, `ing`, wordsigns, capital/number signs).

## Recommended bring-up order
1. **PC sim** — `python run_tests.py` passes.
2. **Keyboard** — flash firmware with `DIAG_MODE true`, verify each key.
3. **Voice** — enable STT/TTS/AI correction, test mic/speaker/Claude.
4. **Display** — `ENABLE_DISPLAY true`, verify servos form cells.
5. **Emotion** — `emotion_inference.py --no-serial`, then wire UART and confirm
   prefixes appear on the cell.
6. **Cloud logging** — enable `cloud_sync` last.

## 🔐 Secrets / Security
No keys live in source. `config.h` does `#include "secrets.h"`, and **`secrets.h`
is git-ignored** (see `.gitignore`). Use `secrets.example.h` as the template.

> ⚠️ The keys previously committed in plaintext (Claude, Deepgram) and the Wi-Fi
> password are **compromised** — **revoke/rotate them**, change the Wi-Fi password,
> and put the new values in `esp32_firmware/secrets.h`.
