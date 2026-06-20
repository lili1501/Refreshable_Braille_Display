# BrailleAI — Python Reference Implementation

A PC-runnable, fully-testable Python prototype of the **BrailleAI** bidirectional
Braille AI communicator. It mirrors the logic the ESP32-S3 firmware + cloud stack
will run, so every algorithm can be validated before hardware bring-up.

## Modules (build order)

| # | File | Outline section | Responsibility |
|---|------|-----------------|----------------|
| 1 | `braille_core.py`   | III – Software | ASCII⇄Braille, half-cell states (0–7), servo angles, Perkins chord decode, emotion prefixes |
| 2 | `display_driver.py` | II/III – Hardware | Refreshable display: PCA9685 PWM duty, paging long text, servo settle timing, idle blank |
| 3 | `input_handler.py`  | III – Software | Perkins keyboard 80 ms chord state machine, number/capital modes |
| 4 | `ai_backend.py`     | IV – Cloud AI | STT→LLM→TTS orchestration + emotion classifier; one call per device mode |
| 5 | `cloud_sync.py`     | V – Monitoring | SD-card JSON logging + Google Apps Script → Google Sheets sync (offline-tolerant) |
| 6 | `device.py`         | III – Firmware loop | Main controller/state machine integrating Modules 1–5; 3 modes |

## Three device modes
1. **Face-to-Face** — hearing person speaks → emotion + summary on Braille; user types (Perkins) → spoken via TTS.
2. **AI Braille Tutor** — interactive drilling.
3. **AI Assistant** — answers typed questions.

## Run
```bash
python braille_core.py      # engine self-test
python display_driver.py    # display paging demo
python input_handler.py     # types "Hi 42" via chords
python ai_backend.py        # all three AI modes (mock providers)
python cloud_sync.py        # offline-tolerant sync demo
python device.py            # full integrated device walkthrough
python run_tests.py         # runs every module self-test
```

## Porting to the ESP32-S3
All hardware/cloud touch-points are behind injectable interfaces:
- `PWMBackend` → swap `SimulatedPCA9685` for the real Adafruit PCA9685 driver.
- `STTProvider`/`LLMProvider`/`TTSProvider` → Whisper/Claude/ElevenLabs clients.
- `HttpPoster` → `urequests.post` to the Apps Script webhook URL.

The pure logic (Modules 1 & 3) ports unchanged to MicroPython or C++.