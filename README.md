# BrailleAI — Emotion-Aware Braille Communicator

A bidirectional communication aid for **DeafBlind** users that pairs a
**refreshable Braille device** with an **audio-visual emotion model**. When a
hearing person speaks, the device shows a tactile **emotion prefix** + a short
**Braille summary**; the user types back on a Perkins keyboard and is **spoken
aloud**.

This repository is a **monorepo** with two subprojects:

| Folder | What it is |
|--------|-----------|
| [`AV/`](AV/) | **Emotion recognition** — trains the audio+video model (CREMA-D) that produces `ft_best.pt`. |
| [`Braille TTS and STT/`](Braille%20TTS%20and%20STT/) | **The BrailleAI device** — ESP32-S3 firmware + Raspberry Pi service that *uses* that model. |

```
Actual Project/
├── AV/                       # CREMA-D emotion recognition (training + inference)
│   ├── final_1.py            #   train + save the deployable model
│   ├── final_2.py            #   interval inference + ground-truth scoring
│   ├── Crema_run_final.py    #   final pipeline (train + live/video inference)
│   ├── iter1/ iter2/ iter3/  #   the three research iterations (+ result graphs)
│   └── README_CREMA_Emotion.md
│
├── Braille TTS and STT/      # The communicator device
│   ├── esp32_firmware/       #   Arduino sketch for the ESP32-S3
│   ├── raspberry_pi/         #   Python: emotion_inference.py + reference sim
│   └── README.md             #   device build/run guide
│
├── BrailleAI_Project_Outline.pdf
├── README.md                 # (this file)
└── .gitignore
```

---

## How the two halves connect

```
   AV/  (train on a GPU)                 Braille TTS and STT/  (the device)
 ┌─────────────────────────┐          ┌───────────────────────────────────────┐
 │ final_1.py / iter3       │  ft_best │ raspberry_pi/emotion_inference.py      │
 │  AVEmotionNet            │ ───.pt──▶│  loads the model, reads camera+mic     │
 │  (wav2vec2 + ViT)        │          │  └─ "EMOTION:happy:0.62\n" via UART ──▶ │
 └─────────────────────────┘          │ esp32_firmware/  → tactile prefix +     │
                                       │   Braille text on the servo cell        │
                                       └───────────────────────────────────────┘
```

The emotion classes (`anger, disgust, fear, happy, neutral, sad`) map to the
Braille prefixes the firmware flashes before each message.

---

## Quick start

**1. Train / obtain the emotion model** — see [`AV/README_CREMA_Emotion.md`](AV/README_CREMA_Emotion.md)
```bash
cd AV
python final_1.py     # trains and saves the model checkpoint
```
> The Raspberry Pi service can also **auto-download** a prebuilt `ft_best.pt`
> from Google Drive, so you don't have to retrain to run the device.

**2. Build the device** — see [`Braille TTS and STT/README.md`](Braille%20TTS%20and%20STT/README.md)
- Flash `esp32_firmware/` to the ESP32-S3 (Arduino IDE).
- Run `raspberry_pi/emotion_inference.py` on Raspberry Pi OS (Linux).
- Wire **Pi TX → ESP32 RX** + common GND (115200 baud).

---

## What is NOT in this repo (git-ignored)
To keep the repository light and under GitHub's file-size limits, large,
regenerable, or sensitive files are excluded (see `.gitignore`):
- **Model checkpoints / caches:** `*.pt`, `*.joblib`, `*.npy`, `*.npz`
  (e.g. `ft_best.pt`, iter3 audio cache, embeddings).
- **Media / data:** `*.mp4`, `*.wav` (e.g. `camera_test.mp4`, CREMA-D clips).
- **Secrets:** `secrets.h` (ESP32 Wi-Fi + API keys) — use `secrets.example.h`.
- **Logs:** `*.jsonl`, Python `__pycache__/`.

Result graphs (`*.png`) and metrics (`*_results.txt`) **are** kept so the
iteration results are visible.

## 🔐 Security
ESP32 secrets live in `Braille TTS and STT/esp32_firmware/secrets.h`
(git-ignored). The Claude/Deepgram keys and Wi-Fi password that were previously
committed in plaintext are **compromised** — revoke/rotate them and change the
Wi-Fi password before sharing.

## Credits
See `BrailleAI_Project_Outline.pdf` for the full hardware/software design.
