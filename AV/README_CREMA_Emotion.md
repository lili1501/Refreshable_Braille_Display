# CREMA-D Emotion Recognition — Audio + Video (Three Approaches)

Predicts the **6 emotions** in the
[CREMA-D dataset](https://github.com/CheyneyComputerScience/CREMA-D)
(anger, disgust, fear, happy, neutral, sad) by **fusing audio and video**.
CREMA-D is audio-visual: every clip exists as both `AudioWAV/<id>.wav` and
`VideoFlash/<id>.flv`, and the label is parsed from the filename.

This project is a **progression of three self-contained scripts** (one per
`iterN/` folder), from hand-crafted features to fully fine-tuned deep models:

| # | Folder / script | Approach | Backbones | Test acc | Test macro-F1 |
|---|-----------------|----------|-----------|---------:|--------------:|
| 1 | `iter1/iter1_classical_pipeline.py` | Classical feature-level fusion | hand-crafted (librosa/Parselmouth + MediaPipe) | **0.751** | **0.749** |
| 2 | `iter2/iter2_deep_embeddings.py` | Frozen deep-embedding fusion | wav2vec2 + ViT (frozen) | **0.801** | **0.801** |
| 3 | `iter3/iter3_finetune.py` | **End-to-end fine-tuning** | wav2vec2 + ViT (trainable) | **0.873** | **0.873** |

All numbers above use the **same clip-level stratified split** — 7,442 clips →
**train 5,209 / val 1,116 / test 1,117** (seed 42), with scaling/normalization
fit on **train only**. Chance level = 1/6 ≈ 16.7%. **Macro-F1** is the headline
metric (all six emotions weighted equally).

> **Final progression: 0.749 → 0.801 → 0.873 macro-F1.** Fine-tuning both
> backbones end-to-end (script 3) is the best model, run on a B200 GPU.

A fourth script, **`Crema_run_final.py`** (in the `AV/` root), is the **final
deployable pipeline**: it trains the best (0.873) fine-tuned model *and then
applies it* to brand-new inputs — a **prerecorded video file**, a **live webcam +
microphone** stream, or **both**.

---

## Project layout

Each iteration lives in its **own folder containing both the code and all its
outputs**, so reruns stay self-contained:

```
AV/
├── iter1/                       # Iteration 1 — classical fusion
│   ├── iter1_classical_pipeline.py
│   └── iter1_features.csv, iter1_*.png, iter1_results.txt, iter1_model.joblib, iter1_predictions.csv
├── iter2/                       # Iteration 2 — frozen deep embeddings
│   ├── iter2_deep_embeddings.py
│   └── iter2_embeddings.npz, iter2_*.png, iter2_results.txt, iter2_model.joblib, iter2_predictions.csv
├── iter3/                       # Iteration 3 — end-to-end fine-tuning (BEST)
│   ├── iter3_finetune.py
│   └── iter3_<tag>_model.pt, iter3_<tag>_*.png, iter3_<tag>_results.txt, iter3_*.npy/.npz
├── Crema_run_final.py           # Final deployable pipeline (train + live/video inference)
└── README_CREMA_Emotion.md
```

Each script's `OUT_DIR` already points at its own `iterN/` folder — just run it.

---

## Common setup

### Labels (from the filename)
`ActorID_Sentence_Emotion_Intensity.wav`, e.g. `1001_DFA_ANG_XX.wav`:

| Token | Example | Used as |
|-------|---------|---------|
| ActorID | `1001` | grouping key (for the actor-level split in script 3) |
| Sentence | `DFA` | ignored |
| Emotion | `ANG` | the **label** (`ANG→anger, DIS→disgust, FEA→fear, HAP→happy, NEU→neutral, SAD→sad`) |
| Intensity | `XX` | ignored |

### The data split
Default is **clip-level, stratified by emotion** (70/15/15). The same actor may
appear in more than one split, so this is *easier* than a speaker-independent
split — expect these numbers to sit a little higher than actor-wise evaluation.
**Script 3 runs both clip-level and actor-level and compares them directly.**

---

## 1. `iter1/iter1_classical_pipeline.py` — Classical feature-level fusion

A single line-by-line file (**no functions**) in three parts:
**PART 1** extract → `iter1_features.csv`; **PART 2** train/val/test + save best;
**PART 3** apply the saved model to held-out clips.

### Features
- **Audio (~75, `aud_`)** — RMS energy, spectral (centroid/bandwidth/roll-off/ZCR),
  13 mel + 13 MFCC (mean/std), voice activity, pitch/intensity (Parselmouth).
- **Video (~26, `vid_`)** — EAR, MAR, gaze H/V, head pose yaw/pitch/roll
  (`cv2.solvePnP`), yaw velocity, attention score, blink/yawn counts & rates,
  face-detection ratio — via **MediaPipe Face Landmarker**, aggregated per clip.

If OpenCV/MediaPipe or the model file are missing, video columns become zeros and
the script runs **audio-only** with a warning.

### Models
Five scikit-learn heads, best **by validation macro-F1** auto-selected:
`LogisticRegression`, `SVM (RBF)`, `RandomForest`, `HistGradientBoosting`, `MLP`.

### Results (clip-level test)
**Best = HistGradientBoosting → TEST accuracy 0.751, macro-F1 0.749.**

| Modality (val macro-F1) | audio | video | **fused** |
|---|---:|---:|---:|
| HistGradientBoosting | 0.569 | 0.624 | **0.744** |

Per-class F1 (test): anger **0.86**, happy 0.81, disgust 0.78, neutral 0.72,
sad 0.69, **fear 0.64** (weakest). Fusion clearly beats either modality alone.

### Install
```bash
pip install numpy pandas scikit-learn matplotlib joblib librosa soundfile \
            praat-parselmouth "opencv-python==4.10.0.84" mediapipe "numpy<2"
```

> **MediaPipe model file:** point `MODEL_PATH` at `face_landmarker.task`
> (download: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`).

### Outputs (all prefixed `iter1_`, in `iter1/`)
`iter1_features.csv` (cache), `iter1_class_distribution.png`,
`iter1_model_comparison.png`, `iter1_modality_comparison.png`,
`iter1_confusion_matrix_test.png`, `iter1_per_class_f1_test.png`,
`iter1_feature_importance.png`, `iter1_results.txt`, `iter1_model.joblib`,
`iter1_sample_prediction_proba.png`, `iter1_predictions.csv`.

> Extraction is slow (MediaPipe over ~7,400 clips). Start with `LIMIT = 300`,
> then `LIMIT = 0`. `iter1_features.csv` is cached so training re-runs instantly.

---

## 2. `iter2/iter2_deep_embeddings.py` — Frozen deep-embedding fusion

Same 3-part structure, but replaces hand-crafted features with **pretrained deep
embeddings** (backbones frozen, used purely as feature extractors), then trains a
small classifier head on the fused embeddings.

- **Audio** — `superb/wav2vec2-base-superb-er` → mean-pooled hidden states → **768-d**.
- **Video** — `trpakov/vit-face-expression` → CLS token per frame, averaged over
  frames → **768-d**.
- **Fusion** — concatenate → **1,536-d** → `LogisticRegression` / `SVM` / `MLP`,
  best by validation.
- Embeddings are cached to `deep_embeddings.npz` (extraction is the slow part:
  ~2.8 h on CPU for the full set; instant on re-runs / for training).

### Results (clip-level test)
**Best = MLP → TEST accuracy 0.801, macro-F1 0.801.**

| Modality (val macro-F1) | audio | video | **fused** |
|---|---:|---:|---:|
| MLP | 0.642 | 0.682 | **0.798** |

Per-class F1 (test): happy **0.91**, anger 0.85, disgust 0.82, neutral 0.80,
fear 0.74, **sad 0.70**.

**Why it beats the classical pipeline (+5 pts macro-F1):** learned
representations are far richer than hand-crafted features (audio 0.642 vs 0.569,
video 0.682 vs 0.624), and fusing two stronger modalities compounds the gain.

### Install (adds the deep stack)
```bash
pip install torch torchvision transformers pillow   # + the script-1 deps
```

### Outputs (all prefixed `iter2_`, in `iter2/`)
`iter2_embeddings.npz` (cache), `iter2_class_distribution.png`,
`iter2_model_comparison.png`, `iter2_modality_comparison.png`,
`iter2_confusion_matrix_test.png`, `iter2_per_class_f1_test.png`,
`iter2_sample_prediction_proba.png`, `iter2_results.txt`,
`iter2_model.joblib`, `iter2_predictions.csv`.

---

## 3. `iter3/iter3_finetune.py` — End-to-end fine-tuning (GPU) — **BEST MODEL**

Fine-tunes **both backbones jointly** with a fusion head — the real lever for
higher accuracy. Run on a **B200 GPU**.

- **Audio** — `microsoft/wavlm-large` by default (trainable; conv feature-extractor
  frozen) → masked mean pool. **The headline 0.873 was obtained with
  `superb/wav2vec2-base-superb-er` + mean pooling** — that exact proven config is
  preserved as the default in `Crema_run_final.py`.
- **Video** — `trpakov/vit-face-expression` (trainable) → per-frame CLS →
  **temporal aggregation** over frames.
- **`VIDEO_TEMPORAL`** — `mean` (used for the 0.873 run) | `lstm` (BiLSTM) |
  `attention` (this file's default; learned per-frame weights).
- **Fusion head** → 6-way softmax. Trained with AdamW (split LRs for backbone vs
  head), cosine schedule + warmup, **bf16 autocast**, class weights, label
  smoothing, gradient clipping, and early stopping on val macro-F1.
- **One-time frame/audio cache** to disk → fast epochs after the first.
- **Runs BOTH splits** (clip-level *and* actor-level) and writes a side-by-side
  comparison.

### Results (clip-level test) — best of all three approaches
**TEST accuracy 0.873, macro-F1 0.873** (best val macro-F1 0.893).

Per-class F1 (test): happy **0.97**, anger 0.92, disgust 0.89, neutral 0.87,
fear 0.82, **sad 0.76**. Fine-tuning lifts *every* class vs the frozen-embedding
run — most dramatically the hard ones (sad 0.70 → 0.76, fear 0.74 → 0.82).

> **Optional higher-ceiling experiment:** set `AUDIO_MODEL = "microsoft/wavlm-large"`
> and `VIDEO_TEMPORAL = "attention"` for a stronger (untested, ~3x params) variant;
> lower `BATCH` to 24/32 if you hit OOM.

### Setup (B200 / Blackwell needs a new CUDA build)
```bash
conda create -n crema_ft python=3.10 -y && conda activate crema_ft
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers pillow "opencv-python==4.10.0.84" librosa soundfile \
            scikit-learn matplotlib joblib pandas "numpy<2"
```
Update `AUDIO_DIR` / `VIDEO_DIR` / `OUT_DIR` for the GPU machine before running.

### Tuning levers (if accuracy stalls)
- Increase `N_FRAMES` (more temporal info); raise `EPOCHS`.
- `LR_BACKBONE = 2e-5` if underfitting, or `5e-6` if it diverges early.
- `VIDEO_TEMPORAL = "lstm"`/`"attention"` for a temporal model over frames.

### Both splits + comparison
The finalized script runs **both** the clip-level and actor-level (speaker-
independent) splits and writes per-split metrics plus a side-by-side comparison.
Note: the clip-level **0.873** is optimistic (an actor can appear in train+test);
the actor-level number is the honest, harder benchmark and will be lower.

### Outputs (per split `<tag>` = `clip` / `actor`, in `iter3/`)
`iter3_<tag>_model.pt`, `iter3_<tag>_results.txt`, `iter3_<tag>_val_curve.png`,
`iter3_<tag>_confusion_matrix_test.png`, `iter3_<tag>_per_class_f1_test.png`,
`iter3_<tag>_predictions.csv`, plus the comparison: `iter3_comparison.txt`,
`iter3_compare_overall.png`, `iter3_compare_per_class.png`. (Decode caches:
`iter3_frames_u8.npy`, `iter3_audio_f16.npy`, `iter3_audio_len.npy`, `iter3_meta.npz`.)

---

## 4. `Crema_run_final.py` — Final deployable pipeline (train + live inference)

The **single final deliverable**. It reuses the proven fine-tuning recipe
(`wav2vec2-base` + ViT, mean-pool, clip-level split → **0.873** test macro-F1)
and adds a **PART 4** that applies the trained model to *unknown* inputs.

### Four parts
- **PART 1** — one-time cache of frames + audio.
- **PART 2** — fine-tune (train/val), save best to `ft_best.pt`.
- **PART 3** — test evaluation + graphs.
- **PART 4** — **apply to unknown input**: prerecorded video / live / both.

### How to run inference (set at the top of the file)
| Setting | Effect |
|---------|--------|
| `TRAIN = True/False` | `False` skips training and loads existing `ft_best.pt` |
| `INFER_MODE = "none"` | train + evaluate only (default) |
| `INFER_MODE = "video"` | classify `INFER_VIDEO_PATH` (a prerecorded clip, uses its audio track) |
| `INFER_MODE = "live"` | webcam + microphone, `LIVE_ROUNDS` short windows (press **q** to stop) |
| `INFER_MODE = "both"` | run the video file **and** the live stream |

The model needs **both** modalities, so:
- **Video file** — frames sampled with OpenCV; audio read from the same file via
  `librosa` (needs an ffmpeg backend). No audio track → predicts on video only.
- **Live** — records `LIVE_WINDOW_S` seconds of mic audio (`sounddevice`) while
  grabbing webcam frames, then predicts and overlays the verdict on the preview.

### Extra dependency for live mic capture
```bash
pip install sounddevice          # plus the script-3 deep stack
```

### Inference outputs (in `crema_run_final_output/`)
`ft_best.pt`, training/test graphs + `ft_results.txt`, and
`infer_video_prediction.csv` (probabilities for the supplied video). Live
predictions are printed to the console and overlaid on the webcam window.

> **Note:** live audio+video must be reasonably synchronized; speak and stay in
> frame for the full window. CREMA-D is acted, close-up, frontal speech — live
> webcam conditions that differ a lot from that will lower confidence.

---

## How to read the results

- **Macro-F1** is the headline (all six emotions weighted equally).
- The persistent hard cluster across **all** approaches is **sad ↔ fear ↔ neutral**;
  **happy** and **anger** are the easiest.
- **Fusion beats either modality alone** in every approach (see the modality graphs).
- **clip-level vs actor-level:** clip-level is easier (an actor can leak across
  splits), so actor-level scores are typically lower — script 3 quantifies that gap.

---

## Troubleshooting

- **`numpy` 2 vs OpenCV/TensorFlow conflicts** — pin `numpy==1.26.4` and
  `opencv-python==4.10.0.84` (newer OpenCV demands numpy ≥ 2, which breaks
  MediaPipe/TensorFlow).
- **`No .wav files found`** — fix `AUDIO_DIR`.
- **Video features all zero (script 1)** — `opencv-python`/`mediapipe` missing or
  `MODEL_PATH` wrong; the script prints which and continues audio-only.
- **`.flv` won't open** — OpenCV needs an ffmpeg backend (the full pip wheel
  includes it); or point `VIDEO_DIR` at `.mp4` copies.
- **Extraction too slow** — use `LIMIT` for testing; results cache to
  `features_fused.csv` (script 1) / `deep_embeddings.npz` (script 2).
- **B200 GPU** — Blackwell (sm_100) needs a `cu128` (or nightly) torch build;
  `cu121` wheels won't run on it.
