"""
================================================================================
FINAL_2 — APPLY THE TRAINED MODEL TO A RECORDED VIDEO + AUDIO,
          SCORE IT AGAINST A GROUND-TRUTH FILE (5-second intervals)
================================================================================

Loads the checkpoint saved by `final_1.py`, walks a recorded video in
**5-second intervals**, and for each interval predicts the emotion using
**majority voting** over several overlapping sub-windows. It then compares the
per-interval predictions to a **ground-truth Excel file** and reports accuracy +
macro-F1 + a confusion matrix.

HOW MAJORITY VOTING WORKS (per 5-second interval)
    The interval is split into N_SUBCLIPS overlapping sub-windows (each
    SUB_WIN_S seconds). The model predicts on each sub-window (its own audio
    slice + sampled frames). The interval's final label = the most frequent
    prediction across the sub-windows (ties broken by summed class probability).

GROUND-TRUTH FILE FORMAT (Excel .xlsx, 2 columns)
    interval        | Mood
    00:00 - 00:05   | Happy
    00:05 - 00:10   | Fear
    00:10 - 00:15   | Sad
    ...
    `interval` is an MM:SS - MM:SS time range (5 seconds each); `Mood` is one of:
    Anger, Disgust, Fear, Happy, Neutral, Sad (case-insensitive).
    (A dummy file `dummy_ground_truth.xlsx` is provided.)

--------------------------------------------------------------------------------
USAGE
    1) Run final_1.py first so final_output/final_model.pt exists.
    2) Set VIDEO_PATH to your recorded clip (with an audio track).
    3) Set GROUND_TRUTH_XLSX (defaults to the bundled dummy file).
    4) python final_2.py

DEPENDENCIES
    pip install torch transformers opencv-python==4.10.0.84 librosa soundfile \
                scikit-learn pandas openpyxl "numpy<2"   # openpyxl reads .xlsx
--------------------------------------------------------------------------------
"""

import os
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import librosa

import torch
import torch.nn as nn
from transformers import AutoModel, AutoFeatureExtractor, AutoImageProcessor
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import cv2
warnings.filterwarnings("ignore")


# ================================================================== #
#  CONFIG                                                            #
# ================================================================== #

AV_DIR = r"C:\Users\shubh\Desktop\Hard disk\College(PG)\Non Academic at UCSD\Hackathon\Berkeley June 20-21\Actual Project\AV"
MODEL_PATH       = os.path.join(AV_DIR, "final_output", "final_model.pt")
VIDEO_PATH       = os.path.join(AV_DIR, "recorded_clip.mp4")        # <-- your recorded video
GROUND_TRUTH_XLSX = os.path.join(AV_DIR, "dummy_ground_truth.xlsx")  # <-- ground truth (Excel)

INTERVAL_S   = 5      # length of each scoring interval (seconds)
SUB_WIN_S    = 3.0    # length of each voting sub-window (seconds)
N_SUBCLIPS   = 3      # number of sub-windows voted per interval

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BF16 = DEVICE == "cuda" and torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if BF16 else torch.float16


# ================================================================== #
#  LOAD CHECKPOINT + REBUILD MODEL                                   #
# ================================================================== #

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\nRun final_1.py first to train and save it.")

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
CLASS_LABELS = list(ckpt["classes"])
N_CLASSES    = len(CLASS_LABELS)
AUDIO_MODEL  = ckpt["audio_model"]
VIDEO_MODEL  = ckpt["video_model"]
N_FRAMES     = int(ckpt["n_frames"])
IMG_SIZE     = int(ckpt["img_size"])
SAMPLE_RATE  = int(ckpt["sample_rate"])
MAX_AUDIO_S  = int(ckpt["max_audio_s"])
MAX_AUDIO_LEN = SAMPLE_RATE * MAX_AUDIO_S

audio_fe = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL)
img_proc = AutoImageProcessor.from_pretrained(VIDEO_MODEL)
IMG_MEAN = torch.tensor(img_proc.image_mean).view(1, 1, 3, 1, 1)
IMG_STD  = torch.tensor(img_proc.image_std).view(1, 1, 3, 1, 1)


class AVEmotionNet(nn.Module):
    """Must match final_1.py exactly so the checkpoint loads."""
    def __init__(self):
        super().__init__()
        self.audio = AutoModel.from_pretrained(AUDIO_MODEL)
        self.video = AutoModel.from_pretrained(VIDEO_MODEL)
        a_dim = self.audio.config.hidden_size
        v_dim = self.video.config.hidden_size
        self.register_buffer("img_mean", IMG_MEAN)
        self.register_buffer("img_std", IMG_STD)
        self.head = nn.Sequential(
            nn.LayerNorm(a_dim + v_dim), nn.Dropout(0.3),
            nn.Linear(a_dim + v_dim, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, N_CLASSES),
        )

    def forward(self, input_values, attn, frames_u8):
        ao = self.audio(input_values, attention_mask=attn).last_hidden_state
        if attn is not None:
            fm = self.audio._get_feature_vector_attention_mask(
                ao.shape[1], attn).unsqueeze(-1).to(ao.dtype)
            a = (ao * fm).sum(1) / fm.sum(1).clamp(min=1.0)
        else:
            a = ao.mean(1)
        B, Nf = frames_u8.shape[0], frames_u8.shape[1]
        px = frames_u8.permute(0, 1, 4, 2, 3).float() / 255.0
        px = (px - self.img_mean) / self.img_std
        vo = self.video(px.flatten(0, 1)).last_hidden_state[:, 0]
        v = vo.view(B, Nf, -1).mean(1)
        return self.head(torch.cat([a, v], dim=1))


model = AVEmotionNet().to(DEVICE)
model.load_state_dict(ckpt["state_dict"])
model.eval()
print(f"Loaded model ({AUDIO_MODEL} + {VIDEO_MODEL}) | classes: {CLASS_LABELS}")
print(f"Device: {DEVICE}")


# ================================================================== #
#  HELPERS: slice a time window of the recording                     #
# ================================================================== #

def frames_in_window(cap, fps, t0, t1, n_frames=N_FRAMES):
    """Sample n_frames evenly between t0 and t1 seconds -> (n,H,W,3) uint8."""
    frames = np.zeros((n_frames, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    f0, f1 = int(t0 * fps), max(int(t1 * fps), int(t0 * fps) + 1)
    grab = np.linspace(f0, f1 - 1, n_frames).astype(int)
    got = []
    for gi in grab:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(gi))
        ret, fr = cap.read()
        if not ret:
            continue
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        got.append(cv2.resize(fr, (IMG_SIZE, IMG_SIZE)))
    for k in range(n_frames):
        if got:
            frames[k] = got[min(k, len(got) - 1)]
    return frames


def audio_in_window(full_wav, t0, t1):
    """Return a padded waveform buffer for [t0, t1) seconds -> (wav, true_len)."""
    s0, s1 = int(t0 * SAMPLE_RATE), int(t1 * SAMPLE_RATE)
    seg = full_wav[s0:s1]
    wav = np.zeros(MAX_AUDIO_LEN, dtype=np.float32)
    L = int(min(len(seg), MAX_AUDIO_LEN))
    wav[:L] = seg[:L]
    return wav, max(1, L)


@torch.no_grad()
def predict_window(frames_u8, wav, true_len):
    """Return (pred_idx, probs) for one sub-window."""
    a = audio_fe([wav[:max(1, true_len)]], sampling_rate=SAMPLE_RATE,
                 return_tensors="pt", padding=True)
    iv = a.input_values.to(DEVICE)
    attn = a.get("attention_mask", None)
    attn = attn.to(DEVICE) if attn is not None else None
    fr = torch.from_numpy(frames_u8[None]).to(DEVICE)
    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=(DEVICE == "cuda")):
        logits = model(iv, attn, fr)
    probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
    return int(probs.argmax()), probs


def predict_interval(cap, fps, full_wav, start_s):
    """Majority-vote over N_SUBCLIPS sub-windows inside [start_s, start_s+INTERVAL_S)."""
    end_s = start_s + INTERVAL_S
    # evenly spaced sub-window start offsets so the sub-windows tile the interval
    if N_SUBCLIPS > 1:
        offsets = np.linspace(0, max(0.0, INTERVAL_S - SUB_WIN_S), N_SUBCLIPS)
    else:
        offsets = np.array([max(0.0, (INTERVAL_S - SUB_WIN_S) / 2)])

    preds, prob_sum = [], np.zeros(N_CLASSES, dtype=np.float64)
    for off in offsets:
        a0 = start_s + off
        a1 = min(a0 + SUB_WIN_S, end_s)
        frames = frames_in_window(cap, fps, a0, a1)
        wav, L = audio_in_window(full_wav, a0, a1)
        pidx, probs = predict_window(frames, wav, L)
        preds.append(pidx)
        prob_sum += probs

    # majority vote; break ties by highest summed probability
    counts = Counter(preds)
    top = max(counts.values())
    candidates = [c for c, n in counts.items() if n == top]
    if len(candidates) == 1:
        winner = candidates[0]
    else:
        winner = max(candidates, key=lambda c: prob_sum[c])
    votes = {CLASS_LABELS[c]: counts.get(c, 0) for c in sorted(counts)}
    return CLASS_LABELS[winner], votes


# ================================================================== #
#  RUN OVER THE RECORDING                                            #
# ================================================================== #

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}  (set VIDEO_PATH).")

# load the full audio track of the recording (librosa reads video audio via ffmpeg)
try:
    full_wav, _ = librosa.load(VIDEO_PATH, sr=SAMPLE_RATE, mono=True)
except Exception as e:
    raise RuntimeError(f"Could not read audio from {VIDEO_PATH}: {e}\n"
                       "Ensure ffmpeg is installed so librosa can decode the audio track.")

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
video_dur = total_frames / fps if fps else 0.0
audio_dur = len(full_wav) / SAMPLE_RATE
duration = min(video_dur, audio_dur) if video_dur > 0 else audio_dur
n_intervals = int(duration // INTERVAL_S)
print(f"\nRecording: {VIDEO_PATH}")
print(f"  fps={fps:.2f}  video_dur={video_dur:.1f}s  audio_dur={audio_dur:.1f}s")
print(f"  scoring {n_intervals} interval(s) of {INTERVAL_S}s "
      f"(majority vote over {N_SUBCLIPS} x {SUB_WIN_S}s sub-windows)\n")

rows = []
for i in range(n_intervals):
    start_s = i * INTERVAL_S
    pred, votes = predict_interval(cap, fps, full_wav, start_s)
    rows.append({"start_sec": start_s, "predicted": pred, "votes": votes})
cap.release()

pred_df = pd.DataFrame(rows)


# ================================================================== #
#  COMPARE TO GROUND TRUTH                                           #
# ================================================================== #

def interval_to_start_sec(s):
    """'00:00 - 00:05' -> 0 ; '01:05 - 01:10' -> 65 (uses the START of the range)."""
    first = str(s).split("-")[0].strip()
    parts = first.split(":")
    try:
        if len(parts) == 3:   # HH:MM:SS
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        if len(parts) == 2:   # MM:SS
            return int(parts[0]) * 60 + int(parts[1])
        return int(float(first))
    except ValueError:
        return -1

gt = pd.read_excel(GROUND_TRUTH_XLSX)
gt.columns = [str(c).strip().lower() for c in gt.columns]
interval_col = next((c for c in gt.columns if c in ("interval", "time", "range")), gt.columns[0])
mood_col     = next((c for c in gt.columns if c in ("mood", "label", "emotion")), gt.columns[1])
gt = gt[[interval_col, mood_col]].rename(columns={interval_col: "interval", mood_col: "ground_truth"})
gt["start_sec"] = gt["interval"].map(interval_to_start_sec).astype(int)
gt["ground_truth"] = gt["ground_truth"].astype(str).str.strip().str.lower()

merged = pred_df.merge(gt, on="start_sec", how="left")
merged["correct"] = merged["predicted"] == merged["ground_truth"]

print("=" * 70)
print("PER-INTERVAL RESULTS (model vs ground truth)")
print("=" * 70)
hdr = f"{'interval':>12} {'ground_truth':>14} {'predicted':>12} {'match':>7}   votes"
print(hdr)
print("-" * len(hdr))
for _, r in merged.iterrows():
    rng = f"{int(r['start_sec'])}-{int(r['start_sec']) + INTERVAL_S}s"
    gtv = r["ground_truth"] if pd.notna(r["ground_truth"]) else "(none)"
    mark = "OK" if r["correct"] else ("x" if pd.notna(r["ground_truth"]) else "-")
    print(f"{rng:>12} {gtv:>14} {r['predicted']:>12} {mark:>7}   {r['votes']}")

# metrics over intervals that have a ground-truth label
scored = merged.dropna(subset=["ground_truth"])
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
if len(scored) == 0:
    print("No overlapping intervals with ground truth — check start_sec alignment.")
else:
    acc = accuracy_score(scored["ground_truth"], scored["predicted"])
    f1m = f1_score(scored["ground_truth"], scored["predicted"],
                   average="macro", zero_division=0)
    print(f"  intervals scored : {len(scored)}")
    print(f"  accuracy         : {acc:.3f}")
    print(f"  macro-F1         : {f1m:.3f}")
    labs = sorted(set(scored["ground_truth"]) | set(scored["predicted"]))
    cm = confusion_matrix(scored["ground_truth"], scored["predicted"], labels=labs)
    print("\n  Confusion matrix (rows=truth, cols=pred):")
    print("    " + " ".join(f"{l[:4]:>5}" for l in labs))
    for li, l in enumerate(labs):
        print(f"  {l[:4]:>4} " + " ".join(f"{v:>5}" for v in cm[li]))

print("\nDONE.")
