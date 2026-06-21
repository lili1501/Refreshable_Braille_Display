"""
================================================================================
CREMA-D EMOTION RECOGNITION — FINAL END-TO-END SCRIPT (TRAIN + LIVE INFERENCE)
================================================================================

This is the FINAL, self-contained pipeline for 6-class emotion recognition
(anger, disgust, fear, happy, neutral, sad) by fine-tuning an audio + video
model on CREMA-D, then APPLYING it to brand-new, unknown clips — a prerecorded
video file, a live webcam+mic stream, or both.

    AUDIO  : wav2vec2  (transformer fine-tuned; conv feature-extractor frozen)
    VIDEO  : ViT face-expression model (fine-tuned), mean-pooled over frames
    FUSION : concat(audio_emb, video_emb) -> MLP head -> 6-way softmax

This is the PROVEN configuration (wav2vec2-base + mean pooling, clip-level
split) that reached TEST macro-F1 = 0.873 / accuracy 0.873.

    PART 1 — ONE-TIME CACHE: decode frames + audio to disk (fast epochs after)
    PART 2 — FINE-TUNE (train/val), save best by val macro-F1
    PART 3 — TEST evaluation + graphs + sample predictions
    PART 4 — APPLY TO UNKNOWN INPUT: prerecorded video / live webcam / both

--------------------------------------------------------------------------------
USAGE
  1) Train + evaluate only (default):           set INFER_MODE = "none"
  2) Classify a prerecorded video (with audio): set INFER_MODE = "video" and
                                                INFER_VIDEO_PATH = r"...\clip.mp4"
  3) Live webcam + microphone:                  set INFER_MODE = "live"
  4) Both:                                       set INFER_MODE = "both"

  If a trained checkpoint already exists (ft_best.pt), set TRAIN = False to skip
  straight to inference.

DEPENDENCIES
    pip install torch torchvision transformers pillow opencv-python==4.10.0.84 \
                librosa soundfile scikit-learn matplotlib joblib pandas "numpy<2"
    # live microphone capture additionally needs:
    pip install sounddevice
--------------------------------------------------------------------------------
"""

import os
import re
import glob
import time
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModel, AutoFeatureExtractor, AutoImageProcessor

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
warnings.filterwarnings("ignore")


# ================================================================== #
#  CONFIG                                                            #
# ================================================================== #

AUDIO_DIR = r"C:\Users\shubh\Desktop\archive\content\CREMA-D\AudioWAV"
VIDEO_DIR = r"C:\Users\shubh\Desktop\archive\content\CREMA-D\VideoFlash"
OUT_DIR   = r"C:\Users\shubh\Desktop\Hard disk\College(PG)\Non Academic at UCSD\Hackathon\Berkeley June 20-21\Actual Project\AV\crema_run_final_output"

# PROVEN backbone combo (reached TEST macro-F1 = 0.873 on the clip-level split).
AUDIO_MODEL = "superb/wav2vec2-base-superb-er"
VIDEO_MODEL = "trpakov/vit-face-expression"

SAMPLE_RATE  = 16000
MAX_AUDIO_S  = 5            # pad/truncate audio to this many seconds
N_FRAMES     = 8           # frames sampled per clip
IMG_SIZE     = 224

EPOCHS       = 20
BATCH        = 48          # lower to 32/24 if you hit OOM
LR_BACKBONE  = 1e-5
LR_HEAD      = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_FRAC  = 0.1
LABEL_SMOOTH = 0.05
GRAD_CLIP    = 1.0
NUM_WORKERS  = 0           # Windows: 0 ; Linux: raise to #cores (e.g. 8/16)
FREEZE_AUDIO_CNN = True
PATIENCE     = 5           # early stop on val macro-F1

LIMIT        = 0           # 0 = all clips; e.g. 400 for a quick smoke test
RANDOM_SEED  = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

# ---- training switch -------------------------------------------------------
TRAIN = True               # False -> skip training, load ft_best.pt, go to PART 4

# ---- inference (PART 4) ----------------------------------------------------
INFER_MODE       = "none"  # "none" | "video" | "live" | "both"
INFER_VIDEO_PATH = r""     # path to a prerecorded video file (with audio)
LIVE_ROUNDS      = 8       # number of live predictions before quitting
LIVE_WINDOW_S    = MAX_AUDIO_S   # seconds of audio/video captured per prediction

EMOTION_MAP = {"ANG": "anger", "DIS": "disgust", "FEA": "fear",
               "HAP": "happy", "NEU": "neutral", "SAD": "sad"}

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BF16 = DEVICE == "cuda" and torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if BF16 else torch.float16
MAX_AUDIO_LEN = SAMPLE_RATE * MAX_AUDIO_S
BEST_PATH = os.path.join(OUT_DIR, "ft_best.pt")


# ================================================================== #
#  IMAGE / AUDIO PRE-PROCESSORS (needed for both training & inference)
# ================================================================== #

audio_fe = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL)
img_proc = AutoImageProcessor.from_pretrained(VIDEO_MODEL)
IMG_MEAN = torch.tensor(img_proc.image_mean).view(1, 1, 3, 1, 1)
IMG_STD  = torch.tensor(img_proc.image_std).view(1, 1, 3, 1, 1)


# ================================================================== #
#  SHARED HELPERS: decode frames + audio from any video/clip         #
# ================================================================== #

def frames_from_video(vpath, n_frames=N_FRAMES):
    """Sample n_frames evenly across a video file -> (n_frames,H,W,3) uint8."""
    frames = np.zeros((n_frames, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    if not vpath or not os.path.exists(vpath):
        return frames
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        return frames
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    grab = (np.linspace(0, total - 1, n_frames).astype(int)
            if total > 0 else np.arange(n_frames))
    got = []
    for gi in grab:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(gi))
        ret, fr = cap.read()
        if not ret:
            continue
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        fr = cv2.resize(fr, (IMG_SIZE, IMG_SIZE))
        got.append(fr)
    cap.release()
    for k in range(n_frames):
        if got:
            frames[k] = got[min(k, len(got) - 1)]
    return frames


def audio_from_path(path):
    """Load audio (from .wav or the audio track of a video) -> (wav_f32, true_len)."""
    try:
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    except Exception:
        y = np.zeros(SAMPLE_RATE, dtype=np.float32)
    wav = np.zeros(MAX_AUDIO_LEN, dtype=np.float32)
    L = int(min(len(y), MAX_AUDIO_LEN))
    wav[:L] = y[:L]
    return wav, max(1, L)


# ================================================================== #
#                                                                    #
#  PART 1  —  ONE-TIME CACHE (decode frames + audio to disk)         #
#                                                                    #
# ================================================================== #

print("=" * 70)
print("PART 1 — BUILD/LOAD CACHE (frames + audio)")
print("=" * 70)
print(f"Device: {DEVICE} | bf16: {BF16}")

wav_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
if not wav_files:
    raise FileNotFoundError(f"No .wav files in {AUDIO_DIR}")
if LIMIT and LIMIT > 0:
    wav_files = wav_files[:LIMIT]

video_lookup = {}
for ext in ("*.flv", "*.mp4", "*.avi", "*.wmv"):
    for vp in glob.glob(os.path.join(VIDEO_DIR, ext)):
        video_lookup[Path(vp).stem] = vp

fname_re = re.compile(r"^(\d{4})_([A-Z]{3})_([A-Z]{3})_([A-Z]{2})", re.IGNORECASE)
items = []
for wav_path in wav_files:
    stem = Path(wav_path).stem
    m = fname_re.match(stem)
    if not m:
        continue
    emo = m.group(3).upper()
    if emo not in EMOTION_MAP:
        continue
    items.append((wav_path, video_lookup.get(stem), m.group(1),
                  EMOTION_MAP[emo], Path(wav_path).name))
N = len(items)
print(f"Valid clips: {N}")

frames_path = os.path.join(OUT_DIR, "ft_frames_u8.npy")
audio_path  = os.path.join(OUT_DIR, "ft_audio_f16.npy")
alen_path   = os.path.join(OUT_DIR, "ft_audio_len.npy")
meta_path   = os.path.join(OUT_DIR, "ft_meta.npz")


def _decode_one(rec):
    wav_path, vpath, _actor, _label, _fname = rec
    wav, true_len = audio_from_path(wav_path)
    frames = frames_from_video(vpath)
    return frames, wav.astype(np.float16), true_len


class _DecodeDS(Dataset):
    def __init__(self, recs): self.recs = recs
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        f, w, L = _decode_one(self.recs[i])
        return i, f, w, L


labels = np.array([it[3] for it in items])
actors = np.array([it[2] for it in items])
files  = np.array([it[4] for it in items])

if os.path.exists(frames_path) and os.path.exists(audio_path) and os.path.exists(meta_path):
    print(f"[i] Reusing cache in {OUT_DIR} (delete ft_*.npy to rebuild).")
    frames_mm = np.load(frames_path, mmap_mode="r")
    audio_mm  = np.load(audio_path, mmap_mode="r")
    audio_len = np.load(alen_path)
else:
    print("Building cache (one-time decode; parallelized with workers)...")
    frames_mm = np.lib.format.open_memmap(
        frames_path, mode="w+", dtype=np.uint8,
        shape=(N, N_FRAMES, IMG_SIZE, IMG_SIZE, 3))
    audio_mm = np.lib.format.open_memmap(
        audio_path, mode="w+", dtype=np.float16, shape=(N, MAX_AUDIO_LEN))
    audio_len = np.zeros(N, dtype=np.int64)
    loader = DataLoader(_DecodeDS(items), batch_size=1, num_workers=NUM_WORKERS,
                        collate_fn=lambda b: b[0])
    t0 = time.time()
    for cnt, (i, f, w, L) in enumerate(loader, 1):
        frames_mm[i] = f
        audio_mm[i] = w
        audio_len[i] = L
        if cnt % 200 == 0 or cnt == N:
            print(f"  cached {cnt}/{N}  ({time.time() - t0:.0f}s)")
    frames_mm.flush(); audio_mm.flush()
    np.save(alen_path, audio_len)
    np.savez(meta_path, labels=labels, actors=actors, files=files)
    print(f"Cache built in {time.time() - t0:.0f}s -> {OUT_DIR}")


# ================================================================== #
#  SPLIT + ENCODE LABELS  (clip-level stratified)                    #
# ================================================================== #

le = LabelEncoder()
y_int = le.fit_transform(labels)
class_labels = list(le.classes_)
n_classes = len(class_labels)

all_idx = np.arange(N)
train_idx, tmp_idx = train_test_split(
    all_idx, train_size=TRAIN_FRAC, stratify=y_int, random_state=RANDOM_SEED)
rel_val = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
val_idx, test_idx = train_test_split(
    tmp_idx, train_size=rel_val, stratify=y_int[tmp_idx], random_state=RANDOM_SEED)
print(f"Split -> train {len(train_idx)} | val {len(val_idx)} | test {len(test_idx)}")


# ================================================================== #
#  DATASET / DATALOADERS                                             #
# ================================================================== #

class CremaDS(Dataset):
    def __init__(self, indices):
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, j):
        i = self.indices[j]
        frames = np.asarray(frames_mm[i])
        wav = np.asarray(audio_mm[i]).astype(np.float32)
        L = int(audio_len[i])
        return frames, wav, L, int(y_int[i]), int(i)


def collate(batch):
    frames = np.stack([b[0] for b in batch])
    waves = [b[1][:max(1, b[2])] for b in batch]
    a = audio_fe(waves, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    frames_t = torch.from_numpy(frames)
    ys = torch.tensor([b[3] for b in batch], dtype=torch.long)
    idxs = torch.tensor([b[4] for b in batch], dtype=torch.long)
    return (a.input_values, a.get("attention_mask", None), frames_t, ys, idxs)


def make_loader(indices, shuffle):
    return DataLoader(CremaDS(indices), batch_size=BATCH, shuffle=shuffle,
                      num_workers=NUM_WORKERS, collate_fn=collate,
                      pin_memory=(DEVICE == "cuda"), drop_last=False)


# ================================================================== #
#  MODEL                                                             #
# ================================================================== #

class AVEmotionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio = AutoModel.from_pretrained(AUDIO_MODEL)
        self.video = AutoModel.from_pretrained(VIDEO_MODEL)
        if FREEZE_AUDIO_CNN and hasattr(self.audio, "feature_extractor"):
            self.audio.feature_extractor._freeze_parameters()
        a_dim = self.audio.config.hidden_size
        v_dim = self.video.config.hidden_size
        self.register_buffer("img_mean", IMG_MEAN)
        self.register_buffer("img_std", IMG_STD)
        self.head = nn.Sequential(
            nn.LayerNorm(a_dim + v_dim), nn.Dropout(0.3),
            nn.Linear(a_dim + v_dim, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, n_classes),
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


@torch.no_grad()
def evaluate(loader):
    model.eval()
    preds, gts, idxs = [], [], []
    for input_values, attn, frames_u8, ys, ix in loader:
        input_values = input_values.to(DEVICE, non_blocking=True)
        attn = attn.to(DEVICE) if attn is not None else None
        frames_u8 = frames_u8.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=(DEVICE == "cuda")):
            logits = model(input_values, attn, frames_u8)
        preds.append(logits.argmax(1).cpu().numpy())
        gts.append(ys.numpy()); idxs.append(ix.numpy())
    return (np.concatenate(preds), np.concatenate(gts), np.concatenate(idxs))


# ================================================================== #
#                                                                    #
#  PART 2  —  FINE-TUNE                                              #
#                                                                    #
# ================================================================== #

if TRAIN:
    print("\n" + "=" * 70)
    print("PART 2 — FINE-TUNE (train/val)")
    print("=" * 70)

    train_loader = make_loader(train_idx, True)
    val_loader = make_loader(val_idx, False)

    cls_counts = np.bincount(y_int[train_idx], minlength=n_classes).astype(np.float64)
    cls_w = (cls_counts.sum() / (n_classes * np.clip(cls_counts, 1, None)))
    class_weights = torch.tensor(cls_w, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)

    backbone_params = list(model.audio.parameters()) + list(model.video.parameters())
    head_params = list(model.head.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": LR_BACKBONE},
         {"params": head_params, "lr": LR_HEAD}], weight_decay=WEIGHT_DECAY)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = int(WARMUP_FRAC * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda" and not BF16))

    best_f1, bad_epochs = -1.0, 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for bi, (input_values, attn, frames_u8, ys, _ix) in enumerate(train_loader, 1):
            input_values = input_values.to(DEVICE, non_blocking=True)
            attn = attn.to(DEVICE) if attn is not None else None
            frames_u8 = frames_u8.to(DEVICE, non_blocking=True)
            ys = ys.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=(DEVICE == "cuda")):
                logits = model(input_values, attn, frames_u8)
                loss = criterion(logits, ys)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            scheduler.step()
            running += loss.item()
            if bi % 50 == 0:
                print(f"  epoch {epoch} step {bi}/{steps_per_epoch} loss={running / bi:.4f}")

        vp, vg, _ = evaluate(val_loader)
        vacc = accuracy_score(vg, vp)
        vf1 = f1_score(vg, vp, average="macro", zero_division=0)
        history.append((epoch, running / steps_per_epoch, vacc, vf1))
        print(f"[epoch {epoch}] train_loss={running / steps_per_epoch:.4f} "
              f"val_acc={vacc:.3f} val_macroF1={vf1:.3f}  ({time.time() - t0:.0f}s)")

        if vf1 > best_f1:
            best_f1, bad_epochs = vf1, 0
            torch.save({"state_dict": model.state_dict(), "classes": class_labels,
                        "audio_model": AUDIO_MODEL, "video_model": VIDEO_MODEL,
                        "n_frames": N_FRAMES, "img_size": IMG_SIZE}, BEST_PATH)
            print(f"  ** new best (val macroF1={vf1:.3f}) saved -> {BEST_PATH}")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"Early stopping (no val improvement for {PATIENCE} epochs).")
                break

    hist = np.array(history)
    plt.figure(figsize=(7, 4.5))
    plt.plot(hist[:, 0], hist[:, 3], "-o", label="val macro-F1")
    plt.plot(hist[:, 0], hist[:, 2], "-s", label="val acc")
    plt.xlabel("epoch"); plt.ylabel("score"); plt.ylim(0, 1); plt.legend()
    plt.title("Fine-tuning — validation curve")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ft_01_val_curve.png"), dpi=140)
    plt.close()


    # ============================================================== #
    #  PART 3  —  TEST EVALUATION + GRAPHS                           #
    # ============================================================== #

    print("\n" + "=" * 70)
    print("PART 3 — TEST EVALUATION")
    print("=" * 70)

    test_loader = make_loader(test_idx, False)
    ckpt = torch.load(BEST_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])

    tp, tg, tix = evaluate(test_loader)
    test_acc = accuracy_score(tg, tp)
    test_prec = precision_score(tg, tp, average="macro", zero_division=0)
    test_rec = recall_score(tg, tp, average="macro", zero_division=0)
    test_f1 = f1_score(tg, tp, average="macro", zero_division=0)
    print(f"  Accuracy : {test_acc:.3f}")
    print(f"  Precision: {test_prec:.3f} (macro)")
    print(f"  Recall   : {test_rec:.3f} (macro)")
    print(f"  Macro-F1 : {test_f1:.3f}")

    report = classification_report(tg, tp, target_names=class_labels, zero_division=0)
    with open(os.path.join(OUT_DIR, "ft_results.txt"), "w") as f:
        f.write("CREMA-D FINE-TUNED (audio+video) — Results\n")
        f.write(f"best val macroF1={best_f1:.3f}\n")
        f.write(f"TEST acc={test_acc:.3f} prec={test_prec:.3f} "
                f"rec={test_rec:.3f} macroF1={test_f1:.3f}\n\n{report}\n")

    cm = confusion_matrix(tg, tp, labels=list(range(n_classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    ax.set_title(f"Confusion Matrix — TEST (fine-tuned, macroF1={test_f1:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ft_02_confusion_matrix_test.png"), dpi=140)
    plt.close(fig)

    pcf1 = f1_score(tg, tp, average=None, labels=list(range(n_classes)), zero_division=0)
    plt.figure(figsize=(7.5, 4.5))
    bars = plt.bar(class_labels, pcf1, color="#8E7DBE")
    for b, s in zip(bars, pcf1):
        plt.text(b.get_x() + b.get_width() / 2, s, f"{s:.2f}", ha="center", va="bottom")
    plt.title("Per-Emotion F1 on TEST (fine-tuned)")
    plt.ylabel("F1"); plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ft_03_per_class_f1_test.png"), dpi=140)
    plt.close()
    print("Saved training/test graphs + ft_results.txt in", OUT_DIR)
else:
    print("\n[i] TRAIN=False -> skipping training; loading existing checkpoint.")


# ================================================================== #
#                                                                    #
#  PART 4  —  APPLY TO UNKNOWN INPUT (video file / live / both)      #
#                                                                    #
# ================================================================== #

# Make sure a trained model is loaded (covers the TRAIN=False path too).
if os.path.exists(BEST_PATH):
    ckpt = torch.load(BEST_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    class_labels = ckpt.get("classes", class_labels)
    n_classes = len(class_labels)
model.eval()

IMG_MEAN_D = IMG_MEAN.to(DEVICE)
IMG_STD_D = IMG_STD.to(DEVICE)


@torch.no_grad()
def predict_av(frames_u8, wav_f32, true_len):
    """frames_u8: (N_FRAMES,H,W,3) uint8 ; wav_f32: (MAX_AUDIO_LEN,) -> (label, probs)."""
    a = audio_fe([wav_f32[:max(1, true_len)]], sampling_rate=SAMPLE_RATE,
                 return_tensors="pt", padding=True)
    input_values = a.input_values.to(DEVICE)
    attn = a.get("attention_mask", None)
    attn = attn.to(DEVICE) if attn is not None else None
    frames_t = torch.from_numpy(frames_u8[None]).to(DEVICE)   # (1,N,H,W,3)
    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=(DEVICE == "cuda")):
        logits = model(input_values, attn, frames_t)
    probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return class_labels[idx], probs


def _print_probs(label, probs):
    order = np.argsort(probs)[::-1]
    print(f"  -> PREDICTION: {label.upper()}  (p={probs[order[0]]:.2f})")
    print("     " + " | ".join(f"{class_labels[i]} {probs[i]:.2f}" for i in order))


def infer_video_file(path):
    """Classify a prerecorded video file (uses its own audio track if present)."""
    print("\n" + "-" * 70)
    print(f"[VIDEO FILE] {path}")
    if not os.path.exists(path):
        print("  !! file not found — set INFER_VIDEO_PATH to a real file.")
        return
    frames = frames_from_video(path)
    wav, L = audio_from_path(path)          # librosa reads the video's audio via ffmpeg
    if L <= 1:
        print("  [warn] no audio decoded (no track / no ffmpeg) — using video only.")
    label, probs = predict_av(frames, wav, L)
    _print_probs(label, probs)
    pd.DataFrame([{"input": path, "pred": label,
                   **{c: float(probs[i]) for i, c in enumerate(class_labels)}}]
                 ).to_csv(os.path.join(OUT_DIR, "infer_video_prediction.csv"), index=False)
    return label, probs


def infer_live(rounds=LIVE_ROUNDS, window_s=LIVE_WINDOW_S):
    """Live webcam + microphone: capture a short window, predict, repeat.
    Press 'q' in the preview window to stop early."""
    print("\n" + "-" * 70)
    print(f"[LIVE] webcam + mic | {rounds} rounds of {window_s}s each")
    try:
        import sounddevice as sd
    except Exception:
        print("  !! 'sounddevice' not installed. Run: pip install sounddevice")
        print("     (skipping live inference)")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  !! could not open webcam (index 0).")
        return

    try:
        for r in range(1, rounds + 1):
            print(f"\n  Round {r}/{rounds}: recording {window_s}s... look at the camera & speak.")
            # start non-blocking audio recording
            audio_buf = sd.rec(int(window_s * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                               channels=1, dtype="float32")
            collected = []
            t_end = time.time() + window_s
            while time.time() < t_end:
                ret, fr = cap.read()
                if not ret:
                    continue
                collected.append(fr)
                cv2.putText(fr, f"REC {r}/{rounds}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imshow("CREMA live (press q to quit)", fr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt
            sd.wait()

            # build N_FRAMES from the collected webcam frames
            frames = np.zeros((N_FRAMES, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            if collected:
                pick = np.linspace(0, len(collected) - 1, N_FRAMES).astype(int)
                for k, pi in enumerate(pick):
                    fr = cv2.cvtColor(collected[pi], cv2.COLOR_BGR2RGB)
                    frames[k] = cv2.resize(fr, (IMG_SIZE, IMG_SIZE))

            # audio window -> padded buffer
            y = audio_buf.reshape(-1).astype(np.float32)
            wav = np.zeros(MAX_AUDIO_LEN, dtype=np.float32)
            L = int(min(len(y), MAX_AUDIO_LEN))
            wav[:L] = y[:L]

            label, probs = predict_av(frames, wav, max(1, L))
            _print_probs(label, probs)

            # show the verdict on the last frame for ~1s
            if collected:
                shot = collected[-1].copy()
                cv2.putText(shot, f"{label.upper()} ({probs.max():.2f})", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
                cv2.imshow("CREMA live (press q to quit)", shot)
                cv2.waitKey(800)
    except KeyboardInterrupt:
        print("\n  [stopped by user]")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if INFER_MODE in ("video", "both"):
    infer_video_file(INFER_VIDEO_PATH)
if INFER_MODE in ("live", "both"):
    infer_live()
if INFER_MODE == "none":
    print("\n[i] INFER_MODE='none' -> training/eval only. Set 'video'/'live'/'both' to apply.")

print("\n" + "=" * 70)
print("DONE. Outputs in:", os.path.abspath(OUT_DIR))
print("=" * 70)
