"""
================================================================================
CREMA-D EMOTION RECOGNITION — END-TO-END FINE-TUNING (AUDIO + VIDEO), B200/GPU
================================================================================

Instead of freezing the backbones and training a small head (crema_emotion_deep*),
this script FINE-TUNES both backbones jointly on CREMA-D:

    AUDIO  : wav2vec2  (transformer fine-tuned; conv feature-extractor frozen)
    VIDEO  : ViT face-expression model (fine-tuned), mean-pooled over frames
    FUSION : concat(audio_emb, video_emb) -> MLP head -> 6-way softmax

Trained with AdamW + cosine schedule + bf16 autocast + class weights.

    PART 1 — ONE-TIME CACHE: decode frames + audio to disk (fast epochs after)
    PART 2 — FINE-TUNE (train/val), save best by val macro-F1
    PART 3 — TEST evaluation + graphs + sample predictions

--------------------------------------------------------------------------------
B200 SETUP (Linux). Blackwell (sm_100) needs a NEW CUDA build of torch:

    conda create -n crema_ft python=3.10 -y
    conda activate crema_ft
    # CUDA 12.8 build (or newer / nightly) -- cu121 will NOT run on Blackwell
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    pip install transformers pillow opencv-python==4.10.0.84 librosa soundfile \
                scikit-learn matplotlib joblib pandas "numpy<2" decord

    Verify:  python -c "import torch;print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"

IMPORTANT: update AUDIO_DIR / VIDEO_DIR / OUT_DIR below to the paths on the
B200 machine (the Windows paths here are placeholders).
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
from sklearn.model_selection import train_test_split, GroupShuffleSplit
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
#  CONFIG  (update paths for the B200 machine!)                      #
# ================================================================== #

AUDIO_DIR = r"C:\Users\shubh\Desktop\archive\content\CREMA-D\AudioWAV"
VIDEO_DIR = r"C:\Users\shubh\Desktop\archive\content\CREMA-D\VideoFlash"
OUT_DIR   = r"C:\Users\shubh\Desktop\Hard disk\College(PG)\Non Academic at UCSD\Hackathon\Berkeley June 20-21\Actual Project\AV\iter3"

# WavLM is markedly stronger than wav2vec2-base for emotion/paralinguistics
# (your frozen-embedding runs showed WavLM audio ~0.75 vs wav2vec2 ~0.65).
# Use wavlm-large on the B200; switch to "microsoft/wavlm-base-plus" if VRAM-limited.
AUDIO_MODEL = "microsoft/wavlm-large"
VIDEO_MODEL = "trpakov/vit-face-expression"

SAMPLE_RATE  = 16000
MAX_AUDIO_S  = 5            # pad/truncate audio to this many seconds
N_FRAMES     = 8           # frames sampled per clip
IMG_SIZE     = 224

# ---- temporal aggregation over the per-frame ViT embeddings ----
# "mean"      : average the N_FRAMES CLS tokens (cheapest, default).
# "lstm"      : BiLSTM over frames then mean-pool its outputs (learns dynamics).
# "attention" : learned attention-weighted pooling over frames.
VIDEO_TEMPORAL = "attention"   # "mean" | "lstm" | "attention"

# ----------------------------------------------------------------------------
# TUNING LEVERS (if accuracy stalls):
#   * Increase N_FRAMES (more temporal info); raise EPOCHS for longer training.
#   * Try LR_BACKBONE = 2e-5 if underfitting, or 5e-6 if it diverges early.
#   * Set VIDEO_TEMPORAL = "lstm"/"attention" for a temporal model over frames
#     instead of mean-pooling (bigger change, possible extra gain).
# ----------------------------------------------------------------------------
EPOCHS       = 20
BATCH        = 48          # B200 has huge VRAM; lower to 32/24 if you hit OOM
LR_BACKBONE  = 1e-5
LR_HEAD      = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_FRAC  = 0.1
LABEL_SMOOTH = 0.05
GRAD_CLIP    = 1.0
NUM_WORKERS  = 8           # Linux: raise to #cores; Windows: set 0
FREEZE_AUDIO_CNN = True
PATIENCE     = 5           # early stop on val macro-F1

LIMIT        = 0           # 0 = all clips; e.g. 400 for a quick smoke test
RANDOM_SEED  = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

EMOTION_MAP = {"ANG": "anger", "DIS": "disgust", "FEA": "fear",
               "HAP": "happy", "NEU": "neutral", "SAD": "sad"}

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BF16 = DEVICE == "cuda" and torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if BF16 else torch.float16
MAX_AUDIO_LEN = SAMPLE_RATE * MAX_AUDIO_S


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

# build the valid clip list (audio must parse to a known emotion)
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

frames_path = os.path.join(OUT_DIR, "iter3_frames_u8.npy")     # (N, N_FRAMES, H, W, 3)
audio_path  = os.path.join(OUT_DIR, "iter3_audio_f16.npy")     # (N, MAX_AUDIO_LEN)
alen_path   = os.path.join(OUT_DIR, "iter3_audio_len.npy")     # (N,)
meta_path   = os.path.join(OUT_DIR, "iter3_meta.npz")


def _decode_one(rec):
    """Decode one clip -> (frames_uint8 [N_FRAMES,H,W,3], waveform_f16, true_len)."""
    wav_path, vpath, _actor, _label, _fname = rec
    # audio
    try:
        y, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    except Exception:
        y = np.zeros(SAMPLE_RATE, dtype=np.float32)
    true_len = int(min(len(y), MAX_AUDIO_LEN))
    wav = np.zeros(MAX_AUDIO_LEN, dtype=np.float32)
    wav[:true_len] = y[:true_len]
    # video frames
    frames = np.zeros((N_FRAMES, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    if vpath is not None:
        cap = cv2.VideoCapture(vpath)
        if cap.isOpened():
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            grab = (np.linspace(0, total - 1, N_FRAMES).astype(int)
                    if total > 0 else np.arange(N_FRAMES))
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
            for k in range(N_FRAMES):
                if got:
                    frames[k] = got[min(k, len(got) - 1)]
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
    print(f"[i] Reusing cache in {OUT_DIR} (delete iter3_*.npy to rebuild).")
    frames_mm = np.load(frames_path, mmap_mode="r")
    audio_mm  = np.load(audio_path, mmap_mode="r")
    audio_len = np.load(alen_path)
else:
    print("Building cache (one-time decode; parallelized with workers)...")
    frames_mm = np.lib.format.open_memmap(
        frames_path, mode="w+", dtype=np.uint8, shape=(N, N_FRAMES, IMG_SIZE, IMG_SIZE, 3))
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
#  SPLIT + ENCODE LABELS                                             #
# ================================================================== #

le = LabelEncoder()
y_int = le.fit_transform(labels)
class_labels = list(le.classes_)
n_classes = len(class_labels)

def make_clip_split():
    """Clip-level stratified split (same actors may appear across splits)."""
    a = np.arange(N)
    tr, tmp = train_test_split(a, train_size=TRAIN_FRAC, stratify=y_int,
                               random_state=RANDOM_SEED)
    rel = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
    va, te = train_test_split(tmp, train_size=rel, stratify=y_int[tmp],
                              random_state=RANDOM_SEED)
    return tr, va, te


def make_actor_split():
    """Speaker-independent split by actor (no actor shared across splits)."""
    a = np.arange(N)
    gss1 = GroupShuffleSplit(n_splits=1, train_size=TRAIN_FRAC, random_state=RANDOM_SEED)
    tr, tmp = next(gss1.split(a, y_int, groups=actors))
    rel = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=rel, random_state=RANDOM_SEED)
    va_rel, te_rel = next(gss2.split(tmp, y_int[tmp], groups=actors[tmp]))
    return tr, tmp[va_rel], tmp[te_rel]


# ================================================================== #
#  DATASET / DATALOADERS (read from cache -> fast epochs)            #
# ================================================================== #

audio_fe = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL)
img_proc = AutoImageProcessor.from_pretrained(VIDEO_MODEL)
IMG_MEAN = torch.tensor(img_proc.image_mean).view(1, 1, 3, 1, 1)
IMG_STD  = torch.tensor(img_proc.image_std).view(1, 1, 3, 1, 1)


class CremaDS(Dataset):
    def __init__(self, indices):
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, j):
        i = self.indices[j]
        frames = np.asarray(frames_mm[i])                 # (N_FRAMES,H,W,3) uint8
        wav = np.asarray(audio_mm[i]).astype(np.float32)  # (MAX_AUDIO_LEN,)
        L = int(audio_len[i])
        return frames, wav, L, int(y_int[i]), int(i)


def collate(batch):
    frames = np.stack([b[0] for b in batch])              # (B,N,H,W,3) uint8
    waves = [b[1][:max(1, b[2])] for b in batch]          # list of valid-length wavs
    a = audio_fe(waves, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    frames_t = torch.from_numpy(frames)                   # uint8
    ys = torch.tensor([b[3] for b in batch], dtype=torch.long)
    idxs = torch.tensor([b[4] for b in batch], dtype=torch.long)
    return (a.input_values, a.get("attention_mask", None), frames_t, ys, idxs)


def make_loader(indices, shuffle):
    return DataLoader(CremaDS(indices), batch_size=BATCH, shuffle=shuffle,
                      num_workers=NUM_WORKERS, collate_fn=collate,
                      pin_memory=(DEVICE == "cuda"), drop_last=False)


# (data loaders are built per-split inside run_split)


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

        # ----- temporal aggregator over per-frame ViT embeddings -----
        # All variants output a v_dim vector so the fusion head is unchanged.
        self.temporal = VIDEO_TEMPORAL
        if self.temporal == "lstm":
            self.frame_lstm = nn.LSTM(v_dim, v_dim // 2, batch_first=True,
                                      bidirectional=True)
        elif self.temporal == "attention":
            self.frame_attn = nn.Sequential(
                nn.LayerNorm(v_dim), nn.Linear(v_dim, v_dim // 2),
                nn.Tanh(), nn.Linear(v_dim // 2, 1),
            )

        self.head = nn.Sequential(
            nn.LayerNorm(a_dim + v_dim), nn.Dropout(0.3),
            nn.Linear(a_dim + v_dim, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, n_classes),
        )

    def _aggregate_frames(self, vseq):
        """vseq: (B, Nf, v_dim) -> (B, v_dim)."""
        if self.temporal == "lstm":
            out, _ = self.frame_lstm(vseq)          # (B, Nf, v_dim)
            return out.mean(dim=1)
        if self.temporal == "attention":
            scores = self.frame_attn(vseq)          # (B, Nf, 1)
            weights = torch.softmax(scores, dim=1)  # over frames
            return (vseq * weights).sum(dim=1)      # (B, v_dim)
        return vseq.mean(dim=1)                      # "mean"

    def forward(self, input_values, attn, frames_u8):
        # ----- audio: masked mean pool -----
        ao = self.audio(input_values, attention_mask=attn).last_hidden_state
        if attn is not None:
            fm = self.audio._get_feature_vector_attention_mask(
                ao.shape[1], attn).unsqueeze(-1).to(ao.dtype)
            a = (ao * fm).sum(1) / fm.sum(1).clamp(min=1.0)
        else:
            a = ao.mean(1)
        # ----- video: normalize uint8 -> ViT -> CLS per frame -> temporal agg -----
        B, Nf = frames_u8.shape[0], frames_u8.shape[1]
        px = frames_u8.permute(0, 1, 4, 2, 3).float() / 255.0          # (B,N,3,H,W)
        px = (px - self.img_mean) / self.img_std
        vo = self.video(px.flatten(0, 1)).last_hidden_state[:, 0]      # (B*N, v_dim)
        vseq = vo.view(B, Nf, -1)                                      # (B, N, v_dim)
        v = self._aggregate_frames(vseq.float())                      # (B, v_dim)
        return self.head(torch.cat([a, v], dim=1))


@torch.no_grad()
def evaluate(model, loader):
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
#  PART 2  —  FINE-TUNE + TEST FOR ONE SPLIT (returns metrics)       #
#                                                                    #
# ================================================================== #

def run_split(tag, title, train_idx, val_idx, test_idx):
    """Fine-tune + evaluate one split. `tag` prefixes all output files."""
    print("\n" + "=" * 70)
    print(f"RUN [{tag}] — {title}")
    print(f"  train {len(train_idx)} | val {len(val_idx)} | test {len(test_idx)}")
    print("=" * 70)

    train_loader = make_loader(train_idx, True)
    val_loader = make_loader(val_idx, False)
    test_loader = make_loader(test_idx, False)

    model = AVEmotionNet().to(DEVICE)

    # class weights (inverse frequency) on this split's train set
    cls_counts = np.bincount(y_int[train_idx], minlength=n_classes).astype(np.float64)
    cls_w = cls_counts.sum() / (n_classes * np.clip(cls_counts, 1, None))
    class_weights = torch.tensor(cls_w, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)

    # backbones get the small LR; head + temporal aggregator get the larger LR
    backbone_params = list(model.audio.parameters()) + list(model.video.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": LR_BACKBONE},
         {"params": head_params, "lr": LR_HEAD}],
        weight_decay=WEIGHT_DECAY)

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

    best_path = os.path.join(OUT_DIR, f"iter3_{tag}_model.pt")
    best_f1, bad_epochs, history = -1.0, 0, []

    # ---------------- training loop ----------------
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
                print(f"  [{tag}] epoch {epoch} step {bi}/{steps_per_epoch} "
                      f"loss={running / bi:.4f}")

        vp, vg, _ = evaluate(model, val_loader)
        vacc = accuracy_score(vg, vp)
        vf1 = f1_score(vg, vp, average="macro", zero_division=0)
        history.append((epoch, running / steps_per_epoch, vacc, vf1))
        print(f"  [{tag}] epoch {epoch} train_loss={running / steps_per_epoch:.4f} "
              f"val_acc={vacc:.3f} val_macroF1={vf1:.3f}  ({time.time() - t0:.0f}s)")

        if vf1 > best_f1:
            best_f1, bad_epochs = vf1, 0
            torch.save({"state_dict": model.state_dict(), "classes": class_labels,
                        "audio_model": AUDIO_MODEL, "video_model": VIDEO_MODEL,
                        "n_frames": N_FRAMES, "img_size": IMG_SIZE, "split": tag,
                        "video_temporal": VIDEO_TEMPORAL},
                       best_path)
            print(f"    ** [{tag}] new best (val macroF1={vf1:.3f}) -> {best_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"  [{tag}] early stop (no val improvement for {PATIENCE} epochs).")
                break

    # training curve
    hist = np.array(history)
    plt.figure(figsize=(7, 4.5))
    plt.plot(hist[:, 0], hist[:, 3], "-o", label="val macro-F1")
    plt.plot(hist[:, 0], hist[:, 2], "-s", label="val acc")
    plt.xlabel("epoch"); plt.ylabel("score"); plt.ylim(0, 1); plt.legend()
    plt.title(f"Fine-tuning val curve — {title}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"iter3_{tag}_val_curve.png"), dpi=140)
    plt.close()

    # ---------------- test with best checkpoint ----------------
    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    tp, tg, tix = evaluate(model, test_loader)
    acc = accuracy_score(tg, tp)
    prec = precision_score(tg, tp, average="macro", zero_division=0)
    rec = recall_score(tg, tp, average="macro", zero_division=0)
    f1m = f1_score(tg, tp, average="macro", zero_division=0)
    pcf1 = f1_score(tg, tp, average=None, labels=list(range(n_classes)), zero_division=0)
    print(f"  [{tag}] TEST acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} macroF1={f1m:.3f}")

    report = classification_report(tg, tp, target_names=class_labels, zero_division=0)
    with open(os.path.join(OUT_DIR, f"iter3_{tag}_results.txt"), "w") as fh:
        fh.write(f"CREMA-D FINE-TUNED ({title}) — Results\n")
        fh.write(f"audio={AUDIO_MODEL} video={VIDEO_MODEL} "
                 f"temporal={VIDEO_TEMPORAL} n_frames={N_FRAMES}\n")
        fh.write(f"split sizes: train {len(train_idx)} val {len(val_idx)} "
                 f"test {len(test_idx)}\n")
        fh.write(f"best val macroF1={best_f1:.3f}\n")
        fh.write(f"TEST acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} "
                 f"macroF1={f1m:.3f}\n\n{report}\n")

    cm = confusion_matrix(tg, tp, labels=list(range(n_classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    ax.set_title(f"Confusion Matrix — {title} (macroF1={f1m:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"iter3_{tag}_confusion_matrix_test.png"), dpi=140)
    plt.close(fig)

    plt.figure(figsize=(7.5, 4.5))
    bars = plt.bar(class_labels, pcf1, color="#8E7DBE")
    for b, s in zip(bars, pcf1):
        plt.text(b.get_x() + b.get_width() / 2, s, f"{s:.2f}", ha="center", va="bottom")
    plt.title(f"Per-Emotion F1 on TEST — {title}")
    plt.ylabel("F1"); plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"iter3_{tag}_per_class_f1_test.png"), dpi=140)
    plt.close()

    # full test-set predictions
    rows = [{"file": str(files[tix[k]]), "true": class_labels[tg[k]],
             "pred": class_labels[tp[k]]} for k in range(len(tix))]
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"iter3_{tag}_predictions.csv"), index=False)

    return {"tag": tag, "title": title, "best_val_f1": best_f1, "acc": acc,
            "prec": prec, "rec": rec, "macro_f1": f1m, "per_class_f1": pcf1}


# ================================================================== #
#                                                                    #
#  PART 3  —  RUN BOTH SPLITS + COMPARISON                          #
#                                                                    #
# ================================================================== #

SPLITS = [
    ("clip", "clip-level (stratified)", make_clip_split()),
    ("actor", "actor-level (speaker-independent)", make_actor_split()),
]

results = []
for tag, title, (tr_idx, va_idx, te_idx) in SPLITS:
    results.append(run_split(tag, title, tr_idx, va_idx, te_idx))

# ---------------- comparison report ----------------
print("\n" + "=" * 70)
print("COMPARISON — clip-level vs actor-level")
print("=" * 70)
hdr = f"{'split':36} {'val_F1':>7} {'test_acc':>9} {'test_F1':>8}"
print(hdr)
cmp_lines = ["CREMA-D Fine-tuned — split comparison",
             f"audio={AUDIO_MODEL} video={VIDEO_MODEL}", "=" * 64, hdr]
for r in results:
    line = (f"{r['title']:36} {r['best_val_f1']:7.3f} "
            f"{r['acc']:9.3f} {r['macro_f1']:8.3f}")
    print(line); cmp_lines.append(line)

cmp_lines += ["", "Per-class F1 (TEST):",
              f"{'emotion':10}" + "".join(f"{r['tag']:>12}" for r in results)]
for ci, lab in enumerate(class_labels):
    cmp_lines.append(f"{lab:10}" + "".join(f"{r['per_class_f1'][ci]:12.3f}" for r in results))
with open(os.path.join(OUT_DIR, "iter3_comparison.txt"), "w") as fh:
    fh.write("\n".join(cmp_lines) + "\n")

# overall grouped bar (test acc + macro-F1 per split)
xlab = [r["tag"] for r in results]
x = np.arange(len(results)); w = 0.35
plt.figure(figsize=(7, 4.5))
b1 = plt.bar(x - w / 2, [r["acc"] for r in results], w, label="test acc", color="#457B9D")
b2 = plt.bar(x + w / 2, [r["macro_f1"] for r in results], w, label="test macro-F1", color="#2A9D8F")
for bars in (b1, b2):
    for b in bars:
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(),
                 f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)
plt.xticks(x, xlab); plt.ylim(0, 1.0); plt.ylabel("score"); plt.legend()
plt.title("Fine-tuned: clip-level vs actor-level (TEST)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "iter3_compare_overall.png"), dpi=140)
plt.close()

# per-class grouped bar (clip vs actor)
x = np.arange(n_classes); w = 0.8 / max(1, len(results))
plt.figure(figsize=(9, 4.8))
colors = ["#8E7DBE", "#E76F51", "#2A9D8F", "#457B9D"]
for j, r in enumerate(results):
    plt.bar(x + (j - (len(results) - 1) / 2) * w, r["per_class_f1"], w,
            label=r["tag"], color=colors[j % len(colors)])
plt.xticks(x, class_labels); plt.ylim(0, 1.0); plt.ylabel("F1"); plt.legend()
plt.title("Per-Emotion F1 on TEST — clip vs actor")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "iter3_compare_per_class.png"), dpi=140)
plt.close()

print("\nSaved comparison -> iter3_comparison.txt, iter3_compare_overall.png, "
      "iter3_compare_per_class.png")
print("Per-split files: iter3_<tag>_model.pt, iter3_<tag>_results.txt, "
      "iter3_<tag>_val_curve.png,")
print("  iter3_<tag>_confusion_matrix_test.png, iter3_<tag>_per_class_f1_test.png, "
      "iter3_<tag>_predictions.csv")
print("\n" + "=" * 70)
print("DONE (fine-tuned, both splits). Outputs in:", os.path.abspath(OUT_DIR))
print("=" * 70)
