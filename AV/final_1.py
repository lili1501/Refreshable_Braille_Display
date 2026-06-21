"""
================================================================================
FINAL_1 — TRAIN THE AUDIO+VIDEO EMOTION MODEL AND SAVE IT
================================================================================

This script does ONE job: fine-tune the proven CREMA-D emotion model
(wav2vec2 audio + ViT video, mean-pooled, fused) and save the trained
checkpoint to disk. Use `final_2.py` afterwards to apply it to a recorded
video + audio and score it against a ground-truth file.

    PART 1 — one-time cache of frames + audio
    PART 2 — fine-tune (train/val), save best checkpoint by val macro-F1
    PART 3 — quick TEST metric (sanity check)

The saved checkpoint (`final_output/final_model.pt`) stores the weights AND all
config needed to rebuild the model for inference (classes, backbones, n_frames,
img_size, sample_rate, max_audio_s).

--------------------------------------------------------------------------------
DEPENDENCIES
    pip install torch torchvision transformers pillow opencv-python==4.10.0.84 \
                librosa soundfile scikit-learn joblib pandas "numpy<2"
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
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModel, AutoFeatureExtractor, AutoImageProcessor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import cv2
warnings.filterwarnings("ignore")


# ================================================================== #
#  CONFIG                                                            #
# ================================================================== #

AUDIO_DIR = r"C:\Users\shubh\Desktop\archive\content\CREMA-D\AudioWAV"
VIDEO_DIR = r"C:\Users\shubh\Desktop\archive\content\CREMA-D\VideoFlash"
OUT_DIR   = r"C:\Users\shubh\Desktop\Hard disk\College(PG)\Non Academic at UCSD\Hackathon\Berkeley June 20-21\Actual Project\AV\final_output"

# Proven config (TEST macro-F1 = 0.873 on the clip-level split).
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
NUM_WORKERS  = 0           # Windows: 0 ; Linux: raise to #cores
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
MODEL_PATH = os.path.join(OUT_DIR, "final_model.pt")


# ================================================================== #
#  PRE-PROCESSORS                                                    #
# ================================================================== #

audio_fe = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL)
img_proc = AutoImageProcessor.from_pretrained(VIDEO_MODEL)
IMG_MEAN = torch.tensor(img_proc.image_mean).view(1, 1, 3, 1, 1)
IMG_STD  = torch.tensor(img_proc.image_std).view(1, 1, 3, 1, 1)


def frames_from_video(vpath, n_frames=N_FRAMES):
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
        got.append(cv2.resize(fr, (IMG_SIZE, IMG_SIZE)))
    cap.release()
    for k in range(n_frames):
        if got:
            frames[k] = got[min(k, len(got) - 1)]
    return frames


def audio_from_path(path):
    try:
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    except Exception:
        y = np.zeros(SAMPLE_RATE, dtype=np.float32)
    wav = np.zeros(MAX_AUDIO_LEN, dtype=np.float32)
    L = int(min(len(y), MAX_AUDIO_LEN))
    wav[:L] = y[:L]
    return wav, max(1, L)


# ================================================================== #
#  PART 1 — BUILD/LOAD CACHE                                         #
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

frames_path = os.path.join(OUT_DIR, "final_frames_u8.npy")
audio_path  = os.path.join(OUT_DIR, "final_audio_f16.npy")
alen_path   = os.path.join(OUT_DIR, "final_audio_len.npy")
meta_path   = os.path.join(OUT_DIR, "final_meta.npz")


def _decode_one(rec):
    wav_path, vpath, _a, _l, _f = rec
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

if os.path.exists(frames_path) and os.path.exists(audio_path) and os.path.exists(meta_path):
    print(f"[i] Reusing cache in {OUT_DIR} (delete final_*.npy to rebuild).")
    frames_mm = np.load(frames_path, mmap_mode="r")
    audio_mm  = np.load(audio_path, mmap_mode="r")
    audio_len = np.load(alen_path)
else:
    print("Building cache (one-time decode)...")
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
        frames_mm[i] = f; audio_mm[i] = w; audio_len[i] = L
        if cnt % 200 == 0 or cnt == N:
            print(f"  cached {cnt}/{N}  ({time.time() - t0:.0f}s)")
    frames_mm.flush(); audio_mm.flush()
    np.save(alen_path, audio_len)
    np.savez(meta_path, labels=labels)
    print(f"Cache built in {time.time() - t0:.0f}s")


# ---- split + encode ----
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


class CremaDS(Dataset):
    def __init__(self, indices): self.indices = indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, j):
        i = self.indices[j]
        frames = np.asarray(frames_mm[i])
        wav = np.asarray(audio_mm[i]).astype(np.float32)
        return frames, wav, int(audio_len[i]), int(y_int[i])


def collate(batch):
    frames = np.stack([b[0] for b in batch])
    waves = [b[1][:max(1, b[2])] for b in batch]
    a = audio_fe(waves, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    ys = torch.tensor([b[3] for b in batch], dtype=torch.long)
    return a.input_values, a.get("attention_mask", None), torch.from_numpy(frames), ys


def make_loader(indices, shuffle):
    return DataLoader(CremaDS(indices), batch_size=BATCH, shuffle=shuffle,
                      num_workers=NUM_WORKERS, collate_fn=collate,
                      pin_memory=(DEVICE == "cuda"))


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
    preds, gts = [], []
    for iv, attn, fr, ys in loader:
        iv = iv.to(DEVICE)
        attn = attn.to(DEVICE) if attn is not None else None
        fr = fr.to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=(DEVICE == "cuda")):
            logits = model(iv, attn, fr)
        preds.append(logits.argmax(1).cpu().numpy()); gts.append(ys.numpy())
    return np.concatenate(preds), np.concatenate(gts)


# ================================================================== #
#  PART 2 — TRAIN + SAVE                                             #
# ================================================================== #

print("\n" + "=" * 70)
print("PART 2 — FINE-TUNE + SAVE")
print("=" * 70)

train_loader = make_loader(train_idx, True)
val_loader = make_loader(val_idx, False)

cls_counts = np.bincount(y_int[train_idx], minlength=n_classes).astype(np.float64)
cls_w = cls_counts.sum() / (n_classes * np.clip(cls_counts, 1, None))
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(cls_w, dtype=torch.float32, device=DEVICE),
    label_smoothing=LABEL_SMOOTH)

optimizer = torch.optim.AdamW(
    [{"params": list(model.audio.parameters()) + list(model.video.parameters()), "lr": LR_BACKBONE},
     {"params": list(model.head.parameters()), "lr": LR_HEAD}], weight_decay=WEIGHT_DECAY)

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


def save_checkpoint():
    torch.save({
        "state_dict": model.state_dict(),
        "classes": class_labels,
        "audio_model": AUDIO_MODEL, "video_model": VIDEO_MODEL,
        "n_frames": N_FRAMES, "img_size": IMG_SIZE,
        "sample_rate": SAMPLE_RATE, "max_audio_s": MAX_AUDIO_S,
    }, MODEL_PATH)


best_f1, bad_epochs = -1.0, 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    t0 = time.time()
    running = 0.0
    for bi, (iv, attn, fr, ys) in enumerate(train_loader, 1):
        iv = iv.to(DEVICE); attn = attn.to(DEVICE) if attn is not None else None
        fr = fr.to(DEVICE); ys = ys.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=(DEVICE == "cuda")):
            loss = criterion(model(iv, attn, fr), ys)
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

    vp, vg = evaluate(val_loader)
    vacc = accuracy_score(vg, vp)
    vf1 = f1_score(vg, vp, average="macro", zero_division=0)
    print(f"[epoch {epoch}] train_loss={running / steps_per_epoch:.4f} "
          f"val_acc={vacc:.3f} val_macroF1={vf1:.3f}  ({time.time() - t0:.0f}s)")

    if vf1 > best_f1:
        best_f1, bad_epochs = vf1, 0
        save_checkpoint()
        print(f"  ** new best (val macroF1={vf1:.3f}) saved -> {MODEL_PATH}")
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print(f"Early stopping (no val improvement for {PATIENCE} epochs).")
            break


# ================================================================== #
#  PART 3 — QUICK TEST METRIC                                        #
# ================================================================== #

print("\n" + "=" * 70)
print("PART 3 — TEST (sanity check)")
print("=" * 70)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["state_dict"])
tp, tg = evaluate(make_loader(test_idx, False))
print(f"  TEST accuracy : {accuracy_score(tg, tp):.3f}")
print(f"  TEST macro-F1 : {f1_score(tg, tp, average='macro', zero_division=0):.3f}")
print(f"\nDONE. Trained model saved at:\n  {MODEL_PATH}")
print("Use final_2.py to apply it to a recorded video + audio.")
