"""
================================================================================
CREMA-D EMOTION RECOGNITION — DEEP EMBEDDINGS (AUDIO + VIDEO), SINGLE FILE
================================================================================

Same goal as crema_emotion_pipeline.py (predict the 6 CREMA-D emotions), but
instead of hand-crafted features it uses PRETRAINED DEEP EMBEDDINGS:

    AUDIO  -> wav2vec2 (speech model) -> mean-pooled hidden state  (768-d)
    VIDEO  -> ViT face-expression model on sampled frames -> pooled (768-d)

Both embeddings are concatenated into ONE vector per clip (feature-level
fusion) and a single classifier head learns BOTH modalities jointly.

Three parts, line by line:

    PART 1 — DEEP EMBEDDING EXTRACTION   -> iter2_embeddings.npz  (cached)
    PART 2 — TRAIN / VALIDATION / TEST + SAVE BEST HEAD (+ graphs)
    PART 3 — APPLY THE SAVED MODEL on held-out clips

--------------------------------------------------------------------------------
SETUP
    pip install torch torchvision transformers pillow \
                numpy pandas scikit-learn matplotlib joblib librosa soundfile \
                opencv-python

NOTE
    - First run downloads the two pretrained models from Hugging Face (internet
      required once; afterwards they are cached locally).
    - CPU works but the video ViT pass is the slow part. Embeddings are cached
      to iter2_embeddings.npz, so subsequent runs train in seconds.
    - To check for a GPU:  python -c "import torch; print(torch.cuda.is_available())"
--------------------------------------------------------------------------------
"""

import os
import re
import glob
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---- heavy deps (clear message if missing) ----
try:
    import cv2
    from PIL import Image
    import torch
    from transformers import AutoModel, AutoFeatureExtractor, AutoImageProcessor
except Exception as e:
    print("Missing deep-learning dependencies:", e)
    print("Install:  pip install torch torchvision transformers pillow opencv-python")
    raise SystemExit(0)


# ================================================================== #
#  CONFIG                                                            #
# ================================================================== #

AUDIO_DIR = r"C:\Users\shubh\Desktop\archive\content\CREMA-D\AudioWAV"
VIDEO_DIR = r"C:\Users\shubh\Desktop\archive\content\CREMA-D\VideoFlash"
OUT_DIR   = r"C:\Users\shubh\Desktop\Hard disk\College(PG)\Non Academic at UCSD\Hackathon\Berkeley June 20-21\Actual Project\AV\iter2"

# Pretrained models (downloaded once from Hugging Face).
# Audio: a wav2vec2 fine-tuned for speech emotion -> emotion-rich embeddings.
AUDIO_MODEL = "superb/wav2vec2-base-superb-er"
# Video: a ViT fine-tuned on facial-expression recognition.
VIDEO_MODEL = "trpakov/vit-face-expression"

SAMPLE_RATE = 16000     # wav2vec2 expects 16 kHz
N_FRAMES    = 6         # video frames sampled per clip (higher = slower, more detail)
LIMIT       = 0         # 0 = all clips; e.g. 300 for a quick smoke test
RANDOM_SEED = 42

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

EMOTION_MAP = {
    "ANG": "anger", "DIS": "disgust", "FEA": "fear",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad",
}

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================================================================== #
#                                                                    #
#  PART 1  —  DEEP EMBEDDING EXTRACTION                              #
#                                                                    #
# ================================================================== #

print("=" * 70)
print("PART 1 — DEEP EMBEDDING EXTRACTION (AUDIO wav2vec2 + VIDEO ViT)")
print("=" * 70)
print(f"Device: {DEVICE}")

wav_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
if not wav_files:
    raise FileNotFoundError(f"No .wav files found in {AUDIO_DIR}")
if LIMIT and LIMIT > 0:
    wav_files = wav_files[:LIMIT]

video_lookup = {}
for ext in ("*.flv", "*.mp4", "*.avi", "*.wmv"):
    for vp in glob.glob(os.path.join(VIDEO_DIR, ext)):
        video_lookup[Path(vp).stem] = vp
print(f"Found {len(wav_files)} audio clips and {len(video_lookup)} video files.")

emb_path = os.path.join(OUT_DIR, "iter2_embeddings.npz")

if os.path.exists(emb_path):
    print(f"[i] Reusing cached embeddings: {emb_path} (delete it to re-extract).")
    cache = np.load(emb_path, allow_pickle=True)
    X_all = cache["X"]
    y_all = cache["y"]
    actors_all = cache["actors"]
    files_all = cache["files"]
    A_DIM = int(cache["a_dim"])
    V_DIM = int(cache["v_dim"])
else:
    # ---- load the two pretrained models once ----
    print("Loading pretrained models (first run downloads them)...")
    audio_fe = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL)
    audio_model = AutoModel.from_pretrained(AUDIO_MODEL).to(DEVICE).eval()
    video_proc = AutoImageProcessor.from_pretrained(VIDEO_MODEL)
    video_model = AutoModel.from_pretrained(VIDEO_MODEL).to(DEVICE).eval()

    fname_re = re.compile(r"^(\d{4})_([A-Z]{3})_([A-Z]{3})_([A-Z]{2})", re.IGNORECASE)
    X_list, y_list, actor_list, file_list = [], [], [], []
    A_DIM = V_DIM = None
    t0 = time.time()

    for idx, wav_path in enumerate(wav_files, 1):
        fname = Path(wav_path).name
        stem = Path(wav_path).stem
        m = fname_re.match(stem)
        if not m:
            continue
        actor = m.group(1)
        emo_code = m.group(3).upper()
        if emo_code not in EMOTION_MAP:
            continue
        label = EMOTION_MAP[emo_code]

        # ---------- AUDIO embedding (wav2vec2, mean-pooled) ----------
        try:
            y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        except Exception:
            continue
        if len(y) == 0:
            continue
        a_inputs = audio_fe(y, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            a_out = audio_model(a_inputs.input_values.to(DEVICE))
        a_vec = a_out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()  # (768,)

        # ---------- VIDEO embedding (ViT over sampled frames) ----------
        v_vec = None
        vpath = video_lookup.get(stem)
        if vpath is not None:
            cap = cv2.VideoCapture(vpath)
            if cap.isOpened():
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                if total > 0:
                    grab = np.linspace(0, total - 1, N_FRAMES).astype(int)
                else:
                    grab = np.arange(N_FRAMES)
                frames = []
                for gi in grab:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(gi))
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb))
                cap.release()
                if frames:
                    v_inputs = video_proc(images=frames, return_tensors="pt")
                    with torch.no_grad():
                        v_out = video_model(v_inputs.pixel_values.to(DEVICE))
                    # use the CLS token from last_hidden_state (the genuine
                    # pretrained representation). NOTE: do NOT use pooler_output
                    # here -- this checkpoint has no pooler, so transformers
                    # initialises it randomly (see "MISSING pooler.*" load report).
                    v_frame = v_out.last_hidden_state[:, 0]      # CLS token (N, 768)
                    v_vec = v_frame.mean(dim=0).cpu().numpy()    # (768,)

        if A_DIM is None:
            A_DIM = a_vec.shape[0]
        if v_vec is None:
            # set V_DIM once we know it; default to audio model dim if never seen yet
            v_vec = np.zeros(V_DIM if V_DIM is not None else A_DIM, dtype=np.float32)
        if V_DIM is None:
            V_DIM = v_vec.shape[0]

        X_list.append(np.concatenate([a_vec, v_vec]).astype(np.float32))
        y_list.append(label); actor_list.append(actor); file_list.append(fname)

        if idx % 100 == 0 or idx == len(wav_files):
            print(f"  processed {idx}/{len(wav_files)}  ({time.time() - t0:.0f}s elapsed)")

    X_all = np.vstack(X_list)
    y_all = np.array(y_list)
    actors_all = np.array(actor_list)
    files_all = np.array(file_list)
    np.savez_compressed(emb_path, X=X_all, y=y_all, actors=actors_all,
                        files=files_all, a_dim=A_DIM, v_dim=V_DIM)
    print(f"Extracted deep embeddings for {len(X_all)} clips in "
          f"{time.time() - t0:.0f}s -> {emb_path}")

audio_idx = list(range(A_DIM))
video_idx = list(range(A_DIM, A_DIM + V_DIM))
class_labels = sorted(np.unique(y_all).tolist())
print(f"\nDataset: {len(X_all)} clips | audio emb {A_DIM} + video emb {V_DIM} = "
      f"{X_all.shape[1]} dims | {len(np.unique(actors_all))} actors")
print("Emotion counts:")
for lab in class_labels:
    print(f"  {lab:8} {int(np.sum(y_all == lab))}")

# class-distribution graph
plt.figure(figsize=(7, 4.5))
counts = [int(np.sum(y_all == lab)) for lab in class_labels]
bars = plt.bar(class_labels, counts, color="#3F8EFC")
for b, c in zip(bars, counts):
    plt.text(b.get_x() + b.get_width() / 2, c, str(c), ha="center", va="bottom")
plt.title("CREMA-D — Clips per Emotion")
plt.ylabel("number of clips")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "iter2_class_distribution.png"), dpi=140)
plt.close()


# ================================================================== #
#                                                                    #
#  PART 2  —  TRAIN / VALIDATION / TEST  +  SAVE BEST HEAD            #
#                                                                    #
# ================================================================== #

print("\n" + "=" * 70)
print("PART 2 — TRAIN / VALIDATION / TEST  + SAVE BEST HEAD")
print("=" * 70)

# ---- clip-level stratified split (matches the classical pipeline) ----
all_idx = np.arange(len(X_all))
train_idx, tmp_idx = train_test_split(
    all_idx, train_size=TRAIN_FRAC, stratify=y_all, random_state=RANDOM_SEED)
rel_val = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
val_idx, test_idx = train_test_split(
    tmp_idx, train_size=rel_val, stratify=y_all[tmp_idx], random_state=RANDOM_SEED)

test_mask = np.zeros(len(X_all), dtype=bool); test_mask[test_idx] = True

X_train, y_train = X_all[train_idx], y_all[train_idx]
X_val, y_val = X_all[val_idx], y_all[val_idx]
X_test, y_test = X_all[test_idx], y_all[test_idx]
print("Split: clip-level stratified (NOT speaker-independent).")
print(f"Clips -> train {len(y_train)} | val {len(y_val)} | test {len(y_test)}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# ---- classifier heads on top of the deep embeddings ----
candidate_models = {
    "LogisticRegression": LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced"),
    "SVM_RBF": SVC(C=10.0, gamma="scale", kernel="rbf", class_weight="balanced",
                   probability=True, random_state=RANDOM_SEED),
    "MLP": MLPClassifier(hidden_layer_sizes=(512, 256), alpha=1e-4,
                         max_iter=800, n_iter_no_change=20, random_state=RANDOM_SEED),
}

report_lines = ["CREMA-D Emotion (DEEP EMBEDDINGS, audio+video) — Results", "=" * 55,
                f"Clips: {len(X_all)} | audio emb: {A_DIM} | video emb: {V_DIM}",
                f"Audio model: {AUDIO_MODEL}", f"Video model: {VIDEO_MODEL}",
                f"Split: clip-level stratified -> train {len(y_train)} / "
                f"val {len(y_val)} / test {len(y_test)}", ""]

model_names, val_f1_scores = [], []
best_name, best_val_f1, best_fitted = None, -1.0, None
print("\nTraining heads on FUSED deep embeddings (scored on validation):")
for name, model in candidate_models.items():
    model.fit(X_train_s, y_train)
    vp = model.predict(X_val_s)
    vacc = accuracy_score(y_val, vp)
    vf1 = f1_score(y_val, vp, average="macro", zero_division=0)
    model_names.append(name); val_f1_scores.append(vf1)
    print(f"  {name:20}  val_acc={vacc:.3f}  val_macroF1={vf1:.3f}")
    report_lines.append(f"[VAL] {name}: acc={vacc:.3f} macroF1={vf1:.3f}")
    if vf1 > best_val_f1:
        best_val_f1, best_name, best_fitted = vf1, name, model

print(f"\nBest head on validation: {best_name} (macro-F1 = {best_val_f1:.3f})")
report_lines += ["", f"BEST (validation macro-F1): {best_name} ({best_val_f1:.3f})"]

# graph: model comparison
plt.figure(figsize=(7, 4.5))
colors = ["#F4A261" if n != best_name else "#2A9D8F" for n in model_names]
bars = plt.bar(model_names, val_f1_scores, color=colors)
for b, s in zip(bars, val_f1_scores):
    plt.text(b.get_x() + b.get_width() / 2, s, f"{s:.3f}", ha="center", va="bottom")
plt.title("Validation macro-F1 by Head — deep fused (green = best)")
plt.ylabel("macro-F1"); plt.ylim(0, max(val_f1_scores) * 1.2); plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "iter2_model_comparison.png"), dpi=140)
plt.close()

# graph: modality comparison (audio-only vs video-only vs fused)
modality_scores = {}
for mod_name, cols_idx in [("audio", audio_idx), ("video", video_idx),
                           ("fused", list(range(X_all.shape[1])))]:
    sc = StandardScaler().fit(X_train[:, cols_idx])
    mdl = clone(candidate_models[best_name])
    mdl.fit(sc.transform(X_train[:, cols_idx]), y_train)
    pred = mdl.predict(sc.transform(X_val[:, cols_idx]))
    modality_scores[mod_name] = f1_score(y_val, pred, average="macro", zero_division=0)
    report_lines.append(f"[VAL {mod_name:5}] {best_name} macroF1={modality_scores[mod_name]:.3f}")
plt.figure(figsize=(6.5, 4.5))
mk = list(modality_scores.keys()); mv = [modality_scores[k] for k in mk]
bars = plt.bar(mk, mv, color=["#E76F51", "#457B9D", "#2A9D8F"])
for b, s in zip(bars, mv):
    plt.text(b.get_x() + b.get_width() / 2, s, f"{s:.3f}", ha="center", va="bottom")
plt.title(f"Validation macro-F1 by Modality ({best_name}, deep)")
plt.ylabel("macro-F1"); plt.ylim(0, max(mv) * 1.2)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "iter2_modality_comparison.png"), dpi=140)
plt.close()
print("Modality (val macro-F1):", {k: round(v, 3) for k, v in modality_scores.items()})

# ---- honest TEST evaluation ----
test_pred = best_fitted.predict(X_test_s)
test_acc = accuracy_score(y_test, test_pred)
test_prec = precision_score(y_test, test_pred, average="macro", zero_division=0)
test_rec = recall_score(y_test, test_pred, average="macro", zero_division=0)
test_f1 = f1_score(y_test, test_pred, average="macro", zero_division=0)
print("\n--- Held-out TEST performance (best deep head) ---")
print(f"  Accuracy : {test_acc:.3f}")
print(f"  Precision: {test_prec:.3f}  (macro)")
print(f"  Recall   : {test_rec:.3f}  (macro)")
print(f"  Macro-F1 : {test_f1:.3f}")
report_lines += ["", "--- TEST (best deep head) ---",
                 f"accuracy={test_acc:.3f} precision={test_prec:.3f} "
                 f"recall={test_rec:.3f} macroF1={test_f1:.3f}", "",
                 "Per-class report (TEST):",
                 classification_report(y_test, test_pred, zero_division=0)]

# graph: confusion matrix
cm = confusion_matrix(y_test, test_pred, labels=class_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
ax.set_title(f"Confusion Matrix — TEST ({best_name}, deep)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "iter2_confusion_matrix_test.png"), dpi=140)
plt.close(fig)

# graph: per-class F1
pcf1 = f1_score(y_test, test_pred, average=None, labels=class_labels, zero_division=0)
plt.figure(figsize=(7.5, 4.5))
bars = plt.bar(class_labels, pcf1, color="#8E7DBE")
for b, s in zip(bars, pcf1):
    plt.text(b.get_x() + b.get_width() / 2, s, f"{s:.2f}", ha="center", va="bottom")
plt.title(f"Per-Emotion F1 on TEST ({best_name}, deep)")
plt.ylabel("F1"); plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "iter2_per_class_f1_test.png"), dpi=140)
plt.close()

# ---- retrain best on train+val and save ----
X_tv_s = np.vstack([X_train_s, X_val_s]); y_tv = np.concatenate([y_train, y_val])
final_model = clone(candidate_models[best_name])
final_model.fit(X_tv_s, y_tv)
model_path = os.path.join(OUT_DIR, "iter2_model.joblib")
joblib.dump({"model": final_model, "scaler": scaler, "a_dim": A_DIM, "v_dim": V_DIM,
             "labels": class_labels, "audio_model": AUDIO_MODEL,
             "video_model": VIDEO_MODEL, "model_name": best_name}, model_path)
print(f"\nSaved best deep model bundle -> {model_path}")
report_path = os.path.join(OUT_DIR, "iter2_results.txt")
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))
print(f"Saved metrics report         -> {report_path}")


# ================================================================== #
#                                                                    #
#  PART 3  —  APPLY THE SAVED MODEL                                  #
#                                                                    #
# ================================================================== #

print("\n" + "=" * 70)
print("PART 3 — APPLY THE SAVED MODEL ON HELD-OUT CLIPS")
print("=" * 70)

bundle = joblib.load(model_path)
loaded_model = bundle["model"]
loaded_scaler = bundle["scaler"]

rng2 = np.random.default_rng(RANDOM_SEED)
pick = rng2.choice(test_idx, size=min(10, len(test_idx)), replace=False) if len(test_idx) else []

apply_rows = []
print("\nFused deep embeddings -> emotion on sample held-out clips:")
for i in pick:
    x = loaded_scaler.transform(X_all[i].reshape(1, -1))
    pred = loaded_model.predict(x)[0]
    true_label = y_all[i]
    mark = "OK " if pred == true_label else "x  "
    print(f"  {mark}{str(files_all[i]):30}  true={true_label:8}  pred={pred}")
    apply_rows.append({"file": str(files_all[i]), "true": true_label, "pred": pred})

pd.DataFrame(apply_rows).to_csv(os.path.join(OUT_DIR, "iter2_predictions.csv"), index=False)

if hasattr(loaded_model, "predict_proba") and len(apply_rows) > 0:
    i0 = pick[0]
    proba = loaded_model.predict_proba(loaded_scaler.transform(X_all[i0].reshape(1, -1)))[0]
    classes = loaded_model.classes_
    order = np.argsort(proba)[::-1]
    plt.figure(figsize=(7.5, 4.5))
    bars = plt.bar([classes[k] for k in order], [proba[k] for k in order], color="#E76F51")
    for b, k in zip(bars, order):
        plt.text(b.get_x() + b.get_width() / 2, proba[k], f"{proba[k]:.2f}", ha="center", va="bottom")
    plt.title(f"Predicted Emotion Probabilities\n{str(files_all[i0])} (true={y_all[i0]})")
    plt.ylabel("probability"); plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "iter2_sample_prediction_proba.png"), dpi=140)
    plt.close()

print("\nSaved applied predictions -> iter2_predictions.csv")
print("\n" + "=" * 70)
print("DONE. Deep-embedding outputs in:", os.path.abspath(OUT_DIR))
print("  iter2_embeddings.npz, iter2_model.joblib, iter2_results.txt")
print("  iter2_class_distribution.png, iter2_model_comparison.png,")
print("  iter2_modality_comparison.png, iter2_confusion_matrix_test.png,")
print("  iter2_per_class_f1_test.png, iter2_sample_prediction_proba.png")
print("  iter2_predictions.csv")
print("=" * 70)
