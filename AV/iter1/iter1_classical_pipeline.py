"""
================================================================================
CREMA-D EMOTION RECOGNITION — DUAL (AUDIO + VIDEO), SINGLE FILE, LINE-BY-LINE
================================================================================

Runs on the CREMA-D dataset:  https://github.com/CheyneyComputerScience/CREMA-D
Predicts the 6 emotions (anger, disgust, fear, happy, neutral, sad) by FUSING
two modalities for every clip:

    AUDIO  -> ~75 features (energy, spectral/MFCC, voice-activity, pitch)
    VIDEO  -> facial features per frame, aggregated per clip:
              EAR, MAR, gaze (H/V), blink/yawn rates,
              head pose (yaw/pitch/roll), yaw-velocity, attention score
              (MediaPipe Face Landmarker + solvePnP — the same logic as the
               Step 1 / Step 2 visual scripts, reused here)

Both feature sets are concatenated into ONE vector per clip ("classical
feature-level fusion") and a single classifier predicts the emotion.

This is a straight top-to-bottom script (NO functions) in THREE parts:

    PART 1 — FEATURE EXTRACTION (AUDIO + VIDEO)  -> iter1_features.csv
    PART 2 — TRAIN / VALIDATION / TEST + SAVE BEST MODEL (+ performance graphs)
    PART 3 — APPLY THE SAVED MODEL on held-out clips

--------------------------------------------------------------------------------
SETUP
    pip install numpy pandas scikit-learn matplotlib joblib librosa soundfile \
                praat-parselmouth opencv-python mediapipe

DATA (already on disk in the archive folder — see CONFIG below)
    AudioWAV/<id>.wav  and  VideoFlash/<id>.flv   (same <id> in both)

RUN
    python crema_emotion_pipeline.py
--------------------------------------------------------------------------------
"""

import os
import re
import glob
import math
import time
import warnings
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import librosa
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

warnings.filterwarnings("ignore")

# Parselmouth is optional (audio pitch features). Zeros if missing.
try:
    import parselmouth
    HAVE_PARSELMOUTH = True
except ImportError:
    print("parselmouth missing")
    sys.exit(0)

# OpenCV + MediaPipe are needed for the VIDEO half. If missing, the script
# still runs audio-only (video features become zeros) with a clear warning.
try:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    HAVE_VIDEO_LIBS = True
except Exception:
    print("OpenCV + MediaPipe missing")
    sys.exit(0)


# ================================================================== #
#  CONFIG  (edit these few variables, then just run the file)        #
# ================================================================== #

AUDIO_DIR  = r"C:\Users\shubh\Desktop\archive\content\CREMA-D\AudioWAV"     # INPUT: .wav
VIDEO_DIR  = r"C:\Users\shubh\Desktop\archive\content\CREMA-D\VideoFlash"   # INPUT: .flv
OUT_DIR    = r"C:\Users\shubh\Desktop\Hard disk\College(PG)\Non Academic at UCSD\Hackathon\Berkeley June 20-21\Actual Project\AV\iter1"  # OUTPUT + intermediates (this script's folder)

# MediaPipe Face Landmarker model file (download from:
#   https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task )
MODEL_PATH = r"C:\Users\shubh\Desktop\Hard disk\College(PG)\Academics at UCSD\Y1Q3\ECE 228 - Machine Learning for Physical Applications\Project\Codes\Video_Classifier\face_landmarker.task"

SAMPLE_RATE = 16000      # audio resample rate
FRAME_SKIP  = 3          # process every Nth video frame (higher = faster, less detail)
LIMIT       = 0          # 0 = all clips; e.g. 300 for a quick smoke test
RANDOM_SEED = 42

# Actor-level split fractions (must sum to 1.0)
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

# CREMA-D emotion code -> readable label
EMOTION_MAP = {
    "ANG": "anger", "DIS": "disgust", "FEA": "fear",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad",
}

# Head-pose 3D reference model (mm), origin at nose tip, + matching landmark ids
MODEL_3D = np.array([
    (  0.0,    0.0,   0.0),   # nose tip          (landmark 1)
    (  0.0, -330.0, -65.0),   # chin              (landmark 152)
    (-225.0,  170.0,-135.0),  # left eye corner   (landmark 263)
    ( 225.0,  170.0,-135.0),  # right eye corner  (landmark 33)
    (-150.0, -150.0,-125.0),  # left mouth corner (landmark 287)
    ( 150.0, -150.0,-125.0),  # right mouth corner(landmark 57)
], dtype=np.float64)
POSE_IDS  = [1, 152, 263, 33, 287, 57]
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS, RIGHT_IRIS = 468, 473

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)


# ================================================================== #
#                                                                    #
#  PART 1  —  FEATURE EXTRACTION  (AUDIO + VIDEO FUSION)              #
#                                                                    #
# ================================================================== #

print("=" * 70)
print("PART 1 — FEATURE EXTRACTION (AUDIO + VIDEO)")
print("=" * 70)

# --- locate audio files ---
wav_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
if not wav_files:
    raise FileNotFoundError(f"No .wav files found in {AUDIO_DIR}")
if LIMIT and LIMIT > 0:
    wav_files = wav_files[:LIMIT]

# --- map clip-id -> video path (.flv preferred, .mp4/.avi as fallback) ---
video_lookup = {}
for ext in ("*.flv", "*.mp4", "*.avi", "*.wmv"):
    for vp in glob.glob(os.path.join(VIDEO_DIR, ext)):
        video_lookup[Path(vp).stem] = vp

print(f"Found {len(wav_files)} audio clips and {len(video_lookup)} video files.")
if not HAVE_PARSELMOUTH:
    print("[i] parselmouth missing -> audio pitch features = 0 "
          "(pip install praat-parselmouth).")
VIDEO_ON = HAVE_VIDEO_LIBS and os.path.exists(MODEL_PATH)
if not HAVE_VIDEO_LIBS:
    print("[!] opencv/mediapipe missing -> VIDEO features = 0. "
          "pip install opencv-python mediapipe")
elif not os.path.exists(MODEL_PATH):
    print("[!] face_landmarker.task not found at MODEL_PATH -> VIDEO features = 0.")
else:
    print("[i] Video modality ENABLED (MediaPipe Face Landmarker + solvePnP).")

# --- build the MediaPipe detector once (single process, line-by-line) ---
detector = None
if VIDEO_ON:
    _base = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    _opts = mp_vision.FaceLandmarkerOptions(base_options=_base, num_faces=1,
                                            output_face_blendshapes=False,
                                            output_facial_transformation_matrixes=False)
    detector = mp_vision.FaceLandmarker.create_from_options(_opts)

features_csv = os.path.join(OUT_DIR, "iter1_features.csv")

if os.path.exists(features_csv):
    print(f"[i] Reusing cached features: {features_csv} (delete it to re-extract).")
    data = pd.read_csv(features_csv)
else:
    fname_re = re.compile(r"^(\d{4})_([A-Z]{3})_([A-Z]{3})_([A-Z]{2})", re.IGNORECASE)
    rows = []
    t0 = time.time()

    for idx, wav_path in enumerate(wav_files, 1):
        fname = Path(wav_path).name
        stem  = Path(wav_path).stem

        # ---- parse actor + emotion from the filename ----
        m = fname_re.match(stem)
        if not m:
            continue
        actor = m.group(1)
        emo_code = m.group(3).upper()
        if emo_code not in EMOTION_MAP:
            continue
        label = EMOTION_MAP[emo_code]

        feats = {"file": fname, "actor": actor, "label": label}

        # =====================================================
        #  AUDIO FEATURES
        # =====================================================
        try:
            y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        except Exception:
            continue
        if len(y) == 0:
            continue
        feats["aud_duration_sec"] = round(len(y) / sr, 2)

        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        feats["aud_rms_mean"] = float(np.mean(rms)); feats["aud_rms_std"] = float(np.std(rms))
        feats["aud_rms_max"]  = float(np.max(rms));  feats["aud_rms_min"] = float(np.min(rms))
        spk = np.mean(rms) + 2 * np.std(rms)
        feats["aud_energy_spike_count"] = int(np.sum(rms > spk))
        feats["aud_energy_spike_rate"]  = float(np.sum(rms > spk) / len(rms))

        cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        feats["aud_spectral_centroid_mean"] = float(np.mean(cent))
        feats["aud_spectral_centroid_std"]  = float(np.std(cent))
        feats["aud_spectral_bandwidth_mean"] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]))
        feats["aud_spectral_rolloff_mean"]   = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0]))
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        feats["aud_zcr_mean"] = float(np.mean(zcr)); feats["aud_zcr_std"] = float(np.std(zcr))

        mel_db = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=13), ref=np.max)
        for i in range(mel_db.shape[0]):
            feats[f"aud_mel_{i}_mean"] = float(np.mean(mel_db[i]))
            feats[f"aud_mel_{i}_std"]  = float(np.std(mel_db[i]))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(mfcc.shape[0]):
            feats[f"aud_mfcc_{i}_mean"] = float(np.mean(mfcc[i]))
            feats[f"aud_mfcc_{i}_std"]  = float(np.std(mfcc[i]))

        sth = np.percentile(rms, 25) + 0.5 * np.std(rms)
        sf = rms > sth
        feats["aud_speech_ratio"]  = float(np.sum(sf) / len(rms))
        feats["aud_silence_ratio"] = float(1.0 - feats["aud_speech_ratio"])
        lrun = 0; crun = 0
        for s in sf:
            if s:
                crun += 1; lrun = max(lrun, crun)
            else:
                crun = 0
        feats["aud_longest_speech_run_sec"] = round(lrun * (512 / sr), 2)
        feats["aud_speech_segment_count"] = int(np.sum(np.diff(sf.astype(int)) == 1))

        feats["aud_pitch_mean"] = feats["aud_pitch_std"] = 0.0
        feats["aud_pitch_range"] = feats["aud_voiced_ratio"] = 0.0
        feats["aud_intensity_mean"] = feats["aud_intensity_std"] = 0.0
        if HAVE_PARSELMOUTH:
            try:
                snd = parselmouth.Sound(wav_path)
                pv = snd.to_pitch().selected_array["frequency"]
                voiced = pv[pv > 0]
                if len(voiced) > 0:
                    feats["aud_pitch_mean"] = float(np.mean(voiced)); feats["aud_pitch_std"] = float(np.std(voiced))
                    feats["aud_pitch_range"] = float(np.max(voiced) - np.min(voiced))
                    feats["aud_voiced_ratio"] = float(len(voiced) / len(pv))
                iv = snd.to_intensity().values[0]
                feats["aud_intensity_mean"] = float(np.mean(iv)); feats["aud_intensity_std"] = float(np.std(iv))
            except Exception:
                pass

        # =====================================================
        #  VIDEO FEATURES  (per-frame -> aggregated per clip)
        # =====================================================
        # default zeros so columns always exist even with no video / no face
        ear_list, mar_list = [], []
        gh_list, gv_list = [], []
        yaw_list, pitch_list, roll_list, yawvel_list, attn_list = [], [], [], [], []
        blink_count = 0; yawn_count = 0
        eye_closed = False; mouth_open = False
        frames_seen = 0; frames_face = 0
        prev_yaw = None; prev_rvec = None; prev_tvec = None
        yaw_buf = deque(maxlen=5)

        vpath = video_lookup.get(stem)
        if VIDEO_ON and vpath is not None:
            cap = cv2.VideoCapture(vpath)
            if cap.isOpened():
                fw = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640.0
                fh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480.0
                cam = np.array([[fw, 0, fw / 2.0], [0, fw, fh / 2.0], [0, 0, 1.0]], dtype=np.float64)
                dist = np.zeros((4, 1), dtype=np.float64)
                fi = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if fi % FRAME_SKIP != 0:
                        fi += 1
                        continue
                    frames_seen += 1
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mpimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    res = detector.detect(mpimg)
                    if res.face_landmarks:
                        frames_face += 1
                        lms = res.face_landmarks[0]
                        P = np.array([[lm.x, lm.y] for lm in lms])  # normalized (478,2)

                        # ---- EAR (both eyes) ----
                        v1 = np.linalg.norm(P[160] - P[144]); v2 = np.linalg.norm(P[158] - P[153]); h = np.linalg.norm(P[33] - P[133])
                        ear_l = (v1 + v2) / (2.0 * h) if h != 0 else np.nan
                        v1 = np.linalg.norm(P[385] - P[380]); v2 = np.linalg.norm(P[387] - P[373]); h = np.linalg.norm(P[362] - P[263])
                        ear_r = (v1 + v2) / (2.0 * h) if h != 0 else np.nan
                        ear = np.nanmean([ear_l, ear_r])

                        # ---- MAR ----
                        m1 = np.linalg.norm(P[13] - P[14]); m2 = np.linalg.norm(P[81] - P[178]); m3 = np.linalg.norm(P[18] - P[17])
                        mh = np.linalg.norm(P[61] - P[291])
                        mar = (m1 + m2 + m3) / (2.0 * mh) if mh != 0 else np.nan

                        # ---- Gaze (both eyes, iris offset within eye box) ----
                        le = P[LEFT_EYE]; lir = P[LEFT_IRIS]
                        lminx, lminy = le.min(axis=0); lmaxx, lmaxy = le.max(axis=0)
                        gh_l = (lir[0] - lminx) / (lmaxx - lminx) if (lmaxx - lminx) != 0 else 0.5
                        gv_l = (lir[1] - lminy) / (lmaxy - lminy) if (lmaxy - lminy) != 0 else 0.5
                        re = P[RIGHT_EYE]; rir = P[RIGHT_IRIS]
                        rminx, rminy = re.min(axis=0); rmaxx, rmaxy = re.max(axis=0)
                        gh_r = (rir[0] - rminx) / (rmaxx - rminx) if (rmaxx - rminx) != 0 else 0.5
                        gv_r = (rir[1] - rminy) / (rmaxy - rminy) if (rmaxy - rminy) != 0 else 0.5
                        gh = (gh_l + gh_r) / 2.0; gv = (gv_l + gv_r) / 2.0

                        # ---- Blink / Yawn state machines ----
                        if ear < 0.2:
                            if not eye_closed:
                                blink_count += 1; eye_closed = True
                        else:
                            eye_closed = False
                        if mar > 0.5:
                            if not mouth_open:
                                yawn_count += 1; mouth_open = True
                        else:
                            mouth_open = False

                        # ---- Head pose via solvePnP (warm-started) ----
                        img_pts = np.array([[[lms[i].x * fw, lms[i].y * fh]] for i in POSE_IDS], dtype=np.float64)
                        use_guess = prev_rvec is not None
                        try:
                            ok, rvec, tvec = cv2.solvePnP(
                                MODEL_3D, img_pts, cam, dist,
                                rvec=prev_rvec.copy() if use_guess else None,
                                tvec=prev_tvec.copy() if use_guess else None,
                                useExtrinsicGuess=use_guess, flags=cv2.SOLVEPNP_ITERATIVE)
                        except Exception:
                            ok = False
                        if ok:
                            prev_rvec = rvec; prev_tvec = tvec
                            R, _ = cv2.Rodrigues(rvec)
                            sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
                            if sy > 1e-6:
                                roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
                                pitch = math.degrees(math.atan2(-R[2, 0], sy))
                                yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
                            else:
                                roll = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
                                pitch = math.degrees(math.atan2(-R[2, 0], sy)); yaw = 0.0
                            if prev_yaw is not None:
                                d = abs(yaw - prev_yaw)
                                if d > 180.0:
                                    d = 360.0 - d
                                yaw_buf.append(d)
                            prev_yaw = yaw
                            yawvel = float(np.mean(yaw_buf)) if yaw_buf else 0.0
                            ay, ap, T = abs(yaw), abs(pitch), 20.0
                            syaw = 1.0 if ay <= T else (0.0 if ay >= 2 * T else 1.0 - (ay - T) / T)
                            spit = 1.0 if ap <= T else (0.0 if ap >= 2 * T else 1.0 - (ap - T) / T)
                            attn = math.sqrt(max(syaw, 0.0) * max(spit, 0.0))
                            yaw_list.append(yaw); pitch_list.append(pitch); roll_list.append(roll)
                            yawvel_list.append(yawvel); attn_list.append(attn)

                        if not np.isnan(ear): ear_list.append(ear)
                        if not np.isnan(mar): mar_list.append(mar)
                        gh_list.append(gh); gv_list.append(gv)
                    fi += 1
                cap.release()

        # ---- aggregate the per-frame video metrics into per-clip features ----
        dur = max(feats["aud_duration_sec"], 1e-6)
        feats["vid_face_ratio"]   = float(frames_face / frames_seen) if frames_seen else 0.0
        feats["vid_ear_mean"]  = float(np.mean(ear_list)) if ear_list else 0.0
        feats["vid_ear_std"]   = float(np.std(ear_list))  if ear_list else 0.0
        feats["vid_ear_min"]   = float(np.min(ear_list))  if ear_list else 0.0
        feats["vid_ear_max"]   = float(np.max(ear_list))  if ear_list else 0.0
        feats["vid_mar_mean"]  = float(np.mean(mar_list)) if mar_list else 0.0
        feats["vid_mar_std"]   = float(np.std(mar_list))  if mar_list else 0.0
        feats["vid_mar_max"]   = float(np.max(mar_list))  if mar_list else 0.0
        feats["vid_gazeh_mean"] = float(np.mean(gh_list)) if gh_list else 0.0
        feats["vid_gazeh_std"]  = float(np.std(gh_list))  if gh_list else 0.0
        feats["vid_gazev_mean"] = float(np.mean(gv_list)) if gv_list else 0.0
        feats["vid_gazev_std"]  = float(np.std(gv_list))  if gv_list else 0.0
        feats["vid_yaw_mean"]   = float(np.mean(yaw_list)) if yaw_list else 0.0
        feats["vid_yaw_std"]    = float(np.std(yaw_list))  if yaw_list else 0.0
        feats["vid_pitch_mean"] = float(np.mean(pitch_list)) if pitch_list else 0.0
        feats["vid_pitch_std"]  = float(np.std(pitch_list))  if pitch_list else 0.0
        feats["vid_roll_mean"]  = float(np.mean(roll_list)) if roll_list else 0.0
        feats["vid_roll_std"]   = float(np.std(roll_list))  if roll_list else 0.0
        feats["vid_yawvel_mean"] = float(np.mean(yawvel_list)) if yawvel_list else 0.0
        feats["vid_yawvel_max"]  = float(np.max(yawvel_list))  if yawvel_list else 0.0
        feats["vid_attention_mean"] = float(np.mean(attn_list)) if attn_list else 0.0
        feats["vid_attention_std"]  = float(np.std(attn_list))  if attn_list else 0.0
        feats["vid_blink_count"] = int(blink_count)
        feats["vid_yawn_count"]  = int(yawn_count)
        feats["vid_blink_rate"]  = float(blink_count / dur)
        feats["vid_yawn_rate"]   = float(yawn_count / dur)

        rows.append(feats)
        if idx % 100 == 0 or idx == len(wav_files):
            print(f"  processed {idx}/{len(wav_files)}  ({time.time() - t0:.0f}s elapsed)")

    data = pd.DataFrame(rows)
    data.to_csv(features_csv, index=False)
    print(f"Extracted fused features for {len(data)} clips in "
          f"{time.time() - t0:.0f}s -> {features_csv}")

# release detector if it was created
if detector is not None:
    try:
        detector.close()
    except Exception:
        pass

# ---- dataset summary + class-distribution graph ----
meta_cols = ["file", "actor", "label"]
feature_cols = [c for c in data.columns if c not in meta_cols]
audio_cols = [c for c in feature_cols if c.startswith("aud_")]
video_cols = [c for c in feature_cols if c.startswith("vid_")]
print(f"\nDataset: {len(data)} clips | {len(audio_cols)} audio + "
      f"{len(video_cols)} video = {len(feature_cols)} features | "
      f"{data['actor'].nunique()} actors")
print("Emotion counts:")
print(data["label"].value_counts().to_string())
face_cov = (data["vid_face_ratio"] > 0).mean() if "vid_face_ratio" in data else 0.0
print(f"Clips with a detected face: {face_cov * 100:.1f}%")

plt.figure(figsize=(7, 4.5))
order = sorted(data["label"].unique())
counts = [int((data["label"] == lab).sum()) for lab in order]
bars = plt.bar(order, counts, color="#3F8EFC")
for b, c in zip(bars, counts):
    plt.text(b.get_x() + b.get_width() / 2, c, str(c), ha="center", va="bottom")
plt.title("CREMA-D — Clips per Emotion")
plt.ylabel("number of clips")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "iter1_class_distribution.png"), dpi=140)
plt.close()


# ================================================================== #
#                                                                    #
#  PART 2  —  TRAIN / VALIDATION / TEST  +  SAVE BEST MODEL           #
#                                                                    #
# ================================================================== #

print("\n" + "=" * 70)
print("PART 2 — TRAIN / VALIDATION / TEST  + SAVE BEST MODEL")
print("=" * 70)

# ---- clip-level stratified split: random clips into train/val/test ----
# NOTE: this is NOT speaker-independent -- the same actor can appear in more
# than one split. Scores are usually higher than an actor-level split.
n_actors = data["actor"].nunique()
X_all = data[feature_cols].fillna(0.0).values
y_all = data["label"].values
class_labels = sorted(np.unique(y_all).tolist())
all_idx = np.arange(len(data))

# split off TRAIN first, then divide the remainder into VAL and TEST
train_idx, tmp_idx = train_test_split(
    all_idx, train_size=TRAIN_FRAC, stratify=y_all, random_state=RANDOM_SEED)
rel_val = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
val_idx, test_idx = train_test_split(
    tmp_idx, train_size=rel_val, stratify=y_all[tmp_idx], random_state=RANDOM_SEED)

train_mask = np.zeros(len(data), dtype=bool); train_mask[train_idx] = True
val_mask = np.zeros(len(data), dtype=bool); val_mask[val_idx] = True
test_mask = np.zeros(len(data), dtype=bool); test_mask[test_idx] = True

X_train, y_train = X_all[train_idx], y_all[train_idx]
X_val, y_val = X_all[val_idx], y_all[val_idx]
X_test, y_test = X_all[test_idx], y_all[test_idx]

print("Split: clip-level stratified (NOT speaker-independent).")
print(f"Clips  -> train {len(y_train)} | val {len(y_val)} | test {len(y_test)}")

# ---- scale (fit on TRAIN only) ----
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# ---- candidate models; pick the best on VALIDATION (fused features) ----
candidate_models = {
    "LogisticRegression": LogisticRegression(max_iter=4000, class_weight="balanced"),
    "SVM_RBF": SVC(C=10.0, gamma="scale", kernel="rbf", class_weight="balanced",
                   probability=True, random_state=RANDOM_SEED),
    "RandomForest": RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                           n_jobs=-1, random_state=RANDOM_SEED),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, random_state=RANDOM_SEED),
    "MLP": MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-4,
                         max_iter=800, n_iter_no_change=15, random_state=RANDOM_SEED),
}

report_lines = []
report_lines.append("CREMA-D Emotion Recognition (AUDIO + VIDEO) — Results")
report_lines.append("=" * 55)
report_lines.append(f"Clips: {len(data)} | audio feats: {len(audio_cols)} | "
                    f"video feats: {len(video_cols)} | actors: {n_actors}")
report_lines.append(f"Split: clip-level stratified (NOT speaker-independent) -> "
                    f"train {len(y_train)} / val {len(y_val)} / test {len(y_test)}")
report_lines.append("")

model_names, val_f1_scores = [], []
best_name, best_val_f1, best_fitted = None, -1.0, None

print("\nTraining candidates on FUSED features (scored on validation):")
for name, model in candidate_models.items():
    model.fit(X_train_s, y_train)
    vp = model.predict(X_val_s)
    vacc = accuracy_score(y_val, vp)
    vf1 = f1_score(y_val, vp, average="macro", zero_division=0)
    model_names.append(name); val_f1_scores.append(vf1)
    print(f"  {name:22}  val_acc={vacc:.3f}  val_macroF1={vf1:.3f}")
    report_lines.append(f"[VAL fused] {name}: acc={vacc:.3f} macroF1={vf1:.3f}")
    if vf1 > best_val_f1:
        best_val_f1, best_name, best_fitted = vf1, name, model

print(f"\nBest model on validation: {best_name} (macro-F1 = {best_val_f1:.3f})")
report_lines.append("")
report_lines.append(f"BEST (validation macro-F1): {best_name} ({best_val_f1:.3f})")

# ---- graph: model comparison on validation ----
plt.figure(figsize=(8, 4.5))
colors = ["#F4A261" if n != best_name else "#2A9D8F" for n in model_names]
bars = plt.bar(model_names, val_f1_scores, color=colors)
for b, s in zip(bars, val_f1_scores):
    plt.text(b.get_x() + b.get_width() / 2, s, f"{s:.3f}", ha="center", va="bottom")
plt.title("Validation macro-F1 by Model — fused (green = best)")
plt.ylabel("macro-F1"); plt.ylim(0, max(val_f1_scores) * 1.2); plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "iter1_model_comparison.png"), dpi=140)
plt.close()

# ---- graph: modality comparison (audio-only vs video-only vs fused) ----
# Uses the best model family so the comparison is apples-to-apples.
ai = [feature_cols.index(c) for c in audio_cols]
vi = [feature_cols.index(c) for c in video_cols]
modality_scores = {}
for mod_name, cols_idx in [("audio", ai), ("video", vi), ("fused", list(range(len(feature_cols))))]:
    if len(cols_idx) == 0:
        continue
    sc = StandardScaler().fit(X_train[:, cols_idx])
    m = clone(candidate_models[best_name])
    m.fit(sc.transform(X_train[:, cols_idx]), y_train)
    mp_pred = m.predict(sc.transform(X_val[:, cols_idx]))
    modality_scores[mod_name] = f1_score(y_val, mp_pred, average="macro", zero_division=0)
    report_lines.append(f"[VAL {mod_name:5}] {best_name} macroF1={modality_scores[mod_name]:.3f}")

plt.figure(figsize=(6.5, 4.5))
mk = list(modality_scores.keys()); mv = [modality_scores[k] for k in mk]
bars = plt.bar(mk, mv, color=["#E76F51", "#457B9D", "#2A9D8F"][:len(mk)])
for b, s in zip(bars, mv):
    plt.text(b.get_x() + b.get_width() / 2, s, f"{s:.3f}", ha="center", va="bottom")
plt.title(f"Validation macro-F1 by Modality ({best_name})")
plt.ylabel("macro-F1"); plt.ylim(0, max(mv) * 1.2 if mv else 1)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "iter1_modality_comparison.png"), dpi=140)
plt.close()
print("Modality (val macro-F1):", {k: round(v, 3) for k, v in modality_scores.items()})

# ---- honest evaluation of the best fused model on the held-out TEST set ----
test_pred = best_fitted.predict(X_test_s)
test_acc = accuracy_score(y_test, test_pred)
test_prec = precision_score(y_test, test_pred, average="macro", zero_division=0)
test_rec = recall_score(y_test, test_pred, average="macro", zero_division=0)
test_f1 = f1_score(y_test, test_pred, average="macro", zero_division=0)
print("\n--- Held-out TEST performance (best fused model) ---")
print(f"  Accuracy : {test_acc:.3f}")
print(f"  Precision: {test_prec:.3f}  (macro)")
print(f"  Recall   : {test_rec:.3f}  (macro)")
print(f"  Macro-F1 : {test_f1:.3f}")
report_lines.append("")
report_lines.append("--- TEST (best fused model) ---")
report_lines.append(f"accuracy={test_acc:.3f} precision={test_prec:.3f} "
                    f"recall={test_rec:.3f} macroF1={test_f1:.3f}")
report_lines.append("")
report_lines.append("Per-class report (TEST):")
report_lines.append(classification_report(y_test, test_pred, zero_division=0))

# ---- graph: confusion matrix on TEST ----
cm = confusion_matrix(y_test, test_pred, labels=class_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
ax.set_title(f"Confusion Matrix — TEST ({best_name}, fused)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "iter1_confusion_matrix_test.png"), dpi=140)
plt.close(fig)

# ---- graph: per-class F1 on TEST ----
pcf1 = f1_score(y_test, test_pred, average=None, labels=class_labels, zero_division=0)
plt.figure(figsize=(7.5, 4.5))
bars = plt.bar(class_labels, pcf1, color="#8E7DBE")
for b, s in zip(bars, pcf1):
    plt.text(b.get_x() + b.get_width() / 2, s, f"{s:.2f}", ha="center", va="bottom")
plt.title(f"Per-Emotion F1 on TEST ({best_name}, fused)")
plt.ylabel("F1"); plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "iter1_per_class_f1_test.png"), dpi=140)
plt.close()

# ---- graph: feature importance (only if best model exposes it) ----
if hasattr(best_fitted, "feature_importances_"):
    imp = best_fitted.feature_importances_
    top = np.argsort(imp)[::-1][:25]
    # colour audio vs video bars differently to show the dual contribution
    bcolors = ["#457B9D" if feature_cols[i].startswith("vid_") else "#E76F51" for i in top][::-1]
    plt.figure(figsize=(8, 7))
    plt.barh(range(len(top)), imp[top][::-1], color=bcolors)
    plt.yticks(range(len(top)), [feature_cols[i] for i in top][::-1], fontsize=8)
    plt.xlabel("importance")
    plt.title(f"Top 25 Features — {best_name}  (blue=video, orange=audio)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "iter1_feature_importance.png"), dpi=140)
    plt.close()

# ---- retrain best on TRAIN + VAL, then save the deployable bundle ----
X_tv_s = np.vstack([X_train_s, X_val_s])
y_tv = np.concatenate([y_train, y_val])
final_model = clone(candidate_models[best_name])
final_model.fit(X_tv_s, y_tv)

model_path = os.path.join(OUT_DIR, "iter1_model.joblib")
joblib.dump({"model": final_model, "scaler": scaler, "features": feature_cols,
             "audio_cols": audio_cols, "video_cols": video_cols,
             "labels": class_labels, "sample_rate": SAMPLE_RATE,
             "emotion_map": EMOTION_MAP, "model_name": best_name}, model_path)
print(f"\nSaved best model bundle -> {model_path}")

report_path = os.path.join(OUT_DIR, "iter1_results.txt")
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))
print(f"Saved metrics report    -> {report_path}")
print("Saved graphs            -> 01, 02, 02b, 03, 04, 05 PNGs in", OUT_DIR)


# ================================================================== #
#                                                                    #
#  PART 3  —  APPLY THE SAVED MODEL                                   #
#                                                                    #
# ================================================================== #

print("\n" + "=" * 70)
print("PART 3 — APPLY THE SAVED MODEL ON HELD-OUT CLIPS")
print("=" * 70)

# Load the bundle back, exactly as a deployment script would.
bundle = joblib.load(model_path)
loaded_model = bundle["model"]
loaded_scaler = bundle["scaler"]
loaded_cols = bundle["features"]

# Take a few real clips from the held-out TEST actors and classify them using
# their already-extracted fused feature rows (audio + video together).
sample_idx = data.index[test_mask].to_numpy()
rng2 = np.random.default_rng(RANDOM_SEED)
if len(sample_idx) > 0:
    pick = rng2.choice(sample_idx, size=min(10, len(sample_idx)), replace=False)
else:
    pick = []

apply_rows = []
print("\nFused (audio+video) -> emotion on sample held-out clips:")
for i in pick:
    row = data.loc[i]
    x = row[loaded_cols].fillna(0.0).values.astype(float).reshape(1, -1)
    pred = loaded_model.predict(loaded_scaler.transform(x))[0]
    true_label = row["label"]
    mark = "OK " if pred == true_label else "x  "
    print(f"  {mark}{row['file']:30}  true={true_label:8}  pred={pred}")
    apply_rows.append({"file": row["file"], "true": true_label, "pred": pred})

apply_df = pd.DataFrame(apply_rows)
apply_df.to_csv(os.path.join(OUT_DIR, "iter1_predictions.csv"), index=False)

# Probability chart for the first sample clip.
if hasattr(loaded_model, "predict_proba") and len(apply_rows) > 0:
    first = apply_rows[0]["file"]
    r = data.loc[data["file"] == first, loaded_cols].fillna(0.0).values
    proba = loaded_model.predict_proba(loaded_scaler.transform(r))[0]
    classes = loaded_model.classes_
    order = np.argsort(proba)[::-1]
    plt.figure(figsize=(7.5, 4.5))
    bars = plt.bar([classes[i] for i in order], [proba[i] for i in order], color="#E76F51")
    for b, i in zip(bars, order):
        plt.text(b.get_x() + b.get_width() / 2, proba[i], f"{proba[i]:.2f}", ha="center", va="bottom")
    plt.title(f"Predicted Emotion Probabilities\n{first} (true={apply_rows[0]['true']})")
    plt.ylabel("probability"); plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "iter1_sample_prediction_proba.png"), dpi=140)
    plt.close()

print("\nSaved applied predictions -> iter1_predictions.csv")
print("\n" + "=" * 70)
print("DONE. All outputs are in:", os.path.abspath(OUT_DIR))
print("  iter1_features.csv, iter1_model.joblib, iter1_results.txt")
print("  iter1_class_distribution.png (Part 1)")
print("  iter1_model_comparison.png, iter1_modality_comparison.png,")
print("  iter1_confusion_matrix_test.png, iter1_per_class_f1_test.png,")
print("  iter1_feature_importance.png (Part 2)")
print("  iter1_sample_prediction_proba.png, iter1_predictions.csv (Part 3)")
print("=" * 70)
