"""
emotion_inference.py  -  Raspberry Pi Emotion Recognition Service
=================================================================
Runs on the Raspberry Pi connected to:
  - USB/Pi camera (video capture)
  - USB/I2S microphone (audio capture)
  - ESP32-S3 via UART serial (sends emotion label)

Pipeline (runs in a loop during Face-to-Face mode):
  1. Record WINDOW_S seconds of audio + sample N_FRAMES video frames
  2. Run the trained AVEmotionNet (wav2vec2 + ViT, ft_best.pt)
  3. Send the predicted emotion label to the ESP32 over serial

The ESP32 receives the label, maps it to a tactile Braille prefix via
emotion_to_prefix(), and displays it on the refreshable Braille cells.

Hardware requirements on the Pi:
  - Raspberry Pi 4 (4GB+ RAM recommended) or Pi 5
  - USB webcam or Pi Camera Module (via picamera2 or OpenCV)
  - USB microphone or I2S MEMS mic (INMP441)
  - UART serial connection to ESP32 (TX->RX, GND-GND)

Install dependencies:
  pip install torch transformers librosa opencv-python numpy pyserial sounddevice gdown

Usage:
  python emotion_inference.py                        # auto-downloads from GDrive
  python emotion_inference.py --port /dev/ttyAMA0    # Pi hardware UART
  python emotion_inference.py --model /path/to/ft_best.pt  # use local file
  python emotion_inference.py --no-serial            # print only (testing)
"""
from __future__ import annotations

import argparse
import os
import time
import warnings
from typing import Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration (matches training script exactly)
# ---------------------------------------------------------------------------
AUDIO_MODEL_NAME = "superb/wav2vec2-base-superb-er"
VIDEO_MODEL_NAME = "trpakov/vit-face-expression"
SAMPLE_RATE = 16000
MAX_AUDIO_S = 5
N_FRAMES = 8
IMG_SIZE = 224
MAX_AUDIO_LEN = SAMPLE_RATE * MAX_AUDIO_S

# Google Drive model download
# File ID from: https://drive.google.com/file/d/1oRc44WmF5lV7Hjv9qQM9HC50WCRlfNG-/view
GDRIVE_FILE_ID = "1oRc44WmF5lV7Hjv9qQM9HC50WCRlfNG-"
DEFAULT_MODEL_PATH = os.path.expanduser("~/ft_best.pt")

# Serial protocol
SERIAL_BAUD = 115200
# Format sent to ESP32: "EMOTION:<label>:<confidence>\n"
# e.g. "EMOTION:happy:0.62\n"


def download_model_from_gdrive(dest_path: str = DEFAULT_MODEL_PATH) -> str:
    """Download ft_best.pt from Google Drive if not already present locally."""
    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        if size_mb > 500:  # valid model is ~726MB
            print(f"[emotion] Model already exists: {dest_path} ({size_mb:.0f} MB)")
            return dest_path
        else:
            print(f"[emotion] Model file too small ({size_mb:.0f} MB), re-downloading...")
            os.remove(dest_path)

    print(f"[emotion] Downloading model from Google Drive...")
    print(f"[emotion] File ID: {GDRIVE_FILE_ID}")
    print(f"[emotion] Destination: {dest_path}")

    try:
        import gdown
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, dest_path, quiet=False)
    except ImportError:
        # Fallback: use requests directly with Drive's confirm flow
        print("[emotion] gdown not installed, using requests fallback...")
        import requests
        session = requests.Session()
        url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
        response = session.get(url, stream=True)
        # Handle large file confirmation token
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                url = f"{url}&confirm={value}"
                response = session.get(url, stream=True)
                break
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)

    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"[emotion] Download complete: {size_mb:.0f} MB")
    if size_mb < 500:
        raise RuntimeError(
            f"Downloaded model is only {size_mb:.0f} MB (expected ~726 MB). "
            f"The Google Drive link may have expired or the file is not shared publicly. "
            f"Please re-share the file with 'Anyone with the link' access."
        )
    return dest_path


def load_model(checkpoint_path: str):
    """Load the trained AVEmotionNet from ft_best.pt."""
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoFeatureExtractor, AutoImageProcessor

    device = "cpu"  # Pi runs on CPU

    audio_fe = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL_NAME)
    img_proc = AutoImageProcessor.from_pretrained(VIDEO_MODEL_NAME)
    img_mean = torch.tensor(img_proc.image_mean).view(1, 1, 3, 1, 1)
    img_std = torch.tensor(img_proc.image_std).view(1, 1, 3, 1, 1)

    class AVEmotionNet(nn.Module):
        def __init__(self, n_classes):
            super().__init__()
            self.audio = AutoModel.from_pretrained(AUDIO_MODEL_NAME)
            self.video = AutoModel.from_pretrained(VIDEO_MODEL_NAME)
            a_dim = self.audio.config.hidden_size
            v_dim = self.video.config.hidden_size
            self.register_buffer("img_mean", img_mean)
            self.register_buffer("img_std", img_std)
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

    print(f"[emotion] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_labels = [str(c) for c in ckpt["classes"]]
    model = AVEmotionNet(len(class_labels)).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    print(f"[emotion] Model loaded. Classes: {class_labels}")

    return model, class_labels, audio_fe, device


def capture_audio(duration_s: float = MAX_AUDIO_S) -> Tuple[np.ndarray, int]:
    """Record audio from the default microphone."""
    import sounddevice as sd

    print(f"[emotion] Recording {duration_s}s audio...")
    audio_buf = sd.rec(
        int(duration_s * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    y = audio_buf.reshape(-1).astype(np.float32)
    wav = np.zeros(MAX_AUDIO_LEN, dtype=np.float32)
    length = min(len(y), MAX_AUDIO_LEN)
    wav[:length] = y[:length]
    return wav, max(1, length)


def capture_video_frames(
    cap, duration_s: float = MAX_AUDIO_S, n_frames: int = N_FRAMES
) -> np.ndarray:
    """Capture n_frames evenly over duration_s from an open VideoCapture."""
    import cv2

    frames = np.zeros((n_frames, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    collected = []
    t_end = time.time() + duration_s

    while time.time() < t_end:
        ret, fr = cap.read()
        if ret:
            collected.append(fr)
        time.sleep(0.03)  # ~30fps capture rate

    if not collected:
        return frames

    # Sample n_frames evenly from collected
    pick = np.linspace(0, len(collected) - 1, n_frames).astype(int)
    for k, pi in enumerate(pick):
        fr = cv2.cvtColor(collected[pi], cv2.COLOR_BGR2RGB)
        frames[k] = cv2.resize(fr, (IMG_SIZE, IMG_SIZE))

    return frames


def predict_emotion(
    model, class_labels, audio_fe, device, frames_u8, wav_f32, true_len
) -> Tuple[str, float, dict]:
    """Run inference and return (label, confidence, all_probs)."""
    import torch

    with torch.no_grad():
        a = audio_fe(
            [wav_f32[:max(1, true_len)]],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        input_values = a.input_values.to(device)
        attn = a.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(device)
        frames_t = torch.from_numpy(frames_u8[None]).to(device)
        logits = model(input_values, attn, frames_t)
        probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()

    idx = int(probs.argmax())
    label = class_labels[idx]
    confidence = float(probs[idx])
    all_probs = {class_labels[i]: float(probs[i]) for i in range(len(class_labels))}
    return label, confidence, all_probs


def send_emotion_serial(ser, label: str, confidence: float) -> None:
    """Send emotion result to ESP32 over UART serial."""
    msg = f"EMOTION:{label}:{confidence:.2f}\n"
    ser.write(msg.encode("utf-8"))
    ser.flush()
    print(f"[emotion] Sent -> {msg.strip()}")


# ---------------------------------------------------------------------------
# Label mapping: model outputs -> braille_core.py emotion prefixes
# ---------------------------------------------------------------------------
# Model classes:  anger, disgust, fear, happy, neutral, sad
# Braille prefixes: angry, fearful, happy, neutral, sad, question, urgent
LABEL_TO_BRAILLE = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fearful",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
}


def map_label_for_braille(model_label: str) -> str:
    """Map model output label to braille_core.py emotion prefix name."""
    return LABEL_TO_BRAILLE.get(model_label, "neutral")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Raspberry Pi emotion recognition for BrailleAI"
    )
    parser.add_argument(
        "--model", default=None,
        help="Path to ft_best.pt checkpoint. If not provided or not found, "
             "auto-downloads from Google Drive to ~/ft_best.pt"
    )
    parser.add_argument(
        "--port", default="/dev/ttyUSB0",
        help="Serial port to ESP32 (default: /dev/ttyUSB0)"
    )
    parser.add_argument(
        "--baud", type=int, default=SERIAL_BAUD,
        help=f"Serial baud rate (default: {SERIAL_BAUD})"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera index for OpenCV (default: 0)"
    )
    parser.add_argument(
        "--window", type=float, default=MAX_AUDIO_S,
        help=f"Capture window in seconds (default: {MAX_AUDIO_S})"
    )
    parser.add_argument(
        "--interval", type=float, default=1.0,
        help="Pause between predictions in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--no-serial", action="store_true",
        help="Disable serial output (print-only mode for testing)"
    )
    parser.add_argument(
        "--continuous", action="store_true", default=True,
        help="Run continuously (default: True)"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single prediction then exit"
    )
    args = parser.parse_args()

    # Resolve model path (download from GDrive if needed)
    model_path = args.model
    if model_path and os.path.exists(model_path):
        pass  # use provided path
    elif model_path and not os.path.exists(model_path):
        print(f"[emotion] Model not found at {model_path}, downloading...")
        model_path = download_model_from_gdrive()
    else:
        model_path = download_model_from_gdrive()

    # Load model
    model, class_labels, audio_fe, device = load_model(model_path)

    # Open camera
    import cv2
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[emotion] ERROR: Cannot open camera index {args.camera}")
        return

    # Open serial connection to ESP32
    ser = None
    if not args.no_serial:
        import serial
        try:
            ser = serial.Serial(args.port, args.baud, timeout=1)
            print(f"[emotion] Serial open: {args.port} @ {args.baud} baud")
        except Exception as e:
            print(f"[emotion] WARNING: Cannot open serial {args.port}: {e}")
            print("[emotion] Continuing in print-only mode")

    print("[emotion] Ready. Starting emotion recognition loop...")
    print("-" * 60)

    try:
        while True:
            # Capture audio and video simultaneously
            # Audio capture blocks for window seconds; video captured in parallel
            import threading

            frames_result = [None]

            def capture_video_thread():
                frames_result[0] = capture_video_frames(
                    cap, duration_s=args.window, n_frames=N_FRAMES
                )

            video_thread = threading.Thread(target=capture_video_thread)
            video_thread.start()
            wav, audio_len = capture_audio(duration_s=args.window)
            video_thread.join()
            frames = frames_result[0]

            # Run inference
            t0 = time.time()
            label, confidence, all_probs = predict_emotion(
                model, class_labels, audio_fe, device, frames, wav, audio_len
            )
            infer_time = time.time() - t0

            # Map to braille prefix name
            braille_label = map_label_for_braille(label)

            # Print results
            print(f"[emotion] Prediction: {label} ({confidence:.0%}) "
                  f"-> braille prefix: '{braille_label}' "
                  f"[{infer_time:.1f}s inference]")
            sorted_probs = sorted(all_probs.items(), key=lambda x: -x[1])
            print(f"          {' | '.join(f'{k}:{v:.0%}' for k, v in sorted_probs)}")

            # Send to ESP32
            if ser:
                send_emotion_serial(ser, braille_label, confidence)

            if args.once:
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n[emotion] Stopped by user.")
    finally:
        cap.release()
        if ser:
            ser.close()
        print("[emotion] Shutdown complete.")


if __name__ == "__main__":
    main()
