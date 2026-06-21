"""
camera_stream.py  -  Raspberry Pi Live Camera Stream
====================================================
Streams the Pi camera feed over HTTP so you can view it
from any browser on the same network.

Usage:
  python3 camera_stream.py                  # default port 5000
  python3 camera_stream.py --port 8080      # custom port
  python3 camera_stream.py --camera 1       # alternate camera index

Then open in your laptop browser:
  http://<raspberry_pi_ip>:5000
"""
from __future__ import annotations

import argparse
import time

import cv2
from flask import Flask, Response, render_template_string

app = Flask(__name__)

# Global camera reference
camera_index = 0

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>BrailleAI - Pi Camera Stream</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            background: #1a1a2e;
            color: #eee;
            margin: 0;
            padding: 20px;
        }
        h1 { color: #00d4ff; }
        img {
            border: 3px solid #00d4ff;
            border-radius: 8px;
            max-width: 90%;
        }
        .status {
            color: #0f0;
            font-size: 14px;
            margin-top: 10px;
        }
        .info {
            color: #aaa;
            font-size: 12px;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <h1>BrailleAI - Live Camera Feed</h1>
    <img src="/video_feed" alt="Camera Stream">
    <p class="status">&#9679; LIVE</p>
    <p class="info">Raspberry Pi Camera | Press Ctrl+C on the Pi to stop</p>
</body>
</html>
"""


def generate_frames():
    """Yield MJPEG frames from the camera."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[stream] ERROR: Cannot open camera index {camera_index}")
        return

    print(f"[stream] Camera opened successfully (index {camera_index})")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[stream] Failed to read frame, retrying...")
            time.sleep(0.1)
            continue

        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        )


@app.route('/')
def index():
    """Serve the HTML page with embedded video stream."""
    return render_template_string(HTML_PAGE)


@app.route('/video_feed')
def video_feed():
    """Return the MJPEG stream response."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def main():
    parser = argparse.ArgumentParser(
        description="Stream Raspberry Pi camera over HTTP"
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="HTTP port to serve on (default: 5000)"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0 = all interfaces)"
    )
    args = parser.parse_args()

    global camera_index
    camera_index = args.camera

    print("=" * 50)
    print("  BrailleAI - Pi Camera Stream")
    print("=" * 50)
    print(f"  Camera index: {args.camera}")
    print(f"  Server: http://0.0.0.0:{args.port}")
    print()
    print("  Open this URL in your laptop browser:")
    print(f"  http://<your_pi_ip>:{args.port}")
    print()
    print("  To find your Pi's IP, run: hostname -I")
    print("  Press Ctrl+C to stop")
    print("=" * 50)

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
