#!/bin/bash
# setup_pi.sh - One-command setup for Raspberry Pi
# =================================================
# Run this on your Raspberry Pi to install everything needed
# for the BrailleAI camera stream and emotion recognition.
#
# Usage:
#   chmod +x setup_pi.sh
#   ./setup_pi.sh

set -e

echo "========================================"
echo "  BrailleAI - Raspberry Pi Setup"
echo "========================================"
echo ""

# Update system packages
echo "[1/5] Updating system packages..."
sudo apt-get update -qq

# Install system dependencies
echo "[2/5] Installing system dependencies..."
sudo apt-get install -y -qq python3-pip python3-venv libopencv-dev

# Create virtual environment (optional but recommended)
echo "[3/5] Installing Python packages..."
pip3 install --break-system-packages flask opencv-python-headless numpy pyserial 2>/dev/null || \
pip3 install flask opencv-python-headless numpy pyserial

# Test camera
echo "[4/5] Testing camera..."
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    cap.release()
    if ret:
        print('  ✓ Camera is working! (captured a frame)')
    else:
        print('  ✗ Camera opened but cannot read frames')
else:
    print('  ✗ Camera NOT detected. Check connection.')
    print('    Try: ls /dev/video*')
" 2>/dev/null || echo "  ⚠ Camera test skipped (OpenCV import issue)"

# Show network info
echo "[5/5] Network info:"
echo "  Your Pi's IP address(es):"
hostname -I | tr ' ' '\n' | sed 's/^/    /'
echo ""

PI_IP=$(hostname -I | awk '{print $1}')
echo "========================================"
echo "  Setup complete!"
echo ""
echo "  To start the camera stream, run:"
echo "    python3 camera_stream.py"
echo ""
echo "  Then open in your laptop browser:"
echo "    http://${PI_IP}:5000"
echo ""
echo "  To run emotion recognition:"
echo "    python3 emotion_inference.py --no-serial"
echo "========================================"
