#!/usr/bin/env python3
"""
Headless USB webcam object detection for Raspberry Pi.

Uses YOLOv8 (nano) via the ultralytics library and OpenCV for camera access.
Detections are printed to stdout. No display required.

Usage:
    python3 detect.py [device_index_or_path]

    # default: /dev/video0
    python3 detect.py

    # explicit device index or path
    python3 detect.py 0
    python3 detect.py /dev/video2

Environment variables:
    CAMERA_DEVICE   V4L2 device index or path (default: 0)
    CAMERA_WIDTH    capture width  (default: 640)
    CAMERA_HEIGHT   capture height (default: 480)
    DETECT_FPS      target frames per second (default: 2)
    CONFIDENCE      minimum confidence 0.0–1.0 (default: 0.5)
    MODEL           YOLOv8 model variant: n/s/m/l/x (default: n = nano)
"""

import os
import sys
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

# ── Config ─────────────────────────────────────────────────────────────────────

def _parse_device(val: str) -> int | str:
    """Return an int index if val is numeric, otherwise the raw path string."""
    try:
        return int(val)
    except ValueError:
        return val

DEVICE     = _parse_device(sys.argv[1] if len(sys.argv) > 1
                           else os.getenv("CAMERA_DEVICE", "0"))
WIDTH      = int(os.getenv("CAMERA_WIDTH",  "640"))
HEIGHT     = int(os.getenv("CAMERA_HEIGHT", "480"))
FPS        = float(os.getenv("DETECT_FPS",  "2"))
CONFIDENCE = float(os.getenv("CONFIDENCE",  "0.5"))
VARIANT    = os.getenv("MODEL", "n")          # n=nano, s=small, m=medium …

FRAME_INTERVAL = 1.0 / FPS

# ── Model & camera setup ───────────────────────────────────────────────────────

print(f"Loading YOLOv8{VARIANT} model…", flush=True)
model = YOLO(f"yolov8{VARIANT}.pt")   # downloads on first run (~6 MB for nano)

print(f"Opening camera {DEVICE!r} at {WIDTH}x{HEIGHT}…", flush=True)
cap = cv2.VideoCapture(DEVICE)

if not cap.isOpened():
    sys.exit(f"Error: could not open camera {DEVICE!r}\n"
             f"Tip: run `ls /dev/video*` to list available cameras")

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

print(f"Ready — detecting at up to {FPS} fps  |  confidence ≥ {CONFIDENCE:.0%}")
print("─" * 55, flush=True)

# ── Detection loop ─────────────────────────────────────────────────────────────

try:
    while True:
        t0 = time.monotonic()

        ret, frame = cap.read()
        if not ret:
            print("[WARN] failed to grab frame, retrying…", file=sys.stderr, flush=True)
            time.sleep(0.5)
            continue

        results = model(frame, verbose=False, conf=CONFIDENCE)
        ts = datetime.now().isoformat(timespec="seconds")

        for r in results:
            for box in r.boxes:
                label      = model.names[int(box.cls)]
                confidence = float(box.conf)
                print(f"[{ts}]  {label:<20}  {confidence * 100:.1f}%", flush=True)

        # Throttle to target FPS
        elapsed = time.monotonic() - t0
        sleep_for = FRAME_INTERVAL - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

except KeyboardInterrupt:
    print("\nStopped.", flush=True)
finally:
    cap.release()
