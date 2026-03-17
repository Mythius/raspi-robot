#!/usr/bin/env python3
"""
Headless USB webcam object detection for Raspberry Pi.

Uses OpenCV's built-in DNN module with YOLOv3-tiny — no PyTorch or
display required. Model files are downloaded automatically by run.sh.

Usage:
    python3 detect.py [device_index_or_path]

Environment variables:
    CAMERA_DEVICE   V4L2 device index or path (default: 0)
    CAMERA_WIDTH    capture width  (default: 640)
    CAMERA_HEIGHT   capture height (default: 480)
    DETECT_FPS      target frames per second (default: 2)
    CONFIDENCE      minimum confidence 0.0–1.0 (default: 0.5)
    NMS_THRESHOLD   non-maximum suppression threshold (default: 0.4)
"""

import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────────

def _parse_device(val: str) -> int | str:
    try:
        return int(val)
    except ValueError:
        return val

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR    = os.path.join(SCRIPT_DIR, "models")

DEVICE        = _parse_device(sys.argv[1] if len(sys.argv) > 1
                               else os.getenv("CAMERA_DEVICE", "0"))
WIDTH         = int(os.getenv("CAMERA_WIDTH",   "640"))
HEIGHT        = int(os.getenv("CAMERA_HEIGHT",  "480"))
FPS           = float(os.getenv("DETECT_FPS",   "2"))
CONFIDENCE    = float(os.getenv("CONFIDENCE",   "0.5"))
NMS_THRESHOLD = float(os.getenv("NMS_THRESHOLD","0.4"))

FRAME_INTERVAL = 1.0 / FPS

# ── Load model ─────────────────────────────────────────────────────────────────

names_path   = os.path.join(MODELS_DIR, "coco.names")
weights_path = os.path.join(MODELS_DIR, "yolov3-tiny.weights")
cfg_path     = os.path.join(MODELS_DIR, "yolov3-tiny.cfg")

for p in (names_path, weights_path, cfg_path):
    if not os.path.exists(p):
        sys.exit(f"Missing model file: {p}\nRun ./run.sh to download models.")

with open(names_path) as f:
    CLASSES = [line.strip() for line in f if line.strip()]

print("Loading YOLOv3-tiny model…", flush=True)
net = cv2.dnn.readNet(weights_path, cfg_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names   = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# ── Camera setup ───────────────────────────────────────────────────────────────

print(f"Opening camera {DEVICE!r} at {WIDTH}x{HEIGHT}…", flush=True)
cap = cv2.VideoCapture(DEVICE)

if not cap.isOpened():
    sys.exit(f"Error: could not open camera {DEVICE!r}\n"
             "Tip: run `ls /dev/video*` to list available cameras")

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

        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []

        for output in outputs:
            for detection in output:
                scores    = detection[5:]
                class_id  = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence < CONFIDENCE:
                    continue
                cx, cy, bw, bh = (detection[:4] * [w, h, w, h]).astype(int)
                boxes.append([cx - bw // 2, cy - bh // 2, bw, bh])
                confidences.append(confidence)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, NMS_THRESHOLD)
        ts = datetime.now().isoformat(timespec="seconds")

        if len(indices) > 0:
            for i in np.array(indices).flatten():
                label = CLASSES[class_ids[i]]
                conf  = confidences[i]
                print(f"[{ts}]  {label:<20}  {conf * 100:.1f}%", flush=True)

        elapsed = time.monotonic() - t0
        sleep_for = FRAME_INTERVAL - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

except KeyboardInterrupt:
    print("\nStopped.", flush=True)
finally:
    cap.release()
