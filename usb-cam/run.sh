#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
MODELS_DIR="$SCRIPT_DIR/models"

# ── Create venv if it doesn't exist ───────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR …"
    python3 -m venv "$VENV_DIR"
fi

# ── Install / update dependencies ─────────────────────────────────────────────
echo "Checking dependencies …"
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet -r "$SCRIPT_DIR/requirements.txt"

# ── Download YOLOv3-tiny model files if needed ────────────────────────────────
mkdir -p "$MODELS_DIR"

download() {
    local url="$1"
    local dest="$2"
    if [ ! -f "$dest" ]; then
        echo "Downloading $(basename "$dest") …"
        curl -fsSL "$url" -o "$dest"
    fi
}

download \
    "https://pjreddie.com/media/files/yolov3-tiny.weights" \
    "$MODELS_DIR/yolov3-tiny.weights"

download \
    "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg" \
    "$MODELS_DIR/yolov3-tiny.cfg"

download \
    "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names" \
    "$MODELS_DIR/coco.names"

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Starting detector …"
exec "$VENV_DIR/bin/python3" "$SCRIPT_DIR/detect.py" "$@"
