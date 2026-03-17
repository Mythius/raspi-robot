#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# ── Create venv if it doesn't exist ───────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR …"
    python3 -m venv "$VENV_DIR"
fi

# ── Install / update dependencies ─────────────────────────────────────────────
echo "Checking dependencies …"
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet -r "$SCRIPT_DIR/requirements.txt"

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Starting detector …"
exec "$VENV_DIR/bin/python3" "$SCRIPT_DIR/detect.py" "$@"
