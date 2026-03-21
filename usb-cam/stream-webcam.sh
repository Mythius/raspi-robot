#!/bin/bash

PORT=${1:-8080}
DEVICE=${2:-/dev/video0}

if [ ! -e "$DEVICE" ]; then
    echo "Error: Webcam device $DEVICE not found"
    echo "Available devices:"
    ls /dev/video* 2>/dev/null || echo "  None found"
    exit 1
fi

if ! command -v ffmpeg &>/dev/null; then
    echo "Error: ffmpeg not found. Install with: sudo apt install ffmpeg"
    exit 1
fi

if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

echo "Streaming $DEVICE on http://localhost:$PORT/"
echo "Open your browser to http://localhost:$PORT/"
echo "Press Ctrl+C to stop"

python3 - "$DEVICE" "$PORT" <<'EOF'
import subprocess, threading, time, sys
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

DEVICE = sys.argv[1]
PORT   = int(sys.argv[2])

latest_frame = None
frame_lock   = threading.Lock()

HTML = b"""<!DOCTYPE html>
<html>
<head>
  <title>Webcam Stream</title>
  <style>
    body { background: #000; margin: 0; display: flex; justify-content: center; align-items: center; height: 100vh; }
    img  { max-width: 100%; max-height: 100vh; }
  </style>
</head>
<body>
  <img src="/stream">
</body>
</html>"""

class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence access logs

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML)

        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        frame = latest_frame
                    if frame:
                        try:
                            self.wfile.write(
                                b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n\r\n' +
                                frame + b'\r\n'
                            )
                            self.wfile.flush()
                        except BrokenPipeError:
                            break
                    time.sleep(0.033)  # ~30 fps
            except Exception:
                pass

        else:
            self.send_response(404)
            self.end_headers()

def capture_frames():
    global latest_frame
    cmd = [
        'ffmpeg',
        '-f', 'v4l2', '-i', DEVICE,
        '-vf', 'scale=1280:720',
        '-f', 'image2pipe',
        '-vcodec', 'mjpeg',
        '-q:v', '5',
        '-r', '30',
        '-'
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    buf = b''
    while True:
        chunk = proc.stdout.read(65536)
        if not chunk:
            break
        buf += chunk
        # parse JPEG frames by SOI/EOI markers
        while True:
            start = buf.find(b'\xff\xd8')
            if start == -1:
                break
            end = buf.find(b'\xff\xd9', start + 2)
            if end == -1:
                break
            with frame_lock:
                latest_frame = buf[start:end + 2]
            buf = buf[end + 2:]

t = threading.Thread(target=capture_frames, daemon=True)
t.start()

server = ThreadingHTTPServer(('0.0.0.0', PORT), StreamHandler)
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
EOF
