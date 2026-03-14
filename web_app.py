#!/usr/bin/env python3
"""
SplitScreen Marker Detector — Mobile Web App
=============================================

Flask server that runs the detector on frames from the phone's camera.
Open the page on your phone browser → camera feed goes to the server →
annotated result streams back in real-time.

Usage:
    python web_app.py --watch-model 45mm
    python web_app.py --watch-model 45mm --port 5000

Then open http://<your-pc-ip>:5000 on your phone.
"""

import cv2
import numpy as np
import argparse
import time
import threading
import base64
from flask import Flask, render_template_string, request, jsonify, Response

from detect_marker import ScreenDetector, SCREEN_SIZES

# ================================================================== #
#                         Flask App                                    #
# ================================================================== #

app = Flask(__name__)

# Global detector (initialised in main)
detector = None
detector_lock = threading.Lock()

# Latest annotated frame for MJPEG stream
latest_frame = None
latest_frame_lock = threading.Lock()
latest_pose = None


# ================================================================== #
#                    Mobile HTML Page                                   #
# ================================================================== #

MOBILE_PAGE = """
<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>ECG Marker Detector</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    background: #111; color: #eee; font-family: -apple-system, sans-serif;
    overflow: hidden; height: 100vh; display: flex; flex-direction: column;
}
#top-bar {
    background: #1a1a2e; padding: 8px 12px; display: flex;
    justify-content: space-between; align-items: center; z-index: 10;
}
#top-bar h1 { font-size: 16px; color: #0ff; }
#status { font-size: 13px; padding: 3px 10px; border-radius: 12px; }
.status-searching { background: #c33; }
.status-locked { background: #2a2; }
.status-sending { background: #a80; }

#video-container {
    flex: 1; position: relative; overflow: hidden; background: #000;
}
#camera-video {
    width: 100%; height: 100%; object-fit: cover;
}
#result-overlay {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    object-fit: cover; pointer-events: none;
    display: none;  /* shown when we get results */
}
#info-panel {
    background: rgba(0,0,0,0.85); padding: 10px 14px;
    position: absolute; bottom: 0; left: 0; right: 0;
    backdrop-filter: blur(8px);
}
.info-row { display: flex; justify-content: space-between; margin: 4px 0; }
.info-label { color: #aaa; font-size: 13px; }
.info-value { font-size: 15px; font-weight: 600; }
#body-rot-value { font-size: 28px; color: #0f0; }
#ecg-label { color: #ff0; font-size: 16px; }
#fps-display { color: #888; font-size: 11px; }

#controls {
    background: #1a1a2e; padding: 10px; display: flex;
    justify-content: center; gap: 12px;
}
button {
    padding: 10px 24px; border: none; border-radius: 8px;
    font-size: 15px; font-weight: 600; cursor: pointer;
}
#btn-start { background: #0a0; color: #fff; }
#btn-stop { background: #a00; color: #fff; display: none; }
#btn-flip { background: #333; color: #fff; }
</style>
</head>
<body>

<div id="top-bar">
    <h1>ECG Marker Detector</h1>
    <span id="status" class="status-searching">מחפש</span>
</div>

<div id="video-container">
    <video id="camera-video" autoplay playsinline muted></video>
    <img id="result-overlay" />
    <div id="info-panel" style="display:none;">
        <div id="body-rot-value">--°</div>
        <div id="ecg-label">--</div>
        <div class="info-row">
            <span class="info-label">מרחק</span>
            <span class="info-value" id="distance-value">--</span>
        </div>
        <div class="info-row">
            <span class="info-label">ביטחון</span>
            <span class="info-value" id="confidence-value">--</span>
        </div>
        <div id="fps-display"></div>
    </div>
</div>

<div id="controls">
    <button id="btn-start" onclick="startDetection()">התחל</button>
    <button id="btn-stop" onclick="stopDetection()">עצור</button>
    <button id="btn-flip" onclick="flipCamera()">🔄 הפוך מצלמה</button>
</div>

<script>
const video = document.getElementById('camera-video');
const overlay = document.getElementById('result-overlay');
const statusEl = document.getElementById('status');
const infoPanel = document.getElementById('info-panel');
const bodyRotEl = document.getElementById('body-rot-value');
const ecgEl = document.getElementById('ecg-label');
const distEl = document.getElementById('distance-value');
const confEl = document.getElementById('confidence-value');
const fpsEl = document.getElementById('fps-display');

let stream = null;
let running = false;
let facingMode = 'environment';  // back camera
let canvas = document.createElement('canvas');
let ctx = canvas.getContext('2d');
let frameCount = 0;
let fpsTimer = performance.now();
let currentFps = 0;

// Target: send ~10-12 frames per second (phone → server → back)
const SEND_INTERVAL_MS = 80;

async function initCamera() {
    if (stream) {
        stream.getTracks().forEach(t => t.stop());
    }
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: facingMode,
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        });
        video.srcObject = stream;
        await video.play();
    } catch(e) {
        alert('שגיאה בפתיחת מצלמה: ' + e.message);
    }
}

function flipCamera() {
    facingMode = (facingMode === 'environment') ? 'user' : 'environment';
    initCamera();
}

function startDetection() {
    document.getElementById('btn-start').style.display = 'none';
    document.getElementById('btn-stop').style.display = '';
    infoPanel.style.display = '';
    running = true;
    sendFrame();
}

function stopDetection() {
    running = false;
    document.getElementById('btn-start').style.display = '';
    document.getElementById('btn-stop').style.display = 'none';
    overlay.style.display = 'none';
    infoPanel.style.display = 'none';
    statusEl.textContent = 'מחפש';
    statusEl.className = 'status-searching';
}

async function sendFrame() {
    if (!running) return;

    // Capture frame from video
    if (video.videoWidth === 0) {
        setTimeout(sendFrame, 100);
        return;
    }

    canvas.width = Math.min(video.videoWidth, 640);
    canvas.height = Math.round(canvas.width * video.videoHeight / video.videoWidth);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    statusEl.textContent = 'שולח...';
    statusEl.className = 'status-sending';

    try {
        const blob = await new Promise(r => canvas.toBlob(r, 'image/jpeg', 0.75));
        const formData = new FormData();
        formData.append('frame', blob, 'frame.jpg');

        const resp = await fetch('/process_frame', {
            method: 'POST',
            body: formData
        });

        if (!resp.ok) throw new Error('Server error');

        const data = await resp.json();

        // Show annotated frame
        if (data.image) {
            overlay.src = 'data:image/jpeg;base64,' + data.image;
            overlay.style.display = '';
        }

        // Update info panel
        if (data.pose) {
            const p = data.pose;
            bodyRotEl.textContent = p.body_rotation.toFixed(0) + '°';
            ecgEl.textContent = '~ ' + p.ecg_label;
            distEl.textContent = p.distance_cm.toFixed(1) + ' cm';
            confEl.textContent = (p.confidence * 100).toFixed(0) + '%';

            // Color by rotation
            if (p.body_rotation < 30) {
                bodyRotEl.style.color = '#0f0';
            } else if (p.body_rotation < 60) {
                bodyRotEl.style.color = '#ff0';
            } else {
                bodyRotEl.style.color = '#f80';
            }

            statusEl.textContent = 'נעול ✓';
            statusEl.className = 'status-locked';
        } else {
            bodyRotEl.textContent = '--°';
            ecgEl.textContent = 'מחפש סמן...';
            distEl.textContent = '--';
            confEl.textContent = '--';
            statusEl.textContent = 'מחפש';
            statusEl.className = 'status-searching';
        }

        // FPS counter
        frameCount++;
        const elapsed = (performance.now() - fpsTimer) / 1000;
        if (elapsed >= 1.0) {
            currentFps = frameCount / elapsed;
            frameCount = 0;
            fpsTimer = performance.now();
        }
        fpsEl.textContent = currentFps.toFixed(0) + ' FPS';

    } catch(e) {
        console.error('Frame error:', e);
        statusEl.textContent = 'שגיאה';
        statusEl.className = 'status-searching';
    }

    if (running) {
        setTimeout(sendFrame, SEND_INTERVAL_MS);
    }
}

// Auto-init camera on load
initCamera();
</script>
</body>
</html>
"""


# ================================================================== #
#                    Flask Routes                                       #
# ================================================================== #

@app.route('/')
def index():
    return render_template_string(MOBILE_PAGE)


@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Receive a JPEG frame, run detector, return annotated + pose."""
    global latest_frame, latest_pose

    file = request.files.get('frame')
    if file is None:
        return jsonify({'error': 'no frame'}), 400

    # Decode JPEG → numpy
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'bad image'}), 400

    # Run detector
    with detector_lock:
        annotated, pose, mask = detector.process_frame(frame, debug=False)

    # Encode result as JPEG
    _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
    img_b64 = base64.b64encode(buf).decode('ascii')

    # Store for MJPEG stream
    with latest_frame_lock:
        latest_frame = buf.tobytes()
        latest_pose = pose

    # Build response
    result = {'image': img_b64, 'pose': None}
    if pose:
        result['pose'] = {
            'body_rotation': pose['body_rotation'],
            'body_rotation_raw': pose.get('body_rotation_raw', 0),
            'ecg_label': pose['ecg_label'],
            'distance_cm': pose['distance_cm'],
            'angle_x': pose['angle_x'],
            'angle_y': pose['angle_y'],
            'confidence': pose['confidence'],
            'reproj_error': pose.get('reproj_error', 0),
        }

    return jsonify(result)


# ================================================================== #
#                    MJPEG Stream (optional, for PC viewer)            #
# ================================================================== #

def mjpeg_generator():
    """Yield annotated frames as MJPEG for viewing on another device."""
    while True:
        with latest_frame_lock:
            frame_data = latest_frame
        if frame_data:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'
                   + frame_data + b'\r\n')
        time.sleep(0.05)


@app.route('/stream')
def video_stream():
    return Response(mjpeg_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ================================================================== #
#                    Main                                               #
# ================================================================== #

def get_local_ip():
    """Get the machine's local network IP."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'


def main():
    global detector

    parser = argparse.ArgumentParser(
        description="SplitScreen Marker Detector — Mobile Web App")
    parser.add_argument("--watch-model", type=str, default="45mm",
                        choices=list(SCREEN_SIZES.keys()))
    parser.add_argument("--screen-width", type=float, default=None)
    parser.add_argument("--screen-height", type=float, default=None)
    parser.add_argument("--screen-size", type=str, default=None)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    sw, sh = SCREEN_SIZES[args.watch_model]
    if args.screen_size:
        parts = args.screen_size.lower().split('x')
        sw, sh = float(parts[0]), float(parts[1])
    if args.screen_width:
        sw = args.screen_width
    if args.screen_height:
        sh = args.screen_height

    detector = ScreenDetector(screen_width_mm=sw, screen_height_mm=sh)

    local_ip = get_local_ip()

    print()
    print("=" * 56)
    print("  ECG Marker Detector — Mobile Web App")
    print("=" * 56)
    print()
    print(f"  Watch: {args.watch_model}  Screen: {sw:.1f}×{sh:.1f}mm")
    print()
    print(f"  📱  פתח בטלפון (HTTPS):")
    print(f"       https://{local_ip}:{args.port}")
    print()
    print(f"  🖥️  MJPEG stream (PC):")
    print(f"       https://{local_ip}:{args.port}/stream")
    print()
    print("  ודא שהטלפון והמחשב על אותו WiFi!")
    print("  בטלפון: קבל את אזהרת האבטחה (Advanced → Proceed)")
    print("=" * 56)
    print()

    app.run(host=args.host, port=args.port,
            debug=False, threaded=True, ssl_context='adhoc')


if __name__ == "__main__":
    main()
