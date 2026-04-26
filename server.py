"""
Eunice HUD — Local WebSocket Server
Captures camera, runs YOLO (optional), streams to the HUD frontend.
Run: python server.py
Open: http://localhost:8765 in browser
"""

import asyncio
import websockets
import cv2
import base64
import json
import time
import threading
import queue
import datetime
import sys
import os

# ── CONFIG ─────────────────────────────────────────────────────────────────────
CAMERA_SOURCE  = 0          # 0 = laptop webcam, 1/2 = DroidCam virtual, or URL string
DROIDCAM_URL   = "http://192.168.29.167:4747/video"
USE_DROIDCAM   = False      # set True to use DroidCam URL instead of webcam
USE_YOLO       = False      # set True if ultralytics + GPU available
YOLO_MODEL     = "yolo11n.pt"
FRAME_W        = 854
FRAME_H        = 480
JPEG_QUALITY   = 75
WS_PORT        = 8765
HTTP_PORT      = 8766

# ── YOLO INIT (optional) ───────────────────────────────────────────────────────
model = None
if USE_YOLO:
    try:
        from ultralytics import YOLO
        import torch
        model = YOLO(YOLO_MODEL)
        model.fuse()
        print(f"[YOLO] Loaded {YOLO_MODEL}")
    except Exception as e:
        print(f"[YOLO] Not available: {e}. Running without detection.")
        USE_YOLO = False

# ── SHARED STATE ───────────────────────────────────────────────────────────────
frame_data = {
    "frame": None,
    "detections": [],
    "fps": 0,
    "ts": 0,
}
frame_lock     = threading.Lock()
connected_ws   = set()
eunice_messages = queue.Queue()

# ── CAMERA THREAD ──────────────────────────────────────────────────────────────
def camera_thread():
    src = DROIDCAM_URL if USE_DROIDCAM else CAMERA_SOURCE
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG if isinstance(src, str) else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    fps_smooth = 0
    prev_t = time.perf_counter()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            cap.release()
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG if isinstance(src, str) else cv2.CAP_ANY)
            continue

        frame_count += 1
        now = time.perf_counter()
        fps_smooth = fps_smooth * 0.9 + (1.0 / max(now - prev_t, 1e-6)) * 0.1
        prev_t = now

        detections = []
        if USE_YOLO and model and frame_count % 2 == 0:
            try:
                results = model(frame, verbose=False, conf=0.45)
                if results and results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        label  = model.names[int(box.cls[0])]
                        conf   = float(box.conf[0])
                        fw, fh = frame.shape[1], frame.shape[0]
                        detections.append({
                            "label": label,
                            "conf":  round(conf, 2),
                            "box":   [x1/fw, y1/fh, x2/fw, y2/fh],
                        })
            except Exception:
                pass

        # Encode frame as JPEG → base64
        frame_small = cv2.resize(frame, (FRAME_W, FRAME_H))
        _, buf = cv2.imencode(".jpg", frame_small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        b64 = base64.b64encode(buf).decode("ascii")

        with frame_lock:
            frame_data["frame"]      = b64
            frame_data["detections"] = detections
            frame_data["fps"]        = round(fps_smooth, 1)
            frame_data["ts"]         = now

threading.Thread(target=camera_thread, daemon=True, name="Camera").start()

# ── WEBSOCKET SERVER ───────────────────────────────────────────────────────────
async def ws_handler(ws):
    connected_ws.add(ws)
    print(f"[WS] Client connected. Total: {len(connected_ws)}")
    try:
        async for msg in ws:
            # Accept messages from HUD (e.g. Eunice responses piped in)
            try:
                data = json.loads(msg)
                if data.get("type") == "eunice":
                    eunice_messages.put(data.get("text", ""))
            except Exception:
                pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_ws.discard(ws)
        print(f"[WS] Client disconnected. Total: {len(connected_ws)}")

async def broadcast_loop():
    eunice_text = ""
    eunice_ts   = 0

    while True:
        await asyncio.sleep(1/30)  # ~30 fps broadcast

        if not connected_ws:
            continue

        # Pull latest Eunice message if any
        while not eunice_messages.empty():
            try:
                eunice_text = eunice_messages.get_nowait()
                eunice_ts   = time.time()
            except queue.Empty:
                break

        with frame_lock:
            payload = {
                "type":       "frame",
                "frame":      frame_data["frame"],
                "detections": frame_data["detections"],
                "fps":        frame_data["fps"],
                "time":       datetime.datetime.now().strftime("%H:%M:%S"),
                "date":       datetime.datetime.now().strftime("%A, %d %b %Y"),
                "eunice":     eunice_text if (time.time() - eunice_ts) < 8 else "",
            }

        if payload["frame"] is None:
            continue

        msg = json.dumps(payload)
        dead = set()
        for ws in connected_ws.copy():
            try:
                await ws.send(msg)
            except Exception:
                dead.add(ws)
        connected_ws -= dead

# ── HTTP SERVER (serves index.html) ───────────────────────────────────────────
async def http_handler(reader, writer):
    await reader.read(4096)
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "rb") as f:
        body = f.read()
    resp = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/html; charset=utf-8\r\n"
        b"Connection: close\r\n\r\n"
    ) + body
    writer.write(resp)
    await writer.drain()
    writer.close()

async def main():
    print(f"[HUD] WebSocket  → ws://localhost:{WS_PORT}")
    print(f"[HUD] Frontend   → http://localhost:{HTTP_PORT}")
    print("[HUD] Open the frontend URL in your browser.")

    ws_server   = await websockets.serve(ws_handler, "0.0.0.0", WS_PORT)
    http_server = await asyncio.start_server(http_handler, "0.0.0.0", HTTP_PORT)

    await asyncio.gather(
        ws_server.wait_closed(),
        http_server.serve_forever(),
        broadcast_loop(),
    )

if __name__ == "__main__":
    asyncio.run(main())
