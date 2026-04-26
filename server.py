"""
Eunice HUD — Local WebSocket Server
Run: python3 server.py
Open: http://localhost:8766
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
import urllib.request
import os

# ── CONFIG ─────────────────────────────────────────────────────────────────────
CAMERA_SOURCE  = 0                                    # 0 = built-in webcam
DROIDCAM_URL   = "http://192.168.29.167:4747/video"
USE_DROIDCAM   = False        # True = use DroidCam IP instead of webcam
USE_YOLO       = True         # auto-detects GPU (MPS on Apple Silicon, CPU fallback)
YOLO_MODEL     = "yolov8n.pt"
FRAME_W        = 854
FRAME_H        = 480
JPEG_QUALITY   = 75
WS_PORT        = 8765
HTTP_PORT      = 8766
VPS_FEED_URL   = "http://187.124.98.85:9204/latest"   # Eunice message relay on VPS
POLL_INTERVAL  = 1.0                                   # seconds between VPS polls

# ── HAND FX INIT ──────────────────────────────────────────────────────────────
USE_HAND_FX  = True
HAND_EFFECT  = "chromatic"   # chromatic | mirror | glitch | trails | neon
hand_detector = None
trail_frames  = []

def init_hand_detector():
    global hand_detector
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        model_url  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        if not os.path.exists(model_path):
            print("[HandFX] Downloading model...")
            urllib.request.urlretrieve(model_url, model_path)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            num_hands=2, min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5, min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        hand_detector = mp_vision.HandLandmarker.create_from_options(opts)
        print("[HandFX] Hand detector ready")
        return True
    except Exception as e:
        print(f"[HandFX] Not available: {e}")
        return False

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]

def make_mask(landmarks, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(mask, (cx, cy), 60, 255, -1)
    return cv2.GaussianBlur(mask, (31, 31), 0)

def draw_hand(frame, landmarks, h, w):
    pts = {i: (int(lm.x*w), int(lm.y*h)) for i, lm in enumerate(landmarks)}
    for a, b in HAND_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (0,255,80), 2, cv2.LINE_AA)
    for cx, cy in pts.values():
        cv2.circle(frame, (cx,cy), 6, (0,50,255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx,cy), 8, (255,255,255), 1, cv2.LINE_AA)

def apply_fx(frame, mask, h, w):
    import numpy as np
    if HAND_EFFECT == "chromatic":
        if mask is None: return frame
        intensity = 22
        b,g,r = cv2.split(frame)
        glitched = cv2.merge([np.roll(b,-intensity,axis=1), np.roll(g,intensity//2,axis=0), np.roll(r,intensity,axis=1)])
        m3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32)/255.0
        return (frame*(1-m3) + glitched*m3).astype(np.uint8)
    elif HAND_EFFECT == "mirror":
        result = frame.copy()
        for i in range(1,7):
            alpha = 0.14*(7-i)/6
            for s in [i*22, -i*22]:
                M = np.float32([[1,0,s],[0,1,0]])
                cv2.addWeighted(result,1.0,cv2.warpAffine(frame,M,(w,h)),alpha,0,dst=result)
        return result
    elif HAND_EFFECT == "glitch":
        result = frame.copy()
        for _ in range(14):
            y = np.random.randint(0,h)
            result[y] = np.roll(result[y], np.random.randint(-35,35), axis=0)
        return result
    elif HAND_EFFECT == "trails":
        global trail_frames
        trail_frames.append(frame.copy())
        if len(trail_frames) > 10: trail_frames.pop(0)
        result = frame.copy()
        for i,f in enumerate(trail_frames[:-1]):
            cv2.addWeighted(result,1.0,f,(i+1)/len(trail_frames)*0.22,0,dst=result)
        return result
    elif HAND_EFFECT == "neon":
        if mask is None: return frame
        glow = cv2.GaussianBlur(frame,(0,0),18)
        result = cv2.addWeighted(glow,0.45,frame,0.55,0)
        neon = np.zeros_like(result); neon[:,:,1]=mask; neon[:,:,0]=mask//3
        return cv2.addWeighted(result,1.0,neon,0.5,0)
    return frame

import numpy as np
if USE_HAND_FX:
    init_hand_detector()

# ── YOLO INIT ──────────────────────────────────────────────────────────────────
model  = None
device = "cpu"

if USE_YOLO:
    try:
        import torch
        from ultralytics import YOLO
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        model = YOLO(YOLO_MODEL)
        model.fuse()
        print(f"[YOLO] Loaded {YOLO_MODEL} on {device}")
    except Exception as e:
        print(f"[YOLO] Not available ({e}). Running without detection.")
        USE_YOLO = False

# ── SHARED STATE ───────────────────────────────────────────────────────────────
frame_data = {"frame": None, "detections": [], "fps": 0}
frame_lock  = threading.Lock()
connected_ws = set()
eunice_queue = queue.Queue()

# ── CAMERA THREAD ──────────────────────────────────────────────────────────────
def make_cap():
    src = DROIDCAM_URL if USE_DROIDCAM else CAMERA_SOURCE
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG if isinstance(src, str) else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    return cap

def camera_thread():
    cap        = make_cap()
    fps_smooth = 0.0
    prev_t     = time.perf_counter()
    fc         = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            cap.release(); cap = make_cap()
            continue

        fc += 1
        now       = time.perf_counter()
        fps_smooth = fps_smooth * 0.9 + (1.0 / max(now - prev_t, 1e-6)) * 0.1
        prev_t    = now

        detections = []
        if USE_YOLO and model and fc % 1 == 0:
            try:
                results = model(frame, verbose=False, conf=0.25, device=device)
                if results and results[0].boxes is not None:
                    fw, fh = frame.shape[1], frame.shape[0]
                    for box in results[0].boxes:
                        x1,y1,x2,y2 = box.xyxy[0].tolist()
                        detections.append({
                            "label": model.names[int(box.cls[0])],
                            "conf":  round(float(box.conf[0]), 2),
                            "box":   [x1/fw, y1/fh, x2/fw, y2/fh],
                        })
            except Exception:
                pass

        # Hand FX
        if USE_HAND_FX and hand_detector:
            try:
                import mediapipe as mp
                fh2, fw2 = frame.shape[:2]
                rgb2   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb2)
                hres   = hand_detector.detect(mp_img)
                all_lms = hres.hand_landmarks or []
                combined = None
                if all_lms:
                    combined = np.zeros((fh2,fw2),dtype=np.uint8)
                    for lms in all_lms:
                        combined = cv2.bitwise_or(combined, make_mask(lms,fh2,fw2))
                frame = apply_fx(frame, combined, fh2, fw2)
                if all_lms:
                    for lms in all_lms:
                        draw_hand(frame, lms, fh2, fw2)
            except Exception as e:
                pass

        frame_small = cv2.resize(frame, (FRAME_W, FRAME_H))
        _, buf = cv2.imencode(".jpg", frame_small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        b64    = base64.b64encode(buf).decode("ascii")

        with frame_lock:
            frame_data["frame"]      = b64
            frame_data["detections"] = detections
            frame_data["fps"]        = round(fps_smooth, 1)

threading.Thread(target=camera_thread, daemon=True, name="Camera").start()

# ── VPS POLL THREAD ────────────────────────────────────────────────────────────
def vps_poll_thread():
    last_ts = 0
    while True:
        try:
            with urllib.request.urlopen(VPS_FEED_URL, timeout=3) as r:
                data = json.loads(r.read())
                ts   = data.get("ts", 0)
                msg  = data.get("message", "")
                if msg and ts > last_ts:
                    last_ts = ts
                    eunice_queue.put(msg)
        except Exception:
            pass
        time.sleep(POLL_INTERVAL)

threading.Thread(target=vps_poll_thread, daemon=True, name="VPSPoll").start()

# ── WEBSOCKET ──────────────────────────────────────────────────────────────────
async def ws_handler(ws):
    connected_ws.add(ws)
    print(f"[WS] Client connected ({len(connected_ws)} total)")
    try:
        async for _ in ws:
            pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_ws.discard(ws)

async def broadcast_loop():
    eunice_text = ""
    eunice_ts   = 0

    while True:
        await asyncio.sleep(1/30)
        if not connected_ws:
            continue

        while not eunice_queue.empty():
            try:
                eunice_text = eunice_queue.get_nowait()
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
                "date":       datetime.datetime.now().strftime("%A, %d %b %Y").upper(),
                "eunice":     eunice_text if (time.time() - eunice_ts) < 10 else "",
            }

        if payload["frame"] is None:
            continue

        msg  = json.dumps(payload)
        dead = set()
        for ws in list(connected_ws):
            try:
                await ws.send(msg)
            except Exception:
                dead.add(ws)
        connected_ws.difference_update(dead)

# ── HTTP (serves index.html) ───────────────────────────────────────────────────
async def http_handler(reader, writer):
    await reader.read(4096)
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "rb") as f:
        body = f.read()
    writer.write(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nConnection: close\r\n\r\n"
        + body
    )
    await writer.drain()
    writer.close()

async def main():
    print(f"[HUD] WebSocket  → ws://localhost:{WS_PORT}")
    print(f"[HUD] Frontend   → http://localhost:{HTTP_PORT}")
    ws_server   = await websockets.serve(ws_handler, "0.0.0.0", WS_PORT)
    http_server = await asyncio.start_server(http_handler, "0.0.0.0", HTTP_PORT)
    await asyncio.gather(
        ws_server.wait_closed(),
        http_server.serve_forever(),
        broadcast_loop(),
    )

if __name__ == "__main__":
    asyncio.run(main())
