"""
Eunice HUD — Hand Tracking FX (MediaPipe Tasks API)
Real-time hand skeleton + psychedelic glitch effects
Run: python3 hand_fx.py
"""

import cv2
import numpy as np
import time
import urllib.request
import os

# ── MEDIAPIPE TASKS API ────────────────────────────────────────────────────────
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("[HandFX] Downloading hand landmark model (~5MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[HandFX] Model downloaded.")

base_opts = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
options   = mp_vision.HandLandmarkerOptions(
    base_options=base_opts,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=mp_vision.RunningMode.IMAGE,
)
detector = mp_vision.HandLandmarker.create_from_options(options)

# MediaPipe hand connections
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # thumb
    (0,5),(5,6),(6,7),(7,8),       # index
    (5,9),(9,10),(10,11),(11,12),  # middle
    (9,13),(13,14),(14,15),(15,16),# ring
    (13,17),(17,18),(18,19),(19,20),# pinky
    (0,17)                          # palm
]

# ── CONFIG ─────────────────────────────────────────────────────────────────────
CAMERA_SOURCE = 0
DROIDCAM_URL  = "http://192.168.29.167:4747/video"
USE_DROIDCAM  = False
EFFECT        = "chromatic"  # chromatic | mirror | glitch | trails | neon
DOT_COLOR     = (0, 50, 255)
LINE_COLOR    = (0, 255, 80)

# ── EFFECTS ────────────────────────────────────────────────────────────────────
trail_frames = []

def chromatic_aberration(frame, hand_mask, intensity=20):
    if hand_mask is None: return frame
    b, g, r = cv2.split(frame)
    r_s = np.roll(r,  intensity, axis=1)
    b_s = np.roll(b, -intensity, axis=1)
    g_s = np.roll(g,  intensity//2, axis=0)
    glitched = cv2.merge([b_s, g_s, r_s])
    mask3  = cv2.cvtColor(hand_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    return (frame * (1 - mask3) + glitched * mask3).astype(np.uint8)

def mirror_effect(frame, hand_mask, h, w):
    result = frame.copy()
    for i in range(1, 7):
        alpha = 0.15 * (7 - i) / 6
        shift = i * 20
        for s in [shift, -shift]:
            M = np.float32([[1,0,s],[0,1,0]])
            shifted = cv2.warpAffine(frame, M, (w, h))
            cv2.addWeighted(result, 1.0, shifted, alpha, 0, dst=result)
    return result

def trails_effect(frame, hand_mask, max_t=10):
    global trail_frames
    trail_frames.append(frame.copy())
    if len(trail_frames) > max_t: trail_frames.pop(0)
    result = frame.copy()
    for i, f in enumerate(trail_frames[:-1]):
        alpha = (i + 1) / len(trail_frames) * 0.22
        cv2.addWeighted(result, 1.0, f, alpha, 0, dst=result)
    return result

def glitch_effect(frame, hand_mask):
    result = frame.copy()
    h = result.shape[0]
    for _ in range(14):
        y = np.random.randint(0, h)
        s = np.random.randint(-35, 35)
        result[y] = np.roll(result[y], s, axis=0)
    return result

def neon_effect(frame, hand_mask):
    result = frame.copy()
    if hand_mask is None: return result
    glow = cv2.GaussianBlur(frame, (0,0), 18)
    cv2.addWeighted(glow, 0.45, result, 0.55, 0, dst=result)
    neon = np.zeros_like(result)
    neon[:,:,1] = hand_mask
    neon[:,:,0] = hand_mask // 3
    return cv2.addWeighted(result, 1.0, neon, 0.5, 0)

def make_hand_mask(landmarks, h, w, radius=55):
    mask = np.zeros((h, w), dtype=np.uint8)
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(mask, (cx, cy), radius, 255, -1)
    return cv2.GaussianBlur(mask, (31, 31), 0)

def draw_skeleton(frame, landmarks, h, w):
    pts = {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(landmarks)}
    for a, b in HAND_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], LINE_COLOR, 2, cv2.LINE_AA)
    for i, (cx, cy) in pts.items():
        cv2.circle(frame, (cx, cy), 6, DOT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 8, (255,255,255), 1, cv2.LINE_AA)

# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    src = DROIDCAM_URL if USE_DROIDCAM else CAMERA_SOURCE
    cap = cv2.VideoCapture(src)

    effects    = ["chromatic","mirror","glitch","trails","neon"]
    eff_idx    = effects.index(EFFECT)
    current_fx = EFFECT
    show_skel  = True
    fps_s      = 0.0
    prev_t     = time.perf_counter()

    win = "EUNICE — Hand FX"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print(f"[HandFX] Running. Keys: E=cycle effects | S=skeleton | F=fullscreen | Q=quit")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05); continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # Detect
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)
        all_lms = result.hand_landmarks  # list of lists

        # Build combined mask
        combined = None
        if all_lms:
            combined = np.zeros((h, w), dtype=np.uint8)
            for lms in all_lms:
                combined = cv2.bitwise_or(combined, make_hand_mask(lms, h, w))

        # Apply effect
        if   current_fx == "chromatic": frame = chromatic_aberration(frame, combined)
        elif current_fx == "mirror":    frame = mirror_effect(frame, combined, h, w)
        elif current_fx == "glitch":    frame = glitch_effect(frame, combined)
        elif current_fx == "trails":    frame = trails_effect(frame, combined)
        elif current_fx == "neon":      frame = neon_effect(frame, combined)

        # Skeleton
        if show_skel and all_lms:
            for lms in all_lms:
                draw_skeleton(frame, lms, h, w)

        # HUD bar
        now   = time.perf_counter()
        fps_s = fps_s * 0.9 + (1.0 / max(now - prev_t, 1e-6)) * 0.1
        prev_t = now
        cv2.rectangle(frame, (0,0), (w,36), (0,0,0), -1)
        cv2.putText(frame,
            f"EUNICE  FX:{current_fx.upper()}  FPS:{int(fps_s)}  Hands:{len(all_lms)}  [E]=cycle [S]=skel [F]=full [Q]=quit",
            (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,220,255), 1, cv2.LINE_AA)

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF

        if   key in (ord('q'), 27): break
        elif key == ord('e'):
            eff_idx    = (eff_idx + 1) % len(effects)
            current_fx = effects[eff_idx]
            trail_frames.clear()
            print(f"[HandFX] Effect → {current_fx}")
        elif key == ord('s'): show_skel = not show_skel
        elif key == ord('f'):
            fs = cv2.getWindowProperty(win, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fs != cv2.WINDOW_FULLSCREEN else cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
