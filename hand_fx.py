"""
Eunice HUD — Hand Tracking FX
Real-time hand skeleton + psychedelic glitch effects
Run: python3 hand_fx.py
pip3 install mediapipe opencv-python numpy
"""

import cv2
import numpy as np
import mediapipe as mp
import time

# ── CONFIG ─────────────────────────────────────────────────────────────────────
CAMERA_SOURCE   = 0
DROIDCAM_URL    = "http://192.168.29.167:4747/video"
USE_DROIDCAM    = False
EFFECT          = "chromatic"   # "chromatic" | "mirror" | "glitch" | "trails" | "neon"
SHOW_SKELETON   = True
FULLSCREEN      = False

mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles

# ── EFFECTS ────────────────────────────────────────────────────────────────────

def chromatic_aberration(frame, hand_mask, intensity=18):
    """Split RGB channels around hand region — holographic glitch look"""
    result = frame.copy()
    if hand_mask is None:
        return result
    b, g, r = cv2.split(frame)
    shift = intensity
    # Shift channels in opposite directions inside hand region
    r_shift  = np.roll(r,  shift,  axis=1)
    b_shift  = np.roll(b, -shift,  axis=1)
    g_shift  = np.roll(g,  shift//2, axis=0)
    glitched = cv2.merge([b_shift, g_shift, r_shift])
    # Apply only where hand mask is active
    mask3 = cv2.cvtColor(hand_mask, cv2.COLOR_GRAY2BGR) / 255.0
    result = (frame * (1 - mask3) + glitched * mask3).astype(np.uint8)
    return result

def mirror_effect(frame, hand_mask, landmarks_list, h, w):
    """Echo/ghost copies of hands fanning out"""
    result = frame.copy()
    if hand_mask is None or not landmarks_list:
        return result
    for i in range(1, 6):
        alpha  = 0.18 * (6 - i) / 5
        shift  = i * 18
        kernel = np.float32([[1,0,shift],[0,1,0]])
        shifted = cv2.warpAffine(frame, kernel, (w, h))
        cv2.addWeighted(result, 1.0, shifted, alpha, 0, dst=result)
        kernel2 = np.float32([[1,0,-shift],[0,1,0]])
        shifted2 = cv2.warpAffine(frame, kernel2, (w, h))
        cv2.addWeighted(result, 1.0, shifted2, alpha, 0, dst=result)
    return result

trail_frames = []
def trails_effect(frame, hand_mask, max_trails=8):
    """Motion ghost trails"""
    global trail_frames
    trail_frames.append(frame.copy())
    if len(trail_frames) > max_trails:
        trail_frames.pop(0)
    result = frame.copy()
    for i, f in enumerate(trail_frames[:-1]):
        alpha = (i + 1) / len(trail_frames) * 0.25
        cv2.addWeighted(result, 1.0, f, alpha, 0, dst=result)
    return result

def glitch_effect(frame, hand_mask):
    """Random scanline glitches on hand region"""
    result = frame.copy()
    if hand_mask is None:
        return result
    h, w = frame.shape[:2]
    for _ in range(12):
        y     = np.random.randint(0, h)
        shift = np.random.randint(-30, 30)
        if 0 <= y < h:
            result[y] = np.roll(result[y], shift, axis=0)
    return result

def neon_effect(frame, hand_mask):
    """Neon glow over hand"""
    result = frame.copy()
    if hand_mask is None:
        return result
    glow = cv2.GaussianBlur(frame, (0, 0), 15)
    cv2.addWeighted(glow, 0.5, result, 0.5, 0, dst=result)
    neon_layer = np.zeros_like(frame)
    neon_layer[:, :, 1] = hand_mask  # green channel
    neon_layer[:, :, 0] = hand_mask // 2  # slight blue
    cv2.addWeighted(result, 1.0, neon_layer, 0.4, 0, dst=result)
    return result

# ── HAND MASK ──────────────────────────────────────────────────────────────────
def make_hand_mask(landmarks, h, w, radius=60):
    mask = np.zeros((h, w), dtype=np.uint8)
    for lm in landmarks.landmark:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(mask, (cx, cy), radius, 255, -1)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    return mask

# Custom skeleton colors
DOT_COLOR  = (0, 50, 255)    # red dots
LINE_COLOR = (0, 255, 80)    # green lines
DOT_RADIUS = 6
LINE_THICK = 2

CONNECTIONS = mp_hands.HAND_CONNECTIONS

def draw_skeleton(frame, landmarks, h, w):
    pts = {}
    for i, lm in enumerate(landmarks.landmark):
        cx, cy = int(lm.x * w), int(lm.y * h)
        pts[i] = (cx, cy)

    for conn in CONNECTIONS:
        a, b = conn
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], LINE_COLOR, LINE_THICK, cv2.LINE_AA)

    for i, (cx, cy) in pts.items():
        cv2.circle(frame, (cx, cy), DOT_RADIUS, DOT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), DOT_RADIUS + 2, (255, 255, 255), 1, cv2.LINE_AA)

# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    src = DROIDCAM_URL if USE_DROIDCAM else CAMERA_SOURCE
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG if isinstance(src, str) else cv2.CAP_ANY)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    effect_list = ["chromatic", "mirror", "glitch", "trails", "neon"]
    effect_idx  = effect_list.index(EFFECT)
    current_fx  = EFFECT

    prev_t = time.perf_counter()
    fps_s  = 0.0

    win = "EUNICE — Hand FX"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if FULLSCREEN:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print(f"[HandFX] Running. Keys: Q=quit | E=cycle effects | F=fullscreen | S=toggle skeleton")
    print(f"[HandFX] Current effect: {current_fx}")

    show_skeleton = SHOW_SKELETON

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)  # mirror for natural feel
        h, w  = frame.shape[:2]

        # Hand detection
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        combined_mask     = None
        landmarks_list    = results.multi_hand_landmarks or []

        if landmarks_list:
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            for lm in landmarks_list:
                m = make_hand_mask(lm, h, w)
                combined_mask = cv2.bitwise_or(combined_mask, m)

        # Apply effect
        if   current_fx == "chromatic": frame = chromatic_aberration(frame, combined_mask)
        elif current_fx == "mirror":    frame = mirror_effect(frame, combined_mask, landmarks_list, h, w)
        elif current_fx == "glitch":    frame = glitch_effect(frame, combined_mask)
        elif current_fx == "trails":    frame = trails_effect(frame, combined_mask)
        elif current_fx == "neon":      frame = neon_effect(frame, combined_mask)

        # Draw skeleton on top
        if show_skeleton and landmarks_list:
            for lm in landmarks_list:
                draw_skeleton(frame, lm, h, w)

        # HUD overlay
        now  = time.perf_counter()
        fps_s = fps_s * 0.9 + (1.0 / max(now - prev_t, 1e-6)) * 0.1
        prev_t = now

        cv2.rectangle(frame, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.putText(frame, f"EUNICE  FX:{current_fx.upper()}  FPS:{int(fps_s)}  Hands:{len(landmarks_list)}  [E]=cycle  [S]=skeleton  [Q]=quit",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('e'):
            effect_idx  = (effect_idx + 1) % len(effect_list)
            current_fx  = effect_list[effect_idx]
            trail_frames.clear()
            print(f"[HandFX] Effect → {current_fx}")
        elif key == ord('f'):
            fs = cv2.getWindowProperty(win, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fs != cv2.WINDOW_FULLSCREEN else cv2.WINDOW_NORMAL)
        elif key == ord('s'):
            show_skeleton = not show_skeleton

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()
