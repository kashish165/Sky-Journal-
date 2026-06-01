"""
╔══════════════════════════════════════════════════════════════╗
║              SKY JOURNAL — Hand Gesture Collage              ║
╠══════════════════════════════════════════════════════════════╣
║  ☝  ONE FINGER   → Hover over photos to highlight them      ║
║  ✌  TWO FINGERS  → Explode / scatter all photos outward     ║
║  🤚  OPEN PALM   → Pan / drag the whole collage             ║
║  ✊  FIST        → Collapse everything back home            ║
║  👌  PINCH       → Zoom into nearest photo                  ║
║  Q / ESC         → Quit                                      ║
╚══════════════════════════════════════════════════════════════╝
  Just run:  python sky_journal.py
  All dependencies install automatically on first run.
"""

# ══════════════════════════════════════════════════════════════
#  AUTO-INSTALL  — runs before anything else
# ══════════════════════════════════════════════════════════════
import sys, subprocess

REQUIRED = [
    ("cv2",       "opencv-python"),
    ("mediapipe", "mediapipe"),
    ("numpy",     "numpy"),
]

def _install(pkg):
    print(f"  [installing] {pkg} ...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

needs_restart = False
for import_name, pip_name in REQUIRED:
    try:
        __import__(import_name)
    except ImportError:
        print(f"[setup] Missing package: {pip_name}")
        _install(pip_name)
        needs_restart = True

if needs_restart:
    print("\n[setup] All packages installed — restarting...\n")
    subprocess.check_call([sys.executable] + sys.argv)
    sys.exit(0)

# ══════════════════════════════════════════════════════════════
#  IMPORTS (guaranteed to be present now)
# ══════════════════════════════════════════════════════════════
import cv2
import mediapipe as mp
import numpy as np
import os, glob, math, random, time

# ─── CONFIG ──────────────────────────────────────────────────────────────────
GALLERY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gallery")
WIN_W, WIN_H = 1280, 800
CAM_W, CAM_H = 340, 255
CAM_X, CAM_Y = 30, 18
BG_COLOR = (210, 235, 195)        # light lime-yellow BGR

WEATHER_WORDS = [
    "clear", "cloudy", "windy", "breezy", "stormy", "sunny", "rainy",
    "misty", "bright", "gentle", "hopeful", "tired", "dull", "playful",
    "overcast", "peaceful", "golden", "crisp", "hazy", "vivid", "soft",
]
MONTHS = ["jan","feb","mar","apr","may","jun",
          "jul","aug","sep","oct","nov","dec"]

mp_hands = mp.solutions.hands


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def lerp(a, b, t):
    return a + (b - a) * t

def dist2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def load_gallery(path):
    files = []
    for ext in ("*.jpg","*.jpeg","*.png","*.webp","*.bmp",
                "*.JPG","*.PNG","*.JPEG","*.BMP","*.WEBP"):
        files += glob.glob(os.path.join(path, ext))
    imgs = []
    for f in sorted(files):
        im = cv2.imread(f)
        if im is not None:
            imgs.append((os.path.basename(f), im))
    return imgs

def resize_fit(img, w, h):
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    return cv2.resize(img, (max(1, int(iw * s)), max(1, int(ih * s))))

def rotate_image_bg(img, angle, bg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos_a = abs(M[0, 0]); sin_a = abs(M[0, 1])
    nw = int(h * sin_a + w * cos_a)
    nh = int(h * cos_a + w * sin_a)
    M[0, 2] += (nw - w) / 2
    M[1, 2] += (nh - h) / 2
    return cv2.warpAffine(img, M, (nw, nh),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=bg)

def overlay(canvas, img, cx, cy, alpha=1.0):
    rh, rw = img.shape[:2]
    x1 = cx - rw // 2; y1 = cy - rh // 2
    x2 = x1 + rw;      y2 = y1 + rh
    ix1 = max(0, -x1); iy1 = max(0, -y1)
    ix2 = rw - max(0, x2 - canvas.shape[1])
    iy2 = rh - max(0, y2 - canvas.shape[0])
    cx1 = max(0, x1); cy1 = max(0, y1)
    cx2 = cx1 + ix2 - ix1
    cy2 = cy1 + iy2 - iy1
    if cx2 <= cx1 or cy2 <= cy1 or ix2 <= ix1 or iy2 <= iy1:
        return
    roi   = canvas[cy1:cy2, cx1:cx2].astype(np.float32)
    patch = img[iy1:iy2, ix1:ix2].astype(np.float32)
    canvas[cy1:cy2, cx1:cx2] = (roi * (1 - alpha) + patch * alpha).astype(np.uint8)

def txt(canvas, text, x, y, scale=0.5, color=(40, 40, 40), thick=1):
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


# ─── GESTURE DETECTOR ────────────────────────────────────────────────────────
class GestureDetector:
    @staticmethod
    def fingers_up(lm):
        up = [lm[4].x < lm[3].x]   # thumb (mirrored)
        for tip, pip in [(8,6),(12,10),(16,14),(20,18)]:
            up.append(lm[tip].y < lm[pip].y)
        return up

    @staticmethod
    def detect(lm):
        up    = GestureDetector.fingers_up(lm)
        n     = sum(up[1:])
        pinch = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y) < 0.055
        if pinch and n <= 1:             return "pinch"
        if n == 0:                       return "fist"
        if n == 1 and up[1]:             return "one_finger"
        if n == 2 and up[1] and up[2]:   return "two_fingers"
        if n >= 4:                       return "open_palm"
        return "other"

    @staticmethod
    def tip(lm, i=8):
        return lm[i].x, lm[i].y


# ─── PHOTO CARD ──────────────────────────────────────────────────────────────
class PhotoCard:
    def __init__(self, name, img, idx, total):
        self.name = name

        # date label from filename
        stem = os.path.splitext(name)[0].lower().replace("_", " ").replace("-", " ")
        self.date_label = ""
        for token in stem.split():
            for m in MONTHS:
                if m in token:
                    nums = "".join(c for c in token if c.isdigit())
                    self.date_label = f"{m} {nums}" if nums else m
                    break
            if self.date_label:
                break
        if not self.date_label:
            self.date_label = f"day {idx + 1:02d}"
        self.mood = random.choice(WEATHER_WORDS)

        # thumbnail
        tw = random.randint(105, 175)
        th = int(tw * random.uniform(0.6, 0.72))
        self.thumb = resize_fit(img, tw, th)
        self.tw = self.thumb.shape[1]
        self.th = self.thumb.shape[0]

        # home position — scattered ring around canvas center
        angle  = (idx / total) * 2 * math.pi + random.uniform(-0.6, 0.6)
        radius = random.uniform(160, 360)
        cx0, cy0 = WIN_W // 2 + 55, WIN_H // 2 + 90
        self.home_x = float(max(110, min(WIN_W - 110,
                            cx0 + math.cos(angle) * radius + random.randint(-45, 45))))
        self.home_y = float(max(100, min(WIN_H - 100,
                            cy0 + math.sin(angle) * radius + random.randint(-45, 45))))

        self.x, self.y  = float(WIN_W // 2), float(WIN_H // 2)
        self.home_rot   = random.uniform(-28, 28)
        self.rot        = self.home_rot
        self.scale      = 0.0
        self.target_x   = self.home_x
        self.target_y   = self.home_y
        self.target_s   = 1.0
        self.target_rot = self.home_rot
        self.alpha      = 0.0
        self.born       = time.time() + idx * 0.07   # staggered spawn
        self.hovered    = False
        self.z          = random.random()

    def update(self):
        if time.time() < self.born:
            return
        self.alpha = min(1.0, (time.time() - self.born) * 2.2)
        sp = 0.13
        self.x     = lerp(self.x,     self.target_x,   sp)
        self.y     = lerp(self.y,     self.target_y,   sp)
        self.scale = lerp(self.scale, self.target_s,   sp)
        self.rot   = lerp(self.rot,   self.target_rot, sp)

    def rendered(self):
        s  = max(0.05, self.scale)
        sw = max(4, int(self.tw * s))
        sh = max(4, int(self.th * s))
        return rotate_image_bg(cv2.resize(self.thumb, (sw, sh)), self.rot, BG_COLOR)

    def draw(self, canvas):
        if self.alpha < 0.02 or time.time() < self.born:
            return
        img = self.rendered()
        rh, rw = img.shape[:2]
        ix, iy = int(self.x), int(self.y)

        # drop shadow
        overlay(canvas, np.zeros((rh, rw, 3), dtype=np.uint8),
                ix + 6, iy + 6, alpha=self.alpha * 0.22)

        # white polaroid border
        b = max(2, int(5 * self.scale))
        bordered = cv2.copyMakeBorder(img, b, b, b, b,
                                      cv2.BORDER_CONSTANT, value=(252, 252, 252))
        overlay(canvas, bordered, ix, iy, alpha=self.alpha)

        # hover label
        if self.hovered:
            label = f"{self.date_label}, {self.mood}"
            lw    = len(label) * 7 + 8
            lx    = max(4, min(WIN_W - lw - 4, ix - lw // 2))
            ly    = iy + rh // 2 + 26
            cv2.rectangle(canvas, (lx - 2, ly - 15), (lx + lw, ly + 4), (8, 8, 8), -1)
            txt(canvas, label, lx + 2, ly, 0.44, (215, 215, 215), 1)

    def hit(self, px, py):
        ri = self.rendered()
        rh, rw = ri.shape[:2]
        return abs(px - self.x) < rw // 2 + 8 and abs(py - self.y) < rh // 2 + 8


# ─── AMBIENT FLOATING WORDS ───────────────────────────────────────────────────
class FloatingWord:
    def __init__(self):
        self.respawn()

    def respawn(self):
        if random.random() < 0.4:
            self.word = f"{random.choice(MONTHS)} {random.randint(1,28):02d}"
        else:
            self.word = random.choice(WEATHER_WORDS)
        self.x    = random.randint(20, WIN_W - 40)
        self.y    = random.randint(55, WIN_H - 20)
        self.a    = random.uniform(0.07, 0.20)
        self.size = random.uniform(0.45, 0.95)
        self.vx   = random.uniform(-0.04, 0.04)
        self.vy   = random.uniform(-0.025, 0.015)
        self.life = random.uniform(9, 22)
        self.born = time.time()

    def update(self):
        self.x += self.vx
        self.y += self.vy
        if time.time() - self.born > self.life:
            self.respawn()

    def draw(self, canvas):
        c = int(min(255, 50 * self.a * 5))
        txt(canvas, self.word, int(self.x), int(self.y), self.size, (c, c, c), 1)


# ─── CONSTELLATION LINES ──────────────────────────────────────────────────────
def draw_lines(canvas, cards):
    visible = [c for c in cards if c.alpha > 0.35]
    for i, a in enumerate(visible):
        for b in visible[i + 1:]:
            d = dist2((a.x, a.y), (b.x, b.y))
            if d < 210:
                cv2.line(canvas, (int(a.x), int(a.y)), (int(b.x), int(b.y)),
                         (40, 50, 30), 1, cv2.LINE_AA)


# ─── STARBURST EFFECT ────────────────────────────────────────────────────────
class Starburst:
    def __init__(self):
        self.active = False
        self.x = self.y = 0
        self.t = 0.0

    def trigger(self, x, y):
        self.active = True
        self.x, self.y, self.t = x, y, 0.0

    def update(self):
        if self.active:
            self.t += 0.045
            if self.t >= 1.0:
                self.active = False

    def draw(self, canvas):
        if not self.active:
            return
        n     = 20
        max_l = int(220 * math.sin(self.t * math.pi))
        fade  = max(0.0, 1.0 - self.t)
        for i in range(n):
            ang = (i / n) * 2 * math.pi
            ll  = max_l * random.uniform(0.35, 1.0)
            ex  = int(self.x + math.cos(ang) * ll)
            ey  = int(self.y + math.sin(ang) * ll)
            br  = int(255 * fade)
            cv2.line(canvas, (int(self.x), int(self.y)), (ex, ey),
                     (br, br, br), 2, cv2.LINE_AA)


# ─── WEBCAM WINDOW ────────────────────────────────────────────────────────────
def draw_cam_window(canvas, frame):
    pw, ph = CAM_W + 6, CAM_H + 32
    x0, y0 = CAM_X, CAM_Y
    cv2.rectangle(canvas, (x0, y0), (x0 + pw, y0 + ph), (228, 220, 232), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + pw, y0 + 28), (235, 228, 240), -1)
    for bx, col in [(x0 + pw - 24, (100, 80, 200)),
                    (x0 + pw - 46, (80, 80, 80)),
                    (x0 + pw - 68, (60, 60, 60))]:
        cv2.rectangle(canvas, (bx, y0 + 7), (bx + 15, y0 + 21), col, -1)
        cv2.rectangle(canvas, (bx, y0 + 7), (bx + 15, y0 + 21), (180, 180, 180), 1)
    if frame is not None:
        cf = cv2.flip(cv2.resize(frame, (CAM_W, CAM_H)), 1)
        canvas[y0 + 30: y0 + 30 + CAM_H, x0 + 3: x0 + 3 + CAM_W] = cf
    else:
        canvas[y0 + 30: y0 + 30 + CAM_H, x0 + 3: x0 + 3 + CAM_W] = (20, 20, 20)


# ─── GESTURE CURSOR ──────────────────────────────────────────────────────────
CURSOR_COLORS = {
    "one_finger" : (255, 255, 255),
    "two_fingers": (180, 255, 180),
    "fist"       : (180, 180, 255),
    "open_palm"  : (255, 220, 160),
    "pinch"      : (255, 180, 255),
}

def draw_cursor(canvas, gx, gy, gesture, trail):
    for i, (tx, ty) in enumerate(trail):
        r = max(1, int(5 * i / len(trail)))
        cv2.circle(canvas, (tx, ty), r, (255, 255, 255), -1, cv2.LINE_AA)
    col = CURSOR_COLORS.get(gesture, (200, 200, 200))
    cv2.circle(canvas, (gx, gy), 14, col, 2, cv2.LINE_AA)
    cv2.circle(canvas, (gx, gy), 4,  col, -1, cv2.LINE_AA)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    random.seed(7)

    # make gallery folder if it doesn't exist
    os.makedirs(GALLERY_DIR, exist_ok=True)

    gallery = load_gallery(GALLERY_DIR)
    if not gallery:
        print(f"\n[!] No photos found in: {GALLERY_DIR}")
        print("    Add your sky photos to the 'gallery' folder, then rerun.\n")
        input("Press Enter to exit...")
        return
    print(f"[✓] Loaded {len(gallery)} sky photos — opening Sky Journal...\n")

    cards   = [PhotoCard(name, img, i, len(gallery)) for i, (name, img) in enumerate(gallery)]
    cards.sort(key=lambda c: c.z)
    ambient = [FloatingWord() for _ in range(24)]
    burst   = Starburst()

    # state
    gesture = prev_gesture = "none"
    gx, gy  = WIN_W // 2, WIN_H // 2
    trail   = []
    pan_origin = None
    pan_home   = {}
    zoom_card  = None
    two_done   = False
    fist_done  = False

    hands_mp = mp_hands.Hands(
        model_complexity=0, max_num_hands=1,
        min_detection_confidence=0.65, min_tracking_confidence=0.55)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    cv2.namedWindow("Sky Journal", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sky Journal", WIN_W, WIN_H)

    cam_frame = None

    while True:
        # grab + process webcam frame
        ret, raw = cap.read()
        if ret:
            cam_frame = raw.copy()
            res = hands_mp.process(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks:
                lm      = res.multi_hand_landmarks[0].landmark
                gesture = GestureDetector.detect(lm)
                fx, fy  = GestureDetector.tip(lm, 8)
                gx = int((1 - fx) * WIN_W)
                gy = int(fy * WIN_H)
                trail.append((gx, gy))
                if len(trail) > 20:
                    trail.pop(0)
            else:
                gesture = "none"
                trail   = []

        # ── gesture logic ─────────────────────────────────────────────────────
        if gesture == "one_finger":
            two_done = fist_done = False
            for c in cards:
                c.hovered = c.hit(gx, gy)
                if c.hovered:
                    c.target_s   = 1.4
                    c.target_rot = 0.0
                elif zoom_card is None:
                    c.target_s   = 1.0
                    c.target_rot = c.home_rot

        elif gesture == "two_fingers":
            fist_done = False
            for c in cards:
                c.hovered = False
            if not two_done:
                two_done = True
                burst.trigger(WIN_W // 2, WIN_H // 2 + 80)
                for c in cards:
                    ang = random.uniform(0, 2 * math.pi)
                    r   = random.uniform(240, 510)
                    c.target_x   = max(80, min(WIN_W-80, WIN_W//2 + math.cos(ang)*r))
                    c.target_y   = max(80, min(WIN_H-80, WIN_H//2 + 80 + math.sin(ang)*r))
                    c.target_rot = random.uniform(-38, 38)
                    c.target_s   = 1.0

        elif gesture == "open_palm":
            two_done = fist_done = False
            if prev_gesture != "open_palm":
                pan_origin = (gx, gy)
                pan_home   = {id(c): (c.target_x, c.target_y) for c in cards}
            elif pan_origin:
                dx = (gx - pan_origin[0]) * 0.55
                dy = (gy - pan_origin[1]) * 0.55
                for c in cards:
                    hx, hy = pan_home.get(id(c), (c.x, c.y))
                    c.target_x = hx + dx
                    c.target_y = hy + dy
            for c in cards:
                c.hovered  = False
                c.target_s = 1.0

        elif gesture == "fist":
            if not fist_done:
                fist_done = True
                two_done  = False
                zoom_card = None
                for c in cards:
                    c.target_x   = c.home_x
                    c.target_y   = c.home_y
                    c.target_rot = c.home_rot
                    c.target_s   = 1.0
                    c.hovered    = False

        elif gesture == "pinch":
            two_done = fist_done = False
            nearest = min(cards, key=lambda c: dist2((gx, gy), (c.x, c.y)))
            if dist2((gx, gy), (nearest.x, nearest.y)) < 220:
                zoom_card = nearest
                for c in cards:
                    if c is nearest:
                        c.target_s   = 2.9
                        c.target_rot = 0.0
                        c.target_x   = WIN_W // 2 + 60
                        c.target_y   = WIN_H // 2 + 40
                    else:
                        c.target_s = 0.45

        else:  # none / other
            if zoom_card and prev_gesture == "pinch":
                for c in cards:
                    c.target_x   = c.home_x
                    c.target_y   = c.home_y
                    c.target_s   = 1.0
                    c.target_rot = c.home_rot
                zoom_card = None
            if prev_gesture == "open_palm":
                pan_origin = None

        prev_gesture = gesture

        # update
        for c in cards:   c.update()
        for w in ambient: w.update()
        burst.update()

        # draw
        canvas = np.full((WIN_H, WIN_W, 3), BG_COLOR, dtype=np.uint8)
        for w in ambient:  w.draw(canvas)
        draw_lines(canvas, cards)
        burst.draw(canvas)
        for c in sorted(cards, key=lambda c: (c is zoom_card, c.z)):
            c.draw(canvas)
        draw_cam_window(canvas, cam_frame)
        if gesture != "none":
            draw_cursor(canvas, gx, gy, gesture, trail)

        # gesture badge top-right
        labels = {"one_finger":"point","two_fingers":"scatter",
                  "fist":"collapse","open_palm":"pan","pinch":"zoom"}
        if gesture in labels:
            lb = labels[gesture]
            cv2.rectangle(canvas, (WIN_W-120, 10), (WIN_W-10, 34), (10,10,10), -1)
            txt(canvas, lb, WIN_W-112, 28, 0.52, (215,215,215), 1)

        txt(canvas, f"{len(gallery)} entries",
            CAM_X, CAM_Y + CAM_H + 56, 0.46, (80,80,70), 1)

        cv2.imshow("Sky Journal", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands_mp.close()
    print("[✓] Sky Journal closed.")


if __name__ == "__main__":
    main()
