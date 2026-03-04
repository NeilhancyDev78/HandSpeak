"""
collector/screens/update.py — Single-gesture update screen.

Lets the user pick one label from the full list, wipes only its
existing samples, and re-collects fresh ones. All other labels
remain completely untouched.

Flow:
  1. PICK state    — scrollable label grid; click or arrow-key to select
  2. CONFIRM state — shows current sample count; confirm or go back
  3. COUNTDOWN     — 3-second arcane charge-up (SPACE to trigger)
  4. CAPTURING     — live sample collection with progress bar
  5. POSTCOOL      — 2-second buffer, then back to PICK
"""

import pygame
import math
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np


# ── Palette ───────────────────────────────────────────────────────────────────
BG       = (  5,   5,  12)
GOLD     = (212, 175,  55)
GOLD_DIM = ( 90,  70,  18)
CYAN     = (  0, 210, 220)
GREEN    = ( 50, 220,  80)
RED      = (220,  60,  60)
WHITE    = (255, 255, 255)
GREY     = (120, 120, 140)
PANEL_BG = ( 14,  14,  28)
DARK     = (  8,   8,  18)

# ── States ────────────────────────────────────────────────────────────────────
PICK      = "PICK"
CONFIRM   = "CONFIRM"
COUNTDOWN = "COUNTDOWN"
CAPTURING = "CAPTURING"
POSTCOOL  = "POSTCOOL"

PRE_SECS  = 3
POST_SECS = 2


class UpdateScreen:
    """
    Single-gesture re-collection screen.
    Returns "menu" when done or aborted.
    """

    def __init__(self, screen: pygame.Surface,
                 labels: list[str],
                 data_manager,
                 config: dict):
        self._screen  = screen
        self._labels  = labels
        self._dm      = data_manager
        self._cfg     = config
        self._W, self._H = screen.get_size()
        self._cx      = self._W // 2

        col_cfg = config.get("collector", {})
        self._target    = col_cfg.get("samples_per_gesture", 200)

        self._font_huge  = pygame.font.SysFont("consolas", 100, bold=True)
        self._font_large = pygame.font.SysFont("consolas",  34, bold=True)
        self._font_med   = pygame.font.SysFont("consolas",  22)
        self._font_small = pygame.font.SysFont("consolas",  15)

        self._t0         = time.time()
        self._state      = PICK
        self._selected   = 0
        self._state_ts   = time.time()
        self._samples: list[np.ndarray] = []
        self._cam_frame  = None
        self._hand_found = False
        self._last_lms   = None
        self._last_feat  = None

        self._cap        = cv2.VideoCapture(0)
        self._landmarker = self._build_landmarker(config)

        # Grid layout for pick screen
        self._cols   = 7
        self._cell_w = 90
        self._cell_h = 60

    # ── Public entry ──────────────────────────────────────────────────────────

    def run(self) -> str:
        clock = pygame.time.Clock()
        try:
            while True:
                clock.tick(60)
                result = self._handle_events()
                if result:
                    return result
                self._read_camera()
                self._update_state()
                self._draw()
                pygame.display.flip()
        finally:
            self._cap.release()

    # ── Events ────────────────────────────────────────────────────────────────

    def _handle_events(self) -> str | None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "menu"
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    if self._state == PICK:
                        return "menu"
                    else:
                        self._state = PICK
                elif event.key == pygame.K_SPACE:
                    self._on_space()
                elif event.key == pygame.K_r:
                    self._on_redo()
                elif self._state == PICK:
                    self._handle_grid_keys(event.key)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self._state == PICK:
                    self._handle_grid_click(event.pos)
        return None

    def _handle_grid_keys(self, key) -> None:
        n = len(self._labels)
        if key == pygame.K_RIGHT:
            self._selected = (self._selected + 1) % n
        elif key == pygame.K_LEFT:
            self._selected = (self._selected - 1) % n
        elif key == pygame.K_DOWN:
            self._selected = min(self._selected + self._cols, n - 1)
        elif key == pygame.K_UP:
            self._selected = max(self._selected - self._cols, 0)
        elif key == pygame.K_RETURN:
            self._state    = CONFIRM
            self._state_ts = time.time()

    def _handle_grid_click(self, pos) -> None:
        for i, rect in enumerate(getattr(self, "_cell_rects", [])):
            if rect.collidepoint(pos):
                if i < len(self._labels):
                    self._selected = i
                    self._state    = CONFIRM
                    self._state_ts = time.time()

    def _on_space(self) -> None:
        if self._state == CONFIRM:
            self._dm.delete(self._labels[self._selected])
            self._samples  = []
            self._state    = COUNTDOWN
            self._state_ts = time.time()

    def _on_redo(self) -> None:
        if self._state in (CAPTURING, POSTCOOL):
            self._samples  = []
            self._state    = COUNTDOWN
            self._state_ts = time.time()

    # ── Camera ────────────────────────────────────────────────────────────────

    def _read_camera(self) -> None:
        ret, frame = self._cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        self._cam_frame = self._bgr_to_surface(cv2.resize(frame, (480, 270)))
        self._detect(frame)

    def _detect(self, frame) -> None:
        small  = cv2.resize(frame, (480, 360))
        rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res    = self._landmarker.detect(mp_img)
        if res.hand_landmarks:
            lms = res.hand_landmarks[0]
            self._last_lms   = lms
            self._last_feat  = self._normalise(lms)
            self._hand_found = True
        else:
            self._last_lms   = None
            self._last_feat  = None
            self._hand_found = False

    # ── State machine ─────────────────────────────────────────────────────────

    def _update_state(self) -> None:
        now = time.time()
        if self._state == COUNTDOWN:
            if now - self._state_ts >= PRE_SECS:
                self._state    = CAPTURING
                self._state_ts = now

        elif self._state == CAPTURING:
            if self._hand_found and self._last_feat is not None:
                self._samples.append(self._last_feat)
            if len(self._samples) >= self._target:
                self._dm.save(self._labels[self._selected], self._samples)
                self._state    = POSTCOOL
                self._state_ts = now

        elif self._state == POSTCOOL:
            if now - self._state_ts >= POST_SECS:
                self._state = PICK

    # ── Draw dispatcher ───────────────────────────────────────────────────────

    def _draw(self) -> None:
        t = time.time() - self._t0
        self._screen.fill(BG)
        self._draw_bg_mandala(t)
        if self._state == PICK:
            self._draw_pick(t)
        else:
            self._draw_capture_view(t)

    # ── Pick screen ───────────────────────────────────────────────────────────

    def _draw_pick(self, t: float) -> None:
        title = self._font_large.render("SELECT GESTURE TO UPDATE", True, GOLD)
        self._screen.blit(title, title.get_rect(center=(self._cx, 54)))

        rows    = math.ceil(len(self._labels) / self._cols)
        total_w = self._cols * self._cell_w + (self._cols - 1) * 10
        start_x = self._cx - total_w // 2
        start_y = 110

        self._cell_rects = []
        for i, lbl in enumerate(self._labels):
            col_i = i % self._cols
            row_i = i // self._cols
            x = start_x + col_i * (self._cell_w + 10)
            y = start_y + row_i * (self._cell_h + 10)
            rect = pygame.Rect(x, y, self._cell_w, self._cell_h)
            self._cell_rects.append(rect)

            locked  = self._dm.sample_count(lbl) >= self._target
            sel     = (i == self._selected)
            pulse   = 0.7 + 0.3 * math.sin(t * 2.5) if sel else 1.0
            bg_col  = (30, 30, 70) if sel else PANEL_BG
            bd_col  = tuple(int(c * pulse) for c in GOLD) if sel else (GOLD_DIM if locked else GREY)

            pygame.draw.rect(self._screen, bg_col,  rect, border_radius=4)
            pygame.draw.rect(self._screen, bd_col,  rect, 2, border_radius=4)

            cnt     = self._dm.sample_count(lbl)
            lbl_col = WHITE if sel else (GREEN if locked else GREY)
            ls      = self._font_med.render(lbl, True, lbl_col)
            cs      = self._font_small.render(str(cnt), True, CYAN if sel else GOLD_DIM)
            self._screen.blit(ls, ls.get_rect(center=(x + self._cell_w//2, y + 20)))
            self._screen.blit(cs, cs.get_rect(center=(x + self._cell_w//2, y + 42)))

        hint = "[ ← → ↑ ↓ ] NAVIGATE     [ ENTER / CLICK ] SELECT     [ ESC ] BACK"
        hs   = self._font_small.render(hint, True, GOLD_DIM)
        self._screen.blit(hs, hs.get_rect(center=(self._cx, self._H - 22)))

    # ── Capture / confirm view ────────────────────────────────────────────────

    def _draw_capture_view(self, t: float) -> None:
        now   = time.time()
        label = self._labels[self._selected]

        # Camera feed
        if self._cam_frame:
            surf = pygame.transform.scale(self._cam_frame, (480, 270))
            x, y = 20, 20
            self._screen.blit(surf, (x, y))
            pygame.draw.rect(self._screen, GOLD_DIM, (x-2, y-2, 484, 274), 2)
            if self._last_lms:
                for lm in self._last_lms:
                    px = x + int(lm.x * 480)
                    py = y + int(lm.y * 270)
                    pygame.draw.circle(self._screen, GOLD, (px, py), 4)

        cx_hud = self._cx + 60
        cy_hud = 180

        # Big letter
        pulse = 0.85 + 0.15 * math.sin(t * 2.2)
        ltr   = self._font_huge.render(label, True,
                                       tuple(int(c * pulse) for c in GOLD))
        self._screen.blit(ltr, ltr.get_rect(center=(cx_hud, cy_hud)))

        if self._state == CONFIRM:
            cnt     = self._dm.sample_count(label)
            info    = self._font_med.render(
                f"CURRENT SAMPLES: {cnt}  →  WILL BE REPLACED", True, RED)
            confirm = self._font_med.render(
                "PRESS [ SPACE ] TO CONFIRM & START", True, CYAN)
            self._screen.blit(info,    info.get_rect(center=(cx_hud, cy_hud + 90)))
            self._screen.blit(confirm, confirm.get_rect(center=(cx_hud, cy_hud + 122)))

        elif self._state == COUNTDOWN:
            remaining = PRE_SECS - (now - self._state_ts)
            msg = self._font_med.render(
                f"HOLD STILL  ...  {remaining:.1f}s", True, GOLD)
            self._screen.blit(msg, msg.get_rect(center=(cx_hud, cy_hud + 90)))
            self._draw_countdown_ring(cx_hud, cy_hud, remaining / PRE_SECS)

        elif self._state == CAPTURING:
            pct = len(self._samples) / self._target
            msg = self._font_med.render(
                f"CAPTURING  {len(self._samples)} / {self._target}", True, GREEN)
            self._screen.blit(msg, msg.get_rect(center=(cx_hud, cy_hud + 90)))
            self._draw_capture_bar(pct, cx_hud, cy_hud + 118)

        elif self._state == POSTCOOL:
            remaining = POST_SECS - (now - self._state_ts)
            msg = self._font_med.render(
                f"SAVED  ✦  RETURNING IN  {remaining:.1f}s", True, GREEN)
            self._screen.blit(msg, msg.get_rect(center=(cx_hud, cy_hud + 90)))

        hint = "[ R ] REDO     [ ESC ] BACK TO MENU"
        hs   = self._font_small.render(hint, True, GOLD_DIM)
        self._screen.blit(hs, hs.get_rect(center=(self._cx, self._H - 22)))

    def _draw_countdown_ring(self, cx, cy, frac) -> None:
        r        = 70
        arc_rect = pygame.Rect(cx - r, cy - r - 10, r*2, r*2)
        end_a    = math.radians(-90 + 360 * (1 - frac))
        try:
            pygame.draw.arc(self._screen, GOLD, arc_rect,
                            math.radians(-90), end_a, 5)
        except Exception:
            pass

    def _draw_capture_bar(self, pct: float, cx: int, y: int) -> None:
        bw, bh = 300, 12
        x  = cx - bw // 2
        pygame.draw.rect(self._screen, DARK, (x, y, bw, bh), border_radius=4)
        if pct > 0:
            pygame.draw.rect(self._screen, GREEN,
                             (x, y, int(bw * pct), bh), border_radius=4)
        pygame.draw.rect(self._screen, GOLD_DIM, (x, y, bw, bh), 1, border_radius=4)

    def _draw_bg_mandala(self, t: float) -> None:
        cx, cy = self._cx, self._H // 2
        for ring in range(3):
            r   = 280 + ring * 80
            spk = 8 + ring * 4
            ao  = t * (0.08 + ring * 0.03) * (1 if ring % 2 == 0 else -1)
            pts = [
                (cx + r * math.cos(math.radians(360/spk*i) + ao),
                 cy + r * math.sin(math.radians(360/spk*i) + ao))
                for i in range(spk)
            ]
            for i in range(spk):
                pygame.draw.line(self._screen, GOLD_DIM,
                                 (int(pts[i][0]), int(pts[i][1])),
                                 (int(pts[(i+1)%spk][0]), int(pts[(i+1)%spk][1])), 1)

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(landmarks) -> np.ndarray:
        pts = np.array(
            [[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32
        )
        pts -= pts[0]
        scale = np.max(np.abs(pts))
        if scale > 0:
            pts /= scale
        return pts.flatten()

    @staticmethod
    def _bgr_to_surface(frame) -> pygame.Surface:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))

    @staticmethod
    def _build_landmarker(config: dict):
        inf  = config.get("inference", {})
        base = mp_python.BaseOptions(
            model_asset_path=inf.get("landmark_model", "hand_landmarker.task")
        )
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base,
            num_hands=1,
            min_hand_detection_confidence=inf.get("detection_confidence", 0.6),
            min_hand_presence_confidence=inf.get("presence_confidence",   0.6),
            min_tracking_confidence=inf.get("tracking_confidence",        0.5),
        )
        return mp_vision.HandLandmarker.create_from_options(opts)