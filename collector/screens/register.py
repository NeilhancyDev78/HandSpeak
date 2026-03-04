"""
collector/screens/register.py — Full A–Z dataset registration screen.

Flow per gesture:
  1. READY state    — side panel shows locked/pending labels
  2. SPACE pressed  → 3-second arcane countdown begins
  3. CAPTURING      — samples collected at up to config FPS limit
  4. POST-COOLDOWN  — 2-second pause after capture completes
  5. Advance to next label automatically; repeat from step 1

If all samples for a label are already present from a prior run,
that label is shown as locked and skipped automatically.

SPACE = begin / confirm next
R     = redo current gesture (wipes just-captured buffer, not saved data)
ESC/Q = abort and return to menu
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
BG       = (5,   5,  12)
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
READY      = "READY"
COUNTDOWN  = "COUNTDOWN"
CAPTURING  = "CAPTURING"
POSTCOOL   = "POSTCOOL"
DONE       = "DONE"

PRE_SECS   = 3
POST_SECS  = 2


class RegisterScreen:
    """
    Full A–Z (or any label list) collection screen.

    Returns "menu" when complete or aborted.
    """

    def __init__(self, screen: pygame.Surface,
                 labels: list[str],
                 data_manager,
                 config: dict):
        self._screen = screen
        self._labels = labels
        self._dm     = data_manager
        self._cfg    = config
        self._W, self._H = screen.get_size()
        self._cx = self._W // 2

        col_cfg = config.get("collector", {})
        self._target     = col_cfg.get("samples_per_gesture", 200)
        self._fps_limit  = col_cfg.get("capture_fps_limit",    30)

        self._font_huge  = pygame.font.SysFont("consolas", 120, bold=True)
        self._font_large = pygame.font.SysFont("consolas",  38, bold=True)
        self._font_med   = pygame.font.SysFont("consolas",  22)
        self._font_small = pygame.font.SysFont("consolas",  15)

        self._t0    = time.time()
        self._idx   = 0
        self._state = READY
        self._samples: list[np.ndarray] = []
        self._state_ts  = time.time()
        self._cam_frame = None      # latest BGR frame as pygame Surface

        # Skip already-locked labels upfront
        self._advance_to_next_pending()

        # Camera + tracker
        self._cap = cv2.VideoCapture(0)
        self._landmarker = self._build_landmarker(config)

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

    # ── Event handling ────────────────────────────────────────────────────────

    def _handle_events(self) -> str | None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "menu"
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return "menu"
                if event.key == pygame.K_SPACE:
                    self._on_space()
                if event.key == pygame.K_r:
                    self._on_redo()
        return None

    def _on_space(self) -> None:
        if self._state == READY:
            self._state    = COUNTDOWN
            self._state_ts = time.time()
            self._samples  = []

    def _on_redo(self) -> None:
        if self._state in (CAPTURING, POSTCOOL):
            self._samples  = []
            self._state    = READY
            self._state_ts = time.time()

    # ── Camera ────────────────────────────────────────────────────────────────

    def _read_camera(self) -> None:
        ret, frame = self._cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        self._raw_frame  = frame
        self._cam_frame  = self._bgr_to_surface(
            cv2.resize(frame, (480, 270))
        )
        self._detect(frame)

    def _detect(self, frame) -> None:
        small  = cv2.resize(frame, (480, 360))
        rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res    = self._landmarker.detect(mp_img)
        if res.hand_landmarks:
            lms = res.hand_landmarks[0]
            self._last_lms  = lms
            self._last_feat = self._normalise(lms)
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
                self._samples  = []

        elif self._state == CAPTURING:
            if self._hand_found and self._last_feat is not None:
                self._samples.append(self._last_feat)
            if len(self._samples) >= self._target:
                self._dm.save(self._current_label(), self._samples)
                self._state    = POSTCOOL
                self._state_ts = now

        elif self._state == POSTCOOL:
            if now - self._state_ts >= POST_SECS:
                self._idx += 1
                self._advance_to_next_pending()
                if self._idx >= len(self._labels):
                    self._state = DONE
                else:
                    self._state    = READY
                    self._state_ts = now

    # ── Draw ──────────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        t = time.time() - self._t0
        self._screen.fill(BG)
        self._draw_bg_mandala(t)
        self._draw_camera_feed()
        self._draw_side_panel()
        self._draw_center_hud(t)
        self._draw_hints()

    def _draw_bg_mandala(self, t: float) -> None:
        cx, cy = self._cx, self._H // 2
        for ring in range(4):
            r     = 260 + ring * 70
            spk   = 10 + ring * 3
            speed = 0.10 + ring * 0.04
            ao    = t * speed * (1 if ring % 2 == 0 else -1)
            pts   = [
                (cx + r * math.cos(math.radians(360/spk*i) + ao),
                 cy + r * math.sin(math.radians(360/spk*i) + ao))
                for i in range(spk)
            ]
            for i in range(spk):
                pygame.draw.line(self._screen, GOLD_DIM,
                                 (int(pts[i][0]), int(pts[i][1])),
                                 (int(pts[(i+1)%spk][0]), int(pts[(i+1)%spk][1])), 1)

    def _draw_camera_feed(self) -> None:
        if self._cam_frame is None:
            return
        # Place feed top-left with a gold border
        surf = pygame.transform.scale(self._cam_frame, (480, 270))
        x, y = 20, 20
        self._screen.blit(surf, (x, y))
        pygame.draw.rect(self._screen, GOLD_DIM,
                         (x-2, y-2, 484, 274), 2)

        # Overlay hand landmarks
        if self._last_lms:
            fw, fh = 480, 270
            for lm in self._last_lms:
                px = x + int(lm.x * fw)
                py = y + int(lm.y * fh)
                pygame.draw.circle(self._screen, GOLD, (px, py), 4)

    def _draw_side_panel(self) -> None:
        panel_x = self._W - 220
        panel_w = 200
        pygame.draw.rect(self._screen, PANEL_BG,
                         (panel_x, 10, panel_w, self._H - 20), border_radius=6)
        pygame.draw.rect(self._screen, GOLD_DIM,
                         (panel_x, 10, panel_w, self._H - 20), 1, border_radius=6)

        hdr = self._font_small.render("GESTURE INDEX", True, GOLD)
        self._screen.blit(hdr, hdr.get_rect(center=(panel_x + panel_w//2, 30)))

        item_h = 28
        for i, lbl in enumerate(self._labels):
            y_pos  = 54 + i * item_h
            locked = self._dm.sample_count(lbl) >= self._target
            current = (i == self._idx)

            if current:
                pygame.draw.rect(self._screen, (30, 30, 60),
                                 (panel_x + 4, y_pos - 2, panel_w - 8, item_h - 2),
                                 border_radius=3)
                pygame.draw.rect(self._screen, GOLD,
                                 (panel_x + 4, y_pos - 2, panel_w - 8, item_h - 2),
                                 1, border_radius=3)

            icon = "✦" if locked else "○"
            col  = GREEN if locked else (CYAN if current else GREY)
            line = f"{icon} {lbl}"
            surf = self._font_small.render(line, True, col)
            self._screen.blit(surf, (panel_x + 14, y_pos + 4))

    def _draw_center_hud(self, t: float) -> None:
        now    = time.time()
        label  = self._current_label() if self._idx < len(self._labels) else "—"
        cx     = self._cx - 100   # shift left to leave room for panel
        cy_hud = 170

        if self._state == DONE:
            msg = self._font_large.render("ALL DATASETS COLLECTED", True, GREEN)
            self._screen.blit(msg, msg.get_rect(center=(cx, cy_hud)))
            return

        # Current letter — huge
        pulse = 0.85 + 0.15 * math.sin(t * 2)
        col   = tuple(int(c * pulse) for c in GOLD)
        ltr   = self._font_huge.render(label, True, col)
        self._screen.blit(ltr, ltr.get_rect(center=(cx, cy_hud + 80)))

        # Status line
        if self._state == READY:
            msg  = "PRESS  [ SPACE ]  TO  BEGIN"
            col2 = CYAN
        elif self._state == COUNTDOWN:
            remaining = PRE_SECS - (now - self._state_ts)
            msg  = f"HOLD  STILL  ...  {remaining:.1f}s"
            col2 = GOLD
            self._draw_countdown_ring(cx, cy_hud + 80, remaining / PRE_SECS, t)
        elif self._state == CAPTURING:
            pct  = len(self._samples) / self._target
            msg  = f"CAPTURING  {len(self._samples)} / {self._target}"
            col2 = GREEN
            self._draw_capture_bar(pct, cx, cy_hud + 170)
        elif self._state == POSTCOOL:
            remaining = POST_SECS - (now - self._state_ts)
            msg  = f"SAVED  ✦  NEXT IN  {remaining:.1f}s"
            col2 = GREEN
        else:
            msg, col2 = "", WHITE

        status = self._font_med.render(msg, True, col2)
        self._screen.blit(status, status.get_rect(center=(cx, cy_hud + 185)))

        # Progress index
        prog = self._font_small.render(
            f"GESTURE  {self._idx + 1}  /  {len(self._labels)}", True, GREY)
        self._screen.blit(prog, prog.get_rect(center=(cx, cy_hud + 215)))

    def _draw_countdown_ring(self, cx, cy, frac, t) -> None:
        r   = 80
        arc_rect = pygame.Rect(cx - r, cy - r, r*2, r*2)
        end_angle = math.radians(-90 + 360 * (1 - frac))
        try:
            pygame.draw.arc(self._screen, GOLD, arc_rect,
                            math.radians(-90), end_angle, 5)
        except Exception:
            pass

    def _draw_capture_bar(self, pct: float, cx: int, y: int) -> None:
        bw, bh = 340, 14
        x  = cx - bw // 2
        pygame.draw.rect(self._screen, DARK, (x, y, bw, bh), border_radius=4)
        filled = int(bw * pct)
        if filled > 0:
            pygame.draw.rect(self._screen, GREEN, (x, y, filled, bh), border_radius=4)
        pygame.draw.rect(self._screen, GOLD_DIM, (x, y, bw, bh), 1, border_radius=4)

    def _draw_hints(self) -> None:
        hints = "[ SPACE ] START / NEXT     [ R ] REDO     [ ESC ] MENU"
        surf  = self._font_small.render(hints, True, GOLD_DIM)
        self._screen.blit(surf, surf.get_rect(
            center=(self._cx - 100, self._H - 22)))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _current_label(self) -> str:
        if self._idx < len(self._labels):
            return self._labels[self._idx]
        return ""

    def _advance_to_next_pending(self) -> None:
        while (self._idx < len(self._labels) and
               self._dm.sample_count(self._labels[self._idx]) >= self._target):
            self._idx += 1

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
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return pygame.surfarray.make_surface(
            np.transpose(frame_rgb, (1, 0, 2))
        )

    @staticmethod
    def _build_landmarker(config: dict):
        inf = config.get("inference", {})
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