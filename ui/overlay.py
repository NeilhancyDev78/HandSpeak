"""
ui/overlay.py — Dr. Strange themed OpenCV renderer.

Draws everything on top of the camera feed:
  • Animated mandala rings that react to recognition confidence
  • Glowing amber landmark skeleton
  • Sci-fi HUD: FPS, confidence, current gesture label
  • Mystical scroll text box with cursor
  • Cooldown arc + hold-progress bar
  • Clickable DELETE / SPEAK / CLEAR buttons with glow
"""

import cv2
import numpy as np
import math
import time


# ── Palette (BGR) ─────────────────────────────────────────────────────────────
GOLD      = (  55, 175, 212)
GOLD_DIM  = (  18,  70,  90)
CYAN      = ( 220, 220,   0)
GREEN     = (  80, 220,  80)
RED       = (  60,  60, 220)
WHITE     = ( 255, 255, 255)
DARK      = (  18,  18,  18)
PANEL     = (  24,  20,  12)
COMMIT_FX = (  80, 220,  80)


class Button:
    """Themed OpenCV button with hover and flash states."""

    def __init__(self, label: str, x: int, y: int, w: int, h: int,
                 color_normal, color_hover, color_active):
        self.label         = label
        self.x, self.y     = x, y
        self.w, self.h     = w, h
        self._cn           = color_normal
        self._ch           = color_hover
        self._ca           = color_active
        self.hovered       = False
        self._flash_until  = 0.0

    def flash(self) -> None:
        self._flash_until = time.time() + 0.15

    def hit(self, mx: int, my: int) -> bool:
        return self.x <= mx <= self.x + self.w and self.y <= my <= self.y + self.h

    def draw(self, canvas: np.ndarray) -> None:
        now   = time.time()
        color = self._ca if now < self._flash_until else (
                self._ch if self.hovered else self._cn)

        # Shadow
        cv2.rectangle(canvas,
                      (self.x + 3, self.y + 3),
                      (self.x + self.w + 3, self.y + self.h + 3),
                      (0, 0, 0), -1)
        # Body
        cv2.rectangle(canvas,
                      (self.x, self.y),
                      (self.x + self.w, self.y + self.h), color, -1)
        # Border glow
        bd = GOLD if (self.hovered or now < self._flash_until) else GOLD_DIM
        cv2.rectangle(canvas,
                      (self.x, self.y),
                      (self.x + self.w, self.y + self.h), bd, 2)
        # Corner ticks
        sz = 8
        for (cx2, cy2, dx, dy) in [
            (self.x,          self.y,          1,  1),
            (self.x + self.w, self.y,          -1,  1),
            (self.x,          self.y + self.h,  1, -1),
            (self.x + self.w, self.y + self.h, -1, -1),
        ]:
            cv2.line(canvas, (cx2, cy2), (cx2 + dx*sz, cy2),       GOLD, 1)
            cv2.line(canvas, (cx2, cy2), (cx2, cy2 + dy*sz),       GOLD, 1)
        # Label
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs   = 0.62
        (tw, th), _ = cv2.getTextSize(self.label, font, fs, 2)
        tx = self.x + (self.w - tw) // 2
        ty = self.y + (self.h + th) // 2
        cv2.putText(canvas, self.label, (tx, ty), font, fs, WHITE, 2, cv2.LINE_AA)


class Overlay:
    """
    Stateless renderer — call draw() each frame.
    All animation state is derived from wall-clock time so there
    is no frame-count bookkeeping here.
    """

    PANEL_H = 130

    def __init__(self, display_w: int, display_h: int):
        self._W   = display_w
        self._H   = display_h
        self._t0  = time.time()
        self._build_buttons()

    # ── Button setup ──────────────────────────────────────────────────────────

    def _build_buttons(self) -> None:
        btn_w, btn_h = 150, 48
        btn_y = self._H - self.PANEL_H + (self.PANEL_H - btn_h) // 2 + 14

        x3 = self._W - 20 - btn_w
        x2 = x3 - 18 - btn_w
        x1 = x2 - 18 - btn_w

        self.btn_delete = Button(
            "< DELETE", x1, btn_y, btn_w, btn_h,
            color_normal=(130,  40,  50),
            color_hover =(200,  60,  80),
            color_active=(220,  60,  60),
        )
        self.btn_speak = Button(
            "> SPEAK", x2, btn_y, btn_w, btn_h,
            color_normal=( 20,  90,  30),
            color_hover =( 30, 150,  50),
            color_active=( 40, 200,  60),
        )
        self.btn_clear = Button(
            "x CLEAR", x3, btn_y, btn_w, btn_h,
            color_normal=( 60,  40,  20),
            color_hover =(110,  70,  30),
            color_active=(140,  90,  40),
        )
        self.buttons = [self.btn_delete, self.btn_speak, self.btn_clear]

    # ── Main draw call ────────────────────────────────────────────────────────

    def draw(self, canvas: np.ndarray,
             tracker_result: dict | None,
             engine_state:   dict,
             text:           str,
             fps:            float,
             committed:      bool) -> np.ndarray:
        """
        Composite all UI layers onto canvas (in-place + returned).

        engine_state keys expected:
            label, confidence, cooldown_progress, hold_progress
        """
        t = time.time() - self._t0

        self._draw_mandala(canvas, t, engine_state.get("confidence", 0.0))
        self._draw_landmarks(canvas, tracker_result)
        self._draw_top_bar(canvas, fps, engine_state)
        self._draw_commit_flash(canvas, committed)
        self._draw_bottom_panel(canvas, text, engine_state, t)

        for btn in self.buttons:
            btn.draw(canvas)

        return canvas

    # ── Mandala ───────────────────────────────────────────────────────────────

    def _draw_mandala(self, canvas: np.ndarray,
                      t: float, confidence: float) -> None:
        cx, cy = self._W // 2, (self._H - self.PANEL_H) // 2
        rings  = 3
        alpha_map = canvas.copy()

        for ring in range(rings):
            r     = int(160 + ring * 60 + confidence * 40)
            spk   = 12 + ring * 6
            speed = 0.14 + ring * 0.05
            ao    = t * speed * (1 if ring % 2 == 0 else -1)
            pts   = [
                (int(cx + r * math.cos(math.radians(360 / spk * i) + ao)),
                 int(cy + r * math.sin(math.radians(360 / spk * i) + ao)))
                for i in range(spk)
            ]
            for i in range(spk):
                cv2.line(alpha_map,
                         pts[i], pts[(i + 1) % spk],
                         GOLD_DIM, 1, cv2.LINE_AA)

        cv2.addWeighted(alpha_map, 0.55, canvas, 0.45, 0, canvas)

        # Central glyph dots
        for r2 in (24, 40):
            for i in range(8):
                a  = math.radians(45 * i) + t * 0.5
                px = int(cx + r2 * math.cos(a))
                py = int(cy + r2 * math.sin(a))
                cv2.circle(canvas, (px, py), 3, GOLD, -1, cv2.LINE_AA)

    # ── Landmarks ─────────────────────────────────────────────────────────────

    def _draw_landmarks(self, canvas: np.ndarray,
                        result: dict | None) -> None:
        if result is None or not result.get("found"):
            return

        h, w = canvas.shape[:2]
        lms  = result["landmarks"]
        pts  = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17),
        ]
        for a, b in connections:
            cv2.line(canvas, pts[a], pts[b], CYAN, 2, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(canvas, pt, 6, GOLD,  -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 6, WHITE,  1, cv2.LINE_AA)

    # ── Top HUD bar ───────────────────────────────────────────────────────────

    def _draw_top_bar(self, canvas: np.ndarray,
                      fps: float, engine_state: dict) -> None:
        cv2.rectangle(canvas, (0, 0), (self._W, 58), PANEL, -1)
        cv2.line(canvas, (0, 58), (self._W, 58), GOLD_DIM, 1)

        # FPS
        cv2.putText(canvas, f"FPS  {fps:.0f}",
                    (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.68, GOLD_DIM, 1, cv2.LINE_AA)

        # Title
        title = "H A N D S P E A K"
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
        cv2.putText(canvas, title,
                    (self._W // 2 - tw // 2, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, GOLD, 2, cv2.LINE_AA)

        # Gesture + confidence
        label = engine_state.get("label")
        conf  = engine_state.get("confidence", 0.0)
        if label:
            pred_txt = f"{label}  {conf * 100:.0f}%"
            cv2.putText(canvas, pred_txt,
                        (self._W - 180, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, CYAN, 2, cv2.LINE_AA)

        # Hold-progress bar (thin strip under top bar)
        hold  = engine_state.get("hold_progress", 0.0)
        bar_w = int(self._W * hold)
        cv2.rectangle(canvas, (0, 56), (bar_w, 60), GOLD, -1)

    # ── Commit flash ──────────────────────────────────────────────────────────

    def _draw_commit_flash(self, canvas: np.ndarray, committed: bool) -> None:
        if committed:
            cv2.rectangle(canvas, (0, 0), (self._W, self._H),
                          COMMIT_FX, 10)

    # ── Bottom panel ──────────────────────────────────────────────────────────

    def _draw_bottom_panel(self, canvas: np.ndarray,
                           text: str, engine_state: dict, t: float) -> None:
        py = self._H - self.PANEL_H

        # Panel background
        cv2.rectangle(canvas, (0, py), (self._W, self._H), PANEL, -1)
        cv2.line(canvas, (0, py), (self._W, py), GOLD_DIM, 1)

        # Scroll border accents (corner ticks)
        for (x2, y2, dx, dy) in [
            (16,      py + 8,          1,  1),
            (self._W - 16, py + 8,    -1,  1),
            (16,      self._H - 8,     1, -1),
            (self._W - 16, self._H - 8,-1, -1),
        ]:
            cv2.line(canvas, (x2, y2), (x2 + dx*14, y2),      GOLD_DIM, 1)
            cv2.line(canvas, (x2, y2), (x2, y2 + dy*14),      GOLD_DIM, 1)

        # Typed text
        display = (text + "|") if text is not None else "|"
        cv2.putText(canvas, display,
                    (24, py + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, WHITE, 2, cv2.LINE_AA)

        # Cooldown arc (small, bottom-left)
        cooldown = engine_state.get("cooldown_progress", 1.0)
        arc_cx, arc_cy = 28, self._H - 22
        arc_r = 14
        sweep = int(360 * cooldown)
        if sweep > 0:
            cv2.ellipse(canvas,
                        (arc_cx, arc_cy), (arc_r, arc_r),
                        -90, 0, sweep, GOLD, 2, cv2.LINE_AA)

        # Keyboard hints
        hint = "[D] Delete   [S] Speak   [C] Clear   [Q] Quit"
        cv2.putText(canvas, hint,
                    (60, self._H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, GOLD_DIM, 1, cv2.LINE_AA)