"""
collector/screens/menu.py — Dr. Strange themed main menu screen.

Two options:
  [1] Register New Datasets   — full A–Z collection run
  [2] Update a Dataset        — replace samples for one specific gesture

Rendered entirely in Pygame with animated arcane geometry.
"""

import pygame
import math
import time


# ── Palette ───────────────────────────────────────────────────────────────────
BG        = (5,   5,  12)
GOLD      = (212, 175,  55)
GOLD_DIM  = ( 90,  70,  18)
CYAN      = ( 0,  210, 220)
WHITE     = (255, 255, 255)
GREY      = (120, 120, 140)
PANEL_BG  = ( 14,  14,  28)


class MenuScreen:
    """
    Fullscreen Dr. Strange menu.

    Returns:
      "register"  — user chose Register New Datasets
      "update"    — user chose Update a Dataset
      "quit"      — user closed the window
    """

    _TITLE     = "HANDSPEAK"
    _SUBTITLE  = "ASL GESTURE COLLECTION SYSTEM"
    _OPTIONS   = [
        ("REGISTER NEW DATASETS",  "register"),
        ("UPDATE A DATASET",       "update"),
    ]

    def __init__(self, screen: pygame.Surface, data_summary: dict[str, int]):
        self._screen  = screen
        self._summary = data_summary          # { label: sample_count }
        self._W, self._H = screen.get_size()
        self._cx, self._cy = self._W // 2, self._H // 2

        self._font_title  = pygame.font.SysFont("consolas", 52, bold=True)
        self._font_sub    = pygame.font.SysFont("consolas", 18)
        self._font_btn    = pygame.font.SysFont("consolas", 26, bold=True)
        self._font_stat   = pygame.font.SysFont("consolas", 14)

        self._hovered     = -1
        self._t0          = time.time()

        total_lbl = len(data_summary)
        total_smp = sum(data_summary.values())
        self._stat_line   = f"{total_lbl} GESTURES   {total_smp} TOTAL SAMPLES"

    # ── Public entry ──────────────────────────────────────────────────────────

    def run(self) -> str:
        clock = pygame.time.Clock()
        while True:
            dt = clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return "quit"
                if event.type == pygame.MOUSEMOTION:
                    self._hovered = self._hit(event.pos)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    idx = self._hit(event.pos)
                    if idx >= 0:
                        return self._OPTIONS[idx][1]
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        return "register"
                    if event.key == pygame.K_2:
                        return "update"

            self._draw()
            pygame.display.flip()

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        t = time.time() - self._t0
        self._screen.fill(BG)
        self._draw_mandala(t)
        self._draw_title(t)
        self._draw_stats()
        self._draw_buttons(t)
        self._draw_hints()

    def _draw_mandala(self, t: float) -> None:
        surf = self._screen
        cx, cy = self._cx, self._cy

        for ring in range(6):
            r      = 140 + ring * 55
            spokes = 12 + ring * 4
            alpha  = max(0, 90 - ring * 14)
            speed  = 0.18 + ring * 0.06
            angle_offset = t * speed * (1 if ring % 2 == 0 else -1)

            pts = []
            for i in range(spokes):
                a = math.radians(360 / spokes * i) + angle_offset
                pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))

            for i in range(spokes):
                col = (*GOLD_DIM[:3], alpha)
                x0, y0 = int(pts[i][0]),       int(pts[i][1])
                x1, y1 = int(pts[(i+1) % spokes][0]), int(pts[(i+1) % spokes][1])
                _draw_line_aa(surf, GOLD_DIM, (x0, y0), (x1, y1), alpha // 40 + 1)

        # Central glyph
        for r2 in [28, 46]:
            spk = 8
            ao2 = t * 0.4
            for i in range(spk):
                a = math.radians(360 / spk * i) + ao2
                x = cx + r2 * math.cos(a)
                y = cy + r2 * math.sin(a)
                pygame.draw.circle(surf, GOLD, (int(x), int(y)), 3)

    def _draw_title(self, t: float) -> None:
        # Glow pulse
        pulse = 0.85 + 0.15 * math.sin(t * 1.8)
        col   = tuple(int(c * pulse) for c in GOLD)

        title_surf = self._font_title.render(self._TITLE, True, col)
        sub_surf   = self._font_sub.render(self._SUBTITLE, True, GREY)

        self._screen.blit(
            title_surf,
            title_surf.get_rect(center=(self._cx, 120))
        )
        self._screen.blit(
            sub_surf,
            sub_surf.get_rect(center=(self._cx, 170))
        )

        # Horizontal rule
        rule_y = 192
        pygame.draw.line(self._screen, GOLD_DIM,
                         (self._cx - 320, rule_y), (self._cx + 320, rule_y), 1)

    def _draw_stats(self) -> None:
        stat_surf = self._font_stat.render(self._stat_line, True, GREY)
        self._screen.blit(
            stat_surf,
            stat_surf.get_rect(center=(self._cx, 210))
        )

    def _draw_buttons(self, t: float) -> None:
        btn_w, btn_h = 480, 64
        gap          = 28
        total_h      = len(self._OPTIONS) * (btn_h + gap) - gap
        start_y      = self._cy - total_h // 2 + 40

        self._btn_rects = []
        for i, (label, _) in enumerate(self._OPTIONS):
            x   = self._cx - btn_w // 2
            y   = start_y + i * (btn_h + gap)
            rect = pygame.Rect(x, y, btn_w, btn_h)
            self._btn_rects.append(rect)

            hov = (self._hovered == i)
            pulse = 0.6 + 0.4 * math.sin(t * 2.2 + i * math.pi)

            # Outer glow
            if hov:
                glow_surf = pygame.Surface((btn_w + 20, btn_h + 20), pygame.SRCALPHA)
                pygame.draw.rect(glow_surf, (*GOLD, int(60 * pulse)),
                                 (0, 0, btn_w + 20, btn_h + 20), border_radius=6)
                self._screen.blit(glow_surf, (x - 10, y - 10))

            # Panel
            pygame.draw.rect(self._screen, PANEL_BG, rect, border_radius=4)
            border_col = GOLD if hov else GOLD_DIM
            pygame.draw.rect(self._screen, border_col, rect, 2, border_radius=4)

            # Corner accents
            sz = 10
            for cx2, cy2, dx, dy in [
                (rect.left,  rect.top,    1,  1),
                (rect.right, rect.top,   -1,  1),
                (rect.left,  rect.bottom, 1, -1),
                (rect.right, rect.bottom,-1, -1),
            ]:
                pygame.draw.line(self._screen, GOLD,
                                 (cx2, cy2), (cx2 + dx*sz, cy2), 2)
                pygame.draw.line(self._screen, GOLD,
                                 (cx2, cy2), (cx2, cy2 + dy*sz), 2)

            # Number tag
            num_col  = CYAN if hov else GREY
            num_surf = self._font_btn.render(f"[{i+1}]", True, num_col)
            self._screen.blit(num_surf, num_surf.get_rect(
                midleft=(x + 22, y + btn_h // 2)))

            # Label
            lbl_col  = WHITE if hov else GREY
            lbl_surf = self._font_btn.render(label, True, lbl_col)
            self._screen.blit(lbl_surf, lbl_surf.get_rect(
                midleft=(x + 80, y + btn_h // 2)))

    def _draw_hints(self) -> None:
        hint = "[ 1 / 2 ] SELECT     [ ESC ] QUIT"
        surf = self._font_stat.render(hint, True, GOLD_DIM)
        self._screen.blit(surf, surf.get_rect(
            center=(self._cx, self._H - 36)))

    # ── Hit detection ─────────────────────────────────────────────────────────

    def _hit(self, pos) -> int:
        for i, rect in enumerate(getattr(self, "_btn_rects", [])):
            if rect.collidepoint(pos):
                return i
        return -1


# ── Helpers ───────────────────────────────────────────────────────────────────

def _draw_line_aa(surf, color, p1, p2, width=1) -> None:
    pygame.draw.line(surf, color, p1, p2, width)