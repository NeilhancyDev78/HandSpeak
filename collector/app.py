"""
collector/app.py — HandSpeak Collector entry point.

Launches a fullscreen Pygame window and routes between Menu,
Register, and Update screens.

Usage:
    python collector/app.py
"""

import sys
import os
import json
import pygame

# Allow imports from project root regardless of cwd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collector.data_manager          import DataManager
from collector.screens.menu          import MenuScreen
from collector.screens.register      import RegisterScreen
from collector.screens.update        import UpdateScreen
from core.gesture_manager            import GestureManager


def _load_configs() -> tuple[dict, dict]:
    with open("config/app_config.json",     "r") as f:
        app_cfg = json.load(f)
    with open("config/gesture_config.json", "r") as f:
        ges_cfg = json.load(f)
    # Merge so screens only need one config dict
    merged = {**app_cfg, **ges_cfg}
    return merged, app_cfg


def main() -> None:
    pygame.init()
    pygame.display.set_caption("HandSpeak — Collector")

    config, app_cfg = _load_configs()

    info   = pygame.display.Info()
    W, H   = info.current_w, info.current_h
    screen = pygame.display.set_mode((W, H), pygame.FULLSCREEN)

    gm = GestureManager("config/gesture_config.json")
    dm = DataManager(app_cfg["paths"]["data_dir"])

    route = "menu"
    while route != "quit":
        if route == "menu":
            summary = dm.summary(gm.gesture_names)
            route   = MenuScreen(screen, summary).run()

        elif route == "register":
            # Warn user that existing data will be wiped on completion
            route = RegisterScreen(
                screen, gm.gesture_names, dm, config
            ).run()
            # After register finishes, always return to menu
            route = "menu"

        elif route == "update":
            route = UpdateScreen(
                screen, gm.gesture_names, dm, config
            ).run()
            route = "menu"

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()