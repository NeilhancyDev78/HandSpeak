"""
main.py — HandSpeak entry point.

Usage:
    python main.py
"""

import os
import sys
import json


# ── Dependency checks ─────────────────────────────────────────────────────────

def _check() -> None:
    issues = []

    if not os.path.exists("hand_landmarker.task"):
        issues.append(
            "hand_landmarker.task not found.\n"
            "  Download: https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        )
    if not os.path.exists("model/gesture_model.keras"):
        issues.append(
            "model/gesture_model.keras not found.\n"
            "  Run:  python collector/app.py   (collect data)\n"
            "  Then: python train.py           (train model)"
        )

    if issues:
        print("\n[HandSpeak] Cannot start — missing dependencies:\n")
        for issue in issues:
            print(f"  ✗  {issue}\n")
        sys.exit(1)


# ── Boot ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n┌─────────────────────────────────────┐")
    print("│   H A N D S P E A K  —  ASL→Text   │")
    print("└─────────────────────────────────────┘\n")

    _check()

    with open("config/app_config.json",     "r") as f:
        app_cfg = json.load(f)
    with open("config/gesture_config.json", "r") as f:
        ges_cfg = json.load(f)

    merged_cfg = {**app_cfg, **ges_cfg}

    from core.gesture_manager       import GestureManager
    from core.hand_tracker          import HandTracker
    from core.gesture_engine        import GestureEngine
    from core.text_buffer           import TextBuffer
    from core.tts_engine            import TTSEngine
    from core.performance_monitor   import PerformanceMonitor
    from ui.overlay                 import Overlay
    from ui.pipeline                import VideoPipeline

    print("Initialising modules ...")

    gm      = GestureManager("config/gesture_config.json")
    tracker = HandTracker(app_cfg)
    engine  = GestureEngine(gm, merged_cfg)
    buf     = TextBuffer()
    tts     = TTSEngine(app_cfg)
    perf    = PerformanceMonitor(app_cfg)
    overlay = Overlay(app_cfg["camera"]["width"],
                      app_cfg["camera"]["height"])

    print(f"  Gestures  : {', '.join(gm.gesture_names)}")
    print(f"  Classes   : {gm.num_classes}")

    engine.load_model(app_cfg["inference"]["keras_model"])
    print(f"  Model     : {app_cfg['inference']['keras_model']}")
    print("\nStarting camera  (press Q or ESC to quit)\n")

    pipeline = VideoPipeline(
        tracker, engine, gm, buf, tts, perf, overlay, app_cfg
    )
    pipeline.run()

    print("\nHandSpeak closed.")


if __name__ == "__main__":
    main()