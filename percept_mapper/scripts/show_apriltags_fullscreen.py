"""Display the 4 AprilTag corner overlay at full screen size until ESC.

Uses a borderless windowed mode (pygame.NOFRAME) instead of true fullscreen so
the window stays visible when focus moves elsewhere (e.g. Pupil Capture). The
world camera sees the 4 corner tags regardless of which window currently has
focus — only ESC inside this window will close it.

Run from repo root:
    uv run --project percept_mapper python percept_mapper/scripts/show_apriltags_fullscreen.py
"""

import os
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# SDL hints must be set BEFORE pygame.init() / set_mode() to take effect.
os.environ.setdefault("SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS", "0")
os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "0,0")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
import pygame

from scripts.apriltag_overlay import from_config


def main():
    pygame.init()
    info = pygame.display.Info()
    # Borderless window covering the whole screen. Behaves like a regular
    # window for focus/Alt-Tab, so clicking on Pupil Capture won't minimize it.
    screen = pygame.display.set_mode(
        (info.current_w, info.current_h), pygame.NOFRAME
    )
    pygame.display.set_caption("PhosLab AprilTag overlay (ESC para salir)")

    cfg_path = Path(__file__).resolve().parents[1] / "config" / "params.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    overlay = from_config(config)
    if overlay is None:
        print("[show] ⚠ apriltag_overlay.enabled = false. Saliendo.")
        pygame.quit()
        return

    print(
        f"[show] Ventana borderless {info.current_w}x{info.current_h}. "
        "Permanece visible al perder foco. ESC dentro de esta ventana para salir."
    )

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Ignore window-close requests; only ESC may exit.
                continue
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        screen.fill((0, 0, 0))
        overlay.draw(screen)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
