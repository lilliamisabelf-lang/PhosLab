"""Headless verification: open a windowed pygame screen, draw the AprilTag
overlay, do one render+flip, save a screenshot to disk. Confirms the 4 tags
appear in the corners on a black background.

Run from repo root:
    uv run --project percept_mapper python percept_mapper/scripts/verify_overlay.py
"""

import os
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless

# Allow running from repo root or from percept_mapper/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
import pygame

from scripts.apriltag_overlay import from_config


def main():
    pygame.init()
    screen = pygame.display.set_mode((1920, 1080))
    screen.fill((0, 0, 0))

    cfg_path = Path(__file__).resolve().parents[1] / "config" / "params.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    overlay = from_config(config)
    if overlay is None:
        print("[verify] ⚠ apriltag_overlay.enabled = false en params.yaml; nada que verificar.")
        return

    overlay.draw(screen)
    pygame.display.flip()

    out_path = Path(__file__).resolve().parents[1] / "assets" / "apriltag_overlay_preview.png"
    pygame.image.save(screen, str(out_path))
    print(f"[verify] ✓ captura guardada: {out_path}")

    # Quick sanity check: assert non-zero pixels at the 4 corner regions
    import numpy as np
    arr = pygame.surfarray.array3d(screen)  # (W, H, 3)
    W, H = screen.get_size()
    s = overlay.tag_size_px
    m = overlay.margin_px
    regions = {
        "TL": arr[m:m + s, m:m + s],
        "TR": arr[W - m - s:W - m, m:m + s],
        "BL": arr[m:m + s, H - m - s:H - m],
        "BR": arr[W - m - s:W - m, H - m - s:H - m],
    }
    for name, region in regions.items():
        white_frac = (region.mean(axis=2) > 200).mean()
        black_frac = (region.mean(axis=2) < 50).mean()
        ok = white_frac > 0.05 and black_frac > 0.05
        flag = "✓" if ok else "✗"
        print(f"[verify] {flag} esquina {name}: blanco={white_frac:.2f} negro={black_frac:.2f}")


if __name__ == "__main__":
    main()
