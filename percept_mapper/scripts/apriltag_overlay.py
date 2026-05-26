"""AprilTag corner overlay for the experiment screen.

Renders four AprilTag PNGs at the four screen corners on every frame so that
Pupil Capture's Surface Tracker can keep `phoslab_screen` locked throughout
the experiment. Call `overlay.draw(screen)` immediately before
`pygame.display.flip()`.
"""

from pathlib import Path

import pygame


_PERCEPT_MAPPER_ROOT = Path(__file__).resolve().parents[1]


class AprilTagOverlay:
    def __init__(self, png_dir, tag_files, tag_size_px=80, margin_px=20):
        png_dir = Path(png_dir)
        if not png_dir.is_absolute():
            # Resolve relative to percept_mapper/ so the path works regardless
            # of the launching cwd (repo root vs percept_mapper/).
            candidates = [
                _PERCEPT_MAPPER_ROOT / png_dir,
                _PERCEPT_MAPPER_ROOT.parent / png_dir,  # repo root + given path
                Path.cwd() / png_dir,
            ]
            png_dir = next((p for p in candidates if p.exists()), candidates[0])
        if len(tag_files) != 4:
            raise ValueError("apriltag_overlay.tag_files must list exactly 4 PNGs (TL, TR, BL, BR)")

        self.tag_size_px = int(tag_size_px)
        self.margin_px = int(margin_px)

        self._tags = []
        for fname in tag_files:
            path = png_dir / fname
            if not path.exists():
                raise FileNotFoundError(f"AprilTag PNG no encontrado: {path}")
            img = pygame.image.load(str(path))
            img = pygame.transform.scale(img, (self.tag_size_px, self.tag_size_px))
            self._tags.append(img.convert())

    def _corner_positions(self, screen_size):
        w, h = screen_size
        m = self.margin_px
        s = self.tag_size_px
        return [
            (m, m),                      # TL  -> tag_files[0]
            (w - m - s, m),              # TR  -> tag_files[1]
            (m, h - m - s),              # BL  -> tag_files[2]
            (w - m - s, h - m - s),      # BR  -> tag_files[3]
        ]

    def draw(self, screen):
        positions = self._corner_positions(screen.get_size())
        for pos, tag in zip(positions, self._tags):
            screen.blit(tag, pos)

def from_config(config):
    """Build an AprilTagOverlay from the params.yaml `apriltag_overlay:` block.
    Returns None if disabled or block missing — caller should no-op in that case.
    """
    cfg = (config.get("apriltag_overlay") or {})
    if not cfg.get("enabled", False):
        return None

    return AprilTagOverlay(
        png_dir=cfg.get("png_dir", "percept_mapper/assets/apriltags"),
        tag_files=cfg.get("tag_files", ["tag_0.png", "tag_1.png", "tag_2.png", "tag_3.png"]),
        tag_size_px=cfg.get("tag_size_px", 80),
        margin_px=cfg.get("margin_px", 20),
    )
