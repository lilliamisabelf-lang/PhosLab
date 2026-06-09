"""Mapping DEBUG MODE overlay.

When `debug.mapping_debug_mode` is on, this overlay replaces the plain black
background of the mapping screens with a visual-field reference grid (x/y axes
+ iso-eccentricity rings every N degrees) and, once a phosphene has been
shown, keeps an X marker at its true location — annotated with eccentricity,
polar angle, visual-field degrees and pixels — until the next trial begins.

It is drawn on TOP of whatever screen is active (stim, poststim, drawing) via
`_display_flip`, so the marker persists from phosphene onset until the
participant has given their response. Because it draws on the display surface
(never on the drawing tablet's canvas), the saved response PNGs stay clean.

Iso-eccentricity contours are drawn from the mapper's pixels-per-degree. With
the corrected isotropic geometry (ppd_x == ppd_y, since panel pixels are
square) a "ring every N°" is a true CIRCLE. The renderer stays general: if a
mapper ever reports ppd_x != ppd_y (e.g. a legacy/anisotropic session) the
ring becomes an ellipse, faithfully reflecting the actual deg→px scale rather
than hiding it.
"""

from __future__ import annotations

import math

import pygame


# Colores: rejilla en blanco/gris (atenuada para no tapar el fosfeno ni los
# trazos); marcador en magenta para que destaque sobre el fosfeno blanco.
_AXIS_COLOR = (200, 200, 210)
_RING_COLOR = (110, 110, 125)
_LABEL_COLOR = (160, 160, 175)
_MARKER_COLOR = (255, 0, 180)
_TEXT_COLOR = (235, 238, 248)
_BOX_COLOR = (10, 14, 24, 210)  # RGBA, semitransparente


class MappingDebugOverlay:
    def __init__(
        self,
        screen_size,
        center_px,
        ppd_x,
        ppd_y,
        ring_step_deg=5.0,
        background_color=(0, 0, 0),
    ):
        self.width, self.height = int(screen_size[0]), int(screen_size[1])
        self.cx, self.cy = float(center_px[0]), float(center_px[1])
        self.ppd_x = float(ppd_x)
        self.ppd_y = float(ppd_y)
        self.ring_step_deg = max(1e-3, float(ring_step_deg))
        self.background_color = tuple(background_color)

        # Estado del marcador (fosfeno actual). active=False => sólo rejilla.
        self.active = False
        self.phosphene_px = None
        self.phosphene_deg = None
        self.ecc_deg = None
        self.polar_deg = None
        self.electrode_index = None

        self._grid_surface = None  # cache perezoso (necesita pygame.font)
        self._font = None
        self._ring_font = None

    # ── classmethod de construcción ────────────────────────────────────────
    @classmethod
    def from_config_and_mapper(cls, config, mapper, screen_size):
        """Devuelve un overlay si debug.mapping_debug_mode está activo, si no None.

        Usa la geometría del MAPPER (mismo px/grado y centro con que se calculan
        las posiciones de los fosfenos) para que la rejilla quede alineada con
        los marcadores.
        """
        debug_cfg = (config.get("debug") or {})
        if not debug_cfg.get("mapping_debug_mode", False):
            return None
        try:
            ppd_x = float(mapper.pixels_per_degree_x)
            ppd_y = float(mapper.pixels_per_degree_y)
            center = mapper.screen_center
        except AttributeError:
            return None
        ring_step = float(debug_cfg.get("mapping_debug_ring_step_deg", 5.0))
        bg = tuple((config.get("screen") or {}).get("background_color", (0, 0, 0)))
        return cls(
            screen_size=screen_size,
            center_px=center,
            ppd_x=ppd_x,
            ppd_y=ppd_y,
            ring_step_deg=ring_step,
            background_color=bg,
        )

    # ── estado del marcador ─────────────────────────────────────────────────
    def set_phosphene(self, px_xy, deg_xy, ecc_deg, polar_deg, electrode_index=None):
        self.phosphene_px = (float(px_xy[0]), float(px_xy[1]))
        self.phosphene_deg = (float(deg_xy[0]), float(deg_xy[1]))
        self.ecc_deg = float(ecc_deg)
        self.polar_deg = float(polar_deg)
        self.electrode_index = electrode_index
        self.active = True

    def clear(self):
        self.active = False
        self.phosphene_px = None

    # ── helpers ───────────────────────────────────────────────────────────
    def _ensure_fonts(self):
        if self._font is None:
            if not pygame.font.get_init():
                pygame.font.init()
            self._font = pygame.font.Font(None, 24)
            self._ring_font = pygame.font.Font(None, 18)

    def _max_eccentricity_deg(self):
        max_x = (max(self.cx, self.width - self.cx)) / max(self.ppd_x, 1e-6)
        max_y = (max(self.cy, self.height - self.cy)) / max(self.ppd_y, 1e-6)
        return math.hypot(max_x, max_y)

    def _build_grid(self):
        self._ensure_fonts()
        surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        # Ejes X / Y
        cx, cy = int(round(self.cx)), int(round(self.cy))
        pygame.draw.line(surf, _AXIS_COLOR, (0, cy), (self.width, cy), 1)
        pygame.draw.line(surf, _AXIS_COLOR, (cx, 0), (cx, self.height), 1)

        # Anillos de iso-excentricidad (elipses por la anisotropía px/grado)
        max_ecc = self._max_eccentricity_deg()
        n_rings = int(math.ceil(max_ecc / self.ring_step_deg))
        for k in range(1, n_rings + 1):
            ecc = k * self.ring_step_deg
            rx = ecc * self.ppd_x
            ry = ecc * self.ppd_y
            rect = pygame.Rect(0, 0, int(2 * rx), int(2 * ry))
            rect.center = (cx, cy)
            if rect.width >= 2 and rect.height >= 2:
                pygame.draw.ellipse(surf, _RING_COLOR, rect, 1)
            # Etiqueta de grados sobre el eje +X
            label = self._ring_font.render(f"{ecc:g}°", True, _LABEL_COLOR)
            surf.blit(label, (cx + int(rx) + 3, cy - label.get_height() - 1))
        return surf

    def _draw_marker(self, screen):
        if not self.active or self.phosphene_px is None:
            return
        self._ensure_fonts()
        x, y = int(round(self.phosphene_px[0])), int(round(self.phosphene_px[1]))
        s = 9  # semibrazo de la X
        pygame.draw.line(screen, _MARKER_COLOR, (x - s, y - s), (x + s, y + s), 2)
        pygame.draw.line(screen, _MARKER_COLOR, (x - s, y + s), (x + s, y - s), 2)
        pygame.draw.circle(screen, _MARKER_COLOR, (x, y), s + 5, 1)

        # Anotación
        e_tag = f"e{self.electrode_index}  " if self.electrode_index is not None else ""
        lines = [
            f"{e_tag}ecc {self.ecc_deg:.1f}°  pol {self.polar_deg:.1f}°",
            f"vf ({self.phosphene_deg[0]:+.1f}°, {self.phosphene_deg[1]:+.1f}°)",
            f"px ({x}, {y})",
        ]
        rendered = [self._font.render(t, True, _TEXT_COLOR) for t in lines]
        box_w = max(r.get_width() for r in rendered) + 12
        box_h = sum(r.get_height() for r in rendered) + 10

        # Colocar la caja a la derecha del marcador; si se sale, a la izquierda.
        bx = x + s + 8
        by = y - box_h // 2
        if bx + box_w > self.width:
            bx = x - s - 8 - box_w
        by = max(2, min(by, self.height - box_h - 2))

        box = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        box.fill(_BOX_COLOR)
        oy = 5
        for r in rendered:
            box.blit(r, (6, oy))
            oy += r.get_height()
        screen.blit(box, (bx, by))

    # ── API principal ───────────────────────────────────────────────────────
    def draw(self, screen):
        """Dibuja rejilla (+ marcador si activo) sobre `screen`."""
        if self._grid_surface is None:
            self._grid_surface = self._build_grid()
        screen.blit(self._grid_surface, (0, 0))
        self._draw_marker(screen)
