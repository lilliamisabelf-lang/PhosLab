"""Smoke tests for the mapping DEBUG MODE overlay.

Renders offscreen (SDL dummy driver) so it runs headless in CI. Verifies the
grid draws, the marker sets/clears, geometry is sane, and from_config gating
works. Also dumps a PNG to assets/ for eyeballing.

Run:
    uv run --project percept_mapper python percept_mapper/scripts/debug_overlay_smoke_test.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from scripts.debug_overlay import (  # noqa: E402
    MappingDebugOverlay,
    _MARKER_COLOR,
    _RING_COLOR,
)


# Geometría del config "corto" del usuario: 2560x1440, vf_scope 70° (max ecc).
# Escala ISOTRÓPICA anclada al lado menor (igual que DynaphosMapper tras el fix):
# un grado ocupa los mismos píxeles en X e Y, así que los anillos de
# iso-excentricidad son CÍRCULOS reales (no elipses).
W, H = 2560, 1440
VF_SCOPE_DEG = 70.0
PPD = min(W, H) / (2.0 * VF_SCOPE_DEG)     # 1440/140 = 10.29 px/° (X e Y)
PPD_X = PPD_Y = PPD
CENTER = (W / 2, H / 2)


class _FakeMapper:
    pixels_per_degree_x = PPD_X
    pixels_per_degree_y = PPD_Y
    screen_center = CENTER

    def get_electrode_info(self, idx):
        # 10° a la derecha, 5° arriba
        return {"visual_position_deg": [10.0, 5.0], "eccentricity_deg": 11.180}


def _new_overlay():
    return MappingDebugOverlay(
        screen_size=(W, H), center_px=CENTER, ppd_x=PPD_X, ppd_y=PPD_Y,
        ring_step_deg=5.0,
    )


def _count_color(surf, color, tol=20):
    arr = pygame.surfarray.array3d(surf)
    r, g, b = color
    mask = (
        (abs(arr[:, :, 0].astype(int) - r) <= tol)
        & (abs(arr[:, :, 1].astype(int) - g) <= tol)
        & (abs(arr[:, :, 2].astype(int) - b) <= tol)
    )
    return int(mask.sum())


def test_grid_draws():
    surf = pygame.Surface((W, H))
    surf.fill((0, 0, 0))
    ov = _new_overlay()
    ov.draw(surf)  # grid only, no marker
    nonblack = int((pygame.surfarray.array3d(surf).sum(axis=2) > 0).sum())
    assert nonblack > 5000, f"grid barely drew anything ({nonblack} px)"
    # Eje horizontal: la fila central debe tener píxeles del color de eje.
    cy = H // 2
    row = pygame.surfarray.array3d(surf)[:, cy, :]
    assert (row.sum(axis=1) > 0).sum() > W * 0.5, "horizontal axis not spanning width"
    print(f"  ✓ grid draws (nonblack px={nonblack})")


def test_marker_sets_and_clears():
    surf = pygame.Surface((W, H))
    ov = _new_overlay()

    surf.fill((0, 0, 0))
    ov.draw(surf)
    assert _count_color(surf, _MARKER_COLOR) == 0, "marker shown before set"

    ov.set_phosphene(
        px_xy=(1280 + int(10 * PPD_X), 720 - int(5 * PPD_Y)),
        deg_xy=(10.0, 5.0), ecc_deg=11.18, polar_deg=26.57, electrode_index=16,
    )
    assert ov.active
    surf.fill((0, 0, 0))
    ov.draw(surf)
    assert _count_color(surf, _MARKER_COLOR) > 20, "marker not drawn after set"

    ov.clear()
    assert not ov.active
    surf.fill((0, 0, 0))
    ov.draw(surf)
    assert _count_color(surf, _MARKER_COLOR) == 0, "marker not cleared"
    print("  ✓ marker sets and clears")


def test_from_config_gating():
    base = {"screen": {"width": W, "height": H, "background_color": [0, 0, 0]}}
    off = dict(base, debug={"mapping_debug_mode": False})
    assert MappingDebugOverlay.from_config_and_mapper(off, _FakeMapper(), (W, H)) is None
    on = dict(base, debug={"mapping_debug_mode": True, "mapping_debug_ring_step_deg": 5})
    ov = MappingDebugOverlay.from_config_and_mapper(on, _FakeMapper(), (W, H))
    assert ov is not None and abs(ov.ppd_x - PPD_X) < 1e-6
    print("  ✓ from_config_and_mapper gates on debug.mapping_debug_mode")


def test_rings_are_circular():
    """Con escala isotrópica (ppd_x == ppd_y) los anillos de iso-excentricidad
    deben ser CÍRCULOS: el radio del anillo exterior medido sobre el eje X y
    sobre el eje Y debe coincidir. Aislamos los píxeles de anillo por su color
    (distinto del de los ejes) para no confundir las líneas de los ejes."""
    ov = _new_overlay()
    assert abs(ov.ppd_x - ov.ppd_y) < 1e-9, "geometría de test no es isotrópica"
    surf = pygame.Surface((W, H))
    surf.fill((0, 0, 0))
    ov.draw(surf)

    arr = pygame.surfarray.array3d(surf).astype(int)  # (W, H, 3), indexado [x, y]
    r, g, b = _RING_COLOR
    tol = 20
    ring = (
        (abs(arr[:, :, 0] - r) <= tol)
        & (abs(arr[:, :, 1] - g) <= tol)
        & (abs(arr[:, :, 2] - b) <= tol)
    )
    cx, cy = W // 2, H // 2
    xs = np.where(ring[:, cy])[0]   # cruces de anillo sobre el eje horizontal
    ys = np.where(ring[cx, :])[0]   # cruces de anillo sobre el eje vertical
    # Comparar el MISMO anillo (el más interno) en cada eje: la pantalla es más
    # ancha que alta, así que son visibles más anillos en X que en Y; comparar
    # los exteriores mediría anillos distintos. Cada anillo individual, en
    # cambio, debe tener el mismo radio en X e Y si la escala es isotrópica.
    xs_pos = xs[xs > cx]            # cruces a la derecha del centro
    ys_neg = ys[ys < cy]           # cruces por encima del centro
    assert len(xs_pos) and len(ys_neg), "no se encontraron cruces de anillo"
    rx = int(xs_pos.min() - cx)    # radio del anillo más interno en X
    ry = int(cy - ys_neg.max())    # radio del anillo más interno en Y
    assert abs(rx - ry) <= 3, f"anillos no circulares: rx={rx}px, ry={ry}px"
    print(f"  ✓ rings are circular (innermost rx={rx}px ≈ ry={ry}px)")


def test_marker_persists_during_drawing():
    """Issue #2: durante la respuesta, la tablet repinta TODA la pantalla con su
    canvas (negro + trazos) y solo DESPUÉS se dibuja el overlay (vía
    _display_flip). Simulamos ese orden de pintado y comprobamos que el
    marcador del fosfeno sigue visible mientras el participante dibuja."""
    ov = _new_overlay()
    ov.set_phosphene(
        px_xy=(CENTER[0] + int(10 * PPD_X), CENTER[1] - int(5 * PPD_Y)),
        deg_xy=(10.0, 5.0), ecc_deg=11.18, polar_deg=26.57, electrode_index=16,
    )
    surf = pygame.Surface((W, H))
    # 1) la tablet pinta su canvas a pantalla completa (DrawingTablet.draw)
    canvas = pygame.Surface((W, H))
    canvas.fill((0, 0, 0))
    surf.blit(canvas, (0, 0))
    # 2) overlay por encima (phosphene_mapping._display_flip)
    ov.draw(surf)
    assert _count_color(surf, _MARKER_COLOR) > 20, (
        "marcador perdido en el composite de la fase de dibujo"
    )
    print("  ✓ marker persists through drawing-phase composite")


def test_render_png():
    surf = pygame.Surface((W, H))
    surf.fill((0, 0, 0))
    ov = _new_overlay()
    ov.set_phosphene(
        px_xy=(1280 + int(10 * PPD_X), 720 - int(5 * PPD_Y)),
        deg_xy=(10.0, 5.0), ecc_deg=11.18, polar_deg=26.57, electrode_index=16,
    )
    ov.draw(surf)
    out = Path(__file__).resolve().parents[1] / "assets" / "mapping_debug_preview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surf, str(out))
    print(f"  ✓ preview rendered → {out}")


def main():
    pygame.init()
    pygame.font.init()
    print("[debug_overlay_smoke_test] running...")
    test_grid_draws()
    test_marker_sets_and_clears()
    test_from_config_gating()
    test_rings_are_circular()
    test_marker_persists_during_drawing()
    test_render_png()
    pygame.quit()
    print("\nAll debug_overlay smoke tests passed.")


if __name__ == "__main__":
    main()
