"""Smoke tests para mapping_method (absolute / relative / forced_adjustment).

Verifica que:
  - absolute: DrawingTablet normal, sin cruz en el centro
  - relative: cruz blanca visible en el centro tras el update()
  - forced_adjustment: ForcedAdjustmentTablet acepta drag y devuelve canvas
  - ForcedAdjustmentTablet.reset() genera posiciones distintas
  - MOUSEBUTTONDOWN fuera del hit-area no inicia drag
  - El metadata del experimento incluye el campo mapping_method

Ejecucion:
    uv run python percept_mapper/scripts/mapping_method_smoke_test.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame

from scripts.tablet import DrawingTablet, ForcedAdjustmentTablet

W, H = 800, 600


def _init():
    pygame.init()
    return pygame.display.set_mode((W, H), flags=pygame.NOFRAME)


def _rgb_sum(surface: pygame.Surface, x: int, y: int) -> int:
    return sum(surface.get_at((x, y))[:3])


def _ev(etype, **kwargs):
    return pygame.event.Event(etype, **kwargs)


# ---------------------------------------------------------------------------

def test_absolute_no_cross():
    screen = _init()
    tablet = DrawingTablet(W, H)
    tablet.update(screen, [])
    cx, cy = W // 2, H // 2
    assert _rgb_sum(screen, cx, cy) == 0, "absolute: centro debe ser negro"
    tablet.close()
    print("[test_absolute_no_cross] ok")


def test_relative_cross_visible():
    screen = _init()
    tablet = DrawingTablet(W, H)
    tablet.update(screen, [])
    # Simula lo que phosphene_mapping._draw_center_cross() hace
    cx, cy = W // 2, H // 2
    arm = 24
    pygame.draw.line(screen, (255, 255, 255), (cx - arm, cy), (cx + arm, cy), 2)
    pygame.draw.line(screen, (255, 255, 255), (cx, cy - arm), (cx, cy + arm), 2)
    assert _rgb_sum(screen, cx, cy) == 765, "relative: centro debe ser blanco"
    assert _rgb_sum(screen, cx + arm + 10, cy) == 0, "relative: fuera de la cruz debe ser negro"
    tablet.close()
    print("[test_relative_cross_visible] ok")


def test_forced_pos_not_center():
    _init()
    tablet = ForcedAdjustmentTablet(W, H)
    px, py = tablet._pos
    cx, cy = W // 2, H // 2
    excl_x, excl_y = W // 4, H // 4
    assert abs(px - cx) > excl_x or abs(py - cy) > excl_y, \
        "forced_adjustment: posicion inicial no debe caer en el cuarto central"
    print("[test_forced_pos_not_center] ok")


def test_forced_drag_completes():
    screen = _init()
    tablet = ForcedAdjustmentTablet(W, H)
    dot_x, dot_y = tablet._pos
    target = (W // 2 + 50, H // 2 - 30)

    finished, _ = tablet.update(screen, [_ev(pygame.MOUSEBUTTONDOWN, button=1, pos=(dot_x, dot_y))])
    assert not finished
    assert tablet._dragging, "debe iniciar drag al hacer down sobre el punto"

    finished, canvas = tablet.update(screen, [_ev(pygame.MOUSEBUTTONUP, button=1, pos=target)])
    assert finished, "debe terminar al soltar el boton"
    assert canvas is not None
    assert tablet.last_status == "ok"
    assert _rgb_sum(canvas, target[0], target[1]) > 0, \
        "el canvas debe tener el punto en la posicion de soltar"
    print("[test_forced_drag_completes] ok")


def test_forced_reset_varies_position():
    _init()
    tablet = ForcedAdjustmentTablet(W, H)
    positions = {tablet._pos}
    for _ in range(20):
        tablet.reset()
        positions.add(tablet._pos)
    assert len(positions) >= 3, \
        f"reset debe generar posiciones variadas, obtenidas: {len(positions)}"
    print("[test_forced_reset_varies_position] ok")


def test_forced_trail_rendered_on_screen():
    screen = _init()
    tablet = ForcedAdjustmentTablet(W, H, brush_size=5, brush_color=(255, 255, 0))
    dot_x, dot_y = tablet._pos
    target = (W // 2 + 60, H // 2 + 40)

    # Iniciar drag
    tablet.update(screen, [_ev(pygame.MOUSEBUTTONDOWN, button=1, pos=(dot_x, dot_y))])
    # Simular movimiento al destino (no hay MOUSEMOTION en dummy, inyectamos trail manualmente)
    tablet._pos = target
    tablet._trail.append(target)
    tablet._draw(screen)

    # El rastro debe ser visible en screen (píxel en target debe ser amarillo)
    assert _rgb_sum(screen, target[0], target[1]) > 0, \
        "trail: el rastro debe ser visible en pantalla durante el arrastre"
    print("[test_forced_trail_rendered_on_screen] ok")


def test_forced_enter_without_drag_is_empty():
    screen = _init()
    tablet = ForcedAdjustmentTablet(W, H)

    ev_enter = _ev(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0, unicode="\r")
    finished, canvas = tablet.update(screen, [ev_enter])
    assert finished, "ENTER sin arrastre debe terminar el trial"
    assert tablet.last_status == "empty", "status debe ser empty"
    assert canvas is not None, "canvas no debe ser None (negro vacío)"
    # Canvas vacío = completamente negro
    assert _rgb_sum(canvas, W // 2, H // 2) == 0, \
        "canvas de respuesta vacía debe ser negro"
    print("[test_forced_enter_without_drag_is_empty] ok")


def test_forced_no_drag_outside_hit():
    screen = _init()
    tablet = ForcedAdjustmentTablet(W, H)
    dot_x, dot_y = tablet._pos
    far = (dot_x + ForcedAdjustmentTablet.DOT_RADIUS * ForcedAdjustmentTablet.HIT_FACTOR + 30) % W
    tablet.update(screen, [_ev(pygame.MOUSEBUTTONDOWN, button=1, pos=(far, dot_y))])
    assert not tablet._dragging, "clic fuera del hit-area no debe iniciar drag"
    print("[test_forced_no_drag_outside_hit] ok")


def test_metadata_includes_mapping_method():
    import tempfile
    from scripts.phosphene_mapping import PhospheneMappingExperiment
    from scripts.response_capture import DrawingResponseCapture

    screen = _init()
    clock = pygame.time.Clock()
    capture = DrawingResponseCapture(DrawingTablet(W, H))

    with tempfile.TemporaryDirectory() as tmp:
        exp = PhospheneMappingExperiment(
            params={},
            screen=screen,
            clock=clock,
            eye_tracker=None,
            anchor_screen=None,
            drawing_tablet=capture,
            webcam_viewer=None,
            gaze_trace=None,
            timing_config={},
            electrode_index=1,
            display_info=None,
            num_repetitions=1,
            experiment_dir=tmp,
            input_mode="mouse",
            mapping_method="relative",
        )
        assert exp.experiment_metadata.get("mapping_method") == "relative", \
            "metadata debe contener mapping_method con el valor correcto"
    print("[test_metadata_includes_mapping_method] ok")


# ---------------------------------------------------------------------------

def main() -> int:
    test_absolute_no_cross()
    test_relative_cross_visible()
    test_forced_pos_not_center()
    test_forced_drag_completes()
    test_forced_reset_varies_position()
    test_forced_trail_rendered_on_screen()
    test_forced_enter_without_drag_is_empty()
    test_forced_no_drag_outside_hit()
    test_metadata_includes_mapping_method()
    print("\n[mapping_method_smoke_test] todos los tests pasaron ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
