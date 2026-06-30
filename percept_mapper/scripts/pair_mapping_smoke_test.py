"""Smoke tests para el método de mapeo PAREADO (paired).

Verifica, sin hardware ni display real (SDL dummy):
  - LineDrawingTablet: dos clics fijan A y B en orden; ENTER → status=ok
  - LineDrawingTablet: tecla 1/2 fuerza "solo vi uno" → status=partial
  - LineDrawingTablet: ENTER sin clics → status=empty
  - LineDrawingTablet.save_result escribe PNG y devuelve endpoints + Δ en debug
  - DrawingResponseCapture delega save_result al LineDrawingTablet (no pierde Δ)
  - PairMappingExperiment.run_pair recorre A→rest→B→línea y persiste un registro
    con endpoint_a_px / endpoint_b_px / displacement_px en el metadata
  - El metadata pareado sobrevive el round-trip por TrialRecord (extras)

Ejecución:
    uv run python scripts/pair_mapping_smoke_test.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame

from scripts.tablet import LineDrawingTablet
from scripts.response_capture import DrawingResponseCapture

W, H = 800, 600


def _init():
    pygame.init()
    return pygame.display.set_mode((W, H), flags=pygame.NOFRAME)


def _ev(etype, **kwargs):
    return pygame.event.Event(etype, **kwargs)


def _click(pos):
    return _ev(pygame.MOUSEBUTTONDOWN, button=1, pos=pos)


def _enter():
    return _ev(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0, unicode="\r")


def _key(k):
    return _ev(pygame.KEYDOWN, key=k, mod=0, unicode="")


# ---------------------------------------------------------------------------
# LineDrawingTablet
# ---------------------------------------------------------------------------

def test_line_two_clicks_ordered_ok():
    screen = _init()
    t = LineDrawingTablet(W, H)
    a, b = (200, 150), (500, 400)

    fin, _ = t.update(screen, [_click(a)])
    assert not fin and t.endpoint_a == a and t.endpoint_b is None
    fin, _ = t.update(screen, [_click(b)])
    assert not fin and t.endpoint_b == b
    fin, canvas = t.update(screen, [_enter()])
    assert fin and t.last_status == "ok"
    assert canvas is not None
    t.close()
    print("[test_line_two_clicks_ordered_ok] ok")


def test_line_undo_removes_last():
    screen = _init()
    t = LineDrawingTablet(W, H)
    t.update(screen, [_click((100, 100))])
    t.update(screen, [_click((300, 300))])
    t.update(screen, [_key(LineDrawingTablet.R_DELETE_LAST)])
    assert t.endpoint_b is None and t.endpoint_a == (100, 100)
    t.update(screen, [_key(LineDrawingTablet.R_DELETE_LAST)])
    assert t.endpoint_a is None
    t.close()
    print("[test_line_undo_removes_last] ok")


def test_line_partial_saw_only_first():
    screen = _init()
    t = LineDrawingTablet(W, H)
    t.update(screen, [_click((250, 250))])     # A
    t.update(screen, [_click((400, 400))])     # B (será descartado)
    t.update(screen, [_key(pygame.K_1)])       # "solo vi el 1º" → borra B
    assert t.endpoint_b is None and t.endpoint_a is not None
    fin, _ = t.update(screen, [_enter()])
    assert fin and t.last_status == "partial"
    t.close()
    print("[test_line_partial_saw_only_first] ok")


def test_line_empty_enter_without_clicks():
    screen = _init()
    t = LineDrawingTablet(W, H)
    fin, canvas = t.update(screen, [_enter()])
    assert fin and t.last_status == "empty"
    assert canvas is not None
    t.close()
    print("[test_line_empty_enter_without_clicks] ok")


def test_line_save_result_carries_endpoints():
    screen = _init()
    t = LineDrawingTablet(W, H)
    a, b = (120, 130), (470, 360)
    t.update(screen, [_click(a)])
    t.update(screen, [_click(b)])
    t.update(screen, [_enter()])
    with tempfile.TemporaryDirectory() as tmp:
        res = t.save_result(tmp, drawing_filename="pair_001.png")
        assert (Path(tmp) / "pair_001.png").exists(), "debe escribir el PNG"
        assert res.mode == "paired_line" and res.status == "ok"
        assert res.debug["endpoint_a_px"] == list(a)
        assert res.debug["endpoint_b_px"] == list(b)
        assert res.debug["displacement_px"] == [b[0] - a[0], b[1] - a[1]]
        # metadata fold-through: las claves *_px llegan al dict de metadata
        meta = res.to_metadata()
        assert meta["endpoint_a_px"] == list(a)
        assert meta["displacement_px"] == [b[0] - a[0], b[1] - a[1]]
    t.close()
    print("[test_line_save_result_carries_endpoints] ok")


def test_capture_wrapper_delegates_save_result():
    screen = _init()
    inner = LineDrawingTablet(W, H)
    cap = DrawingResponseCapture(inner)
    a, b = (150, 160), (600, 420)
    cap.reset()
    # Conducir el wrapper como lo hace el bucle de respuesta
    assert cap.update(screen, [_click(a)]) is False
    assert cap.update(screen, [_click(b)]) is False
    assert cap.update(screen, [_enter()]) is True
    with tempfile.TemporaryDirectory() as tmp:
        res = cap.save_result(tmp, drawing_filename="pair_002.png")
        # El wrapper NO debe degradar a "drawing" perdiendo los endpoints
        assert res.mode == "paired_line", f"esperado paired_line, got {res.mode}"
        assert res.debug["endpoint_a_px"] == list(a)
        assert res.debug["endpoint_b_px"] == list(b)
    inner.close()
    print("[test_capture_wrapper_delegates_save_result] ok")


# ---------------------------------------------------------------------------
# PairMappingExperiment
# ---------------------------------------------------------------------------

class _FakeStim:
    def __init__(self, idx):
        self.active_electrode_index = idx


def _ok_phase(*args, **kwargs):
    """run_prestim/stim/poststim stub: siempre completa."""
    return True


def test_pair_experiment_runs_and_persists():
    from scripts.pair_mapping import PairMappingExperiment

    screen = _init()
    clock = pygame.time.Clock()
    inner = LineDrawingTablet(W, H)
    cap = DrawingResponseCapture(inner)

    a, b = (180, 220), (540, 380)

    # Inyectar los eventos de respuesta de forma que el bucle de drawing los
    # consuma vía pygame.event.get(): dos clics + ENTER, repartidos en frames.
    posted = {"step": 0}
    frames = [[_click(a)], [_click(b)], [_enter()]]

    def reset_func(tablet):
        tablet.reset()
        posted["step"] = 0

    # Monkeypatch pygame.event.get para alimentar los frames en orden.
    real_get = pygame.event.get

    def fake_get(*a_, **k_):
        s = posted["step"]
        posted["step"] += 1
        return frames[s] if s < len(frames) else []

    pygame.event.get = fake_get
    try:
        with tempfile.TemporaryDirectory() as tmp:
            exp = PairMappingExperiment(
                params={"screen": {"background_color": [0, 0, 0]}},
                screen=screen,
                clock=clock,
                eye_tracker=None,
                anchor_screen=None,
                drawing_tablet=cap,
                webcam_viewer=None,
                gaze_trace=None,
                timing_config={"interstimulation_ms": 0},
                display_info=None,
                experiment_dir=tmp,
                input_mode="mouse",
                coords_csv="synthetic.csv",
                rest_ms=0.0,   # sin espera real en el test
            )
            rec = exp.run_pair(
                pair_index=1,
                electrode_a=3,
                electrode_b=7,
                stim_a=_FakeStim(3),
                stim_b=_FakeStim(7),
                pos_a=(100, 100),
                pos_b=(200, 200),
                current_a=90.0,
                current_b=90.0,
                pulse_width_us=100.0,
                frequency_hz=300.0,
                run_prestim_func=_ok_phase,
                run_stim_func=_ok_phase,
                run_poststim_func=_ok_phase,
                drawing_tablet_reset_func=reset_func,
                FPS=200,
            )
            assert rec is not None, "run_pair no debe devolver None"
            assert rec["electrode_a"] == 3 and rec["electrode_b"] == 7
            assert rec["response_status"] == "ok"
            assert rec["endpoint_a_px"] == list(a)
            assert rec["endpoint_b_px"] == list(b)
            assert rec["displacement_px"] == [b[0] - a[0], b[1] - a[1]]

            exp.finalize()
            meta_path = Path(tmp) / "pairs" / "metadata.json"
            assert meta_path.exists(), "debe escribir pairs/metadata.json"
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            assert data["mapping_method"] == "paired"
            assert len(data["trials"]) == 1
            # round-trip por TrialRecord: los campos pareados sobreviven en extras
            t0 = data["trials"][0]
            assert t0["endpoint_a_px"] == list(a)
            assert t0["displacement_px"] == [b[0] - a[0], b[1] - a[1]]
            assert t0["electrode_a"] == 3 and t0["pair_index"] == 1
            assert (Path(tmp) / "pairs" / "pair_001.png").exists()
    finally:
        pygame.event.get = real_get
    inner.close()
    print("[test_pair_experiment_runs_and_persists] ok")


# ---------------------------------------------------------------------------

def main() -> int:
    test_line_two_clicks_ordered_ok()
    test_line_undo_removes_last()
    test_line_partial_saw_only_first()
    test_line_empty_enter_without_clicks()
    test_line_save_result_carries_endpoints()
    test_capture_wrapper_delegates_save_result()
    test_pair_experiment_runs_and_persists()
    print("\n[pair_mapping_smoke_test] todos los tests pasaron ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
