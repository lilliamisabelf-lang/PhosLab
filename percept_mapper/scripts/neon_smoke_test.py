"""Quick diagnostic: connect to a Pupil Neon via realtime_api and report what's
flowing — device reachable, matched (scene, gaze) arriving, AprilTags detected in
the scene frame, homography computed, and a sample gaze warped to screen pixels.

NOTE: this runs headless (no experiment window), so the 4 corner AprilTags are NOT
on screen unless you separately display them — e.g. in another terminal run:
    uv run --project percept_mapper python percept_mapper/scripts/show_apriltags_fullscreen.py
and point the Neon's scene camera at that monitor. Without tags visible, steps
(c)/(d) below will report 0 tags / no homography, which is expected.

Run from repo root:
    uv run --project percept_mapper python percept_mapper/scripts/neon_smoke_test.py
"""

import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import yaml

# Reuse the tracker's homography/geometry so the test exercises the real code path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.neon_tracker import NeonTracker

WATCH_SECONDS = 5.0


def load_config():
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "params.yaml"
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[smoke] ⚠ no se pudo leer params.yaml: {e}")
        cfg = {}
    return cfg


def main():
    cfg = load_config()
    neon_cfg = cfg.get("neon") or {}
    screen_cfg = cfg.get("screen") or {}
    overlay_cfg = cfg.get("apriltag_overlay") or {}

    screen_size = (
        int(screen_cfg.get("width", 1536)),
        int(screen_cfg.get("height", 864)),
    )

    # Build a NeonTracker but DON'T let it start its own loop driving an experiment;
    # instead we drive the device directly here for a controlled report. We still
    # reuse its detector + homography helpers.
    print("[smoke] Conectando a Neon (puede tardar por descubrimiento)...")
    try:
        tracker = NeonTracker(
            address=neon_cfg.get("address", ""),
            port=neon_cfg.get("port", 8080),
            discover_timeout_s=neon_cfg.get("discover_timeout_s", 10.0),
            min_confidence=neon_cfg.get("min_confidence", 0.7),
            one_euro=neon_cfg.get("one_euro"),
            max_sample_age_s=neon_cfg.get("max_sample_age_s", 0.25),
            apriltag_overlay=None,  # headless; geometry taken from overlay cfg below
            homography_min_tags=neon_cfg.get("homography_min_tags", 4),
        )
    except Exception as e:
        print("[smoke] ✗ No se pudo iniciar Neon:")
        print(f"        {e}")
        sys.exit(2)

    # Align the tracker's screen-tag geometry with params.yaml's overlay block so
    # the homography correspondence matches what the experiment would draw.
    tracker._tag_size_px = int(overlay_cfg.get("tag_size_px", tracker._tag_size_px))
    tracker._margin_px = int(overlay_cfg.get("margin_px", tracker._margin_px))
    if not overlay_cfg.get("enabled", False):
        print("[smoke] ⚠ apriltag_overlay.enabled = false en params.yaml.")
        print("        El experimento real lo requiere para input_mode: neon.")

    print("[smoke] ✓ Neon conectado.")
    print(f"[smoke] Escuchando {WATCH_SECONDS:.0f}s... (apunta la escena al monitor con los tags)\n")

    frames = 0
    with_gaze = 0
    best_tags = 0
    homographies = 0
    sample_warp = None

    deadline = time.time() + WATCH_SECONDS
    while time.time() < deadline:
        try:
            matched = tracker._device.receive_matched_scene_video_frame_and_gaze(
                timeout_seconds=0.5
            )
        except Exception:
            continue
        if matched is None:
            continue
        frames += 1

        g = matched.gaze
        if g is None or not getattr(g, "worn", True):
            continue
        with_gaze += 1

        bgr = matched.frame.bgr_pixels
        H, n_tags = tracker._compute_homography(bgr, screen_size)
        best_tags = max(best_tags, n_tags)
        if H is not None:
            homographies += 1
            sample_warp = tracker._warp_point(H, float(g.x), float(g.y))

    print("[smoke] === RESUMEN ===")
    print(f"[smoke] (a) frames escena+gaze recibidos : {frames}")
    print(f"[smoke] (b) frames con gaze (worn)       : {with_gaze}")
    print(f"[smoke] (c) max tags detectados en escena: {best_tags}/4")
    print(f"[smoke] (d) homografías calculadas       : {homographies}")
    if sample_warp is not None:
        sx, sy = sample_warp
        in_screen = 0 <= sx <= screen_size[0] and 0 <= sy <= screen_size[1]
        print(
            f"[smoke]     gaze -> pantalla (ejemplo)   : ({sx:.0f}, {sy:.0f}) "
            f"{'dentro' if in_screen else 'FUERA'} de {screen_size}"
        )

    print("\n[smoke] === DIAGNOSTICO ===")
    if frames == 0:
        print("[smoke] ✗ No llegaron frames -> ¿cámara de escena activa y gafas puestas?")
    elif with_gaze == 0:
        print("[smoke] ✗ Frames sí, gaze no -> las gafas no están puestas (worn=False).")
    elif best_tags < tracker.homography_min_tags:
        print(f"[smoke] ⚠ Solo {best_tags}/4 tags vistos -> no hay homografía.")
        print("        Muestra los AprilTags en pantalla y apunta la escena al monitor.")
        print("        Sube apriltag_overlay.tag_size_px (→300) si se ven pequeños.")
    elif homographies > 0:
        print("[smoke] ✓ Cadena completa: Neon -> tags -> homografía -> gaze en pantalla.")
        print("[smoke]   PhosLab debería validar la fijación con input_mode: neon.")

    tracker.release()


if __name__ == "__main__":
    main()
