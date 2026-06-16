"""Calibración de gaze — 9 puntos, regresión afín 2D.

Muestra 9 puntos en una cuadrícula 3×3. Para cada punto recoge muestras
de posición del iris (MediaPipe) y ajusta un modelo afín 2D que mapea
coordenadas de iris → coordenadas de pantalla.

El resultado se guarda en percept_mapper/config/gaze_calibration.json
y EyeTracker lo carga automáticamente la próxima vez que se inicialice.

Uso:
    uv run python percept_mapper/calibrate_gaze.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from math import hypot
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pygame

# ── Landmarks de iris MediaPipe FaceMesh ──────────────────────────────
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

SETTLE_S   = 0.8    # espera tras aparecer el punto (tiempo para sacadar)
CAPTURE_S  = 1.5    # ventana de captura de muestras por punto
CAM_W      = 160    # miniatura webcam
CAM_H      = 120

CALIB_PATH = Path(__file__).resolve().parent / "config" / "gaze_calibration.json"


# ── Helpers ────────────────────────────────────────────────────────────

def _iris_center(landmarks, fw: int, fh: int) -> tuple[float, float] | None:
    pts = [(landmarks[i].x * fw, landmarks[i].y * fh)
           for i in LEFT_IRIS + RIGHT_IRIS]
    return (sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts))


def _calibration_points(sw: int, sh: int, margin: int = 200) -> list[tuple[int, int]]:
    """9 puntos en cuadrícula 3×3.  El margen garantiza que ningún punto
    queda bajo la miniatura de webcam (160×120 px en esquina sup. izq.)."""
    xs = [margin, sw // 2, sw - margin]
    ys = [margin, sh // 2, sh - margin]
    return [(x, y) for y in ys for x in xs]


def _draw_target(
    screen: pygame.Surface,
    pos: tuple[int, int],
    frac_settle: float,
    idx: int,
    total: int,
    font: pygame.font.Font,
) -> None:
    screen.fill((0, 0, 0))
    x, y = pos
    outer_r = int(50 - 40 * frac_settle)
    if outer_r > 8:
        pygame.draw.circle(screen, (140, 140, 140), (x, y), outer_r, 2)
    pygame.draw.circle(screen, (255, 255, 255), (x, y), 8)
    lbl = font.render(f"Mira el punto  {idx + 1} / {total}", True, (180, 180, 180))
    screen.blit(lbl, (20, screen.get_height() - 38))


def _draw_thumbnail(screen: pygame.Surface, frame) -> None:
    """Miniatura de webcam en esquina superior izquierda (no tapa ningún punto)."""
    if frame is None:
        return
    display_frame = cv2.flip(frame, 1)          # espejo para el usuario
    small = cv2.resize(display_frame, (CAM_W, CAM_H))
    small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    surf = pygame.surfarray.make_surface(small_rgb.transpose(1, 0, 2))
    screen.blit(surf, (10, 10))
    pygame.draw.rect(screen, (80, 80, 80), (9, 9, CAM_W + 2, CAM_H + 2), 1)


def _fit_affine(
    iris_pts: np.ndarray, screen_pts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Regresión afín 2D: screen ≈ [ix, iy, 1] · coeff.
    Retorna (coeff_x, coeff_y) — vectores de 3 elementos."""
    A = np.column_stack([iris_pts, np.ones(len(iris_pts))])
    coeff_x, _, _, _ = np.linalg.lstsq(A, screen_pts[:, 0], rcond=None)
    coeff_y, _, _, _ = np.linalg.lstsq(A, screen_pts[:, 1], rcond=None)
    return coeff_x, coeff_y


def _predict(
    coeff_x: np.ndarray, coeff_y: np.ndarray, ix: float, iy: float
) -> tuple[float, float]:
    v = np.array([ix, iy, 1.0])
    return float(coeff_x @ v), float(coeff_y @ v)


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    sw, sh = screen.get_size()
    pygame.display.set_caption("PhosLab — Calibración de gaze")
    clock = pygame.time.Clock()
    font     = pygame.font.Font(None, 32)
    font_big = pygame.font.Font(None, 52)

    # ── Cámara ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        pygame.quit()
        sys.exit("[calibrate_gaze] ERROR: no se pudo abrir la cámara.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    points = _calibration_points(sw, sh)

    # ── Pantalla de bienvenida ─────────────────────────────────────────
    screen.fill((0, 0, 0))
    lines = [
        (font_big, "Calibración de seguimiento ocular", (255, 255, 255)),
        (font, "Mantén la cabeza quieta.", (180, 180, 180)),
        (font, "Mira cada punto blanco hasta que desaparezca.", (180, 180, 180)),
        (font, "", (0, 0, 0)),
        (font, "Pulsa  ESPACIO  para comenzar  |  ESC para cancelar", (140, 140, 140)),
    ]
    total_h = sum(f.size("A")[1] + 12 for f, _, _ in lines)
    y0 = sh // 2 - total_h // 2
    for f, text, color in lines:
        surf = f.render(text, True, color)
        screen.blit(surf, surf.get_rect(center=(sw // 2, y0)))
        y0 += f.size("A")[1] + 12
    pygame.display.flip()

    waiting = True
    while waiting:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                cap.release(); face_mesh.close(); pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    cap.release(); face_mesh.close(); pygame.quit(); sys.exit()
                if ev.key == pygame.K_SPACE:
                    waiting = False
        clock.tick(30)

    # ── Bucle de calibración ───────────────────────────────────────────
    all_iris: list[tuple[float, float]] = []
    all_screen: list[tuple[int, int]]   = []

    for idx, (px, py) in enumerate(points):

        # Fase 1: settle — anillo encogiendo, sin captura
        t0 = time.monotonic()
        while True:
            elapsed = time.monotonic() - t0
            if elapsed >= SETTLE_S:
                break
            ret, frame = cap.read()
            raw_frame  = frame if ret else None   # sin flip → consistente con EyeTracker
            _draw_target(screen, (px, py), elapsed / SETTLE_S, idx, len(points), font)
            _draw_thumbnail(screen, raw_frame)
            for ev in pygame.event.get():
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    cap.release(); face_mesh.close(); pygame.quit(); sys.exit()
            pygame.display.flip()
            clock.tick(60)

        # Fase 2: capture — muestras de iris
        t0 = time.monotonic()
        while True:
            elapsed = time.monotonic() - t0
            if elapsed >= CAPTURE_S:
                break
            ret, frame = cap.read()
            raw_frame  = frame if ret else None

            if ret and raw_frame is not None:
                rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    fh, fw = raw_frame.shape[:2]
                    ic = _iris_center(lm, fw, fh)
                    if ic is not None:
                        all_iris.append(ic)
                        all_screen.append((px, py))

            _draw_target(screen, (px, py), 1.0, idx, len(points), font)
            _draw_thumbnail(screen, raw_frame)
            for ev in pygame.event.get():
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    cap.release(); face_mesh.close(); pygame.quit(); sys.exit()
            pygame.display.flip()
            clock.tick(60)

    cap.release()
    face_mesh.close()

    # ── Ajuste del modelo ──────────────────────────────────────────────
    if len(all_iris) < 18:   # mínimo ~2 muestras por punto
        screen.fill((0, 0, 0))
        msg = font_big.render(
            f"Pocas muestras ({len(all_iris)}) — mejora la iluminación y repite.",
            True, (255, 80, 80),
        )
        screen.blit(msg, msg.get_rect(center=(sw // 2, sh // 2)))
        pygame.display.flip()
        time.sleep(4)
        pygame.quit()
        sys.exit(1)

    iris_arr   = np.array(all_iris,   dtype=float)
    screen_arr = np.array(all_screen, dtype=float)
    coeff_x, coeff_y = _fit_affine(iris_arr, screen_arr)

    # Error residual (en píxeles) sobre los datos de calibración
    errors = [
        hypot(*_predict(coeff_x, coeff_y, ix, iy)) -
        hypot(sx, sy)
        for (ix, iy), (sx, sy) in zip(all_iris, all_screen)
    ]
    # Error euclidiano real
    errors = [
        hypot(
            _predict(coeff_x, coeff_y, ix, iy)[0] - sx,
            _predict(coeff_x, coeff_y, ix, iy)[1] - sy,
        )
        for (ix, iy), (sx, sy) in zip(all_iris, all_screen)
    ]
    mean_err_px = float(np.mean(errors))

    # ── Guardar JSON ───────────────────────────────────────────────────
    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    calib = {
        "coeff_x":          coeff_x.tolist(),
        "coeff_y":          coeff_y.tolist(),
        "screen_size":      [sw, sh],
        "n_samples":        len(all_iris),
        "mean_residual_px": round(mean_err_px, 2),
        "timestamp":        datetime.now().isoformat(),
    }
    CALIB_PATH.write_text(json.dumps(calib, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[calibrate_gaze] ✓ Calibración guardada en {CALIB_PATH}")
    print(f"[calibrate_gaze]   Muestras: {len(all_iris)}  |  error residual: {mean_err_px:.1f} px")

    # ── Pantalla de resultado ──────────────────────────────────────────
    screen.fill((0, 0, 0))
    ok     = font_big.render("Calibración completada ✓", True, (100, 230, 100))
    detail = font.render(
        f"{len(all_iris)} muestras  |  error residual: {mean_err_px:.0f} px",
        True, (200, 200, 200),
    )
    hint = font.render("ESC para salir  (cierre automático en 4 s)", True, (120, 120, 120))
    screen.blit(ok,     ok.get_rect(center=(sw // 2, sh // 2 - 50)))
    screen.blit(detail, detail.get_rect(center=(sw // 2, sh // 2 + 10)))
    screen.blit(hint,   hint.get_rect(center=(sw // 2, sh // 2 + 55)))
    pygame.display.flip()

    t_end = time.monotonic() + 4.0
    while time.monotonic() < t_end:
        for ev in pygame.event.get():
            if ev.type in (pygame.QUIT,):
                break
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                t_end = 0
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pygame.quit()
        sys.exit(0)
    except Exception as exc:
        print(f"[calibrate_gaze] ERROR: {exc}")
        import traceback
        traceback.print_exc()
        pygame.quit()
        sys.exit(1)
