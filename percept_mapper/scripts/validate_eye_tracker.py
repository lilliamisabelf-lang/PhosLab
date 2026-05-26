"""Eye-tracker accuracy validation against a 9-point fixation grid.

Independent of Pupil Capture's own calibration / accuracy visualizer. Shows
9 dots in a 3x3 grid, asks the participant to fixate each for `fixate_ms`
ms, records the mean smoothed gaze during the fixation window, and reports
per-point error in degrees of visual angle. Refuses to start an experiment
if mean error exceeds `max_mean_error_deg`.

Run standalone:
    uv run --project percept_mapper python percept_mapper/scripts/validate_eye_tracker.py

Or import `run_validation_grid(config, eye_tracker, screen)` to gate the
experiment from inside main.py.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pygame


@dataclass
class GridPointResult:
    grid_x: int
    grid_y: int
    target_px: tuple[float, float]
    samples: list[tuple[float, float]]
    mean_gaze_px: tuple[float, float] | None
    error_px: float | None
    error_deg: float | None

    def to_dict(self) -> dict:
        return {
            "grid_x": self.grid_x,
            "grid_y": self.grid_y,
            "target_px": list(self.target_px),
            "n_samples": len(self.samples),
            "mean_gaze_px": list(self.mean_gaze_px) if self.mean_gaze_px else None,
            "error_px": self.error_px,
            "error_deg": self.error_deg,
        }


@dataclass
class ValidationReport:
    screen_size: tuple[int, int]
    pixels_per_degree: float
    points: list[GridPointResult]
    mean_error_deg: float | None
    max_error_deg: float | None
    n_points_valid: int
    passed: bool
    threshold_deg: float
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "screen_size": list(self.screen_size),
            "pixels_per_degree": self.pixels_per_degree,
            "points": [p.to_dict() for p in self.points],
            "mean_error_deg": self.mean_error_deg,
            "max_error_deg": self.max_error_deg,
            "n_points_valid": self.n_points_valid,
            "passed": self.passed,
            "threshold_deg": self.threshold_deg,
            "timestamp": self.timestamp,
        }


def _grid_targets(screen_w: int, screen_h: int, inset_px: int = 200) -> list[tuple[int, int, tuple[float, float]]]:
    """Return 9 (grid_x, grid_y, target_px) tuples for a 3x3 grid inside the
    screen with `inset_px` margin."""
    xs = [inset_px, screen_w // 2, screen_w - inset_px]
    ys = [inset_px, screen_h // 2, screen_h - inset_px]
    points = []
    for gy, py in enumerate(ys):
        for gx, px in enumerate(xs):
            points.append((gx, gy, (float(px), float(py))))
    return points


def _draw_target(screen, target_px, frac_settle: float):
    """Draw the validation target: an outer ring that shrinks as the
    settling timer counts down. Encodes "hold fixation" tempo visually."""
    screen.fill((0, 0, 0))
    x, y = int(target_px[0]), int(target_px[1])
    pygame.draw.circle(screen, (255, 255, 255), (x, y), 6, 0)
    radius = int(50 - 40 * frac_settle)
    if radius > 6:
        pygame.draw.circle(screen, (180, 180, 180), (x, y), radius, 2)


def run_validation_grid(
    config: dict,
    eye_tracker,
    screen: pygame.Surface,
    *,
    settle_ms: int = 600,
    capture_ms: int = 700,
    inset_px: int = 200,
    fps: int = 60,
    max_mean_error_deg: float = 1.5,
) -> ValidationReport:
    """Drive the 9-point grid. Returns a ValidationReport.

    `settle_ms` is the lead-in before sampling starts at each point — gives
    the participant time to saccade. `capture_ms` is the sampling window;
    we average all smoothed-gaze readings in that window.
    """
    screen_size = screen.get_size()
    sw, sh = screen_size
    px_per_deg = _pixels_per_degree(config, sw)
    points_def = _grid_targets(sw, sh, inset_px=inset_px)
    clock = pygame.time.Clock()
    results: list[GridPointResult] = []

    for grid_x, grid_y, target_px in points_def:
        # Settling phase: show target, no sampling.
        t0 = time.monotonic()
        while True:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            if elapsed_ms >= settle_ms:
                break
            _maybe_get_frame(eye_tracker)
            _draw_target(screen, target_px, elapsed_ms / settle_ms)
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    return _make_report(screen_size, px_per_deg, results, max_mean_error_deg)
            pygame.display.flip()
            clock.tick(fps)

        # Capture phase: sample smoothed gaze.
        samples: list[tuple[float, float]] = []
        t0 = time.monotonic()
        while True:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            if elapsed_ms >= capture_ms:
                break
            _maybe_get_frame(eye_tracker)
            g = _read_smooth_gaze(eye_tracker)
            if g is not None:
                samples.append((float(g[0]), float(g[1])))
            _draw_target(screen, target_px, 1.0)
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    return _make_report(screen_size, px_per_deg, results, max_mean_error_deg)
            pygame.display.flip()
            clock.tick(fps)

        if samples:
            mean_x = sum(s[0] for s in samples) / len(samples)
            mean_y = sum(s[1] for s in samples) / len(samples)
            mean_gaze = (mean_x, mean_y)
            dx = mean_x - target_px[0]
            dy = mean_y - target_px[1]
            err_px = (dx * dx + dy * dy) ** 0.5
            err_deg = err_px / max(px_per_deg, 1e-9)
        else:
            mean_gaze = None
            err_px = None
            err_deg = None

        results.append(GridPointResult(
            grid_x=grid_x,
            grid_y=grid_y,
            target_px=target_px,
            samples=samples,
            mean_gaze_px=mean_gaze,
            error_px=err_px,
            error_deg=err_deg,
        ))
        print(
            f"[validate_eye_tracker] ({grid_x},{grid_y}) "
            f"n={len(samples)}  error_deg={err_deg if err_deg is None else f'{err_deg:.2f}'}"
        )

    return _make_report(screen_size, px_per_deg, results, max_mean_error_deg)


def _maybe_get_frame(eye_tracker):
    if eye_tracker is not None and hasattr(eye_tracker, "get_frame"):
        try:
            eye_tracker.get_frame()
        except Exception:
            pass


def _read_smooth_gaze(eye_tracker):
    if eye_tracker is None:
        return None
    g = getattr(eye_tracker, "last_smooth_gaze", None)
    if g is not None:
        return g
    g = getattr(eye_tracker, "last_raw_gaze", None)
    return g


def _pixels_per_degree(config: dict, screen_w: int) -> float:
    """Approximate isotropic px/deg from screen geometry in params.yaml."""
    screen_cfg = config.get("screen", {})
    diag_in = float(screen_cfg.get("screen_diagonal_inches", 13.3))
    dist_cm = float(screen_cfg.get("dist_to_screen_cm", 60.0))
    h = int(screen_cfg.get("height", 1080))
    # Estimate horizontal pixels-per-cm from diagonal + aspect ratio
    aspect = screen_w / max(h, 1)
    diag_cm = diag_in * 2.54
    width_cm = diag_cm * aspect / ((1 + aspect ** 2) ** 0.5)
    px_per_cm = screen_w / max(width_cm, 1e-6)
    # 1 deg of visual angle at viewing distance D is D * tan(1 deg) cm
    import math
    cm_per_deg = dist_cm * math.tan(math.radians(1.0))
    return px_per_cm * cm_per_deg


def _make_report(
    screen_size: tuple[int, int],
    px_per_deg: float,
    points: list[GridPointResult],
    threshold_deg: float,
) -> ValidationReport:
    errs = [p.error_deg for p in points if p.error_deg is not None]
    if errs:
        mean_err = sum(errs) / len(errs)
        max_err = max(errs)
        passed = mean_err <= threshold_deg
    else:
        mean_err = None
        max_err = None
        passed = False
    return ValidationReport(
        screen_size=screen_size,
        pixels_per_degree=px_per_deg,
        points=points,
        mean_error_deg=mean_err,
        max_error_deg=max_err,
        n_points_valid=len(errs),
        passed=passed,
        threshold_deg=threshold_deg,
        timestamp=datetime.now().isoformat(),
    )


def save_report(report: ValidationReport, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"eye_tracker_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    return fname


def _standalone_main():
    """Run the grid in standalone mode against the configured tracker."""
    import yaml
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "config" / "params.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pygame.init()
    screen_cfg = config.get("screen", {})
    sw = int(screen_cfg.get("width", 1920))
    sh = int(screen_cfg.get("height", 1080))
    screen = pygame.display.set_mode((sw, sh))
    pygame.display.set_caption("PhosLab eye-tracker validation")

    # Pick a tracker the same way main.py does.
    input_mode = config.get("input_mode", "mouse")
    eye_tracker = None
    if input_mode == "pupil":
        try:
            from core.pupil_tracker import PupilTracker
            pcfg = config.get("pupil", {})
            eye_tracker = PupilTracker(
                address=pcfg.get("address", "127.0.0.1"),
                port=int(pcfg.get("port", 50020)),
                surface_name=pcfg.get("surface_name", "phoslab_screen"),
                min_confidence=float(pcfg.get("min_confidence", 0.7)),
                one_euro=pcfg.get("one_euro"),
            )
        except Exception as e:
            print(f"[validate] PupilTracker no disponible: {e}")
    if eye_tracker is None:
        from core.mouse_tracker import MouseTracker
        eye_tracker = MouseTracker()

    report = run_validation_grid(config, eye_tracker, screen)
    out_dir = repo_root / "logs" / "eye_tracker_validation"
    saved = save_report(report, out_dir)
    print()
    print("=" * 60)
    print(f"Mean error: {report.mean_error_deg}  (threshold={report.threshold_deg}°)")
    print(f"Max error:  {report.max_error_deg}")
    print(f"Pass:       {report.passed}")
    print(f"Saved to:   {saved}")
    print("=" * 60)


if __name__ == "__main__":
    _standalone_main()
