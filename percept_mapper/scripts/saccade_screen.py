"""SaccadeScreen — gaze-based response capture stage.

Drop-in alternative to DrawingTablet for the response phase. Instead of having
the participant draw the perceived phosphene location, this screen captures
the participant's gaze for a fixed window and extracts a single (x, y) point
representing their saccade endpoint (or any of the three extraction methods).

Interface mirrors DrawingTablet:
    reset()
    update(screen, events) -> (finished: bool, payload: dict | None)
    close()

The payload dict has shape:
    {
        "response_xy": (x, y) | None,
        "samples": [{"t": float, "x": float, "y": float}, ...],
        "extraction": str,
        "status": "ok" | "failed_no_fixation" | "failed_no_motion" | ...,
        "attempts": int,
    }

TODO: replace the fixed capture window with a return-to-anchor detector once
gaze quality is good enough — the participant's saccade back to center is a
natural trial-end signal, more ecological than a hard timeout.
"""

from __future__ import annotations

import time
from typing import Any

import pygame

from scripts import saccade_extractors


class SaccadeScreen:
    """Captures gaze samples after a stimulus and extracts a response point."""

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        anchor_xy,
        eye_tracker,
        capture_duration_ms: int = 1500,
        extraction: str = "idt_first_fixation",
        extractor_params: dict | None = None,
        min_response_distance_px: float = 30.0,
        max_attempts: int = 3,
        show_gaze_trace: bool = True,
        allow_mouse_fallback: bool = False,
        audio_cue=None,
        anchor_radius: int = 50,
        anchor_color=(255, 0, 0),
        anchor_thickness: int = 3,
    ):
        print(f"[SaccadeScreen] Inicializando (extraction={extraction})...")

        self.screen_width = int(screen_width)
        self.screen_height = int(screen_height)
        self.anchor_xy = (float(anchor_xy[0]), float(anchor_xy[1]))
        self.eye_tracker = eye_tracker
        self.capture_duration_s = float(capture_duration_ms) / 1000.0
        self.extraction = str(extraction)
        self.extractor_params = dict(extractor_params or {})
        self.min_response_distance_px = float(min_response_distance_px)
        self.max_attempts = int(max(1, max_attempts))
        self.show_gaze_trace = bool(show_gaze_trace)
        self.allow_mouse_fallback = bool(allow_mouse_fallback)
        self.audio_cue = audio_cue
        self.anchor_radius = int(anchor_radius)
        self.anchor_color = tuple(anchor_color)
        self.anchor_thickness = int(anchor_thickness)

        self.font = pygame.font.Font(None, 48)
        self.title_text = "Mira el fosfeno y vuelve al centro"

        # Per-trial state
        self.samples: list[dict[str, Any]] = []
        self._capture_start_t: float | None = None
        self._finished = False
        self._attempts = 0
        self._last_status: str | None = None

        print("[SaccadeScreen] ✓ Inicializado")

    # ---- lifecycle ----------------------------------------------------------

    def reset(self):
        """Arm a new capture window. Audio cue fires on the first update()
        after reset (after the first frame is rendered, so we know pygame
        timing is live)."""
        self.samples = []
        self._capture_start_t = None
        self._finished = False
        self._attempts += 1
        self._last_status = None
        print(
            f"[SaccadeScreen] Reseteado, intento {self._attempts}/{self.max_attempts}"
        )

    def close(self):
        # No persistent resources today; method exists for symmetry with
        # DrawingTablet so callers can swap implementations.
        pass

    # ---- main loop ----------------------------------------------------------

    def update(self, screen, events):
        """Drive one frame. Returns (finished, payload).

        Sampling source priority:
          1. eye_tracker.last_smooth_gaze (if set and not None)
          2. eye_tracker.last_raw_gaze (fallback)
          3. pygame.mouse.get_pos() — debug mode if no tracker / no samples yet
        """
        # Initialize capture clock + play tone on first update after reset
        if self._capture_start_t is None:
            self._capture_start_t = time.monotonic()
            self._play_audio_cue()

        # Process events (allow early abort with ESC, manual rerun with R)
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._last_status = "aborted_by_user"
                    return self._finalize(force_status="aborted_by_user")
                if event.key == pygame.K_r:
                    # Manual rerun — reset clock + samples without bumping
                    # attempts counter past max (user-initiated retry).
                    self.samples = []
                    self._capture_start_t = time.monotonic()
                    print("[SaccadeScreen] Manual retry (R)")

        # Capture one sample this frame
        now = time.monotonic()
        elapsed = now - self._capture_start_t
        pt = self._read_gaze()
        if pt is not None:
            self.samples.append(
                {"t": elapsed, "x": float(pt[0]), "y": float(pt[1])}
            )

        # Draw frame
        self._draw(screen, elapsed)

        # Finished?
        if elapsed >= self.capture_duration_s:
            return self._finalize()

        return (False, None)

    # ---- helpers ------------------------------------------------------------

    def _read_gaze(self):
        et = self.eye_tracker
        if et is not None:
            p = getattr(et, "last_smooth_gaze", None)
            if p is not None:
                return p
            p = getattr(et, "last_raw_gaze", None)
            if p is not None:
                return p
        if self.allow_mouse_fallback:
            # Debug fallback for input_mode: mouse / Wacom tests.
            return pygame.mouse.get_pos()
        return None

    def _play_audio_cue(self):
        if self.audio_cue is None:
            return
        try:
            self.audio_cue.play()
        except Exception as e:
            print(f"[SaccadeScreen] ⚠ audio_cue.play falló: {e}")

    def _draw(self, screen, elapsed):
        screen.fill((0, 0, 0))

        # Anchor circle (return target)
        ax, ay = self.anchor_xy
        pygame.draw.circle(
            screen,
            self.anchor_color,
            (int(ax), int(ay)),
            self.anchor_radius,
            self.anchor_thickness,
        )

        # Optional gaze trace overlay (faint, last ~300 ms)
        if self.show_gaze_trace and len(self.samples) >= 2:
            cutoff = elapsed - 0.3
            recent = [s for s in self.samples if s["t"] >= cutoff]
            if len(recent) >= 2:
                points = [(int(s["x"]), int(s["y"])) for s in recent]
                pygame.draw.lines(screen, (120, 120, 120), False, points, 2)

        # Progress text (lightweight, top of screen)
        remaining = max(0.0, self.capture_duration_s - elapsed)
        title = self.font.render(self.title_text, True, (255, 255, 255))
        screen.blit(
            title,
            title.get_rect(center=(self.screen_width // 2, 50)),
        )

    def _finalize(self, force_status: str | None = None):
        """Run the chosen extractor, build the payload, return (True, payload)."""
        self._finished = True

        if force_status:
            payload = self._payload(None, force_status)
            return (True, payload)

        # Pick extractor
        method = self.extraction
        if method == "peak_distance":
            xy = saccade_extractors.peak_distance(self.samples, self.anchor_xy)
        elif method == "velocity_endpoint":
            params = self.extractor_params.get("velocity", {})
            xy = saccade_extractors.velocity_endpoint(
                self.samples,
                self.anchor_xy,
                onset_threshold_px_s=params.get("onset_threshold_px_s", 1500.0),
                settle_threshold_px_s=params.get("settle_threshold_px_s", 300.0),
                smoothing_window=params.get("smoothing_window", 5),
            )
        else:  # default idt_first_fixation
            params = self.extractor_params.get("idt", {})
            xy = saccade_extractors.idt_first_fixation(
                self.samples,
                self.anchor_xy,
                dispersion_px=params.get("dispersion_px", 60.0),
                min_duration_ms=params.get("min_duration_ms", 100.0),
                skip_anchor_radius_px=params.get(
                    "skip_anchor_radius_px", self.anchor_radius * 0.8
                ),
            )
            method = "idt_first_fixation"

        # Validate response
        status = "ok"
        if xy is None:
            status = "failed_no_fixation" if method == "idt_first_fixation" else "failed_no_endpoint"
        elif not self.samples:
            status = "failed_no_motion"
        else:
            dx = xy[0] - self.anchor_xy[0]
            dy = xy[1] - self.anchor_xy[1]
            if (dx * dx + dy * dy) ** 0.5 < self.min_response_distance_px:
                status = "failed_too_close_to_anchor"

        if status != "ok":
            xy = None

        self._last_status = status
        print(
            f"[SaccadeScreen] Finalizado: status={status} "
            f"response_xy={xy} samples={len(self.samples)} "
            f"attempt={self._attempts}/{self.max_attempts}"
        )
        return (True, self._payload(xy, status))

    def _payload(self, xy, status):
        return {
            "response_xy": xy,
            "samples": list(self.samples),
            "extraction": self.extraction,
            "status": status,
            "attempts": self._attempts,
            "max_attempts": self.max_attempts,
            "capture_duration_ms": int(self.capture_duration_s * 1000),
            "anchor_xy": self.anchor_xy,
        }

    # ---- retry policy support ------------------------------------------------

    def should_rerun(self) -> bool:
        """Caller checks this after a failed extraction. Returns True if the
        retry budget allows another attempt. Silent retry — no UI message."""
        if self._last_status == "ok":
            return False
        return self._attempts < self.max_attempts
