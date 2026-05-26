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
        anchor_color=(180, 180, 180),  # dim white — "response phase, still a saccade target"
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
        if self.allow_mouse_fallback:
            self.title_text = "Mueve el cursor al fosfeno y vuelve al centro"
        else:
            self.title_text = "Mira el fosfeno y vuelve al centro"

        # Title fade: full opacity for first `title_full_alpha_ms` after cue,
        # then linear fade to 0 by `title_fade_done_ms`. After the first 3
        # captures of this SaccadeScreen instance, suppress the title entirely
        # (participant has learned the task).
        self.title_full_alpha_ms = 200
        self.title_fade_done_ms = 500
        self.title_suppress_after_trial = 3

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
        # When the user's input mode is mouse/Wacom, allow_mouse_fallback is
        # True and we must read pygame.mouse.get_pos() LIVE. We cannot trust
        # MouseTracker.last_*_gaze here: that attribute is only refreshed
        # inside MouseTracker.is_looking_at_point(), which is called during
        # prestim/stim/poststim — not during the saccade capture window. If
        # we deferred to it, every trial after the first would render a
        # frozen trace stuck at whatever cursor position prestim happened to
        # cache.
        if self.allow_mouse_fallback:
            return pygame.mouse.get_pos()
        et = self.eye_tracker
        if et is not None:
            p = getattr(et, "last_smooth_gaze", None)
            if p is not None:
                return p
            p = getattr(et, "last_raw_gaze", None)
            if p is not None:
                return p
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

        # Live gaze trace + current-position marker.
        if self.show_gaze_trace and self.samples:
            # Trace: bright cyan polyline over the last N samples, regardless
            # of timestamp — so even a small mouse jitter still draws a line.
            if len(self.samples) >= 2:
                recent = self.samples[-90:]  # ~1.5 s at 60 Hz
                # Drop near-duplicate consecutive points to avoid invisible
                # zero-length segments when the pointer is stationary.
                points = []
                last_pt = None
                for s in recent:
                    p = (int(s["x"]), int(s["y"]))
                    if last_pt is None or p != last_pt:
                        points.append(p)
                        last_pt = p
                if len(points) >= 2:
                    pygame.draw.lines(screen, (0, 220, 255), False, points, 4)

            # Current position: small filled yellow dot
            last = self.samples[-1]
            pygame.draw.circle(
                screen, (255, 255, 0), (int(last["x"]), int(last["y"])), 5
            )

        # Title fade: opaque early, faded by 500ms, suppressed after a few
        # trials. A visible timer or persistent instruction trains the
        # participant to attend the title instead of the percept memory.
        if self._attempts <= self.title_suppress_after_trial:
            elapsed_ms = elapsed * 1000.0
            if elapsed_ms <= self.title_full_alpha_ms:
                alpha = 255
            elif elapsed_ms >= self.title_fade_done_ms:
                alpha = 0
            else:
                span = self.title_fade_done_ms - self.title_full_alpha_ms
                alpha = int(255 * (1.0 - (elapsed_ms - self.title_full_alpha_ms) / span))
            if alpha > 0:
                title = self.font.render(self.title_text, True, (255, 255, 255))
                title.set_alpha(alpha)
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
