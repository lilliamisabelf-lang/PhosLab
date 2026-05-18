"""
Gaze Trace Overlay - Real-time eye gaze visualization on experiment screens

Draws:
  - Thin white line: raw gaze trace (last N ms, configurable)
  - Blue line: filtered/smoothed gaze estimate (filter selectable via config)

Supported filters: none, ema, dema, one_euro, kalman, sma
"""

import time
import pygame
import numpy as np
from collections import deque


# ── Filter implementations ────────────────────────────────────────────

class _EMA:
    """Exponential Moving Average."""
    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self._x = self._y = None

    def step(self, x, y):
        if self._x is None:
            self._x, self._y = x, y
        else:
            self._x = self.alpha * x + (1 - self.alpha) * self._x
            self._y = self.alpha * y + (1 - self.alpha) * self._y
        return self._x, self._y

    def reset(self):
        self._x = self._y = None


class _DEMA:
    """Double EMA — lag-corrected."""
    def __init__(self, alpha=0.15):
        self.e1 = _EMA(alpha)
        self.e2 = _EMA(alpha)

    def step(self, x, y):
        s1x, s1y = self.e1.step(x, y)
        s2x, s2y = self.e2.step(s1x, s1y)
        return 2 * s1x - s2x, 2 * s1y - s2y

    def reset(self):
        self.e1.reset()
        self.e2.reset()


class _SMA:
    """Simple Moving Average."""
    def __init__(self, window=10):
        self.window = max(1, int(window))
        self._buf_x = deque(maxlen=self.window)
        self._buf_y = deque(maxlen=self.window)

    def step(self, x, y):
        self._buf_x.append(x)
        self._buf_y.append(y)
        return sum(self._buf_x) / len(self._buf_x), sum(self._buf_y) / len(self._buf_y)

    def reset(self):
        self._buf_x.clear()
        self._buf_y.clear()


class _OneEuro:
    """One-Euro Filter (Casiez et al. 2012)."""
    def __init__(self, fps=60, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.dt = 1.0 / fps
        self._prev_x = self._prev_y = None
        self._dx = self._dy = 0.0

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def step(self, x, y):
        if self._prev_x is None:
            self._prev_x, self._prev_y = x, y
            return x, y

        dx = (x - self._prev_x) / self.dt
        dy = (y - self._prev_y) / self.dt
        a_d = self._alpha(self.d_cutoff, self.dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        self._dy = a_d * dy + (1 - a_d) * self._dy

        cx = self.min_cutoff + self.beta * abs(self._dx)
        cy = self.min_cutoff + self.beta * abs(self._dy)
        ax = self._alpha(cx, self.dt)
        ay = self._alpha(cy, self.dt)

        self._prev_x = ax * x + (1 - ax) * self._prev_x
        self._prev_y = ay * y + (1 - ay) * self._prev_y
        return self._prev_x, self._prev_y

    def reset(self):
        self._prev_x = self._prev_y = None
        self._dx = self._dy = 0.0


class _Kalman:
    """1D Kalman filter with constant velocity model, applied independently to x and y."""
    def __init__(self, process_noise=1.0, measurement_noise=900.0):
        self.q = process_noise
        self.r = measurement_noise
        self._init = False

    def _reset_state(self):
        self.x_state = np.zeros(2)
        self.y_state = np.zeros(2)
        self.Px = np.eye(2) * 1000
        self.Py = np.eye(2) * 1000
        self._init = False

    def _step_1d(self, state, P, z):
        dt = 1.0
        F = np.array([[1, dt], [0, 1]])
        H = np.array([[1, 0]])
        Q = self.q * np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])
        R = np.array([[self.r]])

        state = F @ state
        P = F @ P @ F.T + Q
        y = z - H @ state
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        state = state + (K @ np.array([y])).flatten()
        P = (np.eye(2) - K @ H) @ P
        return state, P

    def step(self, x, y):
        if not self._init:
            self.x_state = np.array([x, 0.0])
            self.y_state = np.array([y, 0.0])
            self.Px = np.eye(2) * 1000
            self.Py = np.eye(2) * 1000
            self._init = True
            return x, y
        self.x_state, self.Px = self._step_1d(self.x_state, self.Px, x)
        self.y_state, self.Py = self._step_1d(self.y_state, self.Py, y)
        return float(self.x_state[0]), float(self.y_state[0])

    def reset(self):
        self._reset_state()


class _NoFilter:
    """Pass-through (no smoothing)."""
    def step(self, x, y):
        return x, y

    def reset(self):
        pass


def create_filter(filter_name, params):
    """
    Create a filter instance from a name and its parameter dict.

    Args:
        filter_name: 'none', 'ema', 'dema', 'sma', 'one_euro', 'kalman'
        params: dict with filter-specific parameters (from config)
    """
    if filter_name == "none":
        return _NoFilter()
    elif filter_name == "ema":
        return _EMA(alpha=params.get("alpha", 0.15))
    elif filter_name == "dema":
        return _DEMA(alpha=params.get("alpha", 0.15))
    elif filter_name == "sma":
        return _SMA(window=params.get("window", 10))
    elif filter_name == "one_euro":
        return _OneEuro(
            min_cutoff=params.get("min_cutoff", 1.0),
            beta=params.get("beta", 0.007),
            d_cutoff=params.get("d_cutoff", 1.0),
        )
    elif filter_name == "kalman":
        return _Kalman(
            process_noise=params.get("process_noise", 1.0),
            measurement_noise=params.get("measurement_noise", 900.0),
        )
    else:
        print(f"[WARN] Unknown filter '{filter_name}', using none")
        return _NoFilter()


# ── GazeTrace ─────────────────────────────────────────────────────────

class GazeTrace:
    """
    Draws real-time gaze trace overlay on any pygame surface.

    Args:
        trace_duration_ms: Duration of the raw trace tail in milliseconds
        filter_name: Filter type ('none', 'ema', 'dema', 'sma', 'one_euro', 'kalman')
        filter_params: Dict of parameters for the chosen filter
        raw_color: RGB color for the raw trace line
        smooth_color: RGB color for the filtered trace line
        raw_thickness: Line thickness for raw trace
        smooth_thickness: Line thickness for filtered trace
    """

    def __init__(
        self,
        trace_duration_ms=300,
        filter_name="ema",
        filter_params=None,
        raw_color=(255, 255, 255),
        smooth_color=(80, 140, 255),
        raw_thickness=1,
        smooth_thickness=2,
    ):
        self.trace_duration_ms = trace_duration_ms
        self.raw_color = raw_color
        self.smooth_color = smooth_color
        self.raw_thickness = raw_thickness
        self.smooth_thickness = smooth_thickness

        # Create the selected filter
        self.filter_name = filter_name
        self._filter = create_filter(filter_name, filter_params or {})

        # Timestamped points: deque of (timestamp_s, (x, y))
        self.raw_points = deque(maxlen=5000)
        self.smooth_points = deque(maxlen=5000)

    def update(self, raw_gaze):
        """
        Feed a new raw gaze point. Call once per frame.

        Args:
            raw_gaze: (x, y) tuple from eye tracker, or None if no detection
        """
        now = time.time()

        if raw_gaze is None:
            return

        x, y = float(raw_gaze[0]), float(raw_gaze[1])

        # Store raw point
        self.raw_points.append((now, (x, y)))

        # Apply filter
        sx, sy = self._filter.step(x, y)
        self.smooth_points.append((now, (sx, sy)))

    def draw(self, screen):
        """
        Draw the gaze traces on the given pygame surface.
        Call after screen content but before pygame.display.flip().
        """
        now = time.time()
        cutoff = now - (self.trace_duration_ms / 1000.0)

        # --- Raw trace (white, thin) ---
        raw_visible = [(t, p) for t, p in self.raw_points if t >= cutoff]
        if len(raw_visible) >= 2:
            int_points = [(int(p[0]), int(p[1])) for _, p in raw_visible]
            pygame.draw.lines(screen, self.raw_color, False, int_points, self.raw_thickness)

        # --- Filtered trace (blue, thicker) ---
        if self.filter_name != "none":
            smooth_visible = [(t, p) for t, p in self.smooth_points if t >= cutoff]
            if len(smooth_visible) >= 2:
                int_points = [(int(p[0]), int(p[1])) for _, p in smooth_visible]
                pygame.draw.lines(screen, self.smooth_color, False, int_points, self.smooth_thickness)

    def clear(self):
        """Clear all stored trace data (e.g. between experiment phases)."""
        self.raw_points.clear()
        self.smooth_points.clear()
        self._filter.reset()

