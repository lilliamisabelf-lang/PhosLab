"""
Pupil Tracker - Consume calibrated gaze from a running Pupil Capture instance
over its ZMQ Pupil Remote API. Drop-in replacement for EyeTracker / MouseTracker.

Requires Pupil Capture (https://github.com/pupil-labs/pupil) running with the
Surface Tracker plugin enabled and a named screen surface.
"""

import threading
import time
import numpy as np

import zmq
import msgpack

DIAG_PERIOD_S = 2.0  # log diagnostic snapshot every N seconds


class _OneEuro:
    """One-Euro Filter (Casiez et al. 2012). Fixed dt = 1/fps for predictable
    smoothing strength regardless of sample arrival jitter."""
    def __init__(self, fps=60, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.dt = 1.0 / float(fps)
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


class PupilTracker:
    """
    Subscribes to surface-normalized gaze samples published by Pupil Capture
    and exposes the same interface as EyeTracker / MouseTracker.
    """

    def __init__(
        self,
        address: str = "127.0.0.1",
        port: int = 50020,
        surface_name: str = "phoslab_screen",
        min_confidence: float = 0.6,
        one_euro: dict | None = None,
        max_sample_age_s: float = 0.25,
    ):
        print("[PupilTracker] Inicializando...")
        self.address = address
        self.port = int(port)
        self.surface_name = surface_name
        self.min_confidence = float(min_confidence)
        self.max_sample_age_s = float(max_sample_age_s)

        self.last_raw_gaze = None
        self.last_smooth_gaze = None
        self.last_gaze_time = None

        oe_cfg = dict(one_euro or {})
        self._filter = _OneEuro(
            fps=oe_cfg.get("fps", 60),
            min_cutoff=oe_cfg.get("min_cutoff", 1.0),
            beta=oe_cfg.get("beta", 0.007),
            d_cutoff=oe_cfg.get("d_cutoff", 1.0),
        )
        print(
            f"[PupilTracker] One-Euro: min_cutoff={self._filter.min_cutoff} "
            f"beta={self._filter.beta} d_cutoff={self._filter.d_cutoff}"
        )

        self._screen_size = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._req = None
        self._sub = None
        self._thread = None

        self._ctx = zmq.Context.instance()
        try:
            self._req = self._ctx.socket(zmq.REQ)
            self._req.setsockopt(zmq.RCVTIMEO, 2000)
            self._req.setsockopt(zmq.SNDTIMEO, 2000)
            self._req.connect(f"tcp://{self.address}:{self.port}")
            self._req.send_string("SUB_PORT")
            sub_port = self._req.recv_string()
            print(f"[PupilTracker] SUB_PORT={sub_port}")

            self._sub = self._ctx.socket(zmq.SUB)
            self._sub.connect(f"tcp://{self.address}:{sub_port}")
            topic = f"surfaces.{self.surface_name}"
            self._sub.setsockopt_string(zmq.SUBSCRIBE, topic)
            print(f"[PupilTracker] Suscrito a tópico: {topic}")

            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            print("[PupilTracker] ✓ Inicializado correctamente")
        except Exception:
            self._close_sockets()
            raise

    def _loop(self):
        poller = zmq.Poller()
        poller.register(self._sub, zmq.POLLIN)
        # diagnostic counters (reset each DIAG_PERIOD_S)
        diag_t0 = time.time()
        diag_msgs = 0
        diag_with_gaze = 0
        diag_screen_unknown = 0
        diag_last_norm = None
        diag_last_conf = None
        while not self._stop.is_set():
            socks = dict(poller.poll(timeout=100))
            now = time.time()
            if now - diag_t0 >= DIAG_PERIOD_S:
                with self._lock:
                    smooth = self.last_smooth_gaze
                    screen = self._screen_size
                print(
                    f"[PupilTracker][diag] msgs={diag_msgs} con_gaze={diag_with_gaze} "
                    f"sin_screen={diag_screen_unknown} norm={diag_last_norm} "
                    f"conf={diag_last_conf} smooth={smooth} screen={screen}"
                )
                diag_t0 = now
                diag_msgs = 0
                diag_with_gaze = 0
                diag_screen_unknown = 0

            if self._sub not in socks:
                continue
            try:
                topic = self._sub.recv_string(zmq.NOBLOCK)
                payload = self._sub.recv(zmq.NOBLOCK)
            except zmq.Again:
                continue
            try:
                msg = msgpack.unpackb(payload, raw=False)
            except Exception:
                continue
            diag_msgs += 1

            gaze_norm = self._extract_gaze(msg, diag_record=True)
            if gaze_norm is None:
                continue
            diag_with_gaze += 1
            diag_last_norm = (round(gaze_norm[0], 3), round(gaze_norm[1], 3))
            diag_last_conf = self._last_seen_conf

            with self._lock:
                screen = self._screen_size
            if screen is None:
                diag_screen_unknown += 1
                continue

            nx, ny = gaze_norm
            gx = float(nx) * screen[0]
            gy = (1.0 - float(ny)) * screen[1]

            fx, fy = self._filter.step(gx, gy)
            with self._lock:
                self.last_raw_gaze = (gx, gy)
                self.last_smooth_gaze = (float(fx), float(fy))
                self.last_gaze_time = time.monotonic()

    _last_seen_conf = None

    def _extract_gaze(self, msg, diag_record=False):
        """Return (nx, ny) in surface-normalized coords passing the confidence
        threshold, or None. Tolerates the two common payload shapes Pupil
        Capture emits on a surface topic."""
        if not isinstance(msg, dict):
            return None

        gaze_list = msg.get("gaze_on_surfaces") or msg.get("gaze_on_srf") or []
        for g in gaze_list:
            try:
                conf = float(g.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            if diag_record:
                self._last_seen_conf = round(conf, 3)
            if conf < self.min_confidence:
                continue
            pos = g.get("norm_pos")
            if pos and len(pos) >= 2:
                return (pos[0], pos[1])

        fix_list = msg.get("fixations_on_surfaces") or msg.get("fixations_on_srf") or []
        for f in fix_list:
            pos = f.get("norm_pos")
            if pos and len(pos) >= 2:
                return (pos[0], pos[1])

        return None

    def get_frame(self):
        """Pupil Capture owns the cameras; PhosLab does not need raw frames.
        Returns a sentinel so callers' `if eye_tracker else None` checks pass."""
        return "pupil"

    def is_looking_at_point(
        self, frame, target_point, screen_size, tolerance_radius=100
    ):
        with self._lock:
            self._screen_size = (int(screen_size[0]), int(screen_size[1]))
            smooth = self.last_smooth_gaze
            last_gaze_time = self.last_gaze_time

        if smooth is None or last_gaze_time is None:
            return False

        if time.monotonic() - last_gaze_time > self.max_sample_age_s:
            return False

        dx = smooth[0] - target_point[0]
        dy = smooth[1] - target_point[1]
        return (dx * dx + dy * dy) ** 0.5 <= tolerance_radius

    def _close_sockets(self):
        try:
            if self._sub is not None:
                self._sub.close(0)
        except Exception:
            pass
        finally:
            self._sub = None

        try:
            if self._req is not None:
                self._req.close(0)
        except Exception:
            pass
        finally:
            self._req = None

    def release(self):
        print("[PupilTracker] Liberando recursos...")
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._close_sockets()
        print("[PupilTracker] ✓ Recursos liberados")
