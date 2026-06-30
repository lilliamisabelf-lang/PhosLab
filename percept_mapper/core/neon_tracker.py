"""
Neon Tracker - Consume calibrated gaze from a Pupil Labs **Neon** headset over
its Companion-phone realtime API. Drop-in replacement for EyeTracker /
MouseTracker / PupilTracker.

Unlike Pupil *Core* (core/pupil_tracker.py), the Neon does NOT use Pupil Capture
or the ZMQ Network API, and it does NOT do surface tracking. The glasses plug
into a Companion phone (USB-C) running the Neon Companion app; that phone is the
API host, reached over the network by pupil_labs.realtime_api:

    Neon glasses --USB-C--> Companion phone (app) --WiFi--> this PC (realtime_api)

The realtime API gives a time-matched (scene frame, gaze) pair, with gaze in
**scene-camera pixels** — NOT screen pixels. PhosLab's fixation check needs gaze
in screen pixels, so this tracker maps scene -> screen itself: it detects the 4
AprilTags PhosLab already renders at the screen corners every frame (see
scripts/apriltag_overlay.py) in the Neon scene image, computes a homography from
the detected scene-tag centers to the known on-screen tag centers, and warps each
gaze sample through it. That mirrors what Pupil Capture's Surface Tracker does for
the Core path, but self-contained — no extra Pupil software.

    scene frame + gaze(scene px)
        |  cv2.aruco detect tags 0..3  ->  scene centers
        |  cv2.findHomography(scene -> screen)         [screen centers from overlay geometry]
    gaze(scene px) --warp--> gaze(screen px) --One-Euro--> last_smooth_gaze

Requires the Neon Companion app reachable on the network and the AprilTag overlay
enabled (apriltag_overlay.enabled: true), so the tags are actually on screen.

Install the realtime API with:  uv add pupil-labs-realtime-api

Related files:
- core/pupil_tracker.py            Pupil Core path (ZMQ surfaces) — interface mirrored here, _OneEuro reused
- scripts/apriltag_overlay.py      Renders tags 0..3 at screen corners; source of truth for screen geometry
- C:/Users/admin/neurolight2/neurolight2/glasses/neon_camera.py  Working Neon realtime_api code this is ported from
"""

import os
import threading
import time
import numpy as np

import cv2

# Reuse the exact smoothing filter the Core path uses — same semantics, no dup.
from core.pupil_tracker import _OneEuro

# Diagnostic snapshot every N seconds. Off by default — set PHOSLAB_NEON_DEBUG=1
# to enable; useful when AprilTag detection or homography is suspect.
DIAG_PERIOD_S = 2.0
_DIAG_ENABLED = os.environ.get("PHOSLAB_NEON_DEBUG", "").strip() not in ("", "0", "false", "False")

# tag_files[i] is tag36h11 ID i (see scripts/extract_apriltags.py), drawn at
# corner i by AprilTagOverlay._corner_positions: 0=TL, 1=TR, 2=BL, 3=BR.
_DEFAULT_TAG_SIZE_PX = 300
_DEFAULT_MARGIN_PX = 30


class NeonTracker:
    """
    Streams matched (scene frame, gaze) from a Neon over realtime_api, warps gaze
    into screen pixels via the on-screen AprilTags, and exposes the same interface
    as EyeTracker / MouseTracker / PupilTracker.
    """

    def __init__(
        self,
        address: str = "",
        port: int = 8080,
        discover_timeout_s: float = 10.0,
        min_confidence: float = 0.7,  # reserved; Neon has no per-sample confidence
        one_euro: dict | None = None,
        max_sample_age_s: float = 0.25,
        apriltag_overlay=None,
        homography_min_tags: int = 4,
    ):
        print("[NeonTracker] Inicializando...")
        self.address = address or ""
        self.port = int(port)
        self.discover_timeout_s = float(discover_timeout_s)
        self.min_confidence = float(min_confidence)
        self.max_sample_age_s = float(max_sample_age_s)
        self.homography_min_tags = int(homography_min_tags)

        # Screen-corner tag geometry. Derived from the live AprilTagOverlay so it
        # always matches what is actually drawn (apriltag_overlay.py). Fall back to
        # the documented defaults if no overlay was passed.
        if apriltag_overlay is not None:
            self._tag_size_px = int(getattr(apriltag_overlay, "tag_size_px", _DEFAULT_TAG_SIZE_PX))
            self._margin_px = int(getattr(apriltag_overlay, "margin_px", _DEFAULT_MARGIN_PX))
        else:
            self._tag_size_px = _DEFAULT_TAG_SIZE_PX
            self._margin_px = _DEFAULT_MARGIN_PX

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
            f"[NeonTracker] One-Euro: min_cutoff={self._filter.min_cutoff} "
            f"beta={self._filter.beta} d_cutoff={self._filter.d_cutoff}"
        )

        # AprilTag (tag36h11) detector for finding the screen corners in the scene.
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self._aruco_detector = cv2.aruco.ArucoDetector(
            self._aruco_dict, cv2.aruco.DetectorParameters()
        )

        self._screen_size = None
        self._last_H = None  # last good homography; reused when a frame loses tags
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._device = None
        self._thread = None

        try:
            self._connect()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            print("[NeonTracker] ✓ Inicializado correctamente")
        except Exception:
            self._close_device()
            raise

    def _connect(self):
        # Lazy import so importing this module (and main.py) never needs the
        # realtime-api stack until a Neon is actually used. Fail loud with the
        # install hint, never degrade to a mock.
        try:
            from pupil_labs.realtime_api.simple import Device, discover_one_device
        except ImportError as exc:
            raise RuntimeError(
                "pupil_labs.realtime_api not installed — run "
                "`uv add pupil-labs-realtime-api`."
            ) from exc

        # Connect by address if given (reliable when the network blocks mDNS),
        # else auto-discover. Fail loud BEFORE the session if nothing answers.
        if self.address:
            print(f"[NeonTracker] Conectando a {self.address}:{self.port}...")
            self._device = Device(address=self.address, port=self.port)
        else:
            print(f"[NeonTracker] Descubriendo Neon (mDNS, {self.discover_timeout_s:.0f}s)...")
            self._device = discover_one_device(
                max_search_duration_seconds=self.discover_timeout_s
            )
            if self._device is None:
                raise RuntimeError(
                    f"No se encontró Neon por descubrimiento en {self.discover_timeout_s:.0f}s. "
                    "¿Está la Companion app abierta y en esta red? Si la red bloquea mDNS, "
                    "fija neon.address y neon.port desde la pantalla Streaming de la app."
                )

        # Block on the first matched (frame, gaze) so we fail loud here, not mid
        # loop, and learn the scene-frame size before the capture thread starts.
        first = self._device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=5.0)
        if first is None:
            raise RuntimeError(
                "Neon conectado pero no envió escena+gaze en 5s — ¿cámara de escena "
                "activa y gafas puestas?"
            )

    def _loop(self):
        diag_t0 = time.time()
        diag_frames = 0
        diag_with_gaze = 0
        diag_with_homography = 0
        diag_no_homography = 0
        diag_last_tags = 0
        while not self._stop.is_set():
            now = time.time()
            if _DIAG_ENABLED and now - diag_t0 >= DIAG_PERIOD_S:
                with self._lock:
                    smooth = self.last_smooth_gaze
                    screen = self._screen_size
                print(
                    f"[NeonTracker][diag] frames={diag_frames} con_gaze={diag_with_gaze} "
                    f"con_H={diag_with_homography} sin_H={diag_no_homography} "
                    f"tags={diag_last_tags} smooth={smooth} screen={screen}"
                )
                diag_t0 = now
                diag_frames = 0
                diag_with_gaze = 0
                diag_with_homography = 0
                diag_no_homography = 0

            try:
                matched = self._device.receive_matched_scene_video_frame_and_gaze(
                    timeout_seconds=0.5
                )
            except Exception:
                continue
            if matched is None:
                continue
            diag_frames += 1

            g = matched.gaze
            # Drop gaze when the headset is not worn (Neon's wear detection) — the
            # analogue of the Core path's confidence filter.
            if g is None or not getattr(g, "worn", True):
                continue
            diag_with_gaze += 1

            # We need the screen size (from is_looking_at_point) before mapping.
            with self._lock:
                screen = self._screen_size
            if screen is None:
                continue

            bgr = matched.frame.bgr_pixels
            H, n_tags = self._compute_homography(bgr, screen)
            diag_last_tags = n_tags
            if H is None:
                H = self._last_H  # reuse last good homography if this frame lost tags
            else:
                self._last_H = H
            if H is None:
                diag_no_homography += 1
                continue
            diag_with_homography += 1

            gx, gy = self._warp_point(H, float(g.x), float(g.y))
            fx, fy = self._filter.step(gx, gy)
            with self._lock:
                self.last_raw_gaze = (gx, gy)
                self.last_smooth_gaze = (float(fx), float(fy))
                self.last_gaze_time = time.monotonic()

    def _screen_corner_centers(self, screen_size):
        """Centers of the 4 corner tags in SCREEN pixels, indexed by tag id 0..3.
        Mirrors AprilTagOverlay._corner_positions (TL, TR, BL, BR) + half a tag to
        get the center. Keep in sync with scripts/apriltag_overlay.py."""
        w, h = int(screen_size[0]), int(screen_size[1])
        m = self._margin_px
        s = self._tag_size_px
        half = s / 2.0
        tops = [
            (m, m),                  # id 0 -> TL
            (w - m - s, m),          # id 1 -> TR
            (m, h - m - s),          # id 2 -> BL
            (w - m - s, h - m - s),  # id 3 -> BR
        ]
        return [(x + half, y + half) for (x, y) in tops]

    def _compute_homography(self, bgr_frame, screen_size):
        """Detect tags 0..3 in the scene frame and map their centers to the known
        screen-tag centers. Returns (H, n_matched_tags); H is None if fewer than
        homography_min_tags matched."""
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._aruco_detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return None, 0

        screen_centers = self._screen_corner_centers(screen_size)
        scene_pts = []
        screen_pts = []
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            mid = int(marker_id)
            if mid < 0 or mid > 3:
                continue  # not one of our 4 corner tags
            # marker_corners is (1, 4, 2); the tag center is the mean of its corners
            center = marker_corners.reshape(-1, 2).mean(axis=0)
            scene_pts.append((float(center[0]), float(center[1])))
            screen_pts.append(screen_centers[mid])

        if len(scene_pts) < self.homography_min_tags:
            return None, len(scene_pts)

        scene_arr = np.asarray(scene_pts, dtype=np.float32)
        screen_arr = np.asarray(screen_pts, dtype=np.float32)
        H, _ = cv2.findHomography(scene_arr, screen_arr)
        if H is None:
            return None, len(scene_pts)
        return H, len(scene_pts)

    @staticmethod
    def _warp_point(H, x, y):
        """Apply homography H to a single (x, y) point -> (X, Y) screen px."""
        vec = np.array([x, y, 1.0], dtype=np.float64)
        out = H @ vec
        w = out[2] if out[2] != 0 else 1e-9
        return float(out[0] / w), float(out[1] / w)

    def get_frame(self):
        """The Neon owns its scene camera; PhosLab does not need raw frames here.
        Returns a sentinel so callers' `if eye_tracker else None` checks pass."""
        return "neon"

    def is_looking_at_point(
        self, frame, target_point, screen_size, tolerance_radius=100
    ):
        # Same body as PupilTracker.is_looking_at_point — identical contract.
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

    def _close_device(self):
        try:
            if self._device is not None:
                self._device.close()
        except Exception:
            pass
        finally:
            self._device = None

    def release(self):
        print("[NeonTracker] Liberando recursos...")
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._close_device()
        print("[NeonTracker] ✓ Recursos liberados")
