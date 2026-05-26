"""Saccade-response extractors.

Pure functions that take a list of timestamped gaze samples + anchor position
and return a single (x, y) point representing the participant's perceived
phosphene location. Three methods, picked at runtime via config.

Samples format: list of dicts with keys `t` (seconds, monotonic), `x`, `y`.
Returns: (x, y) tuple of floats, or None if extraction fails (no fixation
found / not enough motion / etc.).
"""

from __future__ import annotations

import math
from typing import Iterable


def peak_distance(samples, anchor):
    """Pick the sample whose Euclidean distance from the anchor is maximal.

    Cheap baseline. Vulnerable to overshoot, tracker spikes, and corrective
    saccades — saccades typically undershoot by 5-10% then auto-correct, so
    "peak" tends to be the *farther* of (undershoot endpoint, corrective
    target). For research-grade localization prefer `idt_first_fixation`.
    """
    if not samples:
        return None
    ax, ay = anchor
    best = None
    best_d2 = -1.0
    for s in samples:
        dx = s["x"] - ax
        dy = s["y"] - ay
        d2 = dx * dx + dy * dy
        if d2 > best_d2:
            best_d2 = d2
            best = (float(s["x"]), float(s["y"]))
    return best


def idt_first_fixation(
    samples,
    anchor,
    dispersion_px: float = 60.0,
    min_duration_ms: float = 100.0,
    skip_anchor_radius_px: float = 40.0,
):
    """Identification by Dispersion-Threshold (Salvucci & Goldberg 2000).

    Scans the sample stream for the first contiguous run where
        max(x) - min(x) < dispersion_px  AND  max(y) - min(y) < dispersion_px
    that lasts at least `min_duration_ms`. The centroid of that window is
    returned as the response.

    `skip_anchor_radius_px` rejects fixations that lie inside the central
    anchor circle — i.e., the initial fixation BEFORE the saccade. We want
    the first fixation that's clearly off-anchor.

    Returns None if no qualifying fixation exists.
    """
    if not samples or len(samples) < 2:
        return None

    ax, ay = anchor
    min_dur_s = float(min_duration_ms) / 1000.0
    skip_r2 = float(skip_anchor_radius_px) ** 2

    # Sliding-window scan
    n = len(samples)
    i = 0
    while i < n:
        # Skip samples sitting on the anchor (still in initial fixation).
        sx = samples[i]["x"] - ax
        sy = samples[i]["y"] - ay
        if sx * sx + sy * sy < skip_r2:
            i += 1
            continue

        # Try to extend a fixation window starting at i.
        xmin = xmax = samples[i]["x"]
        ymin = ymax = samples[i]["y"]
        j = i
        while j + 1 < n:
            nx = samples[j + 1]["x"]
            ny = samples[j + 1]["y"]
            new_xmin = min(xmin, nx)
            new_xmax = max(xmax, nx)
            new_ymin = min(ymin, ny)
            new_ymax = max(ymax, ny)
            if (new_xmax - new_xmin) >= dispersion_px or (
                new_ymax - new_ymin
            ) >= dispersion_px:
                break
            xmin, xmax, ymin, ymax = new_xmin, new_xmax, new_ymin, new_ymax
            j += 1

        # Window [i..j] is the longest dispersion-bounded run starting at i.
        duration = samples[j]["t"] - samples[i]["t"]
        if duration >= min_dur_s and (j - i + 1) >= 2:
            cx = sum(s["x"] for s in samples[i : j + 1]) / (j - i + 1)
            cy = sum(s["y"] for s in samples[i : j + 1]) / (j - i + 1)
            return (float(cx), float(cy))

        # Advance past this window's start (don't get stuck on micro runs).
        i = max(j + 1, i + 1)

    return None


def velocity_endpoint(
    samples,
    anchor,
    onset_threshold_px_s: float = 1500.0,
    settle_threshold_px_s: float = 300.0,
    smoothing_window: int = 5,
    min_saccade_duration_ms: float = 20.0,
):
    """Velocity-based saccade landing detector.

    Smooths per-sample velocity, finds the first saccade onset
    (velocity > onset_threshold_px_s), then the first sample after onset
    where velocity drops below settle_threshold_px_s for a few samples.
    Returns that sample's (x, y) as the landing point.

    Thresholds are in px/s. 1500 px/s ≈ 25°/s at 60 px/°; saccades commonly
    peak at 400-600°/s for ~10° amplitudes, so this only needs to catch the
    onset, not the peak. Tune for your monitor.

    Returns None if no clear saccade is found.
    """
    if not samples or len(samples) < smoothing_window + 2:
        return None

    # Per-sample speeds
    speeds = [0.0]
    for k in range(1, len(samples)):
        dx = samples[k]["x"] - samples[k - 1]["x"]
        dy = samples[k]["y"] - samples[k - 1]["y"]
        dt = max(1e-4, samples[k]["t"] - samples[k - 1]["t"])
        speeds.append(math.hypot(dx, dy) / dt)

    # Boxcar smoothing of speed
    w = max(1, int(smoothing_window))
    half = w // 2
    smoothed = []
    for k in range(len(speeds)):
        lo = max(0, k - half)
        hi = min(len(speeds), k + half + 1)
        smoothed.append(sum(speeds[lo:hi]) / (hi - lo))

    # Find saccade onset
    onset = None
    for k, v in enumerate(smoothed):
        if v >= onset_threshold_px_s:
            onset = k
            break
    if onset is None:
        return None

    min_sacc_dur_s = float(min_saccade_duration_ms) / 1000.0

    # After onset, find first sustained drop below settle threshold.
    # Require the velocity to stay low for at least min_saccade_duration_ms
    # to avoid latching onto a brief velocity dip during the saccade itself.
    landing = None
    k = onset + 1
    while k < len(smoothed):
        if smoothed[k] < settle_threshold_px_s:
            # Look ahead to confirm sustained settle
            j = k
            while j + 1 < len(smoothed) and smoothed[j + 1] < settle_threshold_px_s:
                j += 1
                if samples[j]["t"] - samples[k]["t"] >= min_sacc_dur_s:
                    landing = k
                    break
            if landing is not None:
                break
            k = j + 1
        else:
            k += 1

    if landing is None:
        return None

    return (float(samples[landing]["x"]), float(samples[landing]["y"]))


# ---- self-test --------------------------------------------------------------

def _make_synthetic_trial(anchor, target, dt=1 / 60, duration=1.5, sigma=2.0,
                          saccade_latency=0.2, saccade_duration=0.05,
                          fixation_jitter_sigma=3.0):
    """Generate a synthetic gaze trial: fixation at anchor, ballistic saccade
    to target, brief overshoot, corrective saccade back to target with noise.
    Returns a list of {t, x, y} dicts."""
    import random

    random.seed(0)
    samples = []
    n = int(duration / dt)
    t = 0.0
    ax, ay = anchor
    tx, ty = target
    # Slight undershoot then correction to mimic real saccades
    undershoot = (ax + 0.9 * (tx - ax), ay + 0.9 * (ty - ay))

    for i in range(n):
        if t < saccade_latency:
            # Initial fixation at anchor
            x = ax + random.gauss(0, fixation_jitter_sigma)
            y = ay + random.gauss(0, fixation_jitter_sigma)
        elif t < saccade_latency + saccade_duration:
            # Ballistic outbound saccade (linear interp)
            frac = (t - saccade_latency) / saccade_duration
            x = ax + frac * (undershoot[0] - ax) + random.gauss(0, sigma)
            y = ay + frac * (undershoot[1] - ay) + random.gauss(0, sigma)
        elif t < saccade_latency + saccade_duration + 0.06:
            # Correction
            frac = (t - saccade_latency - saccade_duration) / 0.06
            x = undershoot[0] + frac * (tx - undershoot[0]) + random.gauss(0, sigma)
            y = undershoot[1] + frac * (ty - undershoot[1]) + random.gauss(0, sigma)
        else:
            # Steady fixation at target with jitter
            x = tx + random.gauss(0, fixation_jitter_sigma)
            y = ty + random.gauss(0, fixation_jitter_sigma)
        samples.append({"t": t, "x": x, "y": y})
        t += dt
    return samples


def _selftest():
    anchor = (1280, 720)
    target = (1500, 600)

    samples = _make_synthetic_trial(anchor, target)

    print("=== peak_distance ===")
    print("  result:", peak_distance(samples, anchor))
    print("  truth :", target)

    print("=== idt_first_fixation ===")
    p = idt_first_fixation(samples, anchor)
    print("  result:", p)
    print("  truth :", target)
    if p is not None:
        err = math.hypot(p[0] - target[0], p[1] - target[1])
        print(f"  error: {err:.1f} px")

    print("=== velocity_endpoint ===")
    p = velocity_endpoint(samples, anchor)
    print("  result:", p)
    print("  truth :", target)

    # Edge cases
    print("=== edge cases ===")
    print("  empty samples ->", idt_first_fixation([], anchor))
    print("  one sample    ->", idt_first_fixation([{"t": 0, "x": 1280, "y": 720}], anchor))
    print("  no off-anchor ->", idt_first_fixation(
        [{"t": i * 1 / 60, "x": 1280 + i % 3, "y": 720 + i % 3} for i in range(60)],
        anchor,
    ))


if __name__ == "__main__":
    _selftest()
