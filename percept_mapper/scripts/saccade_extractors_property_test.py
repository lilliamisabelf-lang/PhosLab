"""Property-based tests for saccade_extractors.

The smoke tests in saccade_smoke_test.py drive a SaccadeScreen end-to-end on
specific synthetic trials. These pin the *pure-function* invariants of the
extractors directly against random sample streams — Hypothesis searches for
inputs that break them.

Run:
    uv run --project percept_mapper python percept_mapper/scripts/saccade_extractors_property_test.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypothesis import given, settings, strategies as st  # noqa: E402

from scripts.saccade_extractors import (  # noqa: E402
    peak_distance,
    idt_first_fixation,
    velocity_endpoint,
)


# --- sample-stream strategies ------------------------------------------------

screen_coord = st.floats(min_value=0.0, max_value=2560.0, allow_nan=False, allow_infinity=False)
anchor_strategy = st.tuples(screen_coord, screen_coord)


@st.composite
def gaze_stream(draw, min_size=2, max_size=120, dt=1 / 60):
    """A list of {t, x, y} samples with strictly increasing t."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    samples = []
    t = 0.0
    for _ in range(n):
        x = draw(screen_coord)
        y = draw(screen_coord)
        samples.append({"t": t, "x": x, "y": y})
        t += dt
    return samples


# --- peak_distance properties -----------------------------------------------

@given(gaze_stream(min_size=1), anchor_strategy)
@settings(max_examples=200, deadline=None)
def prop_peak_distance_returns_a_sample(samples, anchor):
    """The returned point must be one of the input samples (or be None for
    empty input)."""
    result = peak_distance(samples, anchor)
    if not samples:
        assert result is None
        return
    sample_points = {(float(s["x"]), float(s["y"])) for s in samples}
    assert result in sample_points, (
        f"peak_distance returned {result}, not in input samples"
    )


@given(gaze_stream(min_size=1), anchor_strategy)
@settings(max_examples=200, deadline=None)
def prop_peak_distance_is_maximal(samples, anchor):
    """No sample in the input is farther from the anchor than the returned
    one. (Allows ties.)"""
    result = peak_distance(samples, anchor)
    assert result is not None
    rx, ry = result
    ax, ay = anchor
    result_d2 = (rx - ax) ** 2 + (ry - ay) ** 2
    for s in samples:
        d2 = (s["x"] - ax) ** 2 + (s["y"] - ay) ** 2
        assert d2 <= result_d2 + 1e-9, (
            f"sample at ({s['x']},{s['y']}) is farther from anchor "
            f"({ax},{ay}) than the returned peak {result}"
        )


def prop_peak_distance_empty_returns_none():
    """Empty sample list — must return None, not crash."""
    assert peak_distance([], (0, 0)) is None


# --- idt_first_fixation properties ------------------------------------------

@given(anchor_strategy)
@settings(max_examples=20, deadline=None)
def prop_idt_empty_returns_none(anchor):
    assert idt_first_fixation([], anchor) is None
    assert idt_first_fixation([{"t": 0, "x": anchor[0], "y": anchor[1]}], anchor) is None


@given(gaze_stream(min_size=20, max_size=120), anchor_strategy)
@settings(max_examples=100, deadline=None)
def prop_idt_result_is_inside_sample_bbox(samples, anchor):
    """If IDT returns a centroid, it must lie inside the bounding box of the
    input samples — it's an average of a window of them."""
    result = idt_first_fixation(samples, anchor, dispersion_px=200.0, min_duration_ms=20.0)
    if result is None:
        return
    xs = [s["x"] for s in samples]
    ys = [s["y"] for s in samples]
    rx, ry = result
    assert min(xs) - 1e-6 <= rx <= max(xs) + 1e-6
    assert min(ys) - 1e-6 <= ry <= max(ys) + 1e-6


@given(anchor_strategy, st.integers(min_value=20, max_value=80))
@settings(max_examples=80, deadline=None)
def prop_idt_rejects_constant_anchor_input(anchor, n):
    """All samples sitting on the anchor — no off-anchor fixation exists,
    so IDT must return None."""
    samples = [{"t": i * (1 / 60), "x": anchor[0], "y": anchor[1]} for i in range(n)]
    result = idt_first_fixation(samples, anchor, skip_anchor_radius_px=10.0)
    assert result is None


@given(
    anchor_strategy,
    st.tuples(
        st.floats(min_value=200, max_value=1800, allow_nan=False, allow_infinity=False),
        st.floats(min_value=200, max_value=1000, allow_nan=False, allow_infinity=False),
    ),
    st.integers(min_value=30, max_value=80),
)
@settings(max_examples=80, deadline=None)
def prop_idt_finds_clear_off_anchor_fixation(anchor, off_anchor_offset, fix_n):
    """Construct: K samples on the anchor, then `fix_n` samples clustered at
    anchor + offset (>> skip_anchor_radius). IDT must find the off-anchor
    cluster — not return None and not return a point near the anchor."""
    ox, oy = off_anchor_offset
    target = (anchor[0] + ox, anchor[1] + oy)
    # Skip the case where target lands very near the anchor (degenerate)
    if math.hypot(ox, oy) < 80.0:
        return

    samples = []
    t = 0.0
    for _ in range(20):
        samples.append({"t": t, "x": anchor[0], "y": anchor[1]})
        t += 1 / 60
    for _ in range(fix_n):
        samples.append({"t": t, "x": target[0], "y": target[1]})
        t += 1 / 60

    result = idt_first_fixation(
        samples, anchor,
        dispersion_px=30.0,
        min_duration_ms=80.0,
        skip_anchor_radius_px=50.0,
    )
    assert result is not None, "should find the off-anchor cluster"
    # Centroid of the fix window should be near `target`, not the anchor.
    rx, ry = result
    err_target = math.hypot(rx - target[0], ry - target[1])
    err_anchor = math.hypot(rx - anchor[0], ry - anchor[1])
    assert err_target < err_anchor, (
        f"IDT centroid {result} closer to anchor than to target {target}"
    )


# --- velocity_endpoint properties -------------------------------------------

@given(gaze_stream(min_size=10, max_size=20), anchor_strategy)
@settings(max_examples=80, deadline=None)
def prop_velocity_endpoint_returns_sample_or_none(samples, anchor):
    """Whatever velocity_endpoint returns must be one of the input samples
    (or None)."""
    result = velocity_endpoint(samples, anchor, smoothing_window=3)
    if result is None:
        return
    sample_points = {(float(s["x"]), float(s["y"])) for s in samples}
    assert result in sample_points, (
        f"velocity_endpoint returned {result}, not in input samples"
    )


def prop_velocity_endpoint_empty_returns_none():
    assert velocity_endpoint([], (0, 0)) is None
    assert velocity_endpoint([{"t": 0, "x": 0, "y": 0}], (0, 0)) is None


def main():
    print("[saccade_extractors_property_test] running properties...")
    prop_peak_distance_empty_returns_none()
    prop_peak_distance_returns_a_sample()
    print("  ✓ peak_distance returns one of the input samples")
    prop_peak_distance_is_maximal()
    print("  ✓ peak_distance result is maximal-distance")
    prop_idt_empty_returns_none()
    print("  ✓ idt empty/single-sample → None")
    prop_idt_result_is_inside_sample_bbox()
    print("  ✓ idt centroid lies inside input bbox")
    prop_idt_rejects_constant_anchor_input()
    print("  ✓ idt rejects all-on-anchor input")
    prop_idt_finds_clear_off_anchor_fixation()
    print("  ✓ idt locks onto a clear off-anchor fixation")
    prop_velocity_endpoint_empty_returns_none()
    prop_velocity_endpoint_returns_sample_or_none()
    print("  ✓ velocity_endpoint result is one of the input samples")
    print("All saccade_extractors property tests passed.")


if __name__ == "__main__":
    main()
