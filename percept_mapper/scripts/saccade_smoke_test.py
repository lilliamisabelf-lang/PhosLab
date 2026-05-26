"""Headless smoke test for the saccade response path.

Builds a SaccadeScreen with a mock eye_tracker that returns scripted gaze,
runs the response loop, and verifies the payload + retry logic. Doesn't
launch main.py — just exercises the response stage in isolation.

Run:
    uv run --project percept_mapper python percept_mapper/scripts/saccade_smoke_test.py
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame  # noqa: E402

from scripts.saccade_screen import SaccadeScreen  # noqa: E402
from scripts.audio_cue import from_config as build_audio_cue  # noqa: E402
from scripts.response_capture import SaccadeResponseCapture  # noqa: E402


class MockEyeTracker:
    """Returns scripted gaze positions: fixate anchor for 200 ms, then jump
    to target for the rest of the window."""

    def __init__(self, anchor, target, switch_ms=200):
        self.anchor = anchor
        self.target = target
        self.switch_ms = switch_ms
        self.t0 = time.monotonic()
        self.last_smooth_gaze = anchor
        self.last_raw_gaze = anchor

    def _tick(self):
        elapsed_ms = (time.monotonic() - self.t0) * 1000.0
        if elapsed_ms < self.switch_ms:
            self.last_smooth_gaze = self.anchor
        else:
            self.last_smooth_gaze = self.target
        self.last_raw_gaze = self.last_smooth_gaze


def run_one_trial(extraction, target, anchor, max_attempts=3, audio_cue=None):
    pygame.init()
    pygame.display.set_mode((1920, 1080))

    tracker = MockEyeTracker(anchor, target, switch_ms=200)

    screen_obj = SaccadeScreen(
        screen_width=1920,
        screen_height=1080,
        anchor_xy=anchor,
        eye_tracker=tracker,
        capture_duration_ms=800,
        extraction=extraction,
        extractor_params={
            "idt": {"dispersion_px": 60, "min_duration_ms": 100, "skip_anchor_radius_px": 40},
            "velocity": {"onset_threshold_px_s": 1500, "settle_threshold_px_s": 300, "smoothing_window": 5},
        },
        min_response_distance_px=30,
        max_attempts=max_attempts,
        show_gaze_trace=True,
        audio_cue=audio_cue,
    )
    screen_obj.reset()
    finished = False
    payload = None
    while not finished:
        tracker._tick()
        finished, payload = screen_obj.update(pygame.display.get_surface(), [])
        time.sleep(1 / 60.0)
    return payload


def test_idt_with_clear_saccade():
    print("\n=== test: IDT, clear saccade to (1500, 500) ===")
    anchor = (960, 540)
    target = (1500, 500)
    payload = run_one_trial("idt_first_fixation", target=target, anchor=anchor)
    assert payload["status"] == "ok", f"expected ok, got {payload['status']}"
    rxy = payload["response_xy"]
    err = ((rxy[0] - target[0]) ** 2 + (rxy[1] - target[1]) ** 2) ** 0.5
    print(f"  response_xy={rxy}  target={target}  error={err:.1f} px")
    assert err < 30, f"response too far from target ({err:.1f} px)"
    print("  ✓ PASS")


def test_peak_distance_with_clear_saccade():
    print("\n=== test: peak_distance ===")
    anchor = (960, 540)
    target = (1500, 500)
    payload = run_one_trial("peak_distance", target=target, anchor=anchor)
    assert payload["status"] == "ok", f"expected ok, got {payload['status']}"
    rxy = payload["response_xy"]
    err = ((rxy[0] - target[0]) ** 2 + (rxy[1] - target[1]) ** 2) ** 0.5
    print(f"  response_xy={rxy}  target={target}  error={err:.1f} px")
    assert err < 5, f"peak should land exactly on target ({err:.1f} px)"
    print("  ✓ PASS")


def test_velocity_endpoint():
    print("\n=== test: velocity_endpoint ===")
    anchor = (960, 540)
    target = (1500, 500)
    payload = run_one_trial("velocity_endpoint", target=target, anchor=anchor)
    print(f"  status={payload['status']}  response_xy={payload['response_xy']}")
    # MockEyeTracker is a step function, not smooth — velocity_endpoint may or
    # may not detect a clean landing. Accept both ok and failed_no_endpoint.
    assert payload["status"] in ("ok", "failed_no_endpoint")
    print("  ✓ PASS")


def test_no_saccade_returns_failure():
    print("\n=== test: no saccade (stay at anchor) ===")
    anchor = (960, 540)
    payload = run_one_trial("idt_first_fixation", target=anchor, anchor=anchor)
    assert payload["status"] != "ok", f"should fail when no off-anchor fixation"
    print(f"  status={payload['status']}  ✓ PASS")


def test_retry_budget():
    print("\n=== test: retry budget exhausts ===")
    pygame.init()
    pygame.display.set_mode((1920, 1080))
    anchor = (960, 540)
    tracker = MockEyeTracker(anchor, anchor, switch_ms=200)  # no saccade
    screen_obj = SaccadeScreen(
        screen_width=1920,
        screen_height=1080,
        anchor_xy=anchor,
        eye_tracker=tracker,
        capture_duration_ms=200,
        extraction="idt_first_fixation",
        max_attempts=3,
    )
    # Simulate the trial loop's retry behavior
    attempt_count = 0
    while True:
        screen_obj.reset()
        attempt_count += 1
        finished = False
        payload = None
        while not finished:
            tracker._tick()
            finished, payload = screen_obj.update(pygame.display.get_surface(), [])
            time.sleep(1 / 60.0)
        if payload["status"] == "ok":
            break
        if screen_obj.should_rerun():
            continue
        break
    print(f"  attempts={attempt_count}  final status={payload['status']}")
    assert attempt_count == 3, f"should retry exactly 3 times, got {attempt_count}"
    assert payload["status"] != "ok"
    print("  ✓ PASS")


def test_mouse_fallback_ignores_stale_tracker():
    """Regression: in input_mode=mouse/wacom, allow_mouse_fallback=True must
    bypass the tracker's last_*_gaze attributes. MouseTracker only refreshes
    those inside is_looking_at_point() (prestim/stim/poststim), so after the
    first trial they hold a frozen, prestim-era cursor position. If we honor
    them during the saccade window the trace freezes from trial 2 onward."""
    print("\n=== test: allow_mouse_fallback bypasses stale tracker.last_smooth_gaze ===")
    pygame.init()
    pygame.display.set_mode((1920, 1080))
    anchor = (960, 540)

    class StaleTracker:
        # Simulates MouseTracker AFTER prestim cached a value that won't
        # change during saccade capture.
        last_raw_gaze = (123, 456)
        last_smooth_gaze = (123, 456)

    screen_obj = SaccadeScreen(
        screen_width=1920,
        screen_height=1080,
        anchor_xy=anchor,
        eye_tracker=StaleTracker(),
        capture_duration_ms=150,
        extraction="idt_first_fixation",
        max_attempts=1,
        allow_mouse_fallback=True,
    )
    screen_obj.reset()
    surf = pygame.display.get_surface()

    # Drive a few frames; collect the raw sample x positions.
    finished = False
    while not finished:
        finished, _ = screen_obj.update(surf, [])
        time.sleep(1 / 240.0)

    sample_xs = {int(s["x"]) for s in screen_obj.samples}
    assert 123 not in sample_xs, (
        f"trace must not be pinned to stale tracker value; samples_x={sorted(sample_xs)[:6]}"
    )
    print(f"  samples captured={len(screen_obj.samples)}  stale_x_seen={123 in sample_xs}")
    print("  ✓ PASS")


def test_response_capture_silent_retry():
    """SaccadeResponseCapture.update() must keep returning False (and call
    reset on the underlying SaccadeScreen) while should_rerun() is true, then
    finally return True with a fail payload when the budget runs out."""
    print("\n=== test: SaccadeResponseCapture silent retry through wrapper ===")
    pygame.init()
    pygame.display.set_mode((1920, 1080))
    anchor = (960, 540)
    tracker = MockEyeTracker(anchor, anchor, switch_ms=200)  # no saccade ever
    screen_obj = SaccadeScreen(
        screen_width=1920,
        screen_height=1080,
        anchor_xy=anchor,
        eye_tracker=tracker,
        capture_duration_ms=150,
        extraction="idt_first_fixation",
        max_attempts=3,
    )
    wrapper = SaccadeResponseCapture(screen_obj)
    wrapper.reset()

    surf = pygame.display.get_surface()
    finished_count = 0
    safety_iters = 0
    while True:
        safety_iters += 1
        if safety_iters > 2000:
            raise RuntimeError("wrapper retry loop did not terminate")
        tracker._tick()
        finished = wrapper.update(surf, [])
        if finished:
            finished_count += 1
            break
        time.sleep(1 / 240.0)

    assert finished_count == 1, "wrapper should report finished exactly once"
    assert wrapper.last_status != "ok", (
        f"with no saccade, final wrapper status should not be ok; got {wrapper.last_status}"
    )
    assert screen_obj._attempts == 3, (
        f"wrapper should silently exhaust retries; got attempts={screen_obj._attempts}"
    )
    print(
        f"  attempts={screen_obj._attempts}  wrapper.last_status={wrapper.last_status}"
    )
    print("  ✓ PASS")


def test_audio_cue_builds():
    print("\n=== test: audio_cue from_config ===")
    cue = build_audio_cue({"enabled": True, "frequency_hz": 880, "duration_ms": 80, "volume": 0.4})
    if cue is None:
        print("  ⚠ mixer not available in this env (acceptable in headless)")
    else:
        print(f"  ✓ tone built: {cue}")


def main():
    test_idt_with_clear_saccade()
    test_peak_distance_with_clear_saccade()
    test_velocity_endpoint()
    test_no_saccade_returns_failure()
    test_retry_budget()
    test_mouse_fallback_ignores_stale_tracker()
    test_response_capture_silent_retry()
    test_audio_cue_builds()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
