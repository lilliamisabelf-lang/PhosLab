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
    test_audio_cue_builds()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
