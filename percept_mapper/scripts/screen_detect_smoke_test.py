"""Smoke tests for the pure logic in `screen_detect`.

Detection itself is hardware-dependent, so we pin the two testable
pieces: the diagonal derivation and the comment-preserving params writer.

Run:
    uv run --project percept_mapper python percept_mapper/scripts/screen_detect_smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.screen_detect import DisplayInfo, update_params_screen_block


def test_diagonal_from_physical_size():
    d = DisplayInfo(0, "DELL", 60.0, 34.0, (2560, 1440), is_primary=True)
    assert d.diagonal_inches is not None
    assert abs(d.diagonal_inches - 27.15) < 0.05, d.diagonal_inches
    # Missing physical size -> no guess
    assert DisplayInfo(0, "", None, None, (1920, 1080)).diagonal_inches is None
    print("  ✓ diagonal derived from physical size; None when size unknown")


def test_writer_replaces_direct_children_only():
    text = (
        "screen:\n"
        "  width: 1920\n"
        "  height: 1080\n"
        "  fullscreen: true\n"
        "  screen_diagonal_inches: 13.3\n"
        "  vf_scope_deg: 70\n"
        "  anchor_circle:\n"
        "    radius: 50\n"
        "    width: 999   # nested, must NOT be touched\n"
        "timing:\n"
        "  width: 7   # different block, must NOT be touched\n"
    )
    out = update_params_screen_block(text, width=2560, height=1440, diagonal_inches=27.15)
    assert "  width: 2560\n" in out
    assert "  height: 1440\n" in out
    assert "  screen_diagonal_inches: 27.15\n" in out
    # untouched lines
    assert "  fullscreen: true\n" in out
    assert "  vf_scope_deg: 70\n" in out
    assert "    width: 999   # nested, must NOT be touched\n" in out
    assert "  width: 7   # different block, must NOT be touched\n" in out
    # exactly one screen-level width line
    assert out.count("  width: 2560\n") == 1
    print("  ✓ writer updates only direct screen children, preserves rest")


def test_writer_inserts_missing_key():
    text = (
        "screen:\n"
        "  width: 1920\n"
        "  height: 1080\n"
        "timing:\n"
        "  prestimulation: 200\n"
    )
    out = update_params_screen_block(text, diagonal_inches=27.15)
    assert "  screen_diagonal_inches: 27.15\n" in out
    # inserted before the next top-level block, still inside screen
    assert out.index("screen_diagonal_inches") < out.index("timing:")
    print("  ✓ writer inserts a missing key at child indent inside the block")


def test_writer_noop_when_nothing_to_change():
    text = "screen:\n  width: 1920\n"
    assert update_params_screen_block(text) == text
    print("  ✓ writer is a no-op when no updates are given")


def main():
    print("[screen_detect_smoke_test] running...")
    test_diagonal_from_physical_size()
    test_writer_replaces_direct_children_only()
    test_writer_inserts_missing_key()
    test_writer_noop_when_nothing_to_change()
    print("\nAll screen_detect smoke tests passed.")


if __name__ == "__main__":
    main()
