"""Smoke test for scripts.plots.plot_electrode_map.

Builds a minimal results dict, calls the plot function, checks:
  - it returns a matplotlib Figure,
  - it saves a non-empty PNG when given output_path,
  - it does not crash on the minimum valid input (2 reps).

Run:
    uv run --project percept_mapper python percept_mapper/scripts/plots/plots_smoke_test.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _minimal_results() -> dict:
    """Smallest dict that plot_electrode_map needs to render. Mirrors the
    shape `mapping_analyzer.analyze_electrode_repetitions` returns."""
    centroids = [[1495.0, 540.0], [1500.0, 547.0], [1505.0, 533.0]]
    return {
        "electrode_index": 80,
        "num_valid_repetitions": 3,
        "num_invalid_repetitions": 0,
        "centroids": centroids,
        "mean_position": {"x": 1500.0, "y": 540.0},
        "std_position": {"x": 4.1, "y": 5.7},
        "stimulation_position": [1500.0, 540.0],
        "mean_distance_from_average": 5.5,
        "max_distance_from_average": 9.6,
        "mean_distance_from_average_deg": 0.13,
        "max_distance_from_average_deg": 0.24,
    }


def test_plot_returns_figure():
    print("\n=== test: plot_electrode_map returns a Figure ===")
    # Import deferred so we can prove the top-level import is matplotlib-free.
    assert "matplotlib" not in sys.modules
    from scripts.plots import plot_electrode_map
    fig = plot_electrode_map(
        _minimal_results(),
        screen_size=(1920, 1080),
        pixels_per_degree=(64.0, 36.0),
        output_path=None,
    )
    assert fig is not None
    # Now matplotlib should be loaded (lazy import inside the function).
    assert "matplotlib" in sys.modules
    # Close to free memory.
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("  ✓ figure returned, matplotlib loaded only after the call")


def test_plot_saves_file():
    print("\n=== test: plot_electrode_map saves a non-empty PNG ===")
    from scripts.plots import plot_electrode_map
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test_plot.png"
        fig = plot_electrode_map(
            _minimal_results(),
            screen_size=(1920, 1080),
            pixels_per_degree=(64.0, 36.0),
            output_path=out,
        )
        assert out.exists(), "PNG not written"
        size = out.stat().st_size
        assert size > 1000, f"PNG suspiciously small ({size} bytes)"
        import matplotlib.pyplot as plt
        plt.close(fig)
    print(f"  ✓ PNG written, {size} bytes")


def test_plot_handles_two_reps():
    """Minimum input where ellipse_from_cov gives meaningful output is 2
    reps. Below that it should still render (just no ellipse)."""
    print("\n=== test: 2-rep input renders without crash ===")
    from scripts.plots import plot_electrode_map
    results = _minimal_results()
    results["centroids"] = results["centroids"][:2]
    results["num_valid_repetitions"] = 2
    fig = plot_electrode_map(
        results,
        screen_size=(1920, 1080),
        pixels_per_degree=(64.0, 36.0),
    )
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("  ✓ 2-rep input renders cleanly")


def test_none_input_returns_none():
    print("\n=== test: None input returns None (no crash) ===")
    from scripts.plots import plot_electrode_map
    assert plot_electrode_map(
        None, screen_size=(1920, 1080), pixels_per_degree=(64.0, 36.0)
    ) is None
    print("  ✓ None handled gracefully")


def main():
    print("[plots_smoke_test] running...")
    test_plot_returns_figure()
    test_plot_saves_file()
    test_plot_handles_two_reps()
    test_none_input_returns_none()
    print("\nAll plots smoke tests passed.")


if __name__ == "__main__":
    main()
