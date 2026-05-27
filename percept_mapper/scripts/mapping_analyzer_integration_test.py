"""End-to-end integration test for PhospheneMappingAnalyzer.

This is the test that actually verifies the analyzer works on disk —
the property tests pin individual contracts, the smoke tests pin
isolated subsystems, but until you build a complete electrode_<idx>/
directory and run the analyzer on it, you can't be sure the pieces
fit together.

Synthesises a fixture in a tempdir for every test case, runs the
analyzer, asserts on the returned dict structure + schema version +
matplotlib non-import.

Run:
    uv run --project percept_mapper python percept_mapper/scripts/mapping_analyzer_integration_test.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _build_fixture_metadata(
    electrode_index: int = 80,
    num_reps: int = 5,
    stim_position=(1500.0, 540.0),
    response_xys=None,
    include_catch: bool = False,
    legacy_no_schema_version: bool = False,
) -> dict:
    """Build the dict that gets written to electrode_<idx>/metadata.json.

    Each rep is a saccade-mode trial with a deterministic response_xy.
    """
    if response_xys is None:
        # Cluster around the stim position with small jitter
        cx, cy = stim_position
        response_xys = [
            (cx + 5.0 * (i % 3 - 1), cy + 7.0 * ((i + 1) % 3 - 1))
            for i in range(num_reps)
        ]

    repetitions = []
    for i, xy in enumerate(response_xys, start=1):
        rep = {
            "repetition_number": i,
            "electrode_index": electrode_index,
            "trial_idx": i - 1,
            "is_catch": False,
            "is_practice": False,
            "position": list(stim_position),
            "stimulation_parameters": {
                "current_uA": 90.0,
                "pulse_width_us": 200.0,
                "frequency_hz": 50.0,
            },
            "start_time": f"2026-05-27T10:00:{i:02d}",
            "end_time": f"2026-05-27T10:00:{i + 1:02d}",
            "events": {"prestim_start": f"t{i}.1", "stim_start": f"t{i}.2"},
            "fixation_losses": 0,
            "trial_attempts": 1,
            "gaze_tracking": {"prestim": [], "stim": [], "poststim": [], "drawing": []},
            # Saccade-mode response metadata
            "response_mode": "saccade",
            "response_status": "ok",
            "response_xy": list(xy),
            "response_file": f"saccade_samples_repetition_{i:03d}.json",
            "response_file_type": "json",
            "response_extraction": "idt_first_fixation",
            "response_attempts": 1,
            "saccade_samples_file": f"saccade_samples_repetition_{i:03d}.json",
        }
        if not legacy_no_schema_version:
            rep["schema_version"] = 1
        repetitions.append(rep)

    if include_catch:
        # One catch trial at trial_idx=999 — no stim, no response_xy
        repetitions.append({
            "repetition_number": 999,
            "electrode_index": electrode_index,
            "trial_idx": 999,
            "is_catch": True,
            "is_practice": False,
            "position": list(stim_position),
            "stimulation_parameters": {"current_uA": 0.0, "pulse_width_us": 0.0, "frequency_hz": 0.0},
            "start_time": "2026-05-27T11:00:00",
            "end_time": "2026-05-27T11:00:01",
            "events": {},
            "fixation_losses": 0,
            "trial_attempts": 1,
            "gaze_tracking": {"prestim": [], "stim": [], "poststim": [], "drawing": []},
            "response_mode": "saccade",
            "response_status": "failed_no_fixation",
            "response_xy": None,
            "response_file": None,
            "response_file_type": None,
            "schema_version": 1,
        })

    return {
        "experiment_name": "integration_test",
        "experiment_id": "fixture_001",
        "start_time": "2026-05-27T10:00:00",
        "electrode_index": electrode_index,
        "electrode_info": {"index": electrode_index, "visual_position_deg": [4.0, 0.0]},
        "display": {
            "screen_resolution_px": [1920, 1080],
            "screen_center_px": [960, 540],
            "vf_scope_deg": 15.0,
        },
        "num_repetitions": num_reps,
        "timing": {
            "prestimulation_ms": 200, "stimulation_ms": 400,
            "poststimulation_ms": 100, "interstimulation_ms": 500,
        },
        "repetitions": repetitions,
    }


def _make_fixture_dir(metadata: dict, tmpdir: Path) -> Path:
    electrode_dir = tmpdir / f"electrode_{metadata['electrode_index']:03d}"
    electrode_dir.mkdir(parents=True)
    with open(electrode_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return electrode_dir


def test_end_to_end_saccade_session():
    print("\n=== test: end-to-end analyzer on a 5-rep saccade session ===")
    with tempfile.TemporaryDirectory() as tmp:
        electrode_dir = _make_fixture_dir(_build_fixture_metadata(), Path(tmp))

        # Importing the analyzer should NOT load matplotlib.
        import sys as _sys
        assert "matplotlib" not in _sys.modules, "matplotlib already loaded before analyzer import"
        from scripts.mapping_analyzer import PhospheneMappingAnalyzer
        assert "matplotlib" not in _sys.modules, (
            "BUG: importing PhospheneMappingAnalyzer pulled in matplotlib"
        )

        analyzer = PhospheneMappingAnalyzer(electrode_dir)
        results = analyzer.analyze_electrode_repetitions()

    assert results is not None
    assert results["schema_version"] == 1, f"got schema_version={results.get('schema_version')}"
    assert results["electrode_index"] == 80
    assert results["num_total_repetitions"] == 5
    assert results["num_valid_repetitions"] == 5
    assert results["num_invalid_repetitions"] == 0

    centroids = results["centroids"]
    assert len(centroids) == 5
    # The mean should sit near the stim position we wrote (1500, 540)
    mean = results.get("mean_position") or {}
    assert abs(mean.get("x", 0) - 1500.0) < 20.0, f"mean x off: {mean}"
    assert abs(mean.get("y", 0) - 540.0) < 20.0, f"mean y off: {mean}"

    # Layer 1 + Layer 2 additions surface as extras now (schema only declares
    # a subset of fields; the rest pass through). Validate they're present.
    for k in (
        "stimulation_position", "per_repetition_metrics", "boxplot_stats",
        "catch_trial_stats", "within_electrode_reliability",
    ):
        assert k in results, f"missing key: {k}"

    print(f"  ✓ analyzer ran, schema_version={results['schema_version']}, "
          f"centroids={len(centroids)}, mean=({mean['x']:.1f}, {mean['y']:.1f})")
    print("  ✓ matplotlib was NOT loaded by the analyzer import")


def test_catch_trial_separated_from_centroid():
    print("\n=== test: catch trial does not contaminate centroid ===")
    md = _build_fixture_metadata(include_catch=True)
    with tempfile.TemporaryDirectory() as tmp:
        electrode_dir = _make_fixture_dir(md, Path(tmp))
        from scripts.mapping_analyzer import PhospheneMappingAnalyzer
        analyzer = PhospheneMappingAnalyzer(electrode_dir)
        results = analyzer.analyze_electrode_repetitions()

    # 5 real + 1 catch in metadata, but only 5 should contribute to centroids
    assert results["num_valid_repetitions"] == 5
    assert len(results["centroids"]) == 5
    catch = results.get("catch_trial_stats") or {}
    assert catch.get("n_total") == 1
    # The catch trial has response_xy=None so it counts as "no response"
    assert catch.get("n_with_response") == 0
    print(f"  ✓ catch stats: {catch}")


def test_legacy_no_schema_version_still_loads():
    print("\n=== test: pre-v1 metadata.json (no schema_version on reps) still works ===")
    md = _build_fixture_metadata(legacy_no_schema_version=True)
    with tempfile.TemporaryDirectory() as tmp:
        electrode_dir = _make_fixture_dir(md, Path(tmp))
        from scripts.mapping_analyzer import PhospheneMappingAnalyzer
        analyzer = PhospheneMappingAnalyzer(electrode_dir)
        results = analyzer.analyze_electrode_repetitions()
    # The legacy reps get upgraded on read; the result gets stamped v1.
    assert results["schema_version"] == 1
    assert results["num_valid_repetitions"] == 5
    print("  ✓ legacy session upgrades cleanly to schema v1")


def test_visualize_lazy_loads_matplotlib():
    """Calling visualize_results should be the moment matplotlib loads — but
    not before. Validates the lazy-import contract from Layer 2 part 2."""
    print("\n=== test: visualize_results lazily loads matplotlib ===")
    md = _build_fixture_metadata()
    with tempfile.TemporaryDirectory() as tmp:
        electrode_dir = _make_fixture_dir(md, Path(tmp))

        import sys as _sys
        # Avoid cross-test contamination — pop any prior matplotlib
        # imports so we can observe the lazy load in isolation.
        for name in list(_sys.modules):
            if name.startswith("matplotlib"):
                del _sys.modules[name]

        from scripts.mapping_analyzer import PhospheneMappingAnalyzer
        analyzer = PhospheneMappingAnalyzer(electrode_dir)
        results = analyzer.analyze_electrode_repetitions()
        assert "matplotlib" not in _sys.modules, "stats path leaked matplotlib"

        analyzer.visualize_results(results, output_file="analysis_plot.png")
        assert "matplotlib" in _sys.modules, "BUG: visualize_results did not load matplotlib"

    print("  ✓ matplotlib loaded only after visualize_results was called")


def main():
    print("[mapping_analyzer_integration_test] running...")
    test_end_to_end_saccade_session()
    test_catch_trial_separated_from_centroid()
    test_legacy_no_schema_version_still_loads()
    test_visualize_lazy_loads_matplotlib()
    print("\nAll mapping_analyzer integration tests passed.")


if __name__ == "__main__":
    main()
