"""Plotting helpers for percept_mapper.

Each module here is matplotlib-only — pure visual layer, no IO beyond the
optional `output_path` save. Imports are lazy so users that only need
the numbers (`scripts.stats`, `scripts.schemas`, the analyzer's
`analyze_electrode_repetitions`) never pay the matplotlib import cost.
"""

from scripts.plots.electrode_map import plot_electrode_map

__all__ = ["plot_electrode_map"]
