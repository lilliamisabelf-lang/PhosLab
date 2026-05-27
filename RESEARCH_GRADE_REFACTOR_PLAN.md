# Research-grade refactor plan

Three layers, each shippable on its own, each with tests. The order matters:
each layer assumes the previous one.

## Why now

The cheap-wins pass landed (CI, Python pin, property tests). The next
quality jump requires structural changes: typed data, library-shaped
analysis, declarative experiment spec. None are urgent — the system runs
today — but every additional dataset and every new analyzer compounds
the cost of leaving them dict-based and tangled.

---

## Layer 1 — Typed data contracts  *(start here)*

**Problem.** Every payload moving between phases is a hand-rolled dict:
`phosphene_metadata`, `repetition_metadata`, `session_metadata`, the JSON
files on disk, the analyzer outputs. Analyzers defensively `.get()`
everything because schema drifts silently across releases. There is no
version field, so loading an old dataset under new code is a coin flip.

**Goal.** A `schemas` module with frozen dataclasses for the load-bearing
records, each carrying a `schema_version` field and round-trippable to
JSON. Consumers (writers/analyzers) accept either the typed object or
the dict, so we can migrate incrementally.

### Sub-steps

| # | What | Files | Test |
|---|------|-------|------|
| 1.1 | `percept_mapper/scripts/schemas.py`: `TrialRecord`, `SessionMetadata`, `ElectrodeAnalysisResult` dataclasses + `from_dict` / `to_dict` / `SCHEMA_VERSION` constants. | new | property: round-trip preserves all fields; missing fields → default; extra fields → ignored |
| 1.2 | `SessionMetadata` writer in `main.py` swaps dict construction for the dataclass. JSON output should be byte-identical to today's output for new sessions. | `main.py`, `schemas.py` | golden-file: build a SessionMetadata, serialize, compare against the existing format |
| 1.3 | `TrialRecord` replaces the per-trial `repetition_metadata` dict end-to-end. The boundary that writes `electrode_<idx>/metadata.json` stays JSON. | `phosphene_mapping.py`, `main.py` | smoke: run the saccade smoke test driving SaccadeResponseCapture, confirm output keys unchanged |
| 1.4 | `ElectrodeAnalysisResult` typed return from `mapping_analyzer.analyze_electrode_repetitions`. Plot code stays a consumer. | `mapping_analyzer.py` | property: serializing then deserializing reproduces the same dict the analyzer used to return |
| 1.5 | Loader helpers `load_session(path) -> SessionMetadata`, `load_electrode(path) -> ElectrodeMetadata` with explicit `schema_version` check + back-compat shims for `schema_version=None` (old files). | `schemas.py` | property: synthesize old-format dicts → load → re-serialize → matches the new format |

**Exit criteria.** All existing smoke + property tests still pass. New
round-trip property tests pass with `hypothesis`. Old session JSON files
(no `schema_version`) still load. New writes carry `schema_version: 1`.

**Effort.** ~2 h of focused work. Risk: medium — touches the trial loop,
which is on the critical path of every experiment run.

---

## Layer 2 — Analysis as a library

**Problem.** `mapping_analyzer.py` mixes file IO, statistics, and
matplotlib in a single 1000+-line class. You can't import a stat into a
Jupyter notebook without dragging pygame and matplotlib into the
namespace. Unit-testing the numbers means running through the whole
class.

**Goal.** Three layers separated:
- `stats/` — pure functions taking arrays, returning dataclasses (no IO,
  no plotting, no global state).
- `loaders/` — typed loaders sitting on top of Layer 1 schemas.
- `plots/` — matplotlib-only, takes the typed stat results.
- A thin orchestrator class wires them together and is the only
  remaining entry point for `main.py` / the launcher.

### Sub-steps

| # | What | Files | Test |
|---|------|-------|------|
| 2.1 | `scripts/stats/centroid_stats.py`: `compute_centroid_stats(centroids: ndarray, anchor: ndarray, px_per_deg: tuple) -> CentroidStats`. Pure function. | new | property: deterministic given same input; std ≥ 0; centroid ∈ bbox of inputs |
| 2.2 | `scripts/stats/confidence_ellipse.py`: extract `_ellipse_from_cov` into its own file with tests | new (split from mapping_analyzer) | property: ellipse contains > 90% of a synthetic Gaussian sample (sanity) |
| 2.3 | `scripts/stats/catch_response.py`: `summarize_catch(trials, electrode_dir) -> CatchStats`. Pure. | new | smoke: 0 catches → None response_rate; mix → expected rate |
| 2.4 | `scripts/plots/electrode_map.py`: receives `ElectrodeAnalysisResult` + stats, returns a `matplotlib.Figure`. No file IO. | new | smoke: builds without crashing on synthetic input |
| 2.5 | `mapping_analyzer.PhospheneMappingAnalyzer` becomes a thin orchestrator: load → compute stats → optional plot → return. Most of the math moves out. | `mapping_analyzer.py` | existing analyzer-shape tests still pass; output JSON byte-identical to pre-refactor for the same input fixtures |

**Exit criteria.** Stats can be imported from Jupyter without pygame
or matplotlib loading. Plot code is a separate, optional dependency at
the orchestrator level. Smoke tests still green. New unit tests on each
pure function.

**Effort.** ~3 h. Risk: low if Layer 1 is done first — the typed
schemas form the contract between layers.

---

## Layer 3 — Experiment-as-data  *(defer; scope after Layers 1 & 2)*

**Problem.** `main.py` is ~2500 lines specifying *what the experiment is*
in Python: phase ordering, timing, response capture, presentation. To
pre-register a design, share a protocol, or run a variant, you read
Python.

**Goal.** A declarative `protocol.yaml` consumed by a thin runner:

```yaml
protocol:
  schema_version: 1
  name: phosphene_mapping_v1
  phases:
    - { name: prestim, screen: anchor, gate: continuous_fixation_ms, value: 200 }
    - { name: stim,    screen: stimulation, gate: time_or_lost_fixation_ms, value: 400 }
    - { name: poststim, screen: anchor, gate: time_ms, value: 100 }
    - { name: response, screen: saccade | drawing }
  trial_sequence:
    randomize: true
    catch_trial_rate: 0.15
    num_practice_trials: 2
```

**Why defer.** The runner needs Layer 1's typed records (to know what
each phase emits) and Layer 2's library shape (to know how to consume
the records). Doing it before 1 and 2 means we'd rewrite it twice.

**Effort.** ~1 full day. Will scope separately once Layers 1 & 2 land.

---

## Execution order this session

1. Layer 1 in 5 sub-step commits. Each ends with green tests.
2. Layer 2 in 5 sub-step commits. Each ends with green tests.
3. If time and context remain, sketch the Layer 3 protocol schema; do
   not implement.
