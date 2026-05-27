"""Headless, deterministic implementation of the protocol runner.

The `phase_runner` module defines the *contract* (HandlerRegistry,
PhaseStatus, TrialRunResult). This module supplies a concrete set of
handlers that run a real ProtocolSpec to completion **without pygame
and without a display** — they simulate the timing and the fixation
state deterministically, driven by a `HeadlessContext` you control.

What this is for:
- Proves the on-disk protocol + the phase_runner are executable as a
  pipeline. Anything that breaks the phase ordering or gate semantics
  fails a CI test, not just a property test on synthetic data.
- Lets a researcher dry-run a protocol change (new YAML, new gate)
  before risking it on a real participant.
- Establishes the shape of the production runner that Layer 3c will
  build to bind to actual pygame screens.

What this is *not*:
- Not a research tool — no real timing, no rendering, no participant.
- Not a benchmark — uses `time.sleep` for time-based gates with a
  config-controllable speed multiplier (default fast).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

from scripts.phase_runner import HandlerRegistry, PhaseStatus
from scripts.protocol import PhaseSpec


@dataclass
class HeadlessContext:
    """Per-trial state the headless handlers read.

    `fixation_lost_probability` simulates a flaky tracker — each
    fixation-aware phase rolls against this probability to decide
    whether to fire FIXATION_LOST. Set to 0.0 for a clean trial and
    1.0 to force a failure.

    `time_multiplier` scales every gate's `value` ms when sleeping.
    Default 0.0 (no real sleep — instant). Set to 1.0 if you want the
    runner to honor wall-clock timing (useful only for ad-hoc demos).

    `cancelled` short-circuits the trial after the next phase boundary
    when set to True — mimics the participant pressing ESC.
    """
    fixation_lost_probability: float = 0.0
    time_multiplier: float = 0.0
    cancelled: bool = False
    rng_seed: int = 0
    _rng: random.Random = field(init=False)
    # Mutable record of what the runner observed: list of dicts, one
    # per phase invocation. Useful for assertions in tests.
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self._rng = random.Random(self.rng_seed)


def _sleep_for(phase: PhaseSpec, ctx: HeadlessContext) -> None:
    """Honor `time_multiplier` × phase.value (ms). Zero = no sleep."""
    if ctx.time_multiplier > 0 and phase.value > 0:
        time.sleep(ctx.time_multiplier * float(phase.value) / 1000.0)


def _roll_fixation_lost(ctx: HeadlessContext) -> bool:
    """Return True iff this phase should simulate a fixation loss."""
    if ctx.fixation_lost_probability <= 0:
        return False
    if ctx.fixation_lost_probability >= 1:
        return True
    return ctx._rng.random() < ctx.fixation_lost_probability


def _record(ctx: HeadlessContext, phase: PhaseSpec, status: PhaseStatus, note: str = "") -> None:
    ctx.history.append({
        "phase": phase.name,
        "screen": phase.screen,
        "gate": phase.gate,
        "value": phase.value,
        "status": status.value,
        "note": note,
    })


# ---- per-(screen, gate) handlers ------------------------------------------


def _handler_time_ms(phase: PhaseSpec, ctx: dict[str, Any]) -> PhaseStatus:
    """`time_ms` gate: just waits, then OK. No fixation check."""
    hctx = ctx["headless"]
    if hctx.cancelled:
        _record(hctx, phase, PhaseStatus.CANCELLED, "cancelled before sleep")
        return PhaseStatus.CANCELLED
    _sleep_for(phase, hctx)
    _record(hctx, phase, PhaseStatus.OK)
    return PhaseStatus.OK


def _handler_continuous_fixation_ms(phase: PhaseSpec, ctx: dict[str, Any]) -> PhaseStatus:
    """`continuous_fixation_ms` gate: simulate the participant fixating
    for `value` ms. Rolls against `fixation_lost_probability` to decide
    whether the fixation broke before the budget elapsed."""
    hctx = ctx["headless"]
    if hctx.cancelled:
        _record(hctx, phase, PhaseStatus.CANCELLED)
        return PhaseStatus.CANCELLED
    if _roll_fixation_lost(hctx):
        _record(hctx, phase, PhaseStatus.FIXATION_LOST, "rolled fixation loss")
        return PhaseStatus.FIXATION_LOST
    _sleep_for(phase, hctx)
    _record(hctx, phase, PhaseStatus.OK)
    return PhaseStatus.OK


def _handler_time_or_lost_fixation_ms(phase: PhaseSpec, ctx: dict[str, Any]) -> PhaseStatus:
    """`time_or_lost_fixation_ms` gate: wait `value` ms, but abort early
    on simulated fixation loss. Behaviorally identical to
    `continuous_fixation_ms` in this headless model — both can return
    FIXATION_LOST. The retry policy is the differentiator and is owned
    by phase_runner via on_fixation_lost."""
    hctx = ctx["headless"]
    if hctx.cancelled:
        _record(hctx, phase, PhaseStatus.CANCELLED)
        return PhaseStatus.CANCELLED
    if _roll_fixation_lost(hctx):
        _record(hctx, phase, PhaseStatus.FIXATION_LOST, "rolled fixation loss")
        return PhaseStatus.FIXATION_LOST
    _sleep_for(phase, hctx)
    _record(hctx, phase, PhaseStatus.OK)
    return PhaseStatus.OK


def _handler_response_finished(phase: PhaseSpec, ctx: dict[str, Any]) -> PhaseStatus:
    """`response_finished` gate: simulate the participant completing
    their response. No fixation requirement during response phase."""
    hctx = ctx["headless"]
    if hctx.cancelled:
        _record(hctx, phase, PhaseStatus.CANCELLED)
        return PhaseStatus.CANCELLED
    _sleep_for(phase, hctx)
    _record(hctx, phase, PhaseStatus.OK)
    return PhaseStatus.OK


def build_headless_registry() -> HandlerRegistry:
    """Build a HandlerRegistry covering every (screen, gate) pair in
    SCREEN_TYPES × GATE_TYPES that appears in the default protocol.

    Adding a new (screen, gate) to a protocol requires either
    registering a handler here or extending the registry inline.
    Missing handlers raise UnknownHandlerError at run_trial time.
    """
    reg = HandlerRegistry()
    # Anchor + each gate
    for gate, fn in (
        ("time_ms", _handler_time_ms),
        ("continuous_fixation_ms", _handler_continuous_fixation_ms),
        ("time_or_lost_fixation_ms", _handler_time_or_lost_fixation_ms),
    ):
        reg.register("anchor", gate, fn)
    # Stimulation only ever uses time_or_lost_fixation_ms in the default
    # protocol, but we register it across the same gate set for futures.
    for gate, fn in (
        ("time_ms", _handler_time_ms),
        ("continuous_fixation_ms", _handler_continuous_fixation_ms),
        ("time_or_lost_fixation_ms", _handler_time_or_lost_fixation_ms),
    ):
        reg.register("stimulation", gate, fn)
    # Response screens use response_finished.
    reg.register("saccade", "response_finished", _handler_response_finished)
    reg.register("drawing", "response_finished", _handler_response_finished)
    return reg
