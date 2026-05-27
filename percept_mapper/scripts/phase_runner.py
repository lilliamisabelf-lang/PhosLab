"""Skeletal protocol runner.

Step 3c of `RESEARCH_GRADE_REFACTOR_PLAN.md` is "rewrite main.py to
consume a ProtocolSpec." That's invasive enough to merit its own
session. This module is the foundation that future session will build
on: a *pure-Python* dispatcher that iterates a ProtocolSpec and
invokes pluggable handlers, with no pygame / no I/O.

`main.py` will eventually register its real screen + gate callbacks
with this runner. Today the runner is testable in isolation against
mock handlers, so the contract is pinned before we wire in the
production callbacks.

Contract:
- A *handler* is `Callable[[PhaseSpec, ContextDict], PhaseStatus]`.
- The handler returns one of the `PhaseStatus` values to tell the
  runner what to do next.
- The runner enforces phase order and the `on_fixation_lost` policy.
- No magic: every handler dispatch is name-based via the registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

from scripts.protocol import PhaseSpec, ProtocolSpec


class PhaseStatus(str, Enum):
    """What the handler signals back to the runner.

    OK              — phase completed normally; advance to the next.
    FIXATION_LOST   — fixation broken mid-phase; runner consults
                      `phase.on_fixation_lost` to decide what to do.
    CANCELLED       — user pressed ESC / closed the window. The runner
                      stops the trial immediately.
    """
    OK = "ok"
    FIXATION_LOST = "fixation_lost"
    CANCELLED = "cancelled"


class TrialStatus(str, Enum):
    """Aggregate status for a single trial after run_trial returns."""
    COMPLETED = "completed"
    ABORTED = "aborted"
    CANCELLED = "cancelled"


Handler = Callable[[PhaseSpec, dict[str, Any]], PhaseStatus]


class UnknownHandlerError(KeyError):
    """Raised when a protocol references a (screen, gate) pair that the
    registry doesn't know how to dispatch."""


@dataclass
class HandlerRegistry:
    """Maps (screen, gate) → handler callable.

    Keys are stringly-typed (matching ProtocolSpec.PhaseSpec.{screen,gate})
    so a YAML edit can swap implementations without touching Python.
    """
    handlers: dict[tuple[str, str], Handler] = field(default_factory=dict)

    def register(self, screen: str, gate: str, handler: Handler) -> None:
        self.handlers[(screen, gate)] = handler

    def resolve(self, phase: PhaseSpec) -> Handler:
        key = (phase.screen, phase.gate)
        if key not in self.handlers:
            raise UnknownHandlerError(
                f"no handler registered for screen={phase.screen!r}, gate={phase.gate!r}"
            )
        return self.handlers[key]


@dataclass
class TrialRunResult:
    """Outcome of running one trial through the runner.

    `phases_run` is the list of (phase_name, status) tuples in the order
    the runner saw them — useful for both metadata and post-mortem.
    `retry_count` is how many times a phase fired the `retry_phase`
    fixation-loss policy. A trial that retries many times is suspicious.
    """
    status: TrialStatus
    phases_run: list[tuple[str, PhaseStatus]] = field(default_factory=list)
    retry_count: int = 0
    aborted_at_phase: str | None = None


def run_trial(
    protocol: ProtocolSpec,
    registry: HandlerRegistry,
    context: dict[str, Any] | None = None,
    max_retries_per_phase: int = 10,
) -> TrialRunResult:
    """Drive a single trial: invoke each phase's handler in order, apply
    the `on_fixation_lost` policy, return a result.

    `context` is a free-form dict passed to every handler; production
    code stuffs it with screens, eye_tracker, pygame_screen, etc. Tests
    can pass {} or a small dict that mocks need to read.

    `max_retries_per_phase` is a safety guard against an infinite
    retry loop when a flaky tracker keeps losing fixation. The trial
    aborts after that many retries.
    """
    context = dict(context or {})
    result = TrialRunResult(status=TrialStatus.COMPLETED)
    i = 0
    phase_retry_count: dict[str, int] = {}

    while i < len(protocol.phases):
        phase = protocol.phases[i]
        handler = registry.resolve(phase)
        status = handler(phase, context)
        result.phases_run.append((phase.name, status))

        if status is PhaseStatus.OK:
            i += 1
            continue

        if status is PhaseStatus.CANCELLED:
            result.status = TrialStatus.CANCELLED
            result.aborted_at_phase = phase.name
            return result

        # FIXATION_LOST: dispatch on the phase's policy.
        if phase.on_fixation_lost == "retry_phase":
            phase_retry_count[phase.name] = phase_retry_count.get(phase.name, 0) + 1
            result.retry_count += 1
            if phase_retry_count[phase.name] > max_retries_per_phase:
                result.status = TrialStatus.ABORTED
                result.aborted_at_phase = phase.name
                return result
            # Stay at i — the handler is re-invoked.
            continue
        if phase.on_fixation_lost == "abort_trial":
            result.status = TrialStatus.ABORTED
            result.aborted_at_phase = phase.name
            return result
        # "continue" — silently advance despite the loss.
        i += 1

    return result


def run_protocol(
    protocol: ProtocolSpec,
    registry: HandlerRegistry,
    *,
    trial_contexts: Iterable[dict[str, Any]],
    max_retries_per_phase: int = 10,
) -> list[TrialRunResult]:
    """Run the protocol once per item in `trial_contexts`.

    `trial_contexts` is typically built from a `trial_sequence.build_trial_list`
    output — one context per trial, containing the per-trial state the
    handlers need to read.
    """
    return [
        run_trial(
            protocol, registry, context=ctx,
            max_retries_per_phase=max_retries_per_phase,
        )
        for ctx in trial_contexts
    ]
