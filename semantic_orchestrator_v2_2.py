"""Semantic Operations v2.2: atomic classification with fail-soft execution.

The trusted capability, demographic, state and reference guards remain ahead
of the model. The residual model surface is a fixed atomic classification
vector. Invalid model output is never repaired by a second generation; the
orchestrator preserves deterministic grounding when safe and otherwise asks a
clarifying question.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Mapping

import suitability_clarification
from command_orchestrator import CommandOrchestrator, CommandTurnResult
from semantic_atomic_v2_2 import AtomicFrameError, AtomicSemanticFrame, build_atomic_frame_payload
from semantic_capability_v2_1 import evaluate_capability
from semantic_demographic_v2_1 import needs_relational_demographic_clarification
from semantic_orchestrator_v2_1 import SemanticOperationsOrchestratorV21
from semantic_state_v2_2 import AtomicReduction, fail_soft_reduce, grounded_slots_from_query_v22, reduce_atomic_frame


FrameCall = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True)
class AtomicFrameGeneration:
    frame: AtomicSemanticFrame | None
    attempts: int
    error: str | None = None
    raw_output: str | None = None


@dataclass(frozen=True)
class SemanticV22Result:
    status: str
    flow: str
    slots: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    frame: AtomicSemanticFrame | None = None
    reduction: AtomicReduction | None = None
    command_result: CommandTurnResult | None = None
    frame_attempts: int = 0
    frame_fallback: bool = False
    deterministic_route: str | None = None
    latency_ms: float = 0.0
    handled: bool = True
    frame_error: str | None = None
    raw_output: str | None = None

    @property
    def events(self) -> list[dict[str, Any]]:
        return self.command_result.events if self.command_result is not None else []

    @property
    def near_events(self) -> list[dict[str, Any]]:
        return self.command_result.near_events if self.command_result is not None else []


def _answer(raw: Any) -> str | None:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, Mapping) and isinstance(raw.get("answer"), str):
        return str(raw["answer"])
    return None


def generate_atomic_frame(query: str, state: Mapping[str, Any] | None, *, call: FrameCall) -> AtomicFrameGeneration:
    grounded = grounded_slots_from_query_v22(query)
    payload = build_atomic_frame_payload(query, state, grounded)
    try:
        raw = call(payload)
    except Exception as exc:
        return AtomicFrameGeneration(None, 1, error=f"frame call failed: {type(exc).__name__}")
    answer = _answer(raw)
    if answer is None:
        return AtomicFrameGeneration(None, 1, error="empty atomic frame response")
    try:
        return AtomicFrameGeneration(AtomicSemanticFrame.from_json(answer), 1, raw_output=answer)
    except (AtomicFrameError, TypeError, ValueError) as exc:
        # No repair generation in v2.2. A malformed classifier response should
        # not double latency or erase deterministic work.
        return AtomicFrameGeneration(None, 1, error=f"atomic frame invalid: {exc}", raw_output=answer)


class SemanticOperationsOrchestratorV22(SemanticOperationsOrchestratorV21):
    """Parallel v2.2 evaluator; production paths remain untouched."""

    def _execute_v22_plan(self, query: str, state: Mapping[str, Any] | None, reduction: AtomicReduction, route: str, started: float, *, frame: AtomicSemanticFrame | None, generation: AtomicFrameGeneration | None = None) -> SemanticV22Result:
        assert reduction.plan is not None
        command = CommandOrchestrator(modal_call=None, reference_date=self.reference_date, events=self.events).handle_query(
            query,
            state,
            command_plan=reduction.plan,
        )
        return SemanticV22Result(
            status=command.status,
            flow=command.flow,
            slots=command.command.slots.to_dict(),
            message=command.message,
            frame=frame,
            reduction=reduction,
            command_result=command,
            frame_attempts=generation.attempts if generation is not None else 0,
            frame_fallback=reduction.fail_soft,
            deterministic_route=route,
            latency_ms=(time.perf_counter() - started) * 1000,
            handled=command.handled,
            frame_error=generation.error if generation is not None else reduction.fail_soft_reason,
            raw_output=generation.raw_output if generation is not None else None,
        )

    def _from_reduction(
        self,
        query: str,
        state: Mapping[str, Any] | None,
        reduction: AtomicReduction,
        route: str,
        started: float,
        *,
        frame: AtomicSemanticFrame | None,
        generation: AtomicFrameGeneration | None,
    ) -> SemanticV22Result:
        attempts = generation.attempts if generation is not None else 0
        error = generation.error if generation is not None else reduction.fail_soft_reason
        raw = generation.raw_output if generation is not None else None
        if reduction.status == "clarification":
            return SemanticV22Result(
                "clarification", "unsupported", message=reduction.message, frame=frame,
                reduction=reduction, frame_attempts=attempts, frame_fallback=reduction.fail_soft,
                deterministic_route=route, latency_ms=(time.perf_counter() - started) * 1000,
                frame_error=error, raw_output=raw,
            )
        if reduction.status in {"data_limit", "unsupported"}:
            return SemanticV22Result(
                "unsupported", "unsupported", message=reduction.message, frame=frame,
                reduction=reduction, frame_attempts=attempts, frame_fallback=reduction.fail_soft,
                deterministic_route="data_capability_guard" if reduction.status == "data_limit" else route,
                latency_ms=(time.perf_counter() - started) * 1000,
                frame_error=error, raw_output=raw,
            )
        if not reduction.ready or reduction.plan is None:
            return SemanticV22Result(
                "invalid_command", "unsupported",
                message=reduction.message or "検索条件を確認できませんでした。",
                frame=frame, reduction=reduction, frame_attempts=attempts,
                frame_fallback=reduction.fail_soft, handled=False, deterministic_route=route,
                latency_ms=(time.perf_counter() - started) * 1000,
                frame_error=error, raw_output=raw,
            )
        return self._execute_v22_plan(
            query, state, reduction, route, started, frame=frame, generation=generation,
        )

    def handle_query(self, query: str, state: Mapping[str, Any] | None = None) -> SemanticV22Result:
        started = time.perf_counter()
        value = str(query).strip()
        if not value:
            raise ValueError("query must not be empty")

        capability = evaluate_capability(value)
        if not capability.allowed:
            return SemanticV22Result(
                "unsupported", "unsupported", message=capability.message,
                deterministic_route=f"capability:{capability.reason}",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        security_message = self._security_guard(value)
        if security_message is not None:
            return SemanticV22Result(
                "unsupported", "unsupported", message=security_message,
                deterministic_route="security_or_domain_guard",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        suitability = suitability_clarification.analyze_suitability_request(value)
        if suitability.needs_clarification or needs_relational_demographic_clarification(value):
            return SemanticV22Result(
                "clarification", "unsupported",
                message=suitability_clarification.clarification_message(value),
                deterministic_route="ambiguous_suitability_guard",
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        if suitability.should_strip_suitability_marker:
            value = suitability.sanitized_query

        route_name, trusted_plan, immediate_message, previous_scope = self._deterministic_context_plan(value, state)
        if immediate_message is not None:
            status = "clarification" if route_name == "clarify_reference" else "unsupported"
            return SemanticV22Result(
                status, "unsupported", message=immediate_message,
                deterministic_route=route_name,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        if trusted_plan is not None:
            command = CommandOrchestrator(modal_call=None, reference_date=self.reference_date, events=self.events).handle_query(
                value, state, command_plan=trusted_plan,
            )
            return SemanticV22Result(
                status=command.status, flow=command.flow, slots=command.command.slots.to_dict(),
                message=command.message, command_result=command, deterministic_route=route_name,
                latency_ms=(time.perf_counter() - started) * 1000, handled=command.handled,
            )

        if self.frame_call is None:
            reduction = fail_soft_reduce(
                value, state, reference_date=self.reference_date,
                previous_scope=previous_scope, reason="frame_backend_unavailable",
            )
            return self._from_reduction(
                value, state, reduction, route_name, started,
                frame=None, generation=None,
            )

        generation = generate_atomic_frame(value, state, call=self.frame_call)
        if generation.frame is None:
            reduction = fail_soft_reduce(
                value, state, reference_date=self.reference_date,
                previous_scope=previous_scope, reason=generation.error or "atomic_frame_invalid",
            )
            return self._from_reduction(
                value, state, reduction, route_name, started,
                frame=None, generation=generation,
            )

        reduction = reduce_atomic_frame(
            generation.frame,
            value,
            state,
            reference_date=self.reference_date,
            previous_scope=previous_scope,
        )
        return self._from_reduction(
            value, state, reduction, route_name, started,
            frame=generation.frame, generation=generation,
        )


__all__ = [
    "AtomicFrameGeneration", "SemanticOperationsOrchestratorV22",
    "SemanticV22Result", "generate_atomic_frame",
]
