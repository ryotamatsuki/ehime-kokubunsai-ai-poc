"""Parallel Semantic Operations v2 orchestration path.

This module intentionally does not replace ``CommandOrchestrator`` yet. It
prepares a side-by-side architecture for empirical evaluation against the
frozen Unexpected User Utterances suite. Once a semantic frame is validated
and reduced, execution is delegated to the existing trusted deterministic
CommandOrchestrator with ``command_plan=...`` so no new tool-execution surface
is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import time
from typing import Any, Callable, Mapping

import conversation_recovery
import event_search
import suitability_clarification
from app_config import POC_REFERENCE_DATE
from command_orchestrator import CommandOrchestrator, CommandTurnResult
from semantic_frame_v2 import SemanticFrame, SemanticFrameError, build_semantic_frame_payload
from semantic_state_v2 import SemanticReduction, reduce_semantic_frame


FrameCall = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True)
class FrameGeneration:
    frame: SemanticFrame | None
    attempts: int
    repaired: bool = False
    error: str | None = None
    raw_output: str | None = None


@dataclass(frozen=True)
class SemanticV2Result:
    status: str
    flow: str
    slots: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    frame: SemanticFrame | None = None
    reduction: SemanticReduction | None = None
    command_result: CommandTurnResult | None = None
    frame_attempts: int = 0
    frame_repaired: bool = False
    deterministic_route: str | None = None
    latency_ms: float = 0.0
    handled: bool = True

    @property
    def near_events(self) -> list[dict[str, Any]]:
        return self.command_result.near_events if self.command_result is not None else []

    @property
    def events(self) -> list[dict[str, Any]]:
        return self.command_result.events if self.command_result is not None else []


def _answer_from_call(raw: Any) -> str | None:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, Mapping):
        answer = raw.get("answer")
        if isinstance(answer, str):
            return answer
    return None


def generate_semantic_frame(
    query: str,
    state: Mapping[str, Any] | None,
    *,
    call: FrameCall,
) -> FrameGeneration:
    """Generate a compact frame and repair syntax/contract at most once."""

    payload = build_semantic_frame_payload(query, state)
    try:
        raw = call(payload)
    except Exception as exc:
        return FrameGeneration(None, 1, error=f"frame call failed: {type(exc).__name__}")
    answer = _answer_from_call(raw)
    if answer is None:
        return FrameGeneration(None, 1, error="empty frame response")
    try:
        return FrameGeneration(SemanticFrame.from_json(answer), 1, raw_output=answer)
    except (SemanticFrameError, TypeError, ValueError) as first_error:
        repair_payload = dict(payload)
        repair_payload["repair"] = {
            "invalid_output": answer[:1200],
            "error": str(first_error)[:400],
        }
        try:
            repaired_raw = call(repair_payload)
        except Exception as exc:
            return FrameGeneration(
                None,
                2,
                repaired=True,
                error=f"frame repair call failed: {type(exc).__name__}",
                raw_output=answer,
            )
        repaired_answer = _answer_from_call(repaired_raw)
        if repaired_answer is None:
            return FrameGeneration(
                None,
                2,
                repaired=True,
                error="empty frame repair response",
                raw_output=answer,
            )
        try:
            return FrameGeneration(
                SemanticFrame.from_json(repaired_answer),
                2,
                repaired=True,
                raw_output=repaired_answer,
            )
        except (SemanticFrameError, TypeError, ValueError) as second_error:
            return FrameGeneration(
                None,
                2,
                repaired=True,
                error=f"frame repair failed: {second_error}",
                raw_output=repaired_answer,
            )


class SemanticOperationsOrchestratorV2:
    """Hybrid natural-language -> semantic frame -> state reducer -> executor."""

    def __init__(
        self,
        frame_call: FrameCall | None = None,
        *,
        reference_date: date = POC_REFERENCE_DATE,
        events: list[Mapping[str, Any]] | None = None,
    ) -> None:
        self.frame_call = frame_call
        self.reference_date = reference_date
        self.events = events

    @staticmethod
    def _security_guard(query: str) -> str | None:
        intent = event_search.classify_intent(query)
        if intent == "injection":
            return (
                "この案内の制約を変更したり、掲載されていないイベントを作ったりはできません。"
                "掲載済みのイベントから探してみて。"
            )
        if intent == "out_of_scope" or conversation_recovery.is_domain_out_of_scope(query):
            return "このPoCは文化祭イベントの検索・参加案内が中心です。"
        return None

    def handle_query(
        self,
        query: str,
        state: Mapping[str, Any] | None = None,
    ) -> SemanticV2Result:
        started = time.perf_counter()
        value = str(query).strip()
        if not value:
            raise ValueError("query must not be empty")

        security_message = self._security_guard(value)
        if security_message is not None:
            return SemanticV2Result(
                status="unsupported",
                flow="unsupported",
                message=security_message,
                deterministic_route="security_or_domain_guard",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        suitability = suitability_clarification.analyze_suitability_request(value)
        if suitability.needs_clarification:
            return SemanticV2Result(
                status="clarification",
                flow="unsupported",
                message=suitability_clarification.clarification_message(value),
                deterministic_route="ambiguous_suitability_guard",
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        if suitability.should_strip_suitability_marker:
            value = suitability.sanitized_query

        if self.frame_call is None:
            return SemanticV2Result(
                status="unavailable",
                flow="unsupported",
                message="",
                handled=False,
                deterministic_route="frame_backend_unavailable",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        generation = generate_semantic_frame(value, state, call=self.frame_call)
        if generation.frame is None:
            return SemanticV2Result(
                status="unavailable",
                flow="unsupported",
                message="",
                frame_attempts=generation.attempts,
                frame_repaired=generation.repaired,
                handled=False,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        reduction = reduce_semantic_frame(
            generation.frame,
            value,
            state,
            reference_date=self.reference_date,
        )
        if reduction.status == "clarification":
            return SemanticV2Result(
                status="clarification",
                flow="unsupported",
                message=reduction.message,
                frame=generation.frame,
                reduction=reduction,
                frame_attempts=generation.attempts,
                frame_repaired=generation.repaired,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        if reduction.status == "data_limit":
            return SemanticV2Result(
                status="unsupported",
                flow="unsupported",
                message=reduction.message,
                frame=generation.frame,
                reduction=reduction,
                frame_attempts=generation.attempts,
                frame_repaired=generation.repaired,
                deterministic_route="data_capability_guard",
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        if reduction.status == "unsupported":
            return SemanticV2Result(
                status="unsupported",
                flow="unsupported",
                message=reduction.message,
                frame=generation.frame,
                reduction=reduction,
                frame_attempts=generation.attempts,
                frame_repaired=generation.repaired,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        if not reduction.ready or reduction.plan is None:
            return SemanticV2Result(
                status="invalid_command",
                flow="unsupported",
                message=reduction.message or "検索条件を確認できませんでした。",
                frame=generation.frame,
                reduction=reduction,
                frame_attempts=generation.attempts,
                frame_repaired=generation.repaired,
                handled=False,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        command_result = CommandOrchestrator(
            modal_call=None,
            reference_date=self.reference_date,
            events=self.events,
        ).handle_query(value, state, command_plan=reduction.plan)

        return SemanticV2Result(
            status=command_result.status,
            flow=command_result.flow,
            slots=command_result.command.slots.to_dict(),
            message=command_result.message,
            frame=generation.frame,
            reduction=reduction,
            command_result=command_result,
            frame_attempts=generation.attempts,
            frame_repaired=generation.repaired,
            latency_ms=(time.perf_counter() - started) * 1000,
            handled=command_result.handled,
        )


__all__ = [
    "FrameGeneration",
    "SemanticOperationsOrchestratorV2",
    "SemanticV2Result",
    "generate_semantic_frame",
]
