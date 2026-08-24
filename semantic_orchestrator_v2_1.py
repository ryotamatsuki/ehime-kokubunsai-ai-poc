"""Semantic Operations v2.1 orchestration.

High-confidence conversation state/reference acts are resolved before the LLM.
Only residual semantic ambiguity reaches the sparse frame normalizer.  The
trusted CommandOrchestrator remains the sole execution surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import time
from typing import Any, Callable, Mapping, Sequence

import conversation_recovery
import conversation_router
import event_search
import suitability_clarification
from app_config import POC_REFERENCE_DATE
from command_models import CommandPlan, CommandSlots
from command_orchestrator import CommandOrchestrator, CommandTurnResult
from semantic_frame_v2_1 import SparseFrameError, SparseSemanticFrame, build_sparse_frame_payload
from semantic_state_v2_1 import SparseReduction, reduce_sparse_frame


FrameCall = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True)
class SparseFrameGeneration:
    frame: SparseSemanticFrame | None
    attempts: int
    repaired: bool = False
    error: str | None = None
    raw_output: str | None = None


@dataclass(frozen=True)
class SemanticV21Result:
    status: str
    flow: str
    slots: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    frame: SparseSemanticFrame | None = None
    reduction: SparseReduction | None = None
    command_result: CommandTurnResult | None = None
    frame_attempts: int = 0
    frame_repaired: bool = False
    deterministic_route: str | None = None
    latency_ms: float = 0.0
    handled: bool = True
    frame_error: str | None = None

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


def generate_sparse_frame(query: str, state: Mapping[str, Any] | None, *, call: FrameCall) -> SparseFrameGeneration:
    payload = build_sparse_frame_payload(query, state)
    try:
        raw = call(payload)
    except Exception as exc:
        return SparseFrameGeneration(None, 1, error=f"frame call failed: {type(exc).__name__}")
    answer = _answer(raw)
    if answer is None:
        return SparseFrameGeneration(None, 1, error="empty sparse frame response")
    try:
        return SparseFrameGeneration(SparseSemanticFrame.from_json(answer), 1, raw_output=answer)
    except (SparseFrameError, TypeError, ValueError) as first_error:
        repair_payload = dict(payload)
        repair_payload["repair"] = {"invalid_output": answer[:1200], "error": str(first_error)[:400]}
        try:
            repaired_raw = call(repair_payload)
        except Exception as exc:
            return SparseFrameGeneration(None, 2, repaired=True, error=f"frame repair call failed: {type(exc).__name__}", raw_output=answer)
        repaired_answer = _answer(repaired_raw)
        if repaired_answer is None:
            return SparseFrameGeneration(None, 2, repaired=True, error="empty sparse frame repair response", raw_output=answer)
        try:
            return SparseFrameGeneration(SparseSemanticFrame.from_json(repaired_answer), 2, repaired=True, raw_output=repaired_answer)
        except (SparseFrameError, TypeError, ValueError) as second_error:
            return SparseFrameGeneration(None, 2, repaired=True, error=f"frame repair failed: {second_error}", raw_output=repaired_answer)


def _event_id(event: Mapping[str, Any]) -> str:
    return str(event.get("id") or str(event.get("公式URL", "")).rstrip("/").rsplit("/", 1)[-1])


class SemanticOperationsOrchestratorV21:
    def __init__(
        self,
        frame_call: FrameCall | None = None,
        *,
        reference_date: date = POC_REFERENCE_DATE,
        events: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self.frame_call = frame_call
        self.reference_date = reference_date
        self.events = [dict(event) for event in (events or event_search.load_events())]
        self.by_id = {_event_id(event): event for event in self.events}

    @staticmethod
    def _security_guard(query: str) -> str | None:
        intent = event_search.classify_intent(query)
        if intent == "injection":
            return "この案内の制約を変更したり、掲載されていないイベントを作ったりはできません。掲載済みのイベントから探してみて。"
        if intent == "out_of_scope" or conversation_recovery.is_domain_out_of_scope(query):
            return "このPoCは文化祭イベントの検索・参加案内が中心です。"
        return None

    def _context_events(self, state: Mapping[str, Any] | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        if not isinstance(state, Mapping):
            return [], None
        ids = state.get("last_result_ids")
        last = [self.by_id[str(item)] for item in ids if str(item) in self.by_id] if isinstance(ids, (list, tuple)) else []
        selected_id = state.get("selected_event_id")
        selected = self.by_id.get(str(selected_id)) if selected_id not in (None, "") else None
        return last, selected

    @staticmethod
    def _reference_slots(route: conversation_router.ConversationRoute, last_results: Sequence[Mapping[str, Any]], state: Mapping[str, Any] | None) -> dict[str, Any]:
        values: dict[str, Any] = {}
        if route.reference_index is not None:
            # conversation_router uses a zero-based index internally.
            values["reference_kind"] = "ordinal"
            values["reference_index"] = route.reference_index + 1
        elif route.selected_event is not None:
            selected_id = _event_id(route.selected_event)
            for index, event in enumerate(last_results, start=1):
                if _event_id(event) == selected_id:
                    values["reference_kind"] = "ordinal"
                    values["reference_index"] = index
                    break
            else:
                if isinstance(state, Mapping) and str(state.get("selected_event_id") or "") == selected_id:
                    values["reference_kind"] = "selected"
                else:
                    values["reference_kind"] = "event_name"
                    values["event_name"] = str(route.selected_event.get("イベント名", ""))
        elif isinstance(state, Mapping) and state.get("selected_event_id"):
            values["reference_kind"] = "selected"
        return values

    def _deterministic_context_plan(
        self,
        query: str,
        state: Mapping[str, Any] | None,
    ) -> tuple[str, CommandPlan | None, str | None, bool]:
        """Return (route_name, trusted_plan, immediate_message, previous_scope_hint)."""

        last_results, selected_event = self._context_events(state)
        route = conversation_router.route_conversation(
            query,
            last_results,
            selected_event,
            state.get("last_filters") if isinstance(state, Mapping) else None,
            self.reference_date,
        )
        action = route.action_type

        if action == "clarify_reference":
            return action, None, "基準にするイベントを番号かイベント名で教えてみて。", False
        if action in {"generic_scope", "scope_search"} and not event_search.looks_like_event_query(query):
            return "capability_scope_guard", None, "このPoCは文化祭イベントの検索・参加案内が中心です。", False
        if action == "general_faq":
            return action, CommandPlan(flow="general_faq", slots=CommandSlots(), confidence="high"), None, False
        if action == "explain_search":
            return action, CommandPlan(flow="explain_search", slots=CommandSlots(), confidence="high"), None, False
        if action in {"explain_result", "recommend_next", "recommend_similar"}:
            flow = {
                "explain_result": "explain_result",
                "recommend_next": "recommend_next",
                "recommend_similar": "recommend_similar",
            }[action]
            values = self._reference_slots(route, last_results, state)
            if not values:
                return "clarify_reference", None, "基準にするイベントを番号かイベント名で教えてみて。", False
            return action, CommandPlan(flow=flow, slots=CommandSlots.from_dict(values), confidence="high"), None, False
        if action in {"reference_followup", "detail_followup"} and route.detail_field is not None:
            values = self._reference_slots(route, last_results, state)
            if not values:
                return "clarify_reference", None, "どのイベントのことか、番号かイベント名で教えてみて。", False
            values["detail_fields"] = [route.detail_field]
            return action, CommandPlan(flow="event_detail", slots=CommandSlots.from_dict(values), confidence="high"), None, False

        previous_scope = bool(last_results and action == "search" and event_search.is_refinement_query(query))
        return action, None, None, previous_scope

    @staticmethod
    def _force_previous_scope(frame: SparseSemanticFrame, previous_scope: bool) -> SparseSemanticFrame:
        if not previous_scope or frame.scope == "previous":
            return frame
        values = frame.to_dict(sparse=True)
        values["scope"] = "previous"
        return SparseSemanticFrame.from_dict(values)

    def _execute_plan(self, query: str, state: Mapping[str, Any] | None, plan: CommandPlan, route: str, started: float) -> SemanticV21Result:
        command = CommandOrchestrator(modal_call=None, reference_date=self.reference_date, events=self.events).handle_query(query, state, command_plan=plan)
        return SemanticV21Result(
            status=command.status,
            flow=command.flow,
            slots=command.command.slots.to_dict(),
            message=command.message,
            command_result=command,
            deterministic_route=route,
            latency_ms=(time.perf_counter() - started) * 1000,
            handled=command.handled,
        )

    def handle_query(self, query: str, state: Mapping[str, Any] | None = None) -> SemanticV21Result:
        started = time.perf_counter()
        value = str(query).strip()
        if not value:
            raise ValueError("query must not be empty")

        security_message = self._security_guard(value)
        if security_message is not None:
            return SemanticV21Result("unsupported", "unsupported", message=security_message, deterministic_route="security_or_domain_guard", latency_ms=(time.perf_counter() - started) * 1000)

        suitability = suitability_clarification.analyze_suitability_request(value)
        if suitability.needs_clarification:
            return SemanticV21Result("clarification", "unsupported", message=suitability_clarification.clarification_message(value), deterministic_route="ambiguous_suitability_guard", latency_ms=(time.perf_counter() - started) * 1000)
        if suitability.should_strip_suitability_marker:
            value = suitability.sanitized_query

        route_name, trusted_plan, immediate_message, previous_scope = self._deterministic_context_plan(value, state)
        if immediate_message is not None:
            status = "clarification" if route_name == "clarify_reference" else "unsupported"
            return SemanticV21Result(status, "unsupported", message=immediate_message, deterministic_route=route_name, latency_ms=(time.perf_counter() - started) * 1000)
        if trusted_plan is not None:
            return self._execute_plan(value, state, trusted_plan, route_name, started)

        if self.frame_call is None:
            return SemanticV21Result("unavailable", "unsupported", handled=False, deterministic_route="frame_backend_unavailable", latency_ms=(time.perf_counter() - started) * 1000)

        generation = generate_sparse_frame(value, state, call=self.frame_call)
        if generation.frame is None:
            return SemanticV21Result("unavailable", "unsupported", handled=False, frame_attempts=generation.attempts, frame_repaired=generation.repaired, frame_error=generation.error, latency_ms=(time.perf_counter() - started) * 1000)
        frame = self._force_previous_scope(generation.frame, previous_scope)
        reduction = reduce_sparse_frame(frame, value, state, reference_date=self.reference_date)
        if reduction.status == "clarification":
            return SemanticV21Result("clarification", "unsupported", message=reduction.message, frame=frame, reduction=reduction, frame_attempts=generation.attempts, frame_repaired=generation.repaired, deterministic_route=route_name, latency_ms=(time.perf_counter() - started) * 1000)
        if reduction.status in {"data_limit", "unsupported"}:
            return SemanticV21Result("unsupported", "unsupported", message=reduction.message, frame=frame, reduction=reduction, frame_attempts=generation.attempts, frame_repaired=generation.repaired, deterministic_route="data_capability_guard" if reduction.status == "data_limit" else route_name, latency_ms=(time.perf_counter() - started) * 1000)
        if not reduction.ready or reduction.plan is None:
            return SemanticV21Result("invalid_command", "unsupported", message=reduction.message or "検索条件を確認できませんでした。", frame=frame, reduction=reduction, frame_attempts=generation.attempts, frame_repaired=generation.repaired, handled=False, deterministic_route=route_name, latency_ms=(time.perf_counter() - started) * 1000)

        command = CommandOrchestrator(modal_call=None, reference_date=self.reference_date, events=self.events).handle_query(value, state, command_plan=reduction.plan)
        return SemanticV21Result(
            status=command.status,
            flow=command.flow,
            slots=command.command.slots.to_dict(),
            message=command.message,
            frame=frame,
            reduction=reduction,
            command_result=command,
            frame_attempts=generation.attempts,
            frame_repaired=generation.repaired,
            deterministic_route=route_name,
            latency_ms=(time.perf_counter() - started) * 1000,
            handled=command.handled,
        )


__all__ = ["SemanticOperationsOrchestratorV21", "SemanticV21Result", "SparseFrameGeneration", "generate_sparse_frame"]
