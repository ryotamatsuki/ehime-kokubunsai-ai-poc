"""Deterministic execution of validated semantic commands.

The command generator is an untrusted language boundary.  This module is the
trusted boundary after it: it validates ``CommandPlan`` against the canonical
model and Flow Registry, converts slots to a fixed local filter shape, and
dispatches only to Python functions selected by the registry.  No model output
is used as a tool name, Python expression, event fact, count, date, fee, URL,
or feasibility decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time as clock_time
import re
import time
from typing import Any, Callable, Mapping, Sequence

import agent_tools
import event_details
import event_recommendation
import event_search
import faq_search
from agent_models import SearchSpec, ToolResult
from app_config import POC_REFERENCE_DATE
from command_generator import (
    CommandGenerationResult,
    DEFAULT_COMMAND_FORMAT,
    generate_command,
)
from command_models import (
    CommandPlan,
    CommandSlots,
    CommandValidationError,
    parse_command_plan,
    validate_command_plan,
)
from event_pair_recommendation import EventPair, recommend_event_pairs
from flow_registry import get_flow_spec, required_slots_for


MAX_COMMAND_QUERY_LENGTH = 1200
MAX_LAST_RESULT_IDS = 8
MAX_RESULTS = 8
MAX_PAIR_RESULTS = 3
DEFAULT_PAIR_BUFFER_MINUTES = 30
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


@dataclass(frozen=True)
class CommandState:
    """Small, bounded conversation state used for reference resolution."""

    reference_date: date = POC_REFERENCE_DATE
    selected_event_id: str | None = None
    last_result_ids: tuple[str, ...] = ()
    last_command: Mapping[str, Any] | None = None
    active_flow: str | None = None
    pending_slots: Mapping[str, Any] = field(default_factory=dict)
    pending_required_slots: tuple[str, ...] = ()

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None,
        default_reference_date: date,
    ) -> "CommandState":
        if not isinstance(raw, Mapping):
            return cls(reference_date=default_reference_date)
        reference_date = default_reference_date
        raw_date = raw.get("reference_date")
        if isinstance(raw_date, date) and not isinstance(raw_date, type):
            reference_date = raw_date
        elif isinstance(raw_date, str):
            try:
                reference_date = date.fromisoformat(raw_date)
            except ValueError:
                reference_date = default_reference_date

        selected = raw.get("selected_event_id")
        selected_id = str(selected).strip() if selected not in (None, "") else None
        result_ids_raw = raw.get("last_result_ids", ())
        if isinstance(result_ids_raw, str):
            result_ids_raw = (result_ids_raw,)
        if not isinstance(result_ids_raw, (list, tuple)):
            result_ids_raw = ()
        result_ids = tuple(
            str(value).strip()
            for value in result_ids_raw[:MAX_LAST_RESULT_IDS]
            if str(value).strip()
        )
        last_command = raw.get("last_command")
        if not isinstance(last_command, Mapping):
            last_command = None
        pending_slots = raw.get("pending_slots")
        if not isinstance(pending_slots, Mapping):
            pending_slots = {}
        required = raw.get("pending_required_slots", ())
        if not isinstance(required, (list, tuple)):
            required = ()
        required_slots = tuple(
            str(value)
            for value in required
            if str(value) in {"dates", "time_after", "reference", "event"}
        )
        active_flow = raw.get("active_flow")
        active_flow = str(active_flow) if isinstance(active_flow, str) else None
        return cls(
            reference_date=reference_date,
            selected_event_id=selected_id,
            last_result_ids=result_ids,
            last_command=dict(last_command) if last_command is not None else None,
            active_flow=active_flow,
            pending_slots=dict(pending_slots),
            pending_required_slots=required_slots,
        )


@dataclass(frozen=True)
class CommandLatency:
    """Wall-clock observations for a single bounded command turn."""

    generator_ms: float = 0.0
    execution_ms: float = 0.0
    total_ms: float = 0.0
    generator_calls: int = 0
    repair_calls: int = 0


@dataclass(frozen=True)
class CommandTurnResult:
    """UI-safe result of one command turn.

    All event-shaped values in this object originate from the local JSON
    catalog or deterministic recommendation functions.  ``command`` is the
    validated semantic plan, never a raw model response.
    """

    status: str
    command: CommandPlan
    flow: str
    slots: dict[str, Any] = field(default_factory=dict)
    result: ToolResult | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    near_events: list[dict[str, Any]] = field(default_factory=list)
    pairs: list[EventPair] = field(default_factory=list)
    total_matches: int | None = None
    filters: dict[str, Any] | None = None
    pending: dict[str, Any] | None = None
    question: str | None = None
    message: str = ""
    attempts: int = 0
    repaired: bool = False
    latency: CommandLatency = field(default_factory=CommandLatency)
    handled: bool = True

    @property
    def plan(self) -> CommandPlan:
        """Compatibility alias used by intermediate UI adapters."""

        return self.command

    @property
    def exact_events(self) -> list[dict[str, Any]]:
        return self.events

    @property
    def relaxed_events(self) -> list[dict[str, Any]]:
        return self.near_events

    @property
    def answer(self) -> str:
        return self.message


def _safe_query(query: Any) -> str:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    value = query.strip()
    if not value:
        raise ValueError("query must not be empty")
    if len(value) > MAX_COMMAND_QUERY_LENGTH or _CONTROL_RE.search(value):
        raise ValueError("query is outside the bounded command input")
    return value


def _event_id(event: Mapping[str, Any]) -> str:
    value = event.get("id")
    if value is not None and str(value).strip():
        return str(value).strip()
    url = str(event.get("公式URL", "")).rstrip("/")
    return url.rsplit("/", 1)[-1] if url else ""


def _copy_events(events: Sequence[Mapping[str, Any]], limit: int = MAX_RESULTS) -> list[dict[str, Any]]:
    return [dict(event) for event in list(events)[:limit]]


def _event_reference_text(event: Mapping[str, Any]) -> str:
    return " ".join(
        [
            str(event.get("イベント名", "")),
            *[str(value) for value in event.get("aliases", [])],
            *[str(value) for value in event.get("search_tags", [])],
        ]
    )


def _normalized(value: Any) -> str:
    return event_search.normalize_query(str(value)).replace(" ", "")


class DeterministicAdapters:
    """Trusted adapters from semantic slots to local deterministic tools."""

    def __init__(
        self,
        events: Sequence[Mapping[str, Any]] | None = None,
        reference_date: date = POC_REFERENCE_DATE,
    ) -> None:
        self.events = [dict(event) for event in (events if events is not None else event_search.load_events())]
        self.reference_date = reference_date
        self._by_id = {_event_id(event): event for event in self.events}

    def _tool_filters(self, slots: CommandSlots, flow: str) -> dict[str, Any]:
        parsed = event_search.command_slots_to_search_filters(
            slots,
            flow=flow,
            reference_date=self.reference_date,
        )
        # agent_tools accepts this fixed dictionary.  Do not pass the
        # SearchFilters dataclass wholesale: legacy parser-only fields such as
        # ``keywords`` and ``entity`` are not tool inputs.
        filters: dict[str, Any] = {
            "dates": list(parsed.dates),
            "municipalities": list(slots.municipalities),
            "regions": list(slots.regions),
            "genres": list(parsed.genres),
            "genre_groups": [list(group) for group in parsed.genre_groups],
            "age": parsed.age,
            "age_group": parsed.age_group,
            "age_intent": parsed.age_intent,
            "child_friendly": parsed.child_friendly,
            "venue": parsed.venue,
            "entry_free": parsed.entry_free,
            "paid_only": parsed.paid_only,
            "max_entry_fee": parsed.max_entry_fee,
            "reservation_required": parsed.reservation_required,
            "rain_preferred": parsed.rain_preferred,
            "time_slots": list(parsed.time_slots),
            "time_after": parsed.time_after,
            "soft_terms": list(parsed.soft_terms),
            "reference_date": self.reference_date.isoformat(),
        }
        return {
            key: value
            for key, value in filters.items()
            if value is not None
            and value != []
            and not (value is False and key != "reservation_required")
        }

    def _search(
        self,
        slots: CommandSlots,
        flow: str,
        state: CommandState | None = None,
        *,
        count: bool = False,
    ) -> tuple[ToolResult, dict[str, Any]]:
        filters = self._tool_filters(slots, flow)
        spec = SearchSpec(
            search_id="command-count" if count else "command-search",
            tool="count_events" if count else "search_events",
            purpose="exact",
            filters=filters,
        )
        if slots.refine_previous:
            previous_ids = set(state.last_result_ids if state is not None else ())
            if not previous_ids:
                return (
                    ToolResult(
                        search_id=result.search_id,
                        purpose=result.purpose,
                        total_matches=0,
                        message="前回の検索結果がないため、その中からは絞り込めません。",
                    ),
                    filters,
                )
            source_events = [
                self._by_id[event_id]
                for event_id in state.last_result_ids
                if event_id in self._by_id
            ]
        else:
            source_events = self.events
        result = agent_tools.execute_tool(spec, source_events, self.reference_date)
        return result, filters

    def resolve_reference(self, slots: CommandSlots, state: CommandState) -> dict[str, Any] | None:
        kind = slots.reference_kind
        if kind == "event_name" or slots.event_name:
            needle = _normalized(slots.event_name or "")
            if not needle:
                return None
            exact: list[dict[str, Any]] = []
            containing: list[dict[str, Any]] = []
            for event in self.events:
                labels = [
                    str(event.get("イベント名", "")),
                    *[str(value) for value in event.get("aliases", [])],
                    *[str(value) for value in event.get("search_tags", [])],
                ]
                if any(needle == _normalized(label) for label in labels):
                    exact.append(event)
                if needle in _normalized(_event_reference_text(event)):
                    containing.append(event)
            if len(exact) == 1:
                return exact[0]
            if len(exact) > 1:
                return None
            return containing[0] if len(containing) == 1 else None

        if kind in {"ordinal", "last_result"} or slots.reference_index is not None:
            index = slots.reference_index or 1
            if 1 <= index <= len(state.last_result_ids):
                return self._by_id.get(state.last_result_ids[index - 1])
            return None

        if kind == "selected" or state.selected_event_id:
            return self._by_id.get(state.selected_event_id or "")
        return None

    def detail(self, slots: CommandSlots, state: CommandState) -> tuple[ToolResult, dict[str, Any] | None]:
        selected = self.resolve_reference(slots, state)
        spec = SearchSpec(
            search_id="command-detail",
            tool="get_event_detail",
            purpose="detail",
            filters={"event_ids": [_event_id(selected)]} if selected else {},
        )
        result = agent_tools.execute_tool(spec, self.events, self.reference_date)
        return result, selected

    def next_events(self, slots: CommandSlots, state: CommandState) -> tuple[ToolResult, dict[str, Any] | None]:
        selected = self.resolve_reference(slots, state)
        if selected is None:
            spec = SearchSpec("command-next", "recommend_next_events", "recommendation", {})
            return agent_tools.execute_tool(spec, self.events, self.reference_date), None
        day = self.reference_date
        if slots.dates:
            day = date.fromisoformat(slots.dates[0])
        selected_end_override = None
        if slots.time_after is not None and str(selected.get("参加形式")) == event_recommendation.DROP_IN_ENTRY:
            hours, minutes = divmod(slots.time_after, 60)
            selected_end_override = datetime.combine(day, clock_time(hours, minutes))
        recommendation = event_recommendation.recommend_next_events(
            selected,
            self.events,
            day,
            selected_end_override=selected_end_override,
        )
        result = ToolResult(
            search_id="command-next",
            purpose="recommendation",
            total_matches=len(recommendation.events),
            events=_copy_events(recommendation.events),
            all_event_ids=[_event_id(event) for event in recommendation.events],
            message=recommendation.message,
        )
        return result, selected

    def similar_events(self, slots: CommandSlots, state: CommandState) -> tuple[ToolResult, dict[str, Any] | None]:
        selected = self.resolve_reference(slots, state)
        if selected is None:
            spec = SearchSpec("command-similar", "recommend_similar_events", "recommendation", {})
            return agent_tools.execute_tool(spec, self.events, self.reference_date), None
        preferences: dict[str, Any] = {}
        parsed = event_search.command_slots_to_search_filters(
            slots, flow="recommend_similar", reference_date=self.reference_date
        )
        if parsed.child_friendly is True:
            preferences["child_friendly"] = True
        if parsed.entry_free is True:
            preferences["entry_free"] = True
        if slots.municipalities:
            preferences["scope"] = "city"
        elif slots.regions:
            preferences["scope"] = "region"
        recommendation = event_recommendation.recommend_similar_events(
            selected,
            self.events,
            self.reference_date,
            preferences=preferences,
        )
        result = ToolResult(
            search_id="command-similar",
            purpose="recommendation",
            total_matches=len(recommendation.events),
            events=_copy_events(recommendation.events),
            all_event_ids=[_event_id(event) for event in recommendation.events],
            message=recommendation.message,
        )
        return result, selected

    def pair_events(self, slots: CommandSlots) -> tuple[list[EventPair], list[dict[str, Any]], dict[str, Any]]:
        if len(slots.dates) != 1:
            raise ValueError("plan_event_pair requires exactly one date")
        day = date.fromisoformat(slots.dates[0])
        filters = self._tool_filters(slots, "plan_event_pair")
        pairs = recommend_event_pairs(
            self.events,
            day,
            municipalities=slots.municipalities,
            filters=filters,
            limit=MAX_PAIR_RESULTS,
            same_city_buffer_minutes=DEFAULT_PAIR_BUFFER_MINUTES,
        )
        flattened: list[dict[str, Any]] = []
        seen: set[str] = set()
        for pair in pairs:
            for event_id in (pair.first_event_id, pair.second_event_id):
                if event_id not in seen and event_id in self._by_id:
                    flattened.append(dict(self._by_id[event_id]))
                    seen.add(event_id)
        return pairs, flattened, filters

    def faq(self, query: str) -> ToolResult:
        match = faq_search.find_faq(query)
        return ToolResult(
            search_id="command-faq",
            purpose="faq",
            total_matches=1 if match else 0,
            message=match.answer if match else "一般FAQに該当する回答は見つかりませんでした。",
        )


class FlowDispatcher:
    """Map registry flow names to fixed Python executors."""

    def __init__(self, adapters: DeterministicAdapters) -> None:
        self.adapters = adapters

    def dispatch(
        self,
        plan: CommandPlan,
        state: CommandState,
        query: str,
    ) -> tuple[ToolResult | None, list[EventPair], list[dict[str, Any]], dict[str, Any] | None, str]:
        # Validate against the trusted registry before any executor is chosen.
        spec = get_flow_spec(plan.flow)
        if spec.executor_name == "none":
            return None, [], [], None, "このPoCは文化祭イベントの検索・参加案内が中心です。"
        executor_name = spec.executor_name
        if executor_name in {"search_events", "count_events"}:
            result, filters = self.adapters._search(
                plan.slots,
                plan.flow,
                state,
                count=executor_name == "count_events",
            )
            return result, [], [], filters, result.message
        if executor_name == "get_event_detail":
            result, _ = self.adapters.detail(plan.slots, state)
            return result, [], [], None, result.message
        if executor_name == "recommend_next_events":
            result, _ = self.adapters.next_events(plan.slots, state)
            return result, [], [], None, result.message
        if executor_name == "recommend_similar_events":
            result, _ = self.adapters.similar_events(plan.slots, state)
            return result, [], [], None, result.message
        if executor_name == "recommend_event_pairs":
            pairs, events, filters = self.adapters.pair_events(plan.slots)
            message = (
                "同日・時間順で組み合わせられる候補です。"
                "実際の移動時間ではなく、同市町30分・同地域60分のPoC仮定で判定しています。"
                if pairs
                else "同日・簡易バッファで組み合わせられる候補は見つかりませんでした。"
            )
            return None, pairs, events, filters, message
        if executor_name == "search_faq":
            result = self.adapters.faq(query)
            return result, [], [], None, result.message
        raise CommandValidationError(f"unhandled registered flow {plan.flow!r}", path="flow")


class CommandOrchestrator:
    """Run one bounded command-generation and deterministic-execution turn."""

    def __init__(
        self,
        modal_call: Callable[[Mapping[str, Any]], Any] | None = None,
        *,
        reference_date: date = POC_REFERENCE_DATE,
        events: Sequence[Mapping[str, Any]] | None = None,
        output_format: str = DEFAULT_COMMAND_FORMAT,
    ) -> None:
        self.modal_call = modal_call
        self.reference_date = reference_date
        self.adapters = DeterministicAdapters(events, reference_date)
        self.dispatcher = FlowDispatcher(self.adapters)
        self.output_format = output_format

    @staticmethod
    def _security_guard(query: str) -> str | None:
        intent = event_search.classify_intent(query)
        if intent == "injection":
            return "この案内の制約を変更したり、掲載されていないイベントを作ったりはできません。掲載済みのイベントから探してみて。"
        if intent == "out_of_scope":
            return "このPoCは文化祭イベントの検索・参加案内が中心です。"
        return None

    @staticmethod
    def _reset_query(query: str) -> bool:
        return event_search.normalize_query(query).lower() in {"リセット", "会話をリセット", "reset"}

    def _unsupported_plan(self) -> CommandPlan:
        return CommandPlan(flow="unsupported", slots=CommandSlots(), confidence="low")

    def _result(
        self,
        *,
        plan: CommandPlan,
        status: str,
        message: str = "",
        handled: bool = True,
        **kwargs: Any,
    ) -> CommandTurnResult:
        return CommandTurnResult(
            status=status,
            command=plan,
            flow=plan.flow,
            slots=plan.slots.to_dict(),
            message=message,
            handled=handled,
            **kwargs,
        )

    @staticmethod
    def _pending_date_base(state: CommandState) -> CommandPlan | None:
        if state.active_flow not in {"plan_event_pair", "recommend_next"} or state.last_command is None:
            return None
        pending_slots = state.pending_slots
        if pending_slots.get("dates"):
            return None
        try:
            pending = parse_command_plan(state.last_command)
        except (CommandValidationError, TypeError, ValueError):
            return None
        return None if pending.slots.dates else pending

    def _pending_date_fast_path(
        self,
        query: str,
        state: CommandState,
    ) -> CommandPlan | None:
        pending = self._pending_date_base(state)
        if pending is None:
            return None
        parsed = event_recommendation.parse_recommendation_date_answer(query, state.reference_date)
        if not parsed.is_date_like or parsed.invalid or parsed.value is None:
            return None
        values = pending.slots.to_dict()
        values["dates"] = [parsed.value.isoformat()]
        return CommandPlan(
            flow=pending.flow,
            slots=CommandSlots.from_dict(values),
            confidence=pending.confidence,
        )

    @staticmethod
    def _pending_time_base(state: CommandState) -> CommandPlan | None:
        if state.active_flow != "recommend_next" or state.last_command is None:
            return None
        try:
            pending = parse_command_plan(state.last_command)
        except (CommandValidationError, TypeError, ValueError):
            return None
        if not pending.slots.dates or state.pending_slots.get("time_after") is not None:
            return None
        return pending

    def _pending_time_fast_path(
        self,
        query: str,
        state: CommandState,
    ) -> CommandPlan | None:
        pending = self._pending_time_base(state)
        if pending is None:
            return None
        parsed = event_recommendation.parse_recommendation_time_answer(query)
        if not parsed.is_time_like or parsed.invalid or parsed.value is None:
            return None
        values = pending.slots.to_dict()
        values["time_after"] = parsed.value.hour * 60 + parsed.value.minute
        return CommandPlan(
            flow=pending.flow,
            slots=CommandSlots.from_dict(values),
            confidence=pending.confidence,
        )

    @staticmethod
    def _dynamic_missing(plan: CommandPlan, state: CommandState, adapters: DeterministicAdapters) -> tuple[str, ...]:
        required: list[str] = []
        for slot_name in required_slots_for(plan.flow):
            if slot_name == "dates" and not plan.slots.dates:
                required.append("dates")
        if plan.flow in {"event_detail", "recommend_next", "recommend_similar"}:
            if adapters.resolve_reference(plan.slots, state) is None:
                # This is a clarification rather than a fabricated reference.
                required.append("reference")
        if plan.flow == "recommend_next":
            selected = adapters.resolve_reference(plan.slots, state)
            if selected is not None:
                schedule = event_details.normalize_schedule(selected)
                if schedule.start_date != schedule.end_date and not plan.slots.dates:
                    required.append("dates")
                if (
                    str(selected.get("参加形式")) == event_recommendation.DROP_IN_ENTRY
                    and plan.slots.time_after is None
                ):
                    required.append("time_after")
        return tuple(dict.fromkeys(required))

    def _generation(self, query: str, state: CommandState) -> tuple[CommandPlan | None, CommandGenerationResult | None, float]:
        started = time.perf_counter()
        if self.modal_call is None:
            elapsed = (time.perf_counter() - started) * 1000
            return None, None, elapsed
        generated = generate_command(
            query,
            {
                "reference_date": state.reference_date.isoformat(),
                "selected_event_id": state.selected_event_id,
                "last_result_ids": list(state.last_result_ids),
                "last_command": dict(state.last_command) if state.last_command else None,
                "active_flow": state.active_flow,
                "pending_slots": dict(state.pending_slots),
            },
            call=self.modal_call,
            output_format=self.output_format,
        )
        elapsed = (time.perf_counter() - started) * 1000
        if generated.error and generated.plan.flow == "unsupported":
            return None, generated, elapsed
        return generated.plan, generated, elapsed

    def handle_query(
        self,
        query: str,
        state: Mapping[str, Any] | None = None,
        *,
        command_plan: CommandPlan | Mapping[str, Any] | None = None,
    ) -> CommandTurnResult:
        started = time.perf_counter()
        value = _safe_query(query)
        context = CommandState.from_mapping(state, self.reference_date)

        if self._reset_query(value):
            plan = self._unsupported_plan()
            return self._result(
                plan=plan,
                status="reset",
                message="会話をリセットしました。",
                latency=CommandLatency(total_ms=(time.perf_counter() - started) * 1000),
            )

        security_message = self._security_guard(value)
        if security_message is not None:
            plan = self._unsupported_plan()
            return self._result(
                plan=plan,
                status="unsupported",
                message=security_message,
                latency=CommandLatency(total_ms=(time.perf_counter() - started) * 1000),
            )

        generation_ms = 0.0
        generated: CommandGenerationResult | None = None
        if command_plan is not None:
            try:
                plan = validate_command_plan(command_plan)
            except (CommandValidationError, TypeError, ValueError):
                return self._result(
                    plan=self._unsupported_plan(),
                    status="invalid_command",
                    message="検索条件を確認できませんでした。",
                    handled=False,
                    latency=CommandLatency(total_ms=(time.perf_counter() - started) * 1000),
                )
        else:
            pending_base = self._pending_date_base(context)
            pending_plan = self._pending_date_fast_path(value, context)
            pending_time_base = self._pending_time_base(context)
            pending_time_plan = self._pending_time_fast_path(value, context)
            if pending_plan is not None:
                plan = pending_plan
                generated = None
            elif pending_base is not None:
                parsed_pending = event_recommendation.parse_recommendation_date_answer(
                    value, context.reference_date
                )
                if parsed_pending.is_date_like:
                    # Date-shaped pending replies are consumed locally even
                    # when invalid; never spend a second LLM call on a date
                    # parser problem.
                    question = (
                        "日付を解釈できませんでした。例：11月3日、11/3、3日"
                        if parsed_pending.invalid
                        else "何日に行く予定か教えてみて。"
                    )
                    pending = {
                        "flow": pending_base.flow,
                        "command": pending_base.to_dict(),
                        "missing_slots": ["dates"],
                        "awaiting": "date",
                    }
                    return self._result(
                        plan=pending_base,
                        status="clarification",
                        message=question,
                        question=question,
                        pending=pending,
                        latency=CommandLatency(
                            total_ms=(time.perf_counter() - started) * 1000
                        ),
                    )
                plan, generated, generation_ms = self._generation(value, context)
                if plan is None:
                    return self._result(
                        plan=self._unsupported_plan(),
                        status="unavailable",
                        message="",
                        handled=False,
                        attempts=generated.attempts if generated else 0,
                        repaired=generated.repaired if generated else False,
                        latency=CommandLatency(
                            generator_ms=generation_ms,
                            total_ms=(time.perf_counter() - started) * 1000,
                            generator_calls=generated.attempts if generated else 0,
                            repair_calls=max(0, (generated.attempts if generated else 0) - 1),
                        ),
                    )
            elif pending_time_plan is not None:
                plan = pending_time_plan
                generated = None
            elif pending_time_base is not None:
                parsed_time = event_recommendation.parse_recommendation_time_answer(value)
                if parsed_time.is_time_like:
                    question = (
                        "時刻を解釈できませんでした。例：13時、13:30"
                        if parsed_time.invalid
                        else "何時ごろ見終わる予定？"
                    )
                    pending = {
                        "flow": pending_time_base.flow,
                        "command": pending_time_base.to_dict(),
                        "missing_slots": ["time_after"],
                        "awaiting": "time",
                    }
                    return self._result(
                        plan=pending_time_base,
                        status="clarification",
                        message=question,
                        question=question,
                        pending=pending,
                        latency=CommandLatency(
                            total_ms=(time.perf_counter() - started) * 1000
                        ),
                    )
                plan, generated, generation_ms = self._generation(value, context)
                if plan is None:
                    return self._result(
                        plan=self._unsupported_plan(),
                        status="unavailable",
                        message="",
                        handled=False,
                        attempts=generated.attempts if generated else 0,
                        repaired=generated.repaired if generated else False,
                        latency=CommandLatency(
                            generator_ms=generation_ms,
                            total_ms=(time.perf_counter() - started) * 1000,
                            generator_calls=generated.attempts if generated else 0,
                            repair_calls=max(0, (generated.attempts if generated else 0) - 1),
                        ),
                    )
            else:
                plan, generated, generation_ms = self._generation(value, context)
                if plan is None:
                    # Modal failure is not a semantic unsupported answer; the
                    # caller may route the turn to the conservative legacy path.
                    return self._result(
                        plan=self._unsupported_plan(),
                        status="unavailable",
                        message="",
                        handled=False,
                        attempts=generated.attempts if generated else 0,
                        repaired=generated.repaired if generated else False,
                        latency=CommandLatency(
                            generator_ms=generation_ms,
                            total_ms=(time.perf_counter() - started) * 1000,
                            generator_calls=generated.attempts if generated else 0,
                            repair_calls=max(0, (generated.attempts if generated else 0) - 1),
                        ),
                    )

        missing = self._dynamic_missing(plan, context, self.adapters)
        if missing:
            if "dates" in missing:
                question = "何日に行く予定か教えてみて。"
            elif "time_after" in missing:
                question = "何時ごろ見終わる予定？"
            elif "reference" in missing:
                question = "基準にするイベントを番号かイベント名で教えてみて。"
            else:
                question = "必要な条件をもう少し教えてみて。"
            pending = {
                "flow": plan.flow,
                "command": plan.to_dict(),
                "missing_slots": list(missing),
                "awaiting": (
                    "date"
                    if "dates" in missing
                    else "time"
                    if "time_after" in missing
                    else "reference"
                ),
            }
            return self._result(
                plan=plan,
                status="clarification",
                message=question,
                question=question,
                pending=pending,
                attempts=generated.attempts if generated else 0,
                repaired=generated.repaired if generated else False,
                latency=CommandLatency(
                    generator_ms=generation_ms,
                    total_ms=(time.perf_counter() - started) * 1000,
                    generator_calls=generated.attempts if generated else 0,
                    repair_calls=max(0, (generated.attempts if generated else 0) - 1),
                ),
            )

        execution_started = time.perf_counter()
        try:
            result, pairs, events, filters, message = self.dispatcher.dispatch(plan, context, value)
        except (CommandValidationError, TypeError, ValueError, KeyError) as exc:
            return self._result(
                plan=self._unsupported_plan(),
                status="execution_error",
                message="検索条件を確認できませんでした。",
                handled=False,
                attempts=generated.attempts if generated else 0,
                repaired=generated.repaired if generated else False,
                latency=CommandLatency(
                    generator_ms=generation_ms,
                    execution_ms=(time.perf_counter() - execution_started) * 1000,
                    total_ms=(time.perf_counter() - started) * 1000,
                    generator_calls=generated.attempts if generated else 0,
                    repair_calls=max(0, (generated.attempts if generated else 0) - 1),
                ),
            )
        execution_ms = (time.perf_counter() - execution_started) * 1000
        if result is not None:
            events = result.events
            total_matches = result.total_matches
            if not message:
                message = result.message
        else:
            total_matches = len(events)
        latency = CommandLatency(
            generator_ms=generation_ms,
            execution_ms=execution_ms,
            total_ms=(time.perf_counter() - started) * 1000,
            generator_calls=generated.attempts if generated else 0,
            repair_calls=max(0, (generated.attempts if generated else 0) - 1),
        )
        return self._result(
            plan=plan,
            status="ok",
            result=result,
            events=_copy_events(events),
            pairs=list(pairs),
            total_matches=total_matches,
            filters=filters,
            message=message,
            attempts=generated.attempts if generated else 0,
            repaired=generated.repaired if generated else False,
            latency=latency,
        )


def handle_command_query(
    query: str,
    state: Mapping[str, Any] | None = None,
    *,
    modal_call: Callable[[Mapping[str, Any]], Any] | None = None,
    reference_date: date = POC_REFERENCE_DATE,
    command_plan: CommandPlan | Mapping[str, Any] | None = None,
    events: Sequence[Mapping[str, Any]] | None = None,
    output_format: str = DEFAULT_COMMAND_FORMAT,
    **_: Any,
) -> CommandTurnResult:
    """Functional entrypoint for tests and lightweight integrations."""

    return CommandOrchestrator(
        modal_call,
        reference_date=reference_date,
        events=events,
        output_format=output_format,
    ).handle_query(query, state, command_plan=command_plan)


handle_command = handle_command_query
run_command = handle_command_query
execute_command = handle_command_query


__all__ = [
    "CommandLatency",
    "CommandOrchestrator",
    "CommandState",
    "CommandTurnResult",
    "DeterministicAdapters",
    "FlowDispatcher",
    "execute_command",
    "handle_command",
    "handle_command_query",
    "run_command",
]
