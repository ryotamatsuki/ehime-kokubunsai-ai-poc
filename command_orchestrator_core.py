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
import conversation_recovery
import event_details
import event_recommendation
import event_search
import experience_preferences
import faq_search
from agent_models import SearchSpec, ToolResult
from app_config import MAX_RESULT_SET_SIZE, POC_REFERENCE_DATE
from command_generator import (
    CommandGenerationResult,
    DEFAULT_COMMAND_FORMAT,
    generate_command,
)
from command_observability import TurnObservation
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
# These are bounded data/state limits, not the first-page card count.
MAX_LAST_RESULT_IDS = MAX_RESULT_SET_SIZE
# Compatibility alias for intermediate adapters; result copying uses the
# explicit result-set constant below.
MAX_RESULTS = MAX_RESULT_SET_SIZE
MAX_PAIR_RESULTS = 3
DEFAULT_PAIR_BUFFER_MINUTES = 30
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")

# These are semantic feature groups, not a phrase dictionary.  The guard only
# handles the small set of pair requests whose structure is unambiguous; all
# other language remains the responsibility of the Semantic Command model.
_PAIR_MULTIPLE_MARKERS = (
    "2つ",
    "二つ",
    "2か所",
    "二か所",
    "2箇所",
    "二箇所",
    "複数",
    "いくつか",
    "何個か",
    "1つずつ",
)
_PAIR_ACTION_MARKERS = (
    "回る",
    "回りたい",
    "回れる",
    "巡る",
    "巡りたい",
    "はしご",
    "行く",
    "行きたい",
    "行ける",
    "参加",
    "見たい",
    "見られる",
)
_PAIR_VETO_MARKERS = (
    "2番目",
    "二番目",
    "2つ目",
    "二つ目",
    "2件",
    "二件",
    "2人",
    "二人",
    "2歳",
    "二歳",
    "小2",
    "小学2年",
)


def _flatten_groups(groups: Any) -> list[str]:
    values: list[str] = []
    if not isinstance(groups, (list, tuple)):
        return values
    for group in groups:
        items = group if isinstance(group, (list, tuple)) else (group,)
        for item in items:
            value = str(item).strip()
            if value and value not in values:
                values.append(value)
    return values


def _high_confidence_pair_plan(
    query: str,
    reference_date: date,
) -> CommandPlan | None:
    """Build a trusted pair plan only for an unambiguous multi-visit utterance.

    This function deliberately runs after the security guard and before LLM
    generation.  It extracts only deterministic, contract-valid slots; event
    existence and pair feasibility remain in the trusted executor.
    """

    if event_search.classify_intent(query) == "injection":
        return None
    normalized = event_search.normalize_query(query).replace(" ", "")
    if any(marker in normalized for marker in _PAIR_VETO_MARKERS):
        return None
    has_multiple = any(marker in normalized for marker in _PAIR_MULTIPLE_MARKERS)
    has_action = any(marker in normalized for marker in _PAIR_ACTION_MARKERS)
    has_explicit_pair = "はしご" in normalized or (
        "午前と午後" in normalized and ("1つずつ" in normalized or has_action)
    )
    if not has_explicit_pair and not (has_multiple and has_action):
        return None

    try:
        parsed = event_search.parse_query_strict(query, reference_date)
        dates = list(parsed.dates) if len(parsed.dates) == 1 else []
        topics = [
            term
            for term in event_search.topics_to_soft_terms(parsed.soft_terms)
            if term not in parsed.genres
        ]
        pair_time_slots = list(parsed.time_slots)
        if set(pair_time_slots) >= {"午前", "午後"}:
            # "午前と午後に1つずつ" describes the pair as a whole.  The
            # existing pair executor applies time_slots to each candidate, so
            # passing both values would incorrectly require every event to
            # satisfy both halves of the visit plan.
            pair_time_slots = []
        slots = CommandSlots.from_dict(
            {
                "dates": dates,
                "municipalities": _flatten_groups(parsed.city_groups),
                "regions": _flatten_groups(parsed.region_groups),
                "genres": list(parsed.genres),
                "topics": topics,
                "experience_required": list(parsed.experience_required),
                "experience_preferred": list(parsed.experience_preferred),
                "experience_excluded": list(parsed.experience_excluded),
                "audience": (
                    "family"
                    if parsed.child_friendly is True
                    and parsed.age is None
                    and parsed.age_group is None
                    else None
                ),
                "age": parsed.age,
                "age_group": parsed.age_group,
                "age_intent": parsed.age_intent,
                "venue": {
                    "屋内": "indoor",
                    "屋外": "outdoor",
                }.get(parsed.venue),
                "entry_free": parsed.entry_free,
                "paid_only": True if parsed.paid_only else None,
                "max_entry_fee": parsed.max_entry_fee,
                "reservation_required": parsed.reservation_required,
                "rain_preferred": True if parsed.rain_preferred else None,
                "time_slots": pair_time_slots,
                "time_after": parsed.time_after,
                "visit_count": 2,
            }
        )
        return CommandPlan("plan_event_pair", slots, confidence="high")
    except (CommandValidationError, TypeError, ValueError):
        # A guard must fail open to the normal Semantic Command path when its
        # deterministic extraction cannot satisfy the existing contract.
        return None


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
    last_search_context: conversation_recovery.SearchContext | None = None
    # Recovery metadata only.  The actual SearchContext and public evidence
    # remain in Python/Streamlit state and are never sent to the generator.
    last_action: str | None = None
    has_last_search_context: bool = False
    last_result_count: int = 0

    @property
    def requested_slot(self) -> str | None:
        """The authoritative slot currently requested by the active flow."""

        return self.pending_required_slots[0] if self.pending_required_slots else None

    @property
    def slots(self) -> Mapping[str, Any]:
        """The active flow's already-grounded slots.

        ``pending_slots`` remains the single stored representation.  This
        alias makes the explicit ConversationState contract readable without
        introducing a second mutable slot dictionary.
        """

        return self.pending_slots

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
        last_action = raw.get("last_action")
        last_action = str(last_action)[:64] if isinstance(last_action, str) else None
        last_search_context = conversation_recovery.SearchContext.from_value(
            raw.get("last_search_context")
        )
        # The boolean is useful semantic metadata, but Python must validate
        # recovery against the actual trusted context object. A forged or
        # stale flag alone must never unlock an explanation path.
        has_context = last_search_context is not None
        raw_count = raw.get("last_result_count", len(result_ids))
        if isinstance(raw_count, bool) or not isinstance(raw_count, int):
            raw_count = len(result_ids)
        last_result_count = max(0, min(raw_count, MAX_RESULT_SET_SIZE))
        return cls(
            reference_date=reference_date,
            selected_event_id=selected_id,
            last_result_ids=result_ids,
            last_command=dict(last_command) if last_command is not None else None,
            active_flow=active_flow,
            pending_slots=dict(pending_slots),
            pending_required_slots=required_slots,
            last_search_context=last_search_context,
            last_action=last_action,
            has_last_search_context=has_context,
            last_result_count=(
                last_search_context.total_matches
                if last_search_context is not None
                else last_result_count
            ),
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
    all_event_ids: list[str] = field(default_factory=list)
    all_near_event_ids: list[str] = field(default_factory=list)
    pairs: list[EventPair] = field(default_factory=list)
    total_matches: int | None = None
    filters: dict[str, Any] | None = None
    pending: dict[str, Any] | None = None
    question: str | None = None
    message: str = ""
    attempts: int = 0
    repaired: bool = False
    latency: CommandLatency = field(default_factory=CommandLatency)
    observability: dict[str, Any] = field(default_factory=dict)
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


def _copy_events(
    events: Sequence[Mapping[str, Any]],
    limit: int = MAX_RESULT_SET_SIZE,
) -> list[dict[str, Any]]:
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
            "experience_required": list(parsed.experience_required),
            "experience_preferred": list(parsed.experience_preferred),
            "experience_excluded": list(parsed.experience_excluded),
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
                        search_id=spec.search_id,
                        purpose=spec.purpose,
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

    def resolve_context_reference(
        self,
        slots: CommandSlots,
        state: CommandState,
    ) -> dict[str, Any] | None:
        """Resolve an explanation target only inside the active SearchContext.

        The complete ordered ``result_ids`` list is the source of truth. The
        visible card page and the global event catalog are never allowed to
        widen a follow-up reference.
        """

        context = state.last_search_context
        if context is None:
            return None
        allowed_ids = set(context.result_ids)
        if not allowed_ids:
            return None

        if slots.reference_index is not None or slots.reference_kind in {
            "ordinal",
            "last_result",
        }:
            index = slots.reference_index or 1
            if 1 <= index <= len(context.result_ids):
                return self._by_id.get(context.result_ids[index - 1])
            return None

        event = self.resolve_reference(slots, state)
        if event is None or _event_id(event) not in allowed_ids:
            return None
        return event

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
        schedule = event_details.normalize_schedule(selected)
        day = self.reference_date
        if slots.dates:
            day = date.fromisoformat(slots.dates[0])
        elif schedule.start_date == schedule.end_date:
            # A selected single-day event is itself authoritative for the
            # recommendation day. Never substitute the PoC current date.
            day = schedule.start_date
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
        if executor_name == "explain_search":
            message = conversation_recovery.render_search_explanation(
                state.last_search_context
            )
            return (
                ToolResult(
                    search_id="command-explain-search",
                    purpose="recovery",
                    total_matches=state.last_result_count,
                    message=message,
                ),
                [],
                [],
                None,
                message,
            )
        if executor_name == "explain_result":
            selected = self.adapters.resolve_context_reference(plan.slots, state)
            if selected is None:
                message = conversation_recovery.render_result_explanation(
                    state.last_search_context,
                    None,
                    query=query,
                )
                return (
                    ToolResult(
                        search_id="command-explain-result",
                        purpose="recovery",
                        total_matches=0,
                        message=message,
                    ),
                    [],
                    [],
                    None,
                    message,
                )
            event = dict(selected)
            message = conversation_recovery.render_result_explanation(
                state.last_search_context,
                event,
                query=query,
            )
            return (
                ToolResult(
                    search_id="command-explain-result",
                    purpose="recovery",
                    total_matches=1,
                    events=[event],
                    all_event_ids=[_event_id(event)],
                    message=message,
                ),
                [],
                [],
                None,
                message,
            )
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
        self._current_observation: TurnObservation | None = None

    @staticmethod
    def _security_guard(query: str) -> str | None:
        intent = event_search.classify_intent(query)
        if intent == "injection":
            return "この案内の制約を変更したり、掲載されていないイベントを作ったりはできません。掲載済みのイベントから探してみて。"
        if intent == "out_of_scope" or conversation_recovery.is_domain_out_of_scope(query):
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
        observation = self._current_observation
        if observation is not None:
            observation.finish(flow=plan.flow, status=status)
            if observation.has_search_context is False:
                observation.has_search_context = False
            if status == "clarification" and plan.flow in {"explain_search", "explain_result"}:
                if not observation.has_search_context:
                    observation.mark_fallback("missing_search_context", error_type="missing_search_context")
            if status == "execution_error":
                observation.mark_fallback("execution_error", error_type="execution_error")
            if status == "unavailable":
                observation.mark_fallback(
                    observation.semantic_command_error_type or "semantic_command_unavailable",
                    error_type=observation.semantic_command_error_type,
                )
            kwargs.setdefault("observability", observation.emit())
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
    def _reconcile_generated_slots(
        plan: CommandPlan,
        query: str,
        state: CommandState,
    ) -> CommandPlan:
        """Keep generated slots grounded in the current user utterance.

        The model receives ``reference_date`` as context, but that context is
        not a user-specified date.  In particular, a date-free pair request
        must ask for a day instead of silently planning against the PoC's
        current date.  Explicit dates and high-confidence fee conditions are
        parsed locally and take precedence over omitted or model-generated
        values.
        """

        if plan.flow in {"explain_search", "explain_result"}:
            # Recovery semantics and references come from the command plan;
            # do not reinterpret an explanation utterance as search filters.
            return plan

        parsed = event_search.parse_query(query, state.reference_date)
        experience_query = experience_preferences.resolve_experience_query(query)
        query_dates = list(parsed.dates)
        if plan.flow == "plan_event_pair":
            # A pair is defined for exactly one day.  A range such as
            # "今週末" therefore remains a clarification rather than being
            # truncated to an arbitrary day.
            grounded_dates = query_dates if len(query_dates) == 1 else []
        elif query_dates:
            grounded_dates = query_dates
        elif plan.slots.refine_previous and state.last_command is not None:
            # Refinement is intentionally scoped to the prior result set; do
            # not erase a date carried by that already-validated context.
            grounded_dates = list(plan.slots.dates)
        else:
            grounded_dates = []

        values = plan.slots.to_dict()
        values["dates"] = grounded_dates
        if experience_preferences.has_release_phrase(query):
            values["experience_required"] = []
            values["experience_preferred"] = []
            values["experience_excluded"] = []
        elif experience_query.recognized:
            # Explicit Japanese modifiers are higher-confidence than an LLM's
            # omission or free-form paraphrase.  Only canonical vocabulary IDs
            # reach CommandSlots, and the event matcher resolves their facts.
            values["experience_required"] = list(experience_query.required)
            values["experience_preferred"] = list(experience_query.preferred)
            values["experience_excluded"] = list(experience_query.excluded)
        explicit_cities = [
            str(city)
            for group in (parsed.city_groups or [])
            if isinstance(group, (list, tuple))
            for city in group
            if str(city).strip()
        ]
        if explicit_cities:
            values["municipalities"] = list(dict.fromkeys(explicit_cities))
        explicit_regions = [
            str(region)
            for group in (parsed.region_groups or [])
            if isinstance(group, (list, tuple))
            for region in group
            if str(region).strip()
        ]
        if explicit_regions:
            values["regions"] = list(dict.fromkeys(explicit_regions))
        normalized_query = event_search.normalize_query(query).replace(" ", "")
        free_is_negated = any(
            phrase in normalized_query
            for phrase in ("無料ではない", "無料でない", "無料じゃない", "無料ではありません")
        )
        if (
            parsed.entry_free is True
            and not free_is_negated
            and "有料" not in normalized_query
            and not plan.slots.paid_only
        ):
            # Do not let a valid-but-incomplete model command silently drop an
            # explicit free-entry condition and return paid events.
            values["entry_free"] = True
            values["paid_only"] = False
        elif parsed.paid_only is True and not plan.slots.entry_free:
            values["paid_only"] = True
            values["entry_free"] = False

        if values == plan.slots.to_dict():
            return plan
        return CommandPlan(
            flow=plan.flow,
            slots=CommandSlots.from_dict(values),
            confidence=plan.confidence,
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
        if plan.flow == "explain_result":
            if adapters.resolve_context_reference(plan.slots, state) is None:
                # Explanation references are narrower than ordinary detail
                # references: a named/global event outside the active result
                # set must not be explained as if it were selected.
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
        if self._current_observation is not None:
            self._current_observation.semantic_command_called = True
        generated = generate_command(
            query,
            {
                "reference_date": state.reference_date.isoformat(),
                "selected_event_id": state.selected_event_id,
                "last_result_ids": list(state.last_result_ids),
                "last_command": dict(state.last_command) if state.last_command else None,
                "active_flow": state.active_flow,
                "pending_slots": dict(state.pending_slots),
                "pending_required_slots": list(state.pending_required_slots),
                "requested_slot": state.requested_slot,
                "last_action": state.last_action,
                "has_last_search_context": state.has_last_search_context,
                "last_result_count": state.last_result_count,
            },
            call=self.modal_call,
            output_format=self.output_format,
        )
        elapsed = (time.perf_counter() - started) * 1000
        if self._current_observation is not None:
            self._current_observation.mark_generation(generated, elapsed)
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
        self._current_observation = TurnObservation(
            value,
            has_search_context=context.last_search_context is not None,
            last_result_count=context.last_result_count,
        )

        if self._reset_query(value):
            self._current_observation.deterministic_route = "reset"
            self._current_observation.deterministic_confidence = "high"
            plan = self._unsupported_plan()
            return self._result(
                plan=plan,
                status="reset",
                message="会話をリセットしました。",
                latency=CommandLatency(total_ms=(time.perf_counter() - started) * 1000),
            )

        security_message = self._security_guard(value)
        if security_message is not None:
            self._current_observation.deterministic_route = "security_or_domain_guard"
            self._current_observation.deterministic_confidence = "high"
            plan = self._unsupported_plan()
            return self._result(
                plan=plan,
                status="unsupported",
                message=security_message,
                latency=CommandLatency(total_ms=(time.perf_counter() - started) * 1000),
            )

        generation_ms = 0.0
        generated: CommandGenerationResult | None = None
        pair_guard_plan = (
            None
            if command_plan is not None
            else _high_confidence_pair_plan(value, context.reference_date)
        )
        if pair_guard_plan is not None:
            self._current_observation.deterministic_route = "pair_fast_path"
            self._current_observation.deterministic_confidence = "high"

        def generate_or_use_guard() -> tuple[
            CommandPlan | None,
            CommandGenerationResult | None,
            float,
        ]:
            if pair_guard_plan is not None:
                return pair_guard_plan, None, 0.0
            return self._generation(value, context)

        if command_plan is not None:
            self._current_observation.deterministic_route = "trusted_command_plan"
            self._current_observation.deterministic_confidence = "high"
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
                self._current_observation.deterministic_route = "pending_date_fast_path"
                self._current_observation.deterministic_confidence = "high"
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
                if (
                    event_recommendation.classify_pending_answer(
                        value,
                        awaiting="date",
                        reference_date=context.reference_date,
                    )
                    == event_recommendation.PENDING_AMBIGUOUS
                ):
                    question = "何日に行く予定か、11/4のように教えてみて。"
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
                plan, generated, generation_ms = generate_or_use_guard()
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
                self._current_observation.deterministic_route = "pending_time_fast_path"
                self._current_observation.deterministic_confidence = "high"
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
                if (
                    event_recommendation.classify_pending_answer(
                        value,
                        awaiting="time",
                        reference_date=context.reference_date,
                    )
                    == event_recommendation.PENDING_AMBIGUOUS
                ):
                    question = "何時ごろ見終わる予定か、13時のように教えてみて。"
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
                plan, generated, generation_ms = generate_or_use_guard()
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
                plan, generated, generation_ms = generate_or_use_guard()
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

        if command_plan is None:
            plan = self._reconcile_generated_slots(plan, value, context)

        if plan.flow in {"explain_search", "explain_result"} and context.last_search_context is None:
            if self._current_observation.semantic_command_error_type is None:
                self._current_observation.fallback_reason = "missing_search_context"
            message = (
                "直前の検索結果がないけん、まずイベントを探してみて。"
                if plan.flow == "explain_search"
                else "直前の検索結果がないけん、まずイベントを探してから根拠を確認してみて。"
            )
            return self._result(
                plan=plan,
                status="clarification",
                message=message,
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
            if self._current_observation is not None:
                self._current_observation.semantic_command_error_type = "execution_error"
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
        all_event_ids = (
            list(result.all_event_ids)
            if result is not None and result.all_event_ids
            else [_event_id(event) for event in events[:MAX_RESULT_SET_SIZE]]
        )
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
            all_event_ids=all_event_ids,
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
