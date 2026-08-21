"""Bounded orchestration for ambiguous event discovery.

This layer is deliberately narrow.  Existing event lookup, participation,
FAQ, recommendation, injection, and out-of-scope routes remain Fast Path
branches.  Only under-specified discovery questions use Planner -> fixed
tools -> optional Replan -> Writer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import age_semantics
import conversation_router
import event_search
from agent_models import AgenticLatency, AgenticResponse, MergedResults, SearchPlan, SearchSpec, ToolResult, WriterOutput
from agent_planner import (
    ModalConfig,
    request_replan,
    request_search_plan,
    request_writer,
    validate_replan_plan,
    validate_search_plan,
)
from agent_tools import execute_tool
from app_config import MAX_RESULT_SET_SIZE, MAX_WRITER_CANDIDATES, POC_REFERENCE_DATE


MAX_PLANNER_ROUNDS = 2
MAX_SEARCHES_PER_ROUND = 3
MAX_TOTAL_SEARCHES = 5


# Planner/Replan keeps machine-readable field names for validation and QA.
# Convert them only at the presentation boundary so internal contract names
# such as ``soft_terms`` never leak into the user-facing response.
_RELAXED_FIELD_LABELS = {
    "dates": "日付条件",
    "municipalities": "市町条件",
    "regions": "地域条件",
    "genres": "ジャンル条件",
    "genre_groups": "ジャンル条件",
    "age": "年齢条件",
    "age_group": "対象年齢条件",
    "child_friendly": "子ども向け条件",
    "venue": "屋内外条件",
    "entry_free": "無料条件",
    "paid_only": "有料条件",
    "max_entry_fee": "料金上限",
    "reservation_required": "申込条件",
    "rain_preferred": "雨天対応条件",
    "time_slots": "時間帯条件",
    "time_after": "開始時刻条件",
    "soft_terms": "趣味・キーワード条件",
}


@dataclass(frozen=True)
class FastDecision:
    can_handle: bool
    reason: str = ""


@dataclass(frozen=True)
class ParserCoverage:
    """Explain which meaningful query constraints the legacy parser covered."""

    complete: bool
    recognized_constraints: tuple[str, ...] = ()
    unresolved_constraints: tuple[str, ...] = ()
    semantic_soft_terms: tuple[str, ...] = ()
    reason: str = ""


def humanize_relaxed_fields(fields: Sequence[str]) -> tuple[str, ...]:
    """Return safe, user-facing labels for machine-readable relaxed fields."""

    labels: list[str] = []
    for field in fields:
        label = _RELAXED_FIELD_LABELS.get(str(field), "一部の条件")
        if label not in labels:
            labels.append(label)
    return tuple(labels)


def assess_parser_coverage(query: str, parsed_filters: event_search.SearchFilters) -> ParserCoverage:
    """Separate structured coverage from residual semantic constraints.

    A constraint can be recognized and still require Agentic Search when the
    legacy matcher cannot safely enforce its meaning (age semantics and exact
    reservation state are the important examples here).
    """

    normalized = event_search.normalize_query(query)
    recognized: list[str] = []
    unresolved: list[str] = []
    semantic_soft_terms = tuple(str(value) for value in parsed_filters.soft_terms)

    if parsed_filters.dates:
        recognized.append("dates")
    if parsed_filters.city_groups:
        recognized.append("municipalities")
    if parsed_filters.region_groups:
        recognized.append("regions")
    if parsed_filters.genre_groups:
        recognized.append("genres")
    if parsed_filters.intent == "count":
        recognized.append("answer_type=count")
    if parsed_filters.age is not None:
        recognized.append(f"age={parsed_filters.age}")
    if parsed_filters.age_group:
        recognized.append(f"age_group={parsed_filters.age_group}")
    if parsed_filters.age_intent:
        recognized.append(f"age_intent={parsed_filters.age_intent}")
    if parsed_filters.age is not None or parsed_filters.age_group:
        unresolved.append("age_semantics")
    elif age_semantics.query_age_semantics(normalized).recognized:
        unresolved.append("age_constraint")

    if parsed_filters.venue:
        recognized.append(f"venue={parsed_filters.venue}")
    if any(term in normalized for term in ("建物の中", "建物内", "建物", "中でやる")):
        # Even if the deterministic parser records 屋内, keep this natural
        # phrase on Agentic Search so the semantic trace is explicit.
        unresolved.append("indoor_semantics")
    if parsed_filters.reservation_required is not None:
        recognized.append(f"reservation_required={parsed_filters.reservation_required}")
    if any(term in normalized for term in ("予約なし", "予約不要", "予約はいらない", "予約いらない", "申込不要", "申し込み不要", "申込なし", "申し込みなし")):
        unresolved.append("reservation_semantics")
    if parsed_filters.child_friendly:
        recognized.append("child_friendly")
    if parsed_filters.entry_free:
        recognized.append("entry_free")
    if parsed_filters.paid_only:
        recognized.append("paid_only")
    if parsed_filters.time_slots or parsed_filters.time_after is not None:
        recognized.append("time")
    if parsed_filters.soft_terms:
        recognized.append("soft_terms")

    unresolved = list(dict.fromkeys(unresolved))
    return ParserCoverage(
        complete=not unresolved,
        recognized_constraints=tuple(dict.fromkeys(recognized)),
        unresolved_constraints=tuple(unresolved),
        semantic_soft_terms=semantic_soft_terms,
        reason="complete" if not unresolved else "semantic coverage incomplete",
    )


def parser_confidence_is_high(parsed_filters: event_search.SearchFilters, query: str) -> bool:
    """Identify questions the existing deterministic parser already handles."""

    if not assess_parser_coverage(query, parsed_filters).complete:
        return False
    intent = event_search.classify_intent(query)
    # Numeric age is intentionally reserved for Agentic Search because the
    # legacy parser has no age-range field.  Other concrete v2 filters below
    # are already deterministic and must stay on the Fast Path.
    normalized_query = event_search.normalize_query(query)
    if age_semantics.query_age_semantics(normalized_query).recognized:
        return False
    # The legacy count classifier does not cover every colloquial count
    # phrase (notably 「どれくらい」), so keep those on Agentic Search where
    # the deterministic count tool preserves the full match total.
    if any(term in normalized_query for term in ("どれくらい", "どのくらい", "どの程度")):
        return False
    if intent in {"count", "lookup", "attribute", "refine"}:
        return True
    if parsed_filters.entity and parsed_filters.requested_field:
        return True
    if parsed_filters.requested_field and (
        parsed_filters.soft_terms or parsed_filters.dates or parsed_filters.city_groups
    ):
        return True
    if parsed_filters.dates and (
        parsed_filters.city_groups
        or parsed_filters.region_groups
        or parsed_filters.genres
        or parsed_filters.venue
        or parsed_filters.entry_free
    ):
        return True
    if (
        parsed_filters.city_groups
        or parsed_filters.region_groups
        or parsed_filters.genres
        or parsed_filters.child_friendly is True
        or parsed_filters.venue
        or parsed_filters.rain_preferred
        or parsed_filters.entry_free is not None
        or parsed_filters.paid_only
        or parsed_filters.max_entry_fee is not None
        or parsed_filters.time_slots
        or parsed_filters.time_after is not None
    ):
        return True
    return False


def looks_like_event_discovery(query: str) -> bool:
    normalized = event_search.normalize_query(query)
    if any(term in normalized for term in ("どれくらい", "どのくらい", "どの程度")):
        return True
    if age_semantics.query_age_semantics(normalized).recognized and any(
        term in normalized for term in ("楽しめる", "おすすめ", "ある", "イベント")
    ):
        return True
    discovery_terms = (
        "イベントある",
        "イベントはある",
        "探して",
        "楽しめる",
        "楽しみたい",
        "おすすめ",
        "何か",
        "どんなイベント",
        "興味がある",
        "好き",
    )
    return any(term in normalized for term in discovery_terms)


def evaluate_fast_path(
    query: str,
    conversation_state: Mapping[str, Any],
    *,
    reference_date: date = POC_REFERENCE_DATE,
) -> FastDecision:
    route = conversation_router.route_conversation(
        query,
        conversation_state.get("last_results", []),
        conversation_state.get("selected_event"),
        conversation_state.get("last_filters"),
        reference_date,
    )
    parsed = event_search.parse_query(query, reference_date)
    if route.action_type in {
        "injection",
        "out_of_scope",
        "reference_followup",
        "detail_followup",
        "general_faq",
        "recommend_next",
        "recommend_next_without_selection",
        "recommend_similar",
        "recommend_similar_without_selection",
        "nearby",
    }:
        return FastDecision(True, route.action_type)
    if parsed.intent in {"injection", "out_of_scope", "needs_location", "needs_region"}:
        return FastDecision(True, parsed.intent)
    if parser_confidence_is_high(parsed, query):
        return FastDecision(True, "high_parser_confidence")
    return FastDecision(False, "agentic_discovery")


def should_use_agentic_search(
    query: str,
    route: conversation_router.ConversationRoute,
    parsed_filters: event_search.SearchFilters,
) -> bool:
    """Return true only for ambiguous discovery, never for sensitive routes."""

    if route.action_type in {
        "injection",
        "scope_search",
        "out_of_scope",
        "reference_followup",
        "detail_followup",
        "general_faq",
        "recommend_next",
        "recommend_similar",
        "recommend_next_without_selection",
        "recommend_similar_without_selection",
        "nearby",
    }:
        return False
    if route.action_type not in {"search", "generic_scope"}:
        return False
    if parsed_filters.entity and parsed_filters.requested_field:
        return False
    coverage = assess_parser_coverage(query, parsed_filters)
    if coverage.complete and parser_confidence_is_high(parsed_filters, query):
        return False
    if event_search.classify_intent(query) in {"injection", "out_of_scope"}:
        return False
    return event_search.looks_like_event_query(query) or looks_like_event_discovery(query)


def _event_id(event: Mapping[str, Any]) -> str:
    return str(event.get("id") or str(event["公式URL"]).rstrip("/").rsplit("/", 1)[-1])


def merge_agent_results(results: Sequence[ToolResult]) -> MergedResults:
    exact_by_id: dict[str, dict[str, Any]] = {}
    relaxed_by_id: dict[str, dict[str, Any]] = {}
    for result in results:
        target = relaxed_by_id if result.relaxed else exact_by_id
        for event in result.events:
            event_id = _event_id(event)
            if result.relaxed:
                if event_id not in exact_by_id:
                    relaxed_by_id[event_id] = dict(event)
            else:
                exact_by_id[event_id] = dict(event)
                relaxed_by_id.pop(event_id, None)
    exact = list(exact_by_id.values())
    relaxed = list(relaxed_by_id.values())
    return MergedResults(
        exact_events=exact,
        relaxed_events=relaxed,
        # The Writer sees only a bounded candidate budget.  The complete
        # ordered exact/relaxed sets remain available to the UI and state.
        display_candidates=(exact + relaxed)[:MAX_WRITER_CANDIDATES],
    )


def should_replan(plan: SearchPlan, results: Sequence[ToolResult], planner_round: int) -> bool:
    if planner_round >= MAX_PLANNER_ROUNDS - 1:
        return False
    if len(results) >= MAX_TOTAL_SEARCHES:
        return False
    exact_matches = sum(result.total_matches for result in results if not result.relaxed)
    if exact_matches != 0:
        return False
    return plan.allow_replan and any("soft_terms" in spec.filters for spec in plan.searches)


def summarize_tool_results(results: Sequence[ToolResult]) -> dict[str, Any]:
    return {
        "searches": [
            {
                "search_id": result.search_id,
                "purpose": result.purpose,
                "total_matches": result.total_matches,
                "event_ids": result.all_event_ids[:MAX_RESULT_SET_SIZE],
                "relaxed": result.relaxed,
                "relaxed_fields": list(result.relaxed_fields),
                "strong_event_ids": list(result.strong_event_ids),
                "reference_event_ids": list(result.reference_event_ids),
            }
            for result in results
        ]
    }


def build_deterministic_facts(
    query: str,
    plan: SearchPlan,
    merged_results: MergedResults,
    tool_results: Sequence[ToolResult],
) -> dict[str, Any]:
    exact_results = [result for result in tool_results if not result.relaxed]
    if len(exact_results) == 1:
        # The ToolResult count is calculated before the UI page size and is
        # therefore authoritative for both count and list answers.
        total_matches = exact_results[0].total_matches
    elif exact_results:
        total_matches = len({event_id for result in exact_results for event_id in result.all_event_ids})
    else:
        total_matches = 0
    relaxed_fields: list[str] = []
    strong_event_ids: list[str] = []
    reference_event_ids: list[str] = []
    for result in tool_results:
        if result.relaxed:
            relaxed_fields.extend(result.relaxed_fields)
        strong_event_ids.extend(result.strong_event_ids)
        reference_event_ids.extend(result.reference_event_ids)
    return {
        "query": query,
        "answer_type": plan.answer_type,
        "total_matches": total_matches,
        "exact_event_ids": [_event_id(event) for event in merged_results.exact_events],
        "relaxed_event_ids": [_event_id(event) for event in merged_results.relaxed_events],
        "relaxed_fields": list(dict.fromkeys(relaxed_fields)),
        "strong_event_ids": list(dict.fromkeys(strong_event_ids)),
        "reference_event_ids": list(dict.fromkeys(reference_event_ids)),
    }


def build_writer_payload(
    query: str,
    deterministic_facts: Mapping[str, Any],
    candidate_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    bounded_candidates = list(candidate_events)[:MAX_WRITER_CANDIDATES]
    return {
        "query": query,
        "answer_type": deterministic_facts["answer_type"],
        "candidate_ids": [_event_id(event) for event in bounded_candidates],
        "candidate_summary": [
            {
                "id": _event_id(event),
                "ジャンル": str(event.get("ジャンル", "")),
                "概要": str(event.get("概要", ""))[:400],
            }
            for event in bounded_candidates
        ],
        "relaxed": bool(deterministic_facts["relaxed_event_ids"]),
    }


def deterministic_writer_fallback(facts: Mapping[str, Any]) -> WriterOutput:
    exact_ids = tuple(str(value) for value in facts.get("exact_event_ids", []))
    relaxed_ids = tuple(str(value) for value in facts.get("relaxed_event_ids", []))
    if facts.get("answer_type") == "count":
        lead = "条件に合うイベントを確認しました。気になる候補は下のカードから見てみて。"
    elif exact_ids:
        lead = "条件に合う候補を見つけたよ。気になる番号を教えてみん？"
    elif relaxed_ids:
        lead = "ぴったりの条件では見つからんかったけん、条件を少し広げた参考候補を表示するね。"
    else:
        lead = "条件に合うイベントは見つかりませんでした。条件を少し変えて探してみて。"
    follow_up = "地域や料金でも絞ってみる？" if exact_ids or relaxed_ids else None
    return WriterOutput(
        lead=lead,
        recommended_event_ids=exact_ids + relaxed_ids,
        follow_up=follow_up,
    )


def _order_by_writer(
    events: Sequence[Mapping[str, Any]],
    recommended_ids: Sequence[str],
) -> list[dict[str, Any]]:
    by_id = {_event_id(event): dict(event) for event in events}
    ordered: list[dict[str, Any]] = []
    for event_id in recommended_ids:
        if event_id in by_id:
            ordered.append(by_id.pop(event_id))
    ordered.extend(by_id.values())
    return ordered


def render_agentic_response(response: AgenticResponse) -> str:
    """Render only deterministic count/condition facts plus Writer language."""

    parts: list[str] = []
    if response.answer_type == "count":
        parts.append(f"条件に合うイベントは{response.total_matches}件あります。")
    if response.lead:
        parts.append(response.lead)
    if response.relaxed_events:
        fields = "・".join(humanize_relaxed_fields(response.relaxed_fields)) or "一部の条件"
        parts.append(f"参考候補は「{fields}」を少し広げた結果です。")
    if response.follow_up:
        parts.append(response.follow_up)
    return "\n\n".join(parts)


def execute_existing_fast_path(
    query: str,
    decision: FastDecision,
    conversation_state: Mapping[str, Any],
    *,
    reference_date: date = POC_REFERENCE_DATE,
) -> AgenticResponse:
    result = event_search.search_events(
        query,
        reference_date=reference_date,
        previous_filters=conversation_state.get("last_filters"),
        inherit_previous=False,
        limit=MAX_RESULT_SET_SIZE,
    )
    return AgenticResponse(
        answer_type="count" if result.intent == "count" else "list",
        total_matches=result.total_matches,
        exact_events=list(result.events),
        relaxed_events=list(result.near_matches),
        exact_event_ids=tuple(result.all_event_ids),
        relaxed_event_ids=tuple(result.all_near_event_ids),
        lead=result.message or "条件に合う候補を見つけました。",
        relaxed_fields=(result.relaxed_condition,) if result.relaxed_condition else (),
        planner_used=False,
    )


def handle_agentic_query(
    query: str,
    conversation_state: Mapping[str, Any],
    *,
    reference_date: date = POC_REFERENCE_DATE,
    modal_config: ModalConfig | None = None,
    planner_request: Callable[[Mapping[str, Any]], SearchPlan] | None = None,
    replan_request: Callable[[str, SearchPlan, Mapping[str, Any]], SearchPlan | None] | None = None,
    writer_request: Callable[[Mapping[str, Any]], WriterOutput | None] | None = None,
) -> AgenticResponse:
    total_started = perf_counter()
    fast_decision = evaluate_fast_path(query, conversation_state, reference_date=reference_date)
    if fast_decision.can_handle:
        return execute_existing_fast_path(query, fast_decision, conversation_state, reference_date=reference_date)

    planner_context = {
        "query": query,
        "reference_date": reference_date.isoformat(),
        "selected_event_id": conversation_state.get("selected_event_id"),
        "last_result_ids": conversation_state.get("last_result_ids", []),
        "last_filters": conversation_state.get("last_filters", {}),
    }
    planner_started = perf_counter()
    planner_calls = 1
    if planner_request is None:
        plan = request_search_plan(planner_context, modal_config)
    else:
        candidate_plan = planner_request(planner_context)
        plan = (
            candidate_plan
            if isinstance(candidate_plan, SearchPlan) and validate_search_plan(candidate_plan.to_dict()) is not None
            else request_search_plan({"query": query, "reference_date": reference_date.isoformat()}, None)
        )
    if not isinstance(plan, SearchPlan):
        plan = request_search_plan({"query": query, "reference_date": reference_date.isoformat()}, None)
    planner_ms = (perf_counter() - planner_started) * 1000

    all_results: list[ToolResult] = []
    total_search_count = 0
    planner_rounds = 0
    replan_ms = 0.0
    replan_calls = 0
    current_plan = plan
    for planner_round in range(MAX_PLANNER_ROUNDS):
        planner_rounds = planner_round + 1
        round_searches = current_plan.searches[:MAX_SEARCHES_PER_ROUND]
        for search_spec in round_searches:
            if total_search_count >= MAX_TOTAL_SEARCHES:
                break
            try:
                result = execute_tool(search_spec, reference_date=reference_date)
            except (ValueError, TypeError, KeyError):
                continue
            all_results.append(result)
            total_search_count += 1
        if not should_replan(current_plan, all_results, planner_round):
            break
        result_summary = summarize_tool_results(all_results)
        replan_started = perf_counter()
        replan_calls += 1
        if replan_request is None:
            next_plan = request_replan(query, current_plan, result_summary, modal_config)
        else:
            next_plan = replan_request(query, current_plan, result_summary)
            next_plan = validate_replan_plan(next_plan, current_plan)
        replan_ms += (perf_counter() - replan_started) * 1000
        if next_plan is None:
            break
        current_plan = next_plan

    merged = merge_agent_results(all_results)
    facts = build_deterministic_facts(query, current_plan, merged, all_results)
    writer_input = build_writer_payload(query, facts, merged.display_candidates)
    writer_skipped = str(facts["answer_type"]) == "count" and not facts["relaxed_event_ids"]
    writer_calls = 0
    writer_ms = 0.0
    if writer_skipped:
        writer = None
    else:
        writer_started = perf_counter()
        writer_calls = 1
        if writer_request is None:
            writer = request_writer(writer_input, modal_config)
        else:
            writer = writer_request(writer_input)
        writer_ms = (perf_counter() - writer_started) * 1000
    if writer is None:
        writer = deterministic_writer_fallback(facts)
    # Writer recommendations may explain a subset, but they must not change
    # the deterministic search order used by cards, ordinals, or refinement.
    exact_events = [dict(event) for event in merged.exact_events]
    relaxed_events = [dict(event) for event in merged.relaxed_events]
    # Keep the complete bounded result sets for the UI.  The Writer candidate
    # budget is enforced at ``merge_agent_results``/``build_writer_payload``;
    # it must not become a card or conversation-reference limit.
    exact_events = exact_events[:MAX_RESULT_SET_SIZE]
    relaxed_events = relaxed_events[:MAX_RESULT_SET_SIZE]
    return AgenticResponse(
        answer_type=str(facts["answer_type"]),
        total_matches=int(facts["total_matches"]),
        exact_events=exact_events,
        relaxed_events=relaxed_events,
        exact_event_ids=tuple(str(value) for value in facts["exact_event_ids"]),
        relaxed_event_ids=tuple(str(value) for value in facts["relaxed_event_ids"]),
        lead=writer.lead,
        follow_up=writer.follow_up,
        relaxed_fields=tuple(str(value) for value in facts["relaxed_fields"]),
        planner_used=True,
        planner_rounds=planner_rounds,
        search_count=total_search_count,
        recommended_event_ids=writer.recommended_event_ids,
        latency=AgenticLatency(
            planner_ms=planner_ms,
            replan_ms=replan_ms,
            writer_ms=writer_ms,
            total_ms=(perf_counter() - total_started) * 1000,
            planner_calls=planner_calls,
            replan_calls=replan_calls,
            writer_calls=writer_calls,
        ),
        writer_skipped=writer_skipped,
        strong_event_ids=tuple(str(value) for value in facts["strong_event_ids"]),
        reference_event_ids=tuple(str(value) for value in facts["reference_event_ids"]),
    )
