"""Bounded orchestration for ambiguous event discovery.

This layer is deliberately narrow.  Existing event lookup, participation,
FAQ, recommendation, injection, and out-of-scope routes remain Fast Path
branches.  Only discovery questions whose legacy-parser coverage is incomplete
use Planner -> fixed tools -> optional Replan -> Writer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
from typing import Any, Callable, Mapping, Sequence

import conversation_router
import event_search
from agent_models import AgenticResponse, MergedResults, SearchPlan, SearchSpec, ToolResult, WriterOutput
from agent_planner import (
    ModalConfig,
    request_replan,
    request_search_plan,
    request_writer,
    validate_replan_plan,
    validate_search_plan,
)
from agent_tools import execute_tool
from app_config import MAX_SEARCH_RESULTS, POC_REFERENCE_DATE
from parser_coverage import ParserCoverage, evaluate_parser_coverage


MAX_PLANNER_ROUNDS = 2
MAX_SEARCHES_PER_ROUND = 3
MAX_TOTAL_SEARCHES = 5


@dataclass(frozen=True)
class FastDecision:
    can_handle: bool
    reason: str = ""


def parser_confidence_is_high(parsed_filters: event_search.SearchFilters, query: str) -> bool:
    """Compatibility wrapper: high confidence now means complete coverage."""

    return evaluate_parser_coverage(query, parsed_filters).complete


def looks_like_event_discovery(query: str) -> bool:
    normalized = event_search.normalize_query(query)
    if any(term in normalized for term in ("どれくらい", "どのくらい", "どの程度", "何件", "いくつ", "件数")):
        return True
    if re.search(r"\d{1,2}(?:歳|才)", normalized) and any(
        term in normalized for term in ("楽しめる", "おすすめ", "参加", "行ける", "ある", "イベント")
    ):
        return True
    if any(term in normalized for term in ("幼稚園児", "未就学児", "保育園児", "幼児", "小さい子", "小学生", "中学生", "高校生")):
        return True
    if any(term in normalized for term in ("予約なし", "予約不要", "建物の中", "建物内", "お金のかからない", "お金がかからない")):
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
        "scope_search",
        "out_of_scope",
        "reference_followup",
        "detail_followup",
        "general_faq",
        "recommend_next",
        "recommend_similar",
        "recommend_next_without_selection",
        "recommend_similar_without_selection",
    }:
        return FastDecision(True, route.action_type)
    if parsed.intent in {"injection", "out_of_scope", "needs_location", "needs_region"}:
        return FastDecision(True, parsed.intent)
    # Explicit event-specific facts stay deterministic regardless of discovery
    # coverage because entity + field already identifies the factual lookup.
    if parsed.entity and parsed.requested_field:
        return FastDecision(True, "explicit_event_fact")

    coverage = evaluate_parser_coverage(query, parsed)
    if coverage.complete:
        return FastDecision(True, "complete_parser_coverage")
    if looks_like_event_discovery(query):
        return FastDecision(False, f"agentic_incomplete_coverage:{coverage.reason}")
    return FastDecision(True, f"non_discovery:{coverage.reason}")


def should_use_agentic_search(
    query: str,
    route: conversation_router.ConversationRoute,
    parsed_filters: event_search.SearchFilters,
) -> bool:
    """Use Agentic Search only when discovery coverage is incomplete."""

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
    }:
        return False
    if parsed_filters.entity and parsed_filters.requested_field:
        return False
    if event_search.classify_intent(query) in {"injection", "out_of_scope"}:
        return False

    coverage = evaluate_parser_coverage(query, parsed_filters)
    if coverage.complete:
        return False
    return route.action_type == "search" and looks_like_event_discovery(query)


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
        display_candidates=(exact + relaxed)[:MAX_SEARCH_RESULTS],
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
                "event_ids": result.all_event_ids[:MAX_SEARCH_RESULTS],
                "relaxed": result.relaxed,
                "relaxed_fields": list(result.relaxed_fields),
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
        # The ToolResult count is calculated before the eight-card display
        # limit and is therefore the authoritative total for both count and
        # list answers.
        total_matches = exact_results[0].total_matches
    elif exact_results:
        total_matches = len({event_id for result in exact_results for event_id in result.all_event_ids})
    else:
        total_matches = 0
    relaxed_fields: list[str] = []
    for result in tool_results:
        if result.relaxed:
            relaxed_fields.extend(result.relaxed_fields)
    return {
        "query": query,
        "answer_type": plan.answer_type,
        "total_matches": total_matches,
        "exact_event_ids": [_event_id(event) for event in merged_results.exact_events],
        "relaxed_event_ids": [_event_id(event) for event in merged_results.relaxed_events],
        "relaxed_fields": list(dict.fromkeys(relaxed_fields)),
    }


def build_writer_payload(
    query: str,
    deterministic_facts: Mapping[str, Any],
    candidate_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "query": query,
        "answer_type": deterministic_facts["answer_type"],
        "candidate_ids": [_event_id(event) for event in candidate_events],
        "candidate_summary": [
            {
                "id": _event_id(event),
                "ジャンル": str(event.get("ジャンル", "")),
                "概要": str(event.get("概要", ""))[:400],
            }
            for event in candidate_events
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
        fields = "・".join(response.relaxed_fields) if response.relaxed_fields else "一部の条件"
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
        limit=MAX_SEARCH_RESULTS,
    )
    return AgenticResponse(
        answer_type="count" if result.intent == "count" else "list",
        total_matches=result.total_matches,
        exact_events=list(result.events),
        relaxed_events=list(result.near_matches),
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

    all_results: list[ToolResult] = []
    total_search_count = 0
    planner_rounds = 0
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
        if replan_request is None:
            next_plan = request_replan(query, current_plan, result_summary, modal_config)
        else:
            next_plan = replan_request(query, current_plan, result_summary)
            next_plan = validate_replan_plan(next_plan, current_plan)
        if next_plan is None:
            break
        current_plan = next_plan

    merged = merge_agent_results(all_results)
    facts = build_deterministic_facts(query, current_plan, merged, all_results)
    writer_input = build_writer_payload(query, facts, merged.display_candidates)
    if writer_request is None:
        writer = request_writer(writer_input, modal_config)
    else:
        writer = writer_request(writer_input)
    if writer is None:
        writer = deterministic_writer_fallback(facts)
    exact_events = _order_by_writer(merged.exact_events, writer.recommended_event_ids)
    relaxed_events = _order_by_writer(merged.relaxed_events, writer.recommended_event_ids)
    # The UI has one bounded result-card budget.  Exact matches have priority;
    # relaxed references use only the remaining slots.
    exact_events = exact_events[:MAX_SEARCH_RESULTS]
    relaxed_events = relaxed_events[: max(0, MAX_SEARCH_RESULTS - len(exact_events))]
    return AgenticResponse(
        answer_type=str(facts["answer_type"]),
        total_matches=int(facts["total_matches"]),
        exact_events=exact_events,
        relaxed_events=relaxed_events,
        lead=writer.lead,
        follow_up=writer.follow_up,
        relaxed_fields=tuple(str(value) for value in facts["relaxed_fields"]),
        planner_used=True,
        planner_rounds=planner_rounds,
        search_count=total_search_count,
        recommended_event_ids=writer.recommended_event_ids,
    )
