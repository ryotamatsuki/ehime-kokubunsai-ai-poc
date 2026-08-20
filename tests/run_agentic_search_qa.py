"""Standalone QA for the bounded Agentic Search layer."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_models import SearchPlan, SearchSpec, ToolResult  # noqa: E402
from agent_orchestrator import (  # noqa: E402
    build_writer_payload,
    handle_agentic_query,
    merge_agent_results,
    render_agentic_response,
    should_replan,
    should_use_agentic_search,
)
from agent_planner import (  # noqa: E402
    fallback_search_plan,
    validate_replan_plan,
    validate_search_plan,
    validate_writer_output,
)
from agent_tools import execute_detail_lookup, execute_structured_search, execute_tool  # noqa: E402
from app_config import POC_REFERENCE_DATE  # noqa: E402
from conversation_router import route_conversation  # noqa: E402
from event_search import load_events, parse_query  # noqa: E402


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


EVENTS = load_events()


# 1. Count is calculated before the eight-card display limit.
indoor_spec = SearchSpec(
    "s1",
    "count_events",
    "exact",
    {"venue": "屋内"},
)
indoor = execute_structured_search(indoor_spec, EVENTS)
check(indoor.total_matches == 14, "indoor count must be 14")
check(len(indoor.events) == 8, "display candidates must be capped at 8")
check(len(indoor.all_event_ids) == 14, "all matching IDs must be retained")


# 2. Age-only discovery becomes a structured child-friendly search.
age_plan = fallback_search_plan("5歳が楽しめるイベントある？")
check(age_plan.searches[0].filters["age"] == 5, "age was not extracted")
age_result = execute_tool(age_plan.searches[0], EVENTS)
check(age_result.total_matches == 28, "age recommendation must use child-friendly records")
invalid_date_result = execute_tool(fallback_search_plan("2028年99月1日のイベント").searches[0], EVENTS)
check(invalid_date_result.total_matches == 0, "invalid date broadened the fallback search")


# 3. A soft hobby term gets one exact search and one bounded relaxed search.
dinosaur = handle_agentic_query("恐竜好きの5歳", {})
check(dinosaur.search_count == 2, "soft-term query did not replan exactly once")
check(not dinosaur.exact_events, "unmatched hobby term became an exact event")
check(dinosaur.relaxed_events, "relaxed child-friendly candidates are missing")
check("soft_terms" in dinosaur.relaxed_fields, "relaxed field was not recorded")
check("ぴったりの条件" in render_agentic_response(dinosaur), "relaxed explanation is missing")


# 4. Count and list facts are deterministic, independent of Writer output.
count_response = handle_agentic_query("室内イベントはどれくらいあるか", {})
check(count_response.answer_type == "count", "count intent was not preserved")
check(count_response.total_matches == 14, "Writer changed the deterministic count")
check("14件" in render_agentic_response(count_response), "rendered count is incorrect")


# 5. Fast Path questions never enter Agentic Search.
for query in ("砥部焼はいくら？", "上島の開催日は？", "予約って必要？"):
    route = route_conversation(query, [], None, None, POC_REFERENCE_DATE)
    check(not should_use_agentic_search(query, route, parse_query(query)), f"Fast Path leaked into Agentic Search: {query}")
check(
    should_use_agentic_search(
        "5歳が楽しめるイベントある？",
        route_conversation("5歳が楽しめるイベントある？", [], None, None, POC_REFERENCE_DATE),
        parse_query("5歳が楽しめるイベントある？"),
    ),
    "age discovery did not enter Agentic Search",
)
for query in ("子どもと楽しめるイベント", "雨でも楽しめる屋内イベント"):
    route = route_conversation(query, [], None, None, POC_REFERENCE_DATE)
    check(not should_use_agentic_search(query, route, parse_query(query)), f"deterministic condition entered Agentic Search: {query}")


# 6. Exact results always win when exact and relaxed searches overlap.
merged = merge_agent_results(
    [
        ToolResult("exact", "exact", 1, [EVENTS[6]], ["007"]),
        ToolResult("relaxed", "relaxed", 2, [EVENTS[6], EVENTS[9]], ["007", "010"], True, ("soft_terms",)),
    ]
)
check([event["id"] for event in merged.exact_events] == ["007"], "exact event was lost")
check([event["id"] for event in merged.relaxed_events] == ["010"], "relaxed duplicate was not removed")


# 7. Writer receives only candidate IDs, genre, and overview.
payload = build_writer_payload("室内イベント", {"answer_type": "list", "total_matches": 14, "exact_event_ids": ["007"], "relaxed_event_ids": []}, [EVENTS[6]])
check(set(payload["candidate_summary"][0]) == {"id", "ジャンル", "概要"}, "writer received event facts outside its contract")
check("日時" not in payload and "場所" not in payload and "料金" not in payload and "公式URL" not in payload, "writer payload leaked event facts")
check("total_matches" not in payload, "writer received deterministic count facts")


# 8. Invalid planner tools, filter keys, IDs, and writer facts are rejected.
check(
    validate_search_plan({"intent": "discover", "answer_type": "list", "searches": [{"search_id": "s1", "tool": "os_system", "purpose": "exact", "filters": {}}]}) is None,
    "unknown planner tool was accepted",
)
check(
    validate_search_plan({"intent": "discover", "answer_type": "list", "searches": [{"search_id": "s1", "tool": "search_events", "purpose": "exact", "filters": {"python": "exec"}}]}) is None,
    "unknown filter key was accepted",
)
check(
    validate_search_plan({"intent": "discover", "answer_type": "list", "searches": [{"search_id": "s1", "tool": "search_events", "purpose": "exact", "filters": {}}]}) is None,
    "semantically empty search plan was accepted",
)
check(
    validate_search_plan({"intent": "discover", "answer_type": "list", "searches": [{"search_id": "s1", "tool": "search_events", "purpose": "exact", "filters": {"dates": ["2028-99-01"]}}]}) is None,
    "invalid date was accepted",
)
check(
    validate_search_plan({"intent": "discover", "answer_type": "list", "searches": [{"search_id": "s1", "tool": "search_events", "purpose": "exact", "filters": {"venue": "宇宙"}}]}) is None,
    "unknown venue was accepted",
)
check(
    validate_search_plan({"intent": "discover", "answer_type": "list", "searches": [{"search_id": "s1", "tool": "search_events", "purpose": "exact", "filters": {"age_group": "家族"}}]}) is None,
    "unsupported age group was accepted",
)
check(
    validate_search_plan({"intent": "discover", "answer_type": "list", "searches": [{"search_id": "s1", "tool": "search_events", "purpose": "exact", "filters": {"age_intent": "対象"}}]}) is None,
    "age intent without a child predicate was accepted",
)
check(
    validate_search_plan({"intent": "discover", "answer_type": "list", "searches": [{"search_id": "s1", "tool": "search_events", "purpose": "exact", "filters": {"child_friendly": False}}]}) is None,
    "false-only child filter was accepted",
)
check(validate_writer_output({"lead": "日時は2028-11-03です", "recommended_event_ids": [], "reasons": []}, set()) is None, "writer fact leakage was accepted")
check(validate_writer_output({"lead": "11月3日です", "recommended_event_ids": [], "reasons": []}, set()) is None, "Japanese date leakage was accepted")
check(validate_writer_output({"lead": "13時から参加できます", "recommended_event_ids": [], "reasons": []}, set()) is None, "Japanese time leakage was accepted")
check(validate_writer_output({"lead": "午後1時から参加できます", "recommended_event_ids": [], "reasons": []}, set()) is None, "Japanese meridiem time leakage was accepted")
check(validate_writer_output({"lead": "無料で予約不要の屋内イベントです", "recommended_event_ids": [], "reasons": []}, set()) is None, "writer semantic fact leakage was accepted")
check(validate_writer_output({"lead": "候補です", "recommended_event_ids": ["999"], "reasons": []}, {"007"}) is None, "unknown writer event ID was accepted")
check(execute_detail_lookup(SearchSpec("detail", "get_event_detail", "detail", {}), EVENTS).total_matches == 0, "detail tool without an ID returned all events")

# Replan is allowed only after zero exact hits and only as an explicit weaker plan.
replan_source = SearchPlan(
    "discover",
    "list",
    (SearchSpec("s1", "search_events", "exact", {"soft_terms": ["恐竜"], "child_friendly": True}),),
    allow_replan=True,
)
one_exact = ToolResult("s1", "exact", 1, [EVENTS[0]], [EVENTS[0]["id"]])
check(not should_replan(replan_source, [one_exact], 0), "replan ran despite an exact match")
invalid_replan = SearchPlan(
    "discover",
    "list",
    (SearchSpec("s1-relaxed", "search_events", "exact", {}),),
)
check(validate_replan_plan(invalid_replan, replan_source) is None, "unbounded replan was accepted")
valid_replan = SearchPlan(
    "discover",
    "list",
    (SearchSpec("s1-relaxed", "search_events", "relaxed", {"child_friendly": True}, True, ("soft_terms",)),),
)
check(validate_replan_plan(valid_replan, replan_source) is not None, "valid soft-term relaxation was rejected")
for previous_filters, changed_filters, relaxed_fields, label in (
    ({"genres": ["文学"]}, {"genres": ["アート"]}, ("genres",), "genre substitution"),
    ({"max_entry_fee": 800}, {"max_entry_fee": 100}, ("max_entry_fee",), "lower fee"),
    ({"venue": "屋内"}, {"venue": "屋外"}, ("venue",), "venue substitution"),
    ({"soft_terms": ["恐竜"], "child_friendly": True}, {"soft_terms": ["太鼓"], "child_friendly": True}, ("soft_terms",), "soft-term substitution"),
):
    previous = SearchPlan(
        "discover",
        "list",
        (SearchSpec("s1", "search_events", "exact", previous_filters),),
        allow_replan=True,
    )
    invalid_weaken = SearchPlan(
        "discover",
        "list",
        (SearchSpec("s1-relaxed", "search_events", "relaxed", changed_filters, True, relaxed_fields),),
    )
    check(validate_replan_plan(invalid_weaken, previous) is None, f"non-relaxing {label} was accepted")

# Multiple exact tool results still use one bounded card budget.
multi_plan = SearchPlan(
    "discover",
    "list",
    (
        SearchSpec("s1", "search_events", "exact", {"child_friendly": True}),
        SearchSpec("s2", "search_events", "exact", {"venue": "屋内"}),
    ),
)
bounded = handle_agentic_query(
    "恐竜好きのイベント",
    {},
    planner_request=lambda _context: multi_plan,
    writer_request=lambda _payload: None,
)
check(len(bounded.exact_events) <= 8 and len(bounded.exact_events) + len(bounded.relaxed_events) <= 8, "Agentic response exceeded the card budget")


# 9. Tool names are fixed and no dynamic Python dispatch is needed.
check(execute_tool(indoor_spec, EVENTS).total_matches == 14, "allow-listed dispatcher failed")

print("Agentic Search QA: PASS (9 groups)")
print(f"Indoor total: {indoor.total_matches}; age total: {age_result.total_matches}; replan searches: {dinosaur.search_count}")
