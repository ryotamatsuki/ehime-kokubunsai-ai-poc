"""Standalone QA for the bounded Agentic Search layer."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import age_semantics  # noqa: E402
from agent_models import SearchPlan, SearchSpec, ToolResult  # noqa: E402
from agent_orchestrator import (  # noqa: E402
    assess_parser_coverage,
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
from app_config import (  # noqa: E402
    MAX_RESULT_SET_SIZE,
    MAX_WRITER_CANDIDATES,
    POC_REFERENCE_DATE,
)
from conversation_router import route_conversation  # noqa: E402
from event_search import load_events, parse_query  # noqa: E402


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


EVENTS = load_events()


# 1. Count and the complete bounded result set are both deterministic.  The
#    UI/Writer budgets are applied later, at their respective boundaries.
indoor_spec = SearchSpec(
    "s1",
    "count_events",
    "exact",
    {"venue": "屋内"},
)
indoor = execute_structured_search(indoor_spec, EVENTS)
check(indoor.total_matches == 14, "indoor count must be 14")
check(len(indoor.events) == indoor.total_matches, "tool result events were capped before UI pagination")
check(len(indoor.all_event_ids) == 14, "all matching IDs must be retained")


# 2. Age-only discovery becomes a structured child-friendly search.
age_plan = fallback_search_plan("5歳が楽しめるイベントある？")
check(age_plan.searches[0].filters["age"] == 5, "age was not extracted")
age_result = execute_tool(age_plan.searches[0], EVENTS)
check(age_result.total_matches == 28, "age recommendation must use child-friendly records")
age_classes = [
    age_semantics.event_age_match(event, age=5, age_group="preschool", age_intent="recommended")
    for event in EVENTS
]
check(age_classes.count("strong") == 22, "age strong/reference split changed")
check(age_classes.count("reference") == 6, "recommended age reference candidates changed")
check(age_classes.count("excluded") == 2, "non-child events were not excluded from age candidates")
check(len(age_result.strong_event_ids) == 22, "Tool did not retain strong age IDs")
check(len(age_result.reference_event_ids) == 6, "Tool did not retain reference age IDs")
invalid_date_result = execute_tool(fallback_search_plan("2028年99月1日のイベント").searches[0], EVENTS)
check(invalid_date_result.total_matches == 0, "invalid date broadened the fallback search")


# 2b. The natural-language gates found on the public UI stay structured.
age5_query = "5才が楽しめるイベントある？"
age5_filters = parse_query(age5_query)
age5_route = route_conversation(age5_query, [], None, None, POC_REFERENCE_DATE)
age5_coverage = assess_parser_coverage(age5_query, age5_filters)
age5_plan = fallback_search_plan(age5_query)
check(age5_route.action_type != "generic_scope", "5才 query became generic scope")
check(should_use_agentic_search(age5_query, age5_route, age5_filters), "5才 query missed Agentic entry")
check(age5_plan.searches[0].filters.get("age") == 5, "5才 was not normalized to age=5")
check(age5_plan.searches[0].filters.get("age_group") == "preschool", "5才 group was not normalized")
check(age5_plan.searches[0].filters.get("age_intent") == "recommended", "5才 intent was not normalized")
check("5才" not in age5_plan.searches[0].filters.get("soft_terms", []), "5才 became a soft term")
check("age_semantics" in age5_coverage.unresolved_constraints, "age semantic coverage was not traced")

preschool_query = "幼稚園児が楽しめるイベント"
preschool_plan = fallback_search_plan(preschool_query)
preschool_filters = parse_query(preschool_query)
preschool_route = route_conversation(preschool_query, [], None, None, POC_REFERENCE_DATE)
check(should_use_agentic_search(preschool_query, preschool_route, preschool_filters), "preschool query missed Agentic entry")
check(preschool_plan.searches[0].filters.get("age_group") == "preschool", "幼稚園児 was not mapped to preschool")
check(preschool_plan.searches[0].filters.get("age_intent") == "recommended", "preschool intent was not set")
check("幼稚園児" not in preschool_plan.searches[0].filters.get("soft_terms", []), "幼稚園児 became a soft term")
check(validate_search_plan(preschool_plan.to_dict()) is not None, "preschool plan failed schema validation")
for term in ("5歳", "5才", "未就学児", "幼児", "保育園児", "小学生"):
    semantics = age_semantics.query_age_semantics(f"{term}が楽しめるイベント")
    check(semantics.recognized, f"age vocabulary was not recognized: {term}")
check(age_semantics.query_age_semantics("5歳でも参加できるイベント").age_intent == "eligible", "eligible age intent was lost")
eligible_plan = fallback_search_plan("5歳でも参加できるイベント")
check(validate_search_plan(eligible_plan.to_dict()) is not None, "eligible age plan failed schema validation")

indoor_natural_query = "建物の中でやるイベントはいくつくらい？"
indoor_natural_plan = fallback_search_plan(indoor_natural_query)
indoor_natural_result = execute_tool(indoor_natural_plan.searches[0], EVENTS)
check(indoor_natural_plan.answer_type == "count", "natural indoor query lost count intent")
check(indoor_natural_plan.searches[0].tool == "count_events", "natural indoor query did not use count_events")
check(indoor_natural_plan.searches[0].filters.get("venue") == "屋内", "建物の中 was not mapped to indoor")
check(indoor_natural_result.total_matches == 14, "natural indoor count must be 14")
check(len(indoor_natural_result.all_event_ids) == 14, "natural indoor total was capped at cards")

reservation_query = "予約なしで参加できる松山のイベント"
reservation_plan = fallback_search_plan(reservation_query)
reservation_result = execute_tool(reservation_plan.searches[0], EVENTS)
check(reservation_plan.searches[0].filters.get("municipalities") == ["松山市"], "松山 was not canonicalized")
check(reservation_plan.searches[0].filters.get("reservation_required") is False, "予約なし was not structured")
check("予約なし" not in reservation_plan.searches[0].filters.get("soft_terms", []), "予約なし became a soft term")
check(reservation_result.total_matches == 2, "松山の予約不要 exact count changed")
check({event["id"] for event in reservation_result.events} == {"002", "028"}, "reservation exact candidates changed")
check(not ({"019", "026"} & set(reservation_result.all_event_ids)), "未定 was treated as 不要")


# 3. A soft hobby term gets one exact search and one bounded relaxed search.
dinosaur = handle_agentic_query("恐竜好きの5歳", {})
check(dinosaur.search_count == 2, "soft-term query did not replan exactly once")
check(not dinosaur.exact_events, "unmatched hobby term became an exact event")
check(dinosaur.relaxed_events, "relaxed child-friendly candidates are missing")
check("soft_terms" in dinosaur.relaxed_fields, "relaxed field was not recorded")
dinosaur_text = render_agentic_response(dinosaur)
check("ぴったりの条件" in dinosaur_text, "relaxed explanation is missing")
check("趣味・キーワード条件" in dinosaur_text, "relaxed field label is not user-facing")
check("soft_terms" not in dinosaur_text, "internal relaxed field leaked into response")


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

# Multiple exact tool results retain the complete bounded set while the
# Writer payload stays within its separate candidate budget.
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
check(
    len(bounded.exact_events) + len(bounded.relaxed_events) <= MAX_RESULT_SET_SIZE,
    "Agentic response exceeded the result-set bound",
)
writer_payload = build_writer_payload(
    "恐竜好きのイベント",
    {"answer_type": "list", "exact_event_ids": [], "relaxed_event_ids": []},
    bounded.exact_events + bounded.relaxed_events,
)
check(
    len(writer_payload["candidate_ids"]) <= MAX_WRITER_CANDIDATES,
    "Writer candidate budget was not enforced",
)


# 9. Tool names are fixed and no dynamic Python dispatch is needed.
check(execute_tool(indoor_spec, EVENTS).total_matches == 14, "allow-listed dispatcher failed")

# 10. Planner generation is deterministic and separate from Writer/legacy.
modal_source = (ROOT / "modal_backend.py").read_text(encoding="utf-8")
check('if mode == "planner":' in modal_source, "planner generation branch is missing")
generation_start = modal_source.rfind('if mode == "planner":')
planner_block = modal_source[generation_start:].split('elif mode == "writer":', 1)[0]
check('"do_sample": False' in planner_block, "Planner is still sampling")
check('"max_new_tokens": PLANNER_MAX_NEW_TOKENS' in planner_block and 'PLANNER_MAX_NEW_TOKENS = 256' in modal_source, "Planner token budget is not bounded")
check('"temperature"' not in planner_block and '"top_p"' not in planner_block, "Planner received sampling kwargs")
writer_block = modal_source[generation_start:].split('elif mode == "writer":', 1)[1].split('else:', 1)[0]
check('"do_sample": True' in writer_block and '"temperature": 0.65' in writer_block, "Writer sampling config is not separated")

# 11. Simple count keeps deterministic facts and records bounded latency.
natural_count = handle_agentic_query(indoor_natural_query, {})
check(natural_count.total_matches == 14, "natural count response changed")
check(natural_count.writer_skipped, "simple count did not skip Writer")
check(natural_count.latency.total_ms >= natural_count.latency.planner_ms, "latency trace is inconsistent")
check(natural_count.latency.writer_calls == 0, "simple count still called Writer")

print("Agentic Search QA: PASS (11 groups)")
print(f"Indoor total: {indoor.total_matches}; age total: {age_result.total_matches}; replan searches: {dinosaur.search_count}")
