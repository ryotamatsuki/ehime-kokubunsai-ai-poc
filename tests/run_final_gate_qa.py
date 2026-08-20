"""Final audit gates for parser coverage, age semantics, planner and routing."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from age_semantics import age_match_tier, parse_event_age_semantics  # noqa: E402
from agent_orchestrator import (  # noqa: E402
    evaluate_parser_coverage,
    handle_agentic_query,
    render_agentic_response,
    should_use_agentic_search,
)
from agent_planner import fallback_search_plan  # noqa: E402
from agent_tools import execute_tool  # noqa: E402
from app_config import POC_REFERENCE_DATE  # noqa: E402
from conversation_router import route_conversation  # noqa: E402
from event_search import load_events, parse_query  # noqa: E402


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def event_by_id(events: list[dict[str, object]], event_id: str) -> dict[str, object]:
    return next(event for event in events if event["id"] == event_id)


def route(query: str, last_results=None, selected_event=None):
    return route_conversation(
        query,
        last_results or [],
        selected_event,
        None,
        POC_REFERENCE_DATE,
    )


def is_agentic(query: str) -> bool:
    query_route = route(query)
    return should_use_agentic_search(query, query_route, parse_query(query, POC_REFERENCE_DATE))


EVENTS = load_events()

# Data audit: mapping is intentionally limited to actual current values.
age_values = {str(event["参加案内"]["対象年齢"]) for event in EVENTS}
target_values = {str(event["参加案内"]["対象"]) for event in EVENTS}
check(age_values == {"年齢制限なし", "小学生以上推奨", "高校生以上推奨"}, f"unexpected age labels: {age_values}")
check(target_values == {"どなたでも（子ども・家族での参加を想定）", "一般向け"}, f"unexpected target labels: {target_values}")

# Fast Path coverage: complete means the whole search meaning is represented.
for query in ("無料の松山市のイベント", "室内イベント", "子どもと楽しめるイベント", "雨でも楽しめる屋内イベント"):
    parsed = parse_query(query, POC_REFERENCE_DATE)
    coverage = evaluate_parser_coverage(query, parsed)
    check(coverage.complete, f"complete legacy query became incomplete: {query}: {coverage}")
    check(not is_agentic(query), f"complete legacy query entered Agentic Search: {query}")

# Age gates: numeric spellings and group words are Agentic and canonical.
for query in ("5歳が楽しめるイベントある？", "5才が楽しめるイベントある？"):
    query_route = route(query)
    check(query_route.action_type == "search", f"age query became {query_route.action_type}: {query}")
    check(is_agentic(query), f"age query did not enter Agentic Search: {query}")
    plan = fallback_search_plan(query)
    check(plan.searches[0].filters.get("age") == 5, f"age=5 missing: {query}")
    check(plan.searches[0].filters.get("age_intent") == "recommended", f"recommended missing: {query}")

for query in ("幼稚園児が楽しめるイベント", "未就学児向け"):
    check(route(query).action_type == "search", f"preschool query not routed to search: {query}")
    check(is_agentic(query), f"preschool query did not enter Agentic Search: {query}")
    plan = fallback_search_plan(query)
    check(plan.searches[0].filters.get("age_group") == "preschool", f"preschool missing: {query}")

eligible_plan = fallback_search_plan("5歳でも参加できるイベント")
check(eligible_plan.searches[0].filters.get("age") == 5, "eligible age missing")
check(eligible_plan.searches[0].filters.get("age_intent") == "eligible", "eligible intent missing")
elementary_plan = fallback_search_plan("小学生向け")
check(elementary_plan.searches[0].filters.get("age_group") == "elementary", "elementary group missing")

unknown_event = dict(EVENTS[0])
unknown_event["参加案内"] = dict(EVENTS[0]["参加案内"])
unknown_event["参加案内"]["対象年齢"] = "将来追加される未知表現"
check(parse_event_age_semantics(unknown_event).unknown, "unknown age label was guessed")
small_school = event_by_id(EVENTS, "009")
check(age_match_tier(small_school, age=5, age_intent="recommended") == 1, "小学生以上推奨 was not a reference candidate for age 5")
check(age_match_tier(small_school, age=5, age_intent="eligible") > 0, "小学生以上推奨 was incorrectly treated as ineligible")

# Count gate: colloquial count + understood indoor still needs Planner because
# legacy answer_type coverage is incomplete.  Deterministic tool owns 14.
indoor_colloquial = "室内イベントはどれくらいあるか"
check(is_agentic(indoor_colloquial), "colloquial count skipped Agentic Search")
indoor_plan = fallback_search_plan(indoor_colloquial)
check(indoor_plan.answer_type == "count" and indoor_plan.searches[0].tool == "count_events", "count plan missing")
check(execute_tool(indoor_plan.searches[0], EVENTS).total_matches == 14, "indoor count is not 14")

building_query = "建物の中でやるイベントはいくつくらい？"
building_coverage = evaluate_parser_coverage(building_query, parse_query(building_query, POC_REFERENCE_DATE))
check(not building_coverage.complete and "venue=indoor" in building_coverage.unresolved_terms, "building-inside semantic gap not detected")
check(is_agentic(building_query), "building-inside count did not enter Agentic Search")
building_plan = fallback_search_plan(building_query)
check(building_plan.answer_type == "count", "building count answer_type missing")
check(building_plan.searches[0].tool == "count_events", "building count tool missing")
check(building_plan.searches[0].filters.get("venue") == "indoor", "building-inside did not map to indoor")
building_result = execute_tool(building_plan.searches[0], EVENTS)
check(building_result.total_matches == 14, "building-inside count is not 14")
building_response = handle_agentic_query(building_query, {})
check(building_response.planner_used and building_response.total_matches == 14, "building query trace did not use Planner+tool")
check("14件" in render_agentic_response(building_response), "Python count render is not 14")

# Reservation gate: partial city understanding must not swallow reservation.
reservation_query = "予約なしで参加できる松山のイベント"
reservation_coverage = evaluate_parser_coverage(reservation_query, parse_query(reservation_query, POC_REFERENCE_DATE))
check(not reservation_coverage.complete, "reservation semantic gap was marked complete")
check(is_agentic(reservation_query), "reservation query stayed on Fast Path")
reservation_plan = fallback_search_plan(reservation_query)
reservation_filters = reservation_plan.searches[0].filters
check(reservation_filters.get("municipalities") == ["松山市"], f"Matsuyama mapping failed: {reservation_filters}")
check(reservation_filters.get("reservation_required") is False, "reservation_required=false missing")
reservation_result = execute_tool(reservation_plan.searches[0], EVENTS)
check(reservation_result.total_matches > 0, "reservation-free Matsuyama returned no exact events")
check(all(event["参加案内"]["申込要否"] == "不要" for event in reservation_result.events), "未定/必要 leaked into reservation-free exact results")

# Rain + colloquial free: do not silently retain only the understood rain term.
rain_free_query = "雨の日でも大丈夫で、お金のかからないイベント"
rain_free_coverage = evaluate_parser_coverage(rain_free_query, parse_query(rain_free_query, POC_REFERENCE_DATE))
check(not rain_free_coverage.complete and "entry_free" in rain_free_coverage.unresolved_terms, "colloquial free gap was dropped")
check(is_agentic(rain_free_query), "rain/free query stayed on partial Fast Path")
rain_free_plan = fallback_search_plan(rain_free_query)
check(rain_free_plan.searches[0].filters.get("rain_preferred") is True, "rain condition missing")
check(rain_free_plan.searches[0].filters.get("entry_free") is True, "colloquial free condition missing")

# Recommendation intent precedes pronoun/ordinal reference resolution.
opening = event_by_id(EVENTS, "001")
haiku = event_by_id(EVENTS, "002")
saijo = event_by_id(EVENTS, "008")
tobe = event_by_id(EVENTS, "022")

next_pronoun = route("そのイベントのあと何か行ける？", [haiku], haiku)
check(next_pronoun.action_type == "recommend_next" and next_pronoun.selected_event == haiku, "そのイベント next routing failed")
next_sore = route("それのあと何か行ける？", [haiku], haiku)
check(next_sore.action_type == "recommend_next" and next_sore.selected_event == haiku, "それ next routing failed")
next_ordinal = route("2番目のあと何か行ける？", [opening, saijo], opening)
check(next_ordinal.action_type == "recommend_next", "ordinal next became reference_followup")
check(next_ordinal.reference_index == 1 and next_ordinal.selected_event == saijo, "ordinal next seed is wrong")

similar_pronoun = route("それと似たイベントある？", [haiku], haiku)
check(similar_pronoun.action_type == "recommend_similar" and similar_pronoun.selected_event == haiku, "pronoun similar routing failed")
similar_ordinal = route("2番目と似たイベントある？", [opening, saijo], opening)
check(similar_ordinal.action_type == "recommend_similar", "ordinal similar became reference_followup")
check(similar_ordinal.reference_index == 1 and similar_ordinal.selected_event == saijo, "ordinal similar seed is wrong")
named_stale = route("砥部焼と似たイベントある？", [opening], opening)
check(named_stale.action_type == "recommend_similar" and named_stale.selected_event == tobe, "named similar did not override stale selection")
named_no_selection = route("砥部焼と似たイベントある？")
check(named_no_selection.action_type == "recommend_similar" and named_no_selection.selected_event == tobe, "named similar without selection failed")

# Planner generation must be greedy and omit sampling-only temperature/top_p.
modal_source = (ROOT / "modal_backend.py").read_text(encoding="utf-8")
marker = 'if mode == "planner":\n            # SearchPlan JSON is a control structure'
check(marker in modal_source, "planner generation block not found")
planner_block = modal_source.split(marker, 1)[1].split('elif mode == "writer":', 1)[0]
check('"do_sample": False' in planner_block, "Planner do_sample is not False")
check('"temperature"' not in planner_block and '"top_p"' not in planner_block, "Planner still receives sampling knobs")

print("Final Gate QA: PASS")
print(f"Age labels: {sorted(age_values)}")
print(f"Target labels: {sorted(target_values)}")
print(f"Indoor total: {building_result.total_matches}; reservation-free Matsuyama: {reservation_result.total_matches}")
