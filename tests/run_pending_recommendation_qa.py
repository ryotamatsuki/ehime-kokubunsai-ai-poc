"""Regression QA for pending next-event recommendation turns."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import event_recommendation
import event_search
import recommendation_pending
from app_config import POC_REFERENCE_DATE
from conversation_router import route_conversation


events = event_search.load_events()
period_drop_in = next(event for event in events if event["id"] == "007")
start_time_event = next(event for event in events if event["id"] == "002")


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


pending_date = recommendation_pending.make_pending_state(
    period_drop_in,
    awaiting="date",
)

# 1-4. Date-shaped replies are consumed, while normal questions are not.
for answer in ("11月3日", "11/3", "3日", "今日"):
    parsed = event_recommendation.parse_recommendation_date_answer(answer, POC_REFERENCE_DATE)
    check(parsed.is_date_like and parsed.value is not None, f"date answer was not parsed: {answer}")
non_date = recommendation_pending.resolve_pending_input(
    "砥部焼はいくら？", pending_date, events, POC_REFERENCE_DATE
)
check(not non_date.handled, "named-event interruption was consumed by pending date")

# 5. A valid date for a drop-in period event advances to end_time only.
date_step = recommendation_pending.resolve_pending_input(
    "11月3日", pending_date, events, POC_REFERENCE_DATE
)
check(date_step.handled, "date reply did not continue pending recommendation")
check(date_step.event and date_step.event["id"] == "007", "pending event was not reloaded by id")
check(date_step.next_state and date_step.next_state["awaiting"] == "end_time", "drop-in date did not ask for end time")
check(date_step.next_state and date_step.next_state["date"] == "2028-11-03", "pending date was not retained")

# 6-8. Time-shaped replies are parsed and validated against the daily window.
pending_time = date_step.next_state
check(pending_time is not None, "time pending state missing")
early_time = recommendation_pending.resolve_pending_input("8時", pending_time, events, POC_REFERENCE_DATE)
check(early_time.handled and early_time.next_state == pending_time, "opening-before time was accepted")
late_time = recommendation_pending.resolve_pending_input("18時", pending_time, events, POC_REFERENCE_DATE)
check(late_time.handled and late_time.next_state == pending_time, "closing-after time was accepted")
valid_time = recommendation_pending.resolve_pending_input("13時", pending_time, events, POC_REFERENCE_DATE)
check(valid_time.handled and valid_time.clear, "valid end time did not clear pending state")
check(valid_time.recommendation_date == date(2028, 11, 3), "pending date was lost at time step")
check(valid_time.selected_end_override is not None, "selected end override was not returned")

# 9-10. Invalid date/time remain in the same step for correction.
invalid_date = recommendation_pending.resolve_pending_input("12月32日", pending_date, events, POC_REFERENCE_DATE)
check(invalid_date.handled and invalid_date.next_state == pending_date, "invalid date did not preserve pending state")
invalid_time = recommendation_pending.resolve_pending_input("25時", pending_time, events, POC_REFERENCE_DATE)
check(invalid_time.handled and invalid_time.next_state == pending_time, "invalid time did not preserve pending state")

# 11. Outside-period date clears the flow and never recommends.
outside = recommendation_pending.resolve_pending_input("12月1日", pending_date, events, POC_REFERENCE_DATE)
check(outside.handled and outside.clear and outside.recommendation_date is None, "outside-period date was not rejected")

# 12-13. FAQ and reset interruptions are not consumed as date/time answers.
faq_interrupt = recommendation_pending.resolve_pending_input("これは公式？", pending_date, events, POC_REFERENCE_DATE)
check(not faq_interrupt.handled, "FAQ interruption was consumed by pending date")
check(recommendation_pending.is_reset_query("リセット"), "typed reset was not recognized")

# 14. A start-time seed does not require a user-supplied end time.
start_resolution = event_recommendation.resolve_recommendation_date(
    start_time_event, "11月3日に見たあと何か行ける？", POC_REFERENCE_DATE
)
check(start_resolution.recommendation_date == date(2028, 10, 22), "start-time seed date was not resolved")

# 15-16. The named event has priority over stale selected_event context, and
# the one-turn recommendation intent is recognized.
one_turn = route_conversation(
    "11月3日に別子銅山展を見たあと何か行ける？",
    [],
    start_time_event,
    None,
    POC_REFERENCE_DATE,
)
check(one_turn.action_type == "recommend_next", "one-turn recommendation was not routed")
check(one_turn.selected_event and one_turn.selected_event["id"] == "007", "new named event did not override stale context")
for phrase in ("のあと", "の後", "見たあと", "見た後", "終わったあと", "終わった後", "見終わったあと", "見終わった後"):
    check(event_recommendation.is_next_query(f"別子銅山展{phrase}何か行ける？"), f"intent phrase missing: {phrase}")

# 17. A single-day drop-in seed can still use the same end-time flow.
single_day_drop_in = dict(next(event for event in events if event["id"] == "028"))
single_day_drop_in["id"] = "998"
single_day_drop_in["start_datetime"] = "2028-11-03T09:40:00+09:00"
single_day_drop_in["end_datetime"] = "2028-11-03T18:00:00+09:00"
single_pending = recommendation_pending.make_pending_state(
    single_day_drop_in,
    awaiting="end_time",
    recommendation_date=date(2028, 11, 3),
)
single_time = recommendation_pending.resolve_pending_input(
    "午後1時", single_pending, [single_day_drop_in], POC_REFERENCE_DATE
)
check(single_time.handled and single_time.selected_end_override is not None, "single-day drop-in did not accept end time")

# 18. The selected-end override is applied only to drop-in seed events.
override_result = event_recommendation.recommend_next_events(
    period_drop_in,
    events,
    date(2028, 11, 3),
    selected_end_override=valid_time.selected_end_override,
)
check("010" in [event["id"] for event in override_result.events], "selected end override was not used")
start_override_result = event_recommendation.recommend_next_events(
    start_time_event,
    events,
    date(2028, 10, 22),
    selected_end_override=valid_time.selected_end_override,
)
check(start_override_result.events is not None, "start-time recommendation failed with optional override")

print("Pending Recommendation QA: PASS (18 cases)")
