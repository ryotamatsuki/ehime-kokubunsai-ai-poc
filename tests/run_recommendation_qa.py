"""Standalone QA for recommendation date resolution and entry modes."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from event_details import normalize_schedule, validate_events_v2  # noqa: E402
from event_recommendation import (  # noqa: E402
    DROP_IN_ENTRY,
    DROP_IN_MIN_VISIT_MINUTES,
    START_TIME_ENTRY,
    recommend_next_events,
    resolve_recommendation_date,
)
from event_search import load_events  # noqa: E402


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def event_by_id(events: list[dict[str, object]], event_id: str) -> dict[str, object]:
    return next(event for event in events if event["id"] == event_id)


events = load_events()
validate_events_v2(events)
check(len(events) == 30, "events.json must contain 30 events")

schema = json.loads((ROOT / "data" / "events.schema.json").read_text(encoding="utf-8"))
schema_item = schema["items"]
check("参加形式" in schema_item["required"], "schema must require 参加形式")
check(
    schema_item["properties"]["参加形式"]["enum"] == [START_TIME_ENTRY, DROP_IN_ENTRY],
    "schema entry-mode enum is incorrect",
)
check(
    all(event["参加形式"] in {START_TIME_ENTRY, DROP_IN_ENTRY} for event in events),
    "every event must have a valid entry mode",
)
check(sum(event["参加形式"] == DROP_IN_ENTRY for event in events) == 4, "drop-in count changed")
check(sum(event["参加形式"] == START_TIME_ENTRY for event in events) == 26, "start-time count changed")
check("参加形式" not in json.loads((ROOT / "data" / "search_metadata.json").read_text(encoding="utf-8"))[0], "entry mode leaked to search metadata")

haiku = event_by_id(events, "002")
period = event_by_id(events, "007")
walk = event_by_id(events, "016")
drop_in_art = event_by_id(events, "028")
period_schedule = normalize_schedule(period)
check(period_schedule.start_date == date(2028, 10, 21), "period start changed")
check(period_schedule.end_date == date(2028, 11, 26), "period end changed")
check(period_schedule.daily_start_time.isoformat(timespec="minutes") == "09:30", "daily start changed")
check(period_schedule.daily_end_time.isoformat(timespec="minutes") == "17:00", "daily end changed")

# Single-day events use their own date, even though the PoC reference date is
# 2028-11-03.
single_day = resolve_recommendation_date(haiku, "そのあと何か行ける？", date(2028, 11, 3))
check(single_day.recommendation_date == date(2028, 10, 22), "single-day date was not resolved from event")

# A period event never silently falls back to the PoC reference date.
missing_day = resolve_recommendation_date(period, "このあと何か行ける？", date(2028, 11, 3))
check(missing_day.recommendation_date is None, "period event silently used reference date")
check("何日に" in missing_day.message, "period clarification is missing")

explicit_day = resolve_recommendation_date(
    period,
    "11月3日に別子銅山展を見たあと何か行ける？",
    date(2028, 11, 3),
)
check(explicit_day.recommendation_date == date(2028, 11, 3), "explicit period date was not used")

previous_day = resolve_recommendation_date(
    period,
    "このあと何か行ける？",
    date(2028, 11, 3),
    previous_filters={"dates": ["2028-11-03"]},
)
check(previous_day.recommendation_date == date(2028, 11, 3), "previous search date was not retained")

outside_day = resolve_recommendation_date(
    period,
    "12月1日に別子銅山展のあと何か行ける？",
    date(2028, 11, 3),
)
check(outside_day.recommendation_date is None, "outside period date was accepted")
check("開催期間外" in outside_day.message, "outside period message is missing")

# Existing single-day next-event behavior remains intact.
haiku_next = recommend_next_events(haiku, events, date(2028, 10, 22))
check([event["id"] for event in haiku_next.events] == ["003", "028"], "single-day next ranking changed")

# Event 024 is already open when event 016 ends.  It remains a valid same-
# region drop-in candidate with 60 minutes left after the simple buffer.
walk_next = recommend_next_events(walk, events, date(2028, 11, 3))
check("024" in [event["id"] for event in walk_next.events], "ongoing drop-in event was excluded")
check("028" not in [event["id"] for event in walk_next.events], "different-region event was included")
check("016" not in [event["id"] for event in walk_next.events], "selected event was included")

# Reclassifying the same exhibition as start-time participation must exclude
# it when its start time has already passed.
start_time_copy = dict(drop_in_art)
start_time_copy["id"] = "998"
start_time_copy["参加形式"] = START_TIME_ENTRY
start_time_result = recommend_next_events(haiku, events + [start_time_copy], date(2028, 10, 22))
check("998" not in [event["id"] for event in start_time_result.events], "start-time event allowed late entry")

# The actual drop-in exhibition is open until 18:00; after the same-city
# 30-minute buffer from the haiku event, exactly 60 minutes remain.
drop_in_result = recommend_next_events(haiku, events, date(2028, 10, 22))
check("028" in [event["id"] for event in drop_in_result.events], "drop-in event with minimum visit time was excluded")
check(DROP_IN_MIN_VISIT_MINUTES == 60, "drop-in minimum visit assumption changed")

past_result = recommend_next_events(period, events, date(2028, 11, 3))
check("002" not in [event["id"] for event in past_result.events], "past event was recommended")
check("実際の移動時間は計算していません" in past_result.message, "movement disclaimer disappeared")

print("Recommendation QA: PASS")
print("Entry modes: 開始時刻参加=26, 随時入場=4")
