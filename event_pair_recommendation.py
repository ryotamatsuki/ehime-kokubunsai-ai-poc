"""Deterministic same-day event-pair recommendation for the PoC.

This module deliberately performs only schedule and structured-data checks.
It has no latitude/longitude, road-network, or traffic data, so a feasible
pair is not described as the shortest route or as a measured walking/travel
time.  The same-city buffer is an explicit PoC assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import re
import unicodedata
from typing import Any, Mapping, Sequence

import age_semantics
from app_config import CITY_ALIASES
from event_recommendation import (
    DROP_IN_ENTRY,
    DROP_IN_MIN_VISIT_MINUTES,
    START_TIME_ENTRY,
    normalize_schedule,
)


DEFAULT_SAME_CITY_BUFFER_MINUTES = 30
# Keep the existing next-event convention for events in different
# municipalities of the same region.  This is also a PoC assumption, not a
# route-time measurement.
SAME_REGION_BUFFER_MINUTES = 60

SAME_DAY_REASON = "same_day"
TIME_FEASIBLE_REASON = "time_feasible_under_poc_assumption"
SAME_CITY_BUFFER_REASON = "same_city_buffer_poc_assumption"
SAME_REGION_BUFFER_REASON = "same_region_buffer_poc_assumption"


@dataclass(frozen=True)
class EventPair:
    """An ordered pair whose schedules can be followed on one day."""

    first_event_id: str
    second_event_id: str
    reasons: tuple[str, ...]


def _event_id(event: Mapping[str, Any]) -> str:
    value = event.get("id")
    if value is not None:
        return str(value)
    url = event.get("公式URL")
    if url:
        return str(url).rstrip("/").rsplit("/", 1)[-1]
    raise KeyError("イベントIDまたは公式URLがありません")


def _text(event: Mapping[str, Any], key: str) -> str | None:
    value = event.get(key)
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def _city(event: Mapping[str, Any]) -> str | None:
    return _text(event, "市町")


def _region(event: Mapping[str, Any]) -> str | None:
    return _text(event, "地域")


def _entry_mode(event: Mapping[str, Any]) -> str:
    return str(event.get("参加形式", START_TIME_ENTRY))


def _as_values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = (value,)
    elif not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _canonical_city(value: str) -> str:
    value = value.strip()
    return CITY_ALIASES.get(value, value)


def _event_is_free(event: Mapping[str, Any]) -> bool:
    fee = event.get("料金構造")
    if isinstance(fee, Mapping):
        general_fee = fee.get("一般料金円")
        if isinstance(general_fee, int) and not isinstance(general_fee, bool):
            return general_fee == 0
    return "無料" in str(event.get("料金", ""))


def _entry_fee(event: Mapping[str, Any]) -> int | None:
    fee = event.get("料金構造")
    if isinstance(fee, Mapping):
        general_fee = fee.get("一般料金円")
        if isinstance(general_fee, int) and not isinstance(general_fee, bool):
            return general_fee
    match = re.search(r"\d[\d,]*", str(event.get("料金", "")))
    if match is None:
        return 0 if _event_is_free(event) else None
    return int(match.group(0).replace(",", ""))


def _compact_text(value: Any) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value))).lower()


def _filters_mapping(filters: Mapping[str, Any] | Any | None) -> Mapping[str, Any]:
    if filters is None:
        return {}
    if isinstance(filters, Mapping):
        return filters
    to_dict = getattr(filters, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return value
    raise TypeError("filtersはMappingまたはto_dict()を持つ値で指定してください")


def _matches_filters(
    event: Mapping[str, Any],
    filters: Mapping[str, Any],
    municipalities: Sequence[str],
    day: date,
) -> bool:
    explicit_municipalities = _as_values(municipalities)
    if not explicit_municipalities:
        explicit_municipalities = _as_values(
            filters.get(
                "municipalities",
                filters.get("municipality", filters.get("city")),
            )
        )
    requested_cities = {_canonical_city(value) for value in explicit_municipalities}
    if requested_cities and _city(event) not in requested_cities:
        return False

    requested_regions = set(
        _as_values(filters.get("regions", filters.get("region")))
    )
    if requested_regions and _region(event) not in requested_regions:
        return False

    requested_ids = set(_as_values(filters.get("event_ids", filters.get("event_id"))))
    if requested_ids and _event_id(event) not in requested_ids:
        return False

    genre_groups = filters.get("genre_groups")
    event_genre = str(event.get("ジャンル", ""))
    if isinstance(genre_groups, Sequence) and not isinstance(genre_groups, str) and genre_groups:
        for group in genre_groups:
            values = _as_values(group)
            if values and not any(value in event_genre for value in values):
                return False
    else:
        requested_genres = _as_values(filters.get("genres"))
        if requested_genres and not any(value in event_genre for value in requested_genres):
            return False

    if filters.get("child_friendly") is True and event.get("子ども向け") is not True:
        return False
    age = filters.get("age")
    age_group = filters.get("age_group")
    if age is not None or age_group:
        if age_semantics.event_age_match(
            event,
            age=age if isinstance(age, int) and not isinstance(age, bool) else None,
            age_group=str(age_group) if age_group else None,
            age_intent=str(filters.get("age_intent") or "") or None,
        ) == "excluded":
            return False
    if filters.get("entry_free") is True and not _event_is_free(event):
        return False
    if filters.get("paid_only") is True and _event_is_free(event):
        return False

    soft_terms = _as_values(filters.get("soft_terms"))
    if soft_terms:
        searchable = " ".join(
            [
                str(event.get("イベント名", "")),
                *[str(value) for value in event.get("aliases", [])],
                *[str(value) for value in event.get("search_tags", [])],
                str(event.get("ジャンル", "")),
                str(event.get("概要", "")),
            ]
        )
        compact = _compact_text(searchable)
        if not all(_compact_text(term) in compact for term in soft_terms):
            return False

    max_entry_fee = filters.get("max_entry_fee")
    if max_entry_fee is not None:
        fee = _entry_fee(event)
        try:
            if fee is None or fee > int(max_entry_fee):
                return False
        except (TypeError, ValueError):
            return False

    reservation_required = filters.get("reservation_required")
    if reservation_required is not None:
        guide = event.get("参加案内")
        actual = guide.get("申込要否") if isinstance(guide, Mapping) else None
        if reservation_required is True or str(reservation_required) in {"必要", "required"}:
            if actual != "必要":
                return False
        elif reservation_required is False or str(reservation_required) in {"不要", "not_required"}:
            if actual != "不要":
                return False
        else:
            return False

    venue = str(filters.get("venue") or "").strip().lower()
    event_venue = str(event.get("屋内/屋外", ""))
    if venue in {"屋内", "室内", "indoor"} and event_venue != "屋内":
        return False
    if venue in {"屋外", "outdoor"} and "屋外" not in event_venue:
        return False
    if filters.get("rain_preferred") is True:
        rain_policy = event.get("雨天時対応")
        policy = rain_policy.get("開催方針") if isinstance(rain_policy, Mapping) else ""
        if "屋内" not in event_venue and "決行" not in str(policy):
            return False

    schedule = normalize_schedule(event)
    time_after = filters.get("time_after")
    if time_after is not None:
        try:
            if schedule.ends_at(day).hour * 60 + schedule.ends_at(day).minute < int(time_after):
                return False
        except (TypeError, ValueError):
            return False
    time_slots = _as_values(filters.get("time_slots"))
    start_minutes = schedule.daily_start_time.hour * 60 + schedule.daily_start_time.minute
    end_minutes = schedule.daily_end_time.hour * 60 + schedule.daily_end_time.minute
    for slot in time_slots:
        if slot == "午前" and start_minutes >= 12 * 60:
            return False
        if slot == "午後" and end_minutes <= 12 * 60:
            return False
        if slot == "夕方" and end_minutes < 17 * 60:
            return False

    return True


def _candidate_events(
    events: Sequence[Mapping[str, Any]],
    day: date,
    municipalities: Sequence[str],
    filters: Mapping[str, Any] | Any | None,
) -> list[Mapping[str, Any]]:
    normalized_filters = _filters_mapping(filters)
    candidates: list[Mapping[str, Any]] = []
    for event in events:
        schedule = normalize_schedule(event)
        if schedule.active_on(day) and _matches_filters(
            event, normalized_filters, municipalities, day
        ):
            candidates.append(event)
    return candidates


def _buffer_for(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    same_city_buffer_minutes: int,
) -> tuple[int | None, str | None]:
    first_city = _city(first)
    second_city = _city(second)
    if first_city and first_city == second_city:
        return same_city_buffer_minutes, SAME_CITY_BUFFER_REASON

    first_region = _region(first)
    second_region = _region(second)
    if first_region and first_region == second_region:
        return SAME_REGION_BUFFER_MINUTES, SAME_REGION_BUFFER_REASON
    return None, None


def _validate_buffer_minutes(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("same_city_buffer_minutesは0以上の整数で指定してください")


def can_follow(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    day: date,
    *,
    same_city_buffer_minutes: int = DEFAULT_SAME_CITY_BUFFER_MINUTES,
) -> bool:
    """Return whether ``second`` can follow ``first`` on ``day``.

    A start-time event requires arrival before its start.  A drop-in event can
    be entered after arrival, but must still have
    ``DROP_IN_MIN_VISIT_MINUTES`` remaining.  A drop-in first event is treated
    as lasting through its scheduled end because this API has no user-provided
    leave-time override.
    """

    _validate_buffer_minutes(same_city_buffer_minutes)
    if _event_id(first) == _event_id(second):
        return False

    first_schedule = normalize_schedule(first)
    second_schedule = normalize_schedule(second)
    if not first_schedule.active_on(day) or not second_schedule.active_on(day):
        return False

    buffer_minutes, _ = _buffer_for(first, second, same_city_buffer_minutes)
    if buffer_minutes is None:
        return False

    arrival_time = first_schedule.ends_at(day) + timedelta(minutes=buffer_minutes)
    second_start = second_schedule.starts_at(day)
    second_end = second_schedule.ends_at(day)

    if _entry_mode(second) == DROP_IN_ENTRY:
        visit_start = max(arrival_time, second_start)
        return second_end >= visit_start + timedelta(minutes=DROP_IN_MIN_VISIT_MINUTES)
    return second_start >= arrival_time


def _pair_reasons(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> tuple[str, ...]:
    _, buffer_reason = _buffer_for(
        first,
        second,
        DEFAULT_SAME_CITY_BUFFER_MINUTES,
    )
    if buffer_reason is None:
        # This branch is defensive; can_follow() has already rejected pairs
        # without a same-city or same-region relationship.
        buffer_reason = SAME_REGION_BUFFER_REASON
    return (SAME_DAY_REASON, TIME_FEASIBLE_REASON, buffer_reason)


def recommend_event_pairs(
    events: Sequence[Mapping[str, Any]],
    day: date,
    *,
    municipalities: Sequence[str] = (),
    filters: Mapping[str, Any] | Any | None = None,
    limit: int = 3,
    same_city_buffer_minutes: int = DEFAULT_SAME_CITY_BUFFER_MINUTES,
) -> list[EventPair]:
    """Return deterministic ordered event pairs for one day.

    Candidate filtering is structured and local.  Ranking prefers pairs in
    one municipality, then pairs whose second event starts after the first
    event's scheduled end, and finally preserves schedule/input order.  No
    ranking criterion represents a measured route distance or travel time.
    """

    _validate_buffer_minutes(same_city_buffer_minutes)
    if limit <= 0:
        return []

    candidates = _candidate_events(events, day, municipalities, filters)
    ranked: list[tuple[tuple[Any, ...], EventPair]] = []
    schedules = [normalize_schedule(event) for event in candidates]

    for first_index, first in enumerate(candidates):
        for second_index, second in enumerate(candidates):
            if first_index == second_index or _event_id(first) == _event_id(second):
                continue
            if not can_follow(
                first,
                second,
                day,
                same_city_buffer_minutes=same_city_buffer_minutes,
            ):
                continue

            first_schedule = schedules[first_index]
            second_schedule = schedules[second_index]
            same_city = _city(first) is not None and _city(first) == _city(second)
            second_starts_after_first = (
                second_schedule.starts_at(day) >= first_schedule.ends_at(day)
            )
            rank_key = (
                0 if same_city else 1,
                0 if second_starts_after_first else 1,
                first_schedule.starts_at(day),
                second_schedule.starts_at(day),
                first_index,
                second_index,
            )
            ranked.append(
                (
                    rank_key,
                    EventPair(
                        first_event_id=_event_id(first),
                        second_event_id=_event_id(second),
                        reasons=_pair_reasons(first, second),
                    ),
                )
            )

    ranked.sort(key=lambda item: item[0])
    return [pair for _, pair in ranked[:limit]]


__all__ = [
    "DEFAULT_SAME_CITY_BUFFER_MINUTES",
    "EventPair",
    "SAME_CITY_BUFFER_REASON",
    "SAME_DAY_REASON",
    "SAME_REGION_BUFFER_MINUTES",
    "SAME_REGION_BUFFER_REASON",
    "TIME_FEASIBLE_REASON",
    "can_follow",
    "recommend_event_pairs",
]
