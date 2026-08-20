"""Allow-listed deterministic tools for Agentic Search.

The planner can choose a tool name, but it cannot create Python calls.  This
module is the only dispatcher and every tool reads the existing local JSON
source or an existing deterministic helper.
"""

from __future__ import annotations

from datetime import date
import re
from typing import Any, Iterable, Mapping, Sequence

import event_recommendation
import event_search
import faq_search
from agent_models import SearchSpec, ToolResult
from app_config import CITY_ALIASES, GENRE_ALIASES, MAX_SEARCH_RESULTS, POC_REFERENCE_DATE


TOOL_NAMES = frozenset(
    {
        "search_events",
        "count_events",
        "get_event_detail",
        "recommend_next_events",
        "recommend_similar_events",
        "search_faq",
    }
)


class InvalidToolError(ValueError):
    """Raised when a planner requests a tool outside the fixed allow-list."""


class ToolExecutionError(RuntimeError):
    """Raised for a malformed but allow-listed tool request."""


def _event_id(event: Mapping[str, Any]) -> str:
    return str(event.get("id") or str(event["公式URL"]).rstrip("/").rsplit("/", 1)[-1])


def _normalized(value: Any) -> str:
    return event_search.normalize_query(str(value)).replace(" ", "").lower()


def _as_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [item for item in value if item.strip()]
    return []


def _as_dates(value: Any) -> list[date] | None:
    if value is None:
        return []
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        return None
    result: list[date] = []
    for item in value:
        try:
            result.append(date.fromisoformat(item))
        except ValueError:
            return None
    return result


def _event_active_on(event: Mapping[str, Any], dates: Sequence[date] | None) -> bool:
    if dates is None:
        return False
    if not dates:
        return True
    start, end = event_search.parse_event_dates(str(event["日時"]))
    return any(start <= value <= end for value in dates)


def _event_time_bounds(event: Mapping[str, Any]) -> tuple[int, int]:
    times = re.findall(r"(\d{1,2}):(\d{2})", str(event["日時"]))
    if not times:
        return 0, 24 * 60
    return (
        int(times[0][0]) * 60 + int(times[0][1]),
        int(times[-1][0]) * 60 + int(times[-1][1]),
    )


def _matches_time(event: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    start_minutes, end_minutes = _event_time_bounds(event)
    slots = set(_as_strings(filters.get("time_slots")))
    if "午前" in slots and start_minutes >= 12 * 60:
        return False
    if "午後" in slots and end_minutes <= 12 * 60:
        return False
    if "夕方" in slots and end_minutes < 17 * 60:
        return False
    time_after = filters.get("time_after")
    if time_after is None:
        return True
    try:
        return end_minutes >= int(time_after)
    except (TypeError, ValueError):
        return False


def _matches_genres(event: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    event_genre = str(event["ジャンル"])
    groups = filters.get("genre_groups")
    if not isinstance(groups, list) or not groups:
        genres = _as_strings(filters.get("genres"))
        groups = [[genre] for genre in genres]
    for group in groups:
        if not isinstance(group, list) or not group:
            return False
        if not any(
            genre in GENRE_ALIASES and any(alias in event_genre for alias in GENRE_ALIASES[genre])
            for genre in group
        ):
            return False
    return True


def _matches_soft_terms(event: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    terms = _as_strings(filters.get("soft_terms"))
    if not terms:
        return True
    searchable = " ".join(
        [
            str(event.get("イベント名", "")),
            *[str(value) for value in event.get("aliases", [])],
            *[str(value) for value in event.get("search_tags", [])],
            str(event.get("ジャンル", "")),
            str(event.get("概要", "")),
        ]
    )
    compact = _normalized(searchable)
    # Planner exact searches intentionally require all explicit soft terms.
    # A later relaxed search can remove the terms and show reference cards.
    return all(_normalized(term) in compact for term in terms)


def _matches_age(event: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    age = filters.get("age")
    age_group = _normalized(filters.get("age_group", ""))
    age_intent = _normalized(filters.get("age_intent", ""))
    if age_group and age_group not in {"child", "children", "小学生", "子ども", "こども", "family", "家族"}:
        return False
    if age_intent and age_intent not in {"recommended", "対象", "おすすめ", "推奨"}:
        return False
    wants_child = filters.get("child_friendly") is True or age is not None or age_group in {
        "child",
        "children",
        "小学生",
        "子ども",
        "こども",
    }
    if not wants_child:
        return True
    if event.get("子ども向け") is not True:
        return False
    # The PoC data only guarantees a child-friendly boolean, not a detailed
    # age range.  Do not invent one: a numeric age is therefore interpreted as
    # a recommendation for child-friendly events.
    return age_intent in {"", "recommended", "対象", "おすすめ", "推奨"} or age_intent is None


def _matches_reservation(event: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    requested = filters.get("reservation_required")
    if requested is None:
        return True
    value = event.get("参加案内", {}).get("申込要否")
    if requested is True or str(requested) in {"必要", "required"}:
        return value == "必要"
    if requested is False or str(requested) in {"不要", "not_required"}:
        return value == "不要"
    return False


def _matches_event(event: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    if not _event_active_on(event, _as_dates(filters.get("dates"))):
        return False

    municipalities = {
        CITY_ALIASES.get(value, value)
        for value in _as_strings(filters.get("municipalities"))
    }
    if municipalities and str(event.get("市町")) not in municipalities:
        return False
    regions = set(_as_strings(filters.get("regions")))
    if regions and str(event.get("地域")) not in regions:
        return False
    if not _matches_genres(event, filters):
        return False
    if not _matches_age(event, filters):
        return False

    venue = _normalized(filters.get("venue", ""))
    event_venue = str(event.get("屋内/屋外", ""))
    if venue and venue not in {"屋内", "室内", "indoor", "屋外", "outdoor"}:
        return False
    if venue in {"屋内", "室内", "indoor"} and event_venue != "屋内":
        return False
    if venue in {"屋外", "outdoor"} and "屋外" not in event_venue:
        return False

    if filters.get("rain_preferred") is True:
        if "屋内" not in event_venue and "決行" not in str(event.get("雨天時対応", {}).get("開催方針", "")):
            return False
    if filters.get("entry_free") is True and not event_search.is_entry_free(str(event["料金"])):
        return False
    if filters.get("paid_only") is True and event_search.is_entry_free(str(event["料金"])):
        return False
    if filters.get("max_entry_fee") is not None:
        try:
            if event_search.base_entry_fee(str(event["料金"])) > int(filters["max_entry_fee"]):
                return False
        except (TypeError, ValueError):
            return False
    if not _matches_reservation(event, filters):
        return False
    if not _matches_time(event, filters):
        return False
    return _matches_soft_terms(event, filters)


def _ranking_score(event: Mapping[str, Any], filters: Mapping[str, Any], reference_date: date) -> tuple[int, int]:
    score = 0
    soft_terms = _as_strings(filters.get("soft_terms"))
    searchable = " ".join(
        [str(event.get("イベント名", "")), *map(str, event.get("aliases", [])), *map(str, event.get("search_tags", []))]
    )
    for term in soft_terms:
        if _normalized(term) in _normalized(event.get("イベント名", "")):
            score += 100
        elif _normalized(term) in _normalized(searchable):
            score += 50
    if filters.get("child_friendly") is True and event.get("子ども向け") is True:
        score += 15
    if filters.get("entry_free") is True and event_search.is_entry_free(str(event["料金"])):
        score += 10
    start, end = event_search.parse_event_dates(str(event["日時"]))
    if start <= reference_date <= end:
        distance = 0
    elif start > reference_date:
        distance = (start - reference_date).days
    else:
        distance = 10_000 + (reference_date - end).days
    return score, distance


def execute_structured_search(
    spec: SearchSpec,
    events: Iterable[dict[str, Any]] | None = None,
    reference_date: date = POC_REFERENCE_DATE,
) -> ToolResult:
    source_events = list(event_search.load_events() if events is None else events)
    matched = [event for event in source_events if _matches_event(event, spec.filters)]
    matched.sort(key=lambda event: (-_ranking_score(event, spec.filters, reference_date)[0], _ranking_score(event, spec.filters, reference_date)[1], source_events.index(event)))
    return ToolResult(
        search_id=spec.search_id,
        purpose=spec.purpose,
        total_matches=len(matched),
        events=[dict(event) for event in matched[:MAX_SEARCH_RESULTS]],
        all_event_ids=[_event_id(event) for event in matched],
        relaxed=spec.relaxed,
        relaxed_fields=spec.relaxed_fields,
    )


def execute_count_search(
    spec: SearchSpec,
    events: Iterable[dict[str, Any]] | None = None,
    reference_date: date = POC_REFERENCE_DATE,
) -> ToolResult:
    return execute_structured_search(spec, events, reference_date)


def _result_for_events(spec: SearchSpec, events: Sequence[Mapping[str, Any]], message: str = "") -> ToolResult:
    copied = [dict(event) for event in events]
    return ToolResult(
        search_id=spec.search_id,
        purpose=spec.purpose,
        total_matches=len(copied),
        events=copied[:MAX_SEARCH_RESULTS],
        all_event_ids=[_event_id(event) for event in copied],
        relaxed=spec.relaxed,
        relaxed_fields=spec.relaxed_fields,
        message=message,
    )


def execute_detail_lookup(
    spec: SearchSpec,
    events: Iterable[dict[str, Any]] | None = None,
) -> ToolResult:
    source_events = list(event_search.load_events() if events is None else events)
    requested_ids = set(_as_strings(spec.filters.get("event_ids")))
    if spec.filters.get("event_id") is not None:
        requested_ids.add(str(spec.filters["event_id"]))
    if not requested_ids:
        return _result_for_events(spec, [], "イベントIDが必要です。")
    matches = [event for event in source_events if _event_id(event) in requested_ids]
    return _result_for_events(spec, matches)


def _reference_date(filters: Mapping[str, Any], fallback: date) -> date:
    raw = filters.get("reference_date", fallback.isoformat())
    try:
        return date.fromisoformat(str(raw))
    except ValueError:
        return fallback


def execute_next_recommendation(
    spec: SearchSpec,
    events: Iterable[dict[str, Any]] | None = None,
    reference_date: date = POC_REFERENCE_DATE,
) -> ToolResult:
    source_events = list(event_search.load_events() if events is None else events)
    selected_id = str(spec.filters.get("selected_event_id") or spec.filters.get("event_id") or "")
    selected = next((event for event in source_events if _event_id(event) == selected_id), None)
    if selected is None:
        return _result_for_events(spec, [], "基準イベントを特定できませんでした。")
    target_date = _reference_date(spec.filters, reference_date)
    recommendation = event_recommendation.recommend_next_events(selected, source_events, target_date)
    return _result_for_events(spec, recommendation.events, recommendation.message)


def execute_similar_recommendation(
    spec: SearchSpec,
    events: Iterable[dict[str, Any]] | None = None,
    reference_date: date = POC_REFERENCE_DATE,
) -> ToolResult:
    source_events = list(event_search.load_events() if events is None else events)
    selected_id = str(spec.filters.get("selected_event_id") or spec.filters.get("event_id") or "")
    selected = next((event for event in source_events if _event_id(event) == selected_id), None)
    if selected is None:
        return _result_for_events(spec, [], "基準イベントを特定できませんでした。")
    recommendation = event_recommendation.recommend_similar_events(selected, source_events, _reference_date(spec.filters, reference_date))
    return _result_for_events(spec, recommendation.events, recommendation.message)


def execute_faq_search(spec: SearchSpec) -> ToolResult:
    query = str(spec.filters.get("query", ""))
    match = faq_search.find_faq(query)
    return ToolResult(
        search_id=spec.search_id,
        purpose=spec.purpose,
        total_matches=1 if match else 0,
        events=[],
        all_event_ids=[],
        relaxed=spec.relaxed,
        relaxed_fields=spec.relaxed_fields,
        message=match.answer if match else "一般FAQに該当する回答は見つかりませんでした。",
    )


def execute_tool(
    search_spec: SearchSpec,
    events: Iterable[dict[str, Any]] | None = None,
    reference_date: date = POC_REFERENCE_DATE,
) -> ToolResult:
    """Dispatch only to a statically allow-listed deterministic function."""

    if search_spec.tool not in TOOL_NAMES:
        raise InvalidToolError(search_spec.tool)
    if search_spec.tool == "search_events":
        return execute_structured_search(search_spec, events, reference_date)
    if search_spec.tool == "count_events":
        return execute_count_search(search_spec, events, reference_date)
    if search_spec.tool == "get_event_detail":
        return execute_detail_lookup(search_spec, events)
    if search_spec.tool == "recommend_next_events":
        return execute_next_recommendation(search_spec, events, reference_date)
    if search_spec.tool == "recommend_similar_events":
        return execute_similar_recommendation(search_spec, events, reference_date)
    if search_spec.tool == "search_faq":
        return execute_faq_search(search_spec)
    raise InvalidToolError(search_spec.tool)
