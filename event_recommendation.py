"""Deterministic next-event and similarity ranking for the PoC."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
import re
from typing import Any, Mapping, Sequence

from event_details import EventSchedule, normalize_schedule
from event_search import normalize_query, parse_query


START_TIME_ENTRY = "開始時刻参加"
DROP_IN_ENTRY = "随時入場"
DROP_IN_MIN_VISIT_MINUTES = 60


@dataclass(frozen=True)
class RecommendationResult:
    events: list[dict[str, Any]] = field(default_factory=list)
    reasons: dict[str, tuple[str, ...]] = field(default_factory=dict)
    message: str = ""


@dataclass(frozen=True)
class RecommendationDateResolution:
    """The date selected for a next-event calculation, if it is known."""

    recommendation_date: date | None
    message: str = ""


@dataclass(frozen=True)
class RecommendationDateAnswer:
    """Result of parsing a short answer to a pending date question."""

    is_date_like: bool
    value: date | None = None
    invalid: bool = False


@dataclass(frozen=True)
class RecommendationTimeAnswer:
    """Result of parsing a short answer to a pending end-time question."""

    is_time_like: bool
    value: time | None = None
    invalid: bool = False


_PENDING_ANSWER_PREFIXES = ("たぶん", "じゃあ", "えっと")
_PENDING_DATE_SUFFIXES = tuple(
    sorted(
        (
            "でお願い",
            "に行くよ",
            "に行く",
            "にしよう",
            "がいい",
            "くらい",
            "ごろ",
            "頃",
            "だよ",
            "です",
            "かな",
            "で",
        ),
        key=len,
        reverse=True,
    )
)
_PENDING_TIME_SUFFIXES = tuple(
    sorted(
        (
            "くらいかな",
            "ごろかな",
            "頃かな",
            "でお願い",
            "くらい",
            "ごろ",
            "頃",
            "だよ",
            "です",
            "かな",
            "で",
        ),
        key=len,
        reverse=True,
    )
)
_PENDING_DATE_CORE_RE = re.compile(
    r"(?:今日|本日|明日|(?:(?:20\d{2})年)?\d{1,2}月\d{1,2}日|"
    r"(?:(?:20\d{2})[/-])?\d{1,2}[/-]\d{1,2}|\d{1,2}日)"
)
_PENDING_TIME_CORE_RE = re.compile(
    r"(?:(?:午前|午後))?\d{1,2}(?:(?::\d{2})|時(?:(?:半)|(?:ごろ|頃|くらい)?)?)"
)
PENDING_CONTINUE_CURRENT_FLOW = "continue_current_flow"
PENDING_INTERRUPT_WITH_NEW_FLOW = "interrupt_with_new_flow"
PENDING_AMBIGUOUS = "ambiguous"
_PENDING_NEW_FLOW_MARKERS = (
    "やっぱり",
    "普通に",
    "探して",
    "探す",
    "おすすめ",
    "イベント",
    "県内",
    "無料",
    "有料",
    "予約",
    "子ども",
    "子供",
    "料金",
    "場所",
    "似た",
    "市で",
    "町で",
    "村で",
)
_PENDING_NON_INTENT_TERMS = {
    "いつか",
    "そのうち",
    "まだ",
    "未定",
    "わからない",
    "分からない",
    "いいよ",
}


def _pending_answer_core(
    query: str,
    core_pattern: re.Pattern[str],
    suffixes: Sequence[str],
) -> str | None:
    """Return a date/time core only for an anchored short conversational reply.

    Pending answers are intentionally narrower than general query parsing.  A
    core date or time may have one known conversational prefix and/or suffix,
    but arbitrary text surrounding it is not consumed as a slot answer.
    """

    compact = normalize_query(str(query)).replace(" ", "")
    if not compact:
        return None

    bases = [compact.rstrip("。！？!?"), compact]
    for prefix in _PENDING_ANSWER_PREFIXES:
        for base in tuple(bases):
            if base.startswith(prefix):
                remainder = base[len(prefix) :].lstrip("、,：:")
                bases.append(remainder)

    seen: set[str] = set()
    for base in bases:
        if not base or base in seen:
            continue
        seen.add(base)
        if core_pattern.fullmatch(base):
            return base
        for suffix in suffixes:
            if not base.endswith(suffix):
                continue
            candidate = base[: -len(suffix)].rstrip("。！？!?")
            if core_pattern.fullmatch(candidate):
                return candidate
    return None


def _format_date_ja(value: date) -> str:
    return f"{value.year}年{value.month}月{value.day}日"


def _parsed_query_dates(query: str, reference_date: date) -> list[date]:
    filters = parse_query(query, reference_date)
    return [date.fromisoformat(value) for value in filters.dates]


def parse_recommendation_date_answer(
    query: str,
    reference_date: date,
) -> RecommendationDateAnswer:
    """Parse only a short, date-shaped reply to a pending date question."""

    compact = _pending_answer_core(
        query,
        _PENDING_DATE_CORE_RE,
        _PENDING_DATE_SUFFIXES,
    )
    if compact is None:
        return RecommendationDateAnswer(False)
    day_match = re.fullmatch(r"(\d{1,2})日", compact)
    if day_match is not None:
        try:
            return RecommendationDateAnswer(
                True,
                date(reference_date.year, reference_date.month, int(day_match.group(1))),
            )
        except ValueError:
            return RecommendationDateAnswer(True, invalid=True)
    parsed = _parsed_query_dates(compact, reference_date)
    if len(parsed) == 1:
        return RecommendationDateAnswer(True, parsed[0])
    return RecommendationDateAnswer(True, invalid=True)


def parse_query_text(query: str) -> str:
    """Normalize a short recommendation answer without broad interpretation."""

    return normalize_query(str(query))


def parse_recommendation_time_answer(query: str) -> RecommendationTimeAnswer:
    """Parse 13時, 13:00, １３時, and 午後1時 style replies."""

    compact = _pending_answer_core(
        query,
        _PENDING_TIME_CORE_RE,
        _PENDING_TIME_SUFFIXES,
    )
    if compact is None:
        return RecommendationTimeAnswer(False)
    match = re.fullmatch(
        r"(?:(午前|午後))?(\d{1,2})(?:(?::(\d{2}))|時(?:(半)|(?:ごろ|頃|くらい)?)?)?",
        compact,
    )
    if match is None or (match.group(3) is None and "時" not in compact and ":" not in compact):
        return RecommendationTimeAnswer(False)
    period, hour_text, minute_text, half = match.groups()
    hour = int(hour_text)
    minute = 30 if half else int(minute_text or 0)
    if period:
        if hour < 1 or hour > 12:
            return RecommendationTimeAnswer(True, invalid=True)
        if period == "午前" and hour == 12:
            hour = 0
        elif period == "午後" and hour < 12:
            hour += 12
    elif hour > 23 or minute > 59:
        return RecommendationTimeAnswer(True, invalid=True)
    try:
        return RecommendationTimeAnswer(True, time(hour, minute))
    except ValueError:
        return RecommendationTimeAnswer(True, invalid=True)


def classify_pending_answer(
    query: str,
    *,
    awaiting: str,
    reference_date: date,
) -> str:
    """Classify a turn while a flow is waiting for one slot.

    The caller can therefore preserve the pending state for an ambiguous turn
    instead of treating every parser miss as permission to discard context.
    """

    normalized = normalize_query(str(query))
    if awaiting in {"date", "dates"}:
        if parse_recommendation_date_answer(normalized, reference_date).is_date_like:
            return PENDING_CONTINUE_CURRENT_FLOW
    elif awaiting in {"time", "time_after", "end_time"}:
        if parse_recommendation_time_answer(normalized).is_time_like:
            return PENDING_CONTINUE_CURRENT_FLOW
    else:
        return PENDING_AMBIGUOUS

    if any(marker in normalized for marker in _PENDING_NEW_FLOW_MARKERS):
        return PENDING_INTERRUPT_WITH_NEW_FLOW

    parsed = parse_query(normalized, reference_date, include_unknown_residual=False)
    meaningful_soft_terms = [
        term for term in parsed.soft_terms if term not in _PENDING_NON_INTENT_TERMS
    ]
    meaningful_requested_field = (
        parsed.requested_field
        if normalized not in _PENDING_NON_INTENT_TERMS
        else None
    )
    if (
        parsed.dates
        or parsed.city_groups
        or parsed.region_groups
        or parsed.genres
        or parsed.age is not None
        or parsed.age_group
        or parsed.venue
        or parsed.entry_free is not None
        or parsed.paid_only
        or parsed.reservation_required is not None
        or meaningful_requested_field
        or meaningful_soft_terms
    ):
        return PENDING_INTERRUPT_WITH_NEW_FLOW
    return PENDING_AMBIGUOUS


def _previous_filter_dates(previous_filters: Mapping[str, Any] | None) -> list[date]:
    if not previous_filters:
        return []
    values = previous_filters.get("dates", [])
    if not isinstance(values, (list, tuple)):
        return []
    result: list[date] = []
    for value in values:
        try:
            result.append(date.fromisoformat(str(value)))
        except ValueError:
            continue
    return result


def resolve_recommendation_date(
    selected_event: Mapping[str, Any],
    query: str,
    reference_date: date,
    *,
    previous_filters: Mapping[str, Any] | None = None,
) -> RecommendationDateResolution:
    """Resolve the day used by a next-event recommendation.

    A single-day event supplies its own day.  A period event requires an
    explicit day from the current query or the immediately preceding search;
    the fixed PoC date is never used as a silent fallback for that case.
    """

    schedule = normalize_schedule(selected_event)
    if schedule.start_date == schedule.end_date:
        return RecommendationDateResolution(schedule.start_date)

    query_dates = _parsed_query_dates(query, reference_date)
    if not query_dates:
        query_dates = _previous_filter_dates(previous_filters)
    if len(query_dates) != 1:
        return RecommendationDateResolution(
            None,
            "期間開催のイベントなので、何日に行く予定か教えてみて。",
        )

    requested_date = query_dates[0]
    if not schedule.active_on(requested_date):
        return RecommendationDateResolution(
            None,
            f"{_format_date_ja(requested_date)}は選択中のイベントの開催期間外です。その日の次の候補は計算しません。",
        )
    return RecommendationDateResolution(requested_date)


def is_next_query(query: str) -> bool:
    return any(
        term in query
        for term in (
            "このあと", "この後", "そのあと", "その後", "次に", "もう一つ", "行けそう",
            "のあと", "の後", "見たあと", "見た後", "終わったあと", "終わった後",
            "見終わったあと", "見終わった後",
        )
    )


def is_similar_query(query: str) -> bool:
    return any(term in query for term in ("似た", "似ている", "同じ系統", "他にも", "同じ地域で他"))


def _event_id(event: Mapping[str, Any]) -> str:
    return str(event.get("id") or str(event["公式URL"]).rstrip("/").rsplit("/", 1)[-1])


def _genre_tokens(value: str) -> set[str]:
    return {token for token in re.split(r"[・／/、,，\s]+", value) if token}


def _tags(event: Mapping[str, Any]) -> set[str]:
    return {str(value) for value in event.get("search_tags", [])}


def _region(event: Mapping[str, Any]) -> str | None:
    value = event.get("地域")
    return str(value) if value else None


def _city(event: Mapping[str, Any]) -> str | None:
    value = event.get("市町")
    return str(value) if value else None


def _same_scope_score(seed: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    if _city(seed) and _city(seed) == _city(candidate):
        score += 100
        reasons.append("同じ市町")
    elif _region(seed) and _region(seed) == _region(candidate):
        score += 50
        reasons.append("同じ地域")
    return score, reasons


def _preference_score(
    seed: Mapping[str, Any],
    candidate: Mapping[str, Any],
    preferences: Mapping[str, Any] | None,
) -> tuple[int, list[str]]:
    preferences = preferences or {}
    score, reasons = _same_scope_score(seed, candidate)
    if _genre_tokens(str(seed["ジャンル"])) & _genre_tokens(str(candidate["ジャンル"])):
        score += 35
        reasons.append("ジャンルが近い")
    common_tags = _tags(seed) & _tags(candidate)
    if common_tags:
        score += min(45, 12 * len(common_tags))
        reasons.append("検索タグが共通")
    if preferences.get("child_friendly") is True:
        if candidate.get("子ども向け") is True:
            score += 18
            reasons.append("子ども向け条件に一致")
        else:
            return -10_000, []
    if preferences.get("entry_free") is True:
        if str(candidate.get("料金")) in {"無料", "無料（事前申込制）"}:
            score += 12
            reasons.append("無料条件に一致")
        else:
            return -10_000, []
    scope = preferences.get("scope")
    if scope == "city" and _city(seed) != _city(candidate):
        return -10_000, []
    if scope == "region" and _region(seed) != _region(candidate):
        return -10_000, []
    return score, list(dict.fromkeys(reasons))


def recommend_next_events(
    selected_event: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    reference_date: date,
    *,
    limit: int = 3,
    selected_end_override: datetime | None = None,
) -> RecommendationResult:
    """Find same-day events that start after the selected event can end.

    The 30/60-minute buffers are explicit PoC assumptions, not route times.
    Events in another region are excluded because no movement data is present.
    """

    selected_schedule = normalize_schedule(selected_event)
    if not selected_schedule.active_on(reference_date):
        return RecommendationResult(
            message=f"{_format_date_ja(reference_date)}は選択中のイベントの開催期間外なので、このあとの候補は計算できません。"
        )
    selected_end = selected_schedule.ends_at(reference_date)
    if selected_end_override is not None and str(selected_event.get("参加形式")) == DROP_IN_ENTRY:
        override = selected_end_override.replace(tzinfo=None)
        if override.date() != reference_date or not (
            selected_schedule.starts_at(reference_date)
            <= override
            <= selected_schedule.ends_at(reference_date)
        ):
            return RecommendationResult(
                message="イベントの開催時間内で、何時ごろ見終わる予定か教えてみて。"
            )
        selected_end = override
    ranked: list[tuple[int, int, dict[str, Any], tuple[str, ...]]] = []
    for index, event in enumerate(events):
        if _event_id(event) == _event_id(selected_event):
            continue
        schedule = normalize_schedule(event)
        if not schedule.active_on(reference_date):
            continue
        if _region(selected_event) != _region(event):
            continue
        same_city = _city(selected_event) == _city(event)
        buffer = timedelta(minutes=30 if same_city else 60)
        arrival_time = selected_end + buffer
        candidate_start = schedule.starts_at(reference_date)
        candidate_end = schedule.ends_at(reference_date)
        if str(event.get("参加形式", START_TIME_ENTRY)) == DROP_IN_ENTRY:
            visit_start = max(arrival_time, candidate_start)
            minimum_visit_end = visit_start + timedelta(minutes=DROP_IN_MIN_VISIT_MINUTES)
            if candidate_end < minimum_visit_end:
                continue
        elif candidate_start < arrival_time:
            continue
        score, reasons = _preference_score(selected_event, event, None)
        score += 30 if same_city else 10
        ranked.append((score, index, dict(event), tuple(reasons)))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = ranked[: max(0, limit)]
    reasons = {_event_id(event): reason for _, _, event, reason in selected}
    if not selected:
        return RecommendationResult(
            message="同日で、終了後に簡易バッファを置いて参加できる候補は見つかりませんでした。実際の移動時間は計算していません。"
        )
    return RecommendationResult(
        events=[event for _, _, event, _ in selected],
        reasons=reasons,
        message="同日・終了後・簡易移動バッファで次に行けそうな候補です。実際の移動時間は計算していません。",
    )


def recommend_similar_events(
    selected_event: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    reference_date: date,
    *,
    preferences: Mapping[str, Any] | None = None,
    limit: int = 3,
) -> RecommendationResult:
    """Rank other active/future events by structured similarity."""

    ranked: list[tuple[int, int, dict[str, Any], tuple[str, ...]]] = []
    for index, event in enumerate(events):
        if _event_id(event) == _event_id(selected_event):
            continue
        if normalize_schedule(event).end_date < reference_date:
            continue
        score, reasons = _preference_score(selected_event, event, preferences)
        if score <= -10_000:
            continue
        ranked.append((score, index, dict(event), tuple(reasons)))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = ranked[: max(0, limit)]
    reasons = {_event_id(event): reason for _, _, event, reason in selected}
    if not selected:
        return RecommendationResult(message="条件に合う類似イベントは見つかりませんでした。")
    return RecommendationResult(
        events=[event for _, _, event, _ in selected],
        reasons=reasons,
        message="ジャンル・検索タグ・地域などが近い候補です。",
    )
