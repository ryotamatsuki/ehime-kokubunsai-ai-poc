"""Deterministic next-event and similarity ranking for the PoC."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import re
from typing import Any, Mapping, Sequence

from event_details import EventSchedule, normalize_schedule


@dataclass(frozen=True)
class RecommendationResult:
    events: list[dict[str, Any]] = field(default_factory=list)
    reasons: dict[str, tuple[str, ...]] = field(default_factory=dict)
    message: str = ""


def is_next_query(query: str) -> bool:
    return any(term in query for term in ("このあと", "この後", "次に", "もう一つ", "行けそう"))


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
) -> RecommendationResult:
    """Find same-day events that start after the selected event can end.

    The 30/60-minute buffers are explicit PoC assumptions, not route times.
    Events in another region are excluded because no movement data is present.
    """

    selected_schedule = normalize_schedule(selected_event)
    if not selected_schedule.active_on(reference_date):
        return RecommendationResult(message="選択中のイベントはPoC上の今日には開催されないため、このあとの候補は計算できません。")
    selected_end = selected_schedule.ends_at(reference_date)
    ranked: list[tuple[int, int, dict[str, Any], tuple[str, ...]]] = []
    for index, event in enumerate(events):
        if _event_id(event) == _event_id(selected_event):
            continue
        schedule = normalize_schedule(event)
        if not schedule.active_on(reference_date):
            continue
        if _region(selected_event) != _region(event):
            continue
        candidate_start = schedule.starts_at(reference_date)
        same_city = _city(selected_event) == _city(event)
        buffer = timedelta(minutes=30 if same_city else 60)
        if candidate_start < selected_end + buffer:
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
