"""Pure pending-state transitions for next-event recommendation.

The Streamlit layer owns session_state.  This module only interprets a short
date/time reply and returns the next deterministic action.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Mapping, Sequence

import event_recommendation
from event_details import normalize_schedule


@dataclass(frozen=True)
class PendingDecision:
    """The result of applying one user turn to a pending recommendation."""

    handled: bool
    answer: str | None = None
    next_state: dict[str, str] | None = None
    event: dict[str, Any] | None = None
    recommendation_date: date | None = None
    selected_end_override: datetime | None = None
    clear: bool = False


def make_pending_state(
    event: Mapping[str, Any],
    *,
    awaiting: str,
    recommendation_date: date | None = None,
) -> dict[str, str]:
    """Create the minimal session state needed for one pending turn."""

    state = {
        "mode": "next",
        "event_id": str(event.get("id") or ""),
        "awaiting": awaiting,
    }
    if recommendation_date is not None:
        state["date"] = recommendation_date.isoformat()
    return state


def is_reset_query(query: str) -> bool:
    return event_recommendation.parse_query_text(query).lower() in {
        "リセット",
        "会話をリセット",
        "reset",
    }


def _event_by_id(
    events: Sequence[Mapping[str, Any]],
    event_id: object,
) -> dict[str, Any] | None:
    wanted = str(event_id or "")
    for event in events:
        if str(event.get("id") or "") == wanted:
            return dict(event)
    return None


def _date_label(value: date) -> str:
    return f"{value.year}年{value.month}月{value.day}日"


def resolve_pending_input(
    query: str,
    pending: Mapping[str, Any] | None,
    events: Sequence[Mapping[str, Any]],
    reference_date: date,
) -> PendingDecision:
    """Apply a date/time-shaped reply, or decline to consume the turn.

    A non-date/non-time turn returns ``handled=False`` so the caller can clear
    the pending state and route the new question normally.
    """

    if not pending or pending.get("mode") != "next":
        return PendingDecision(False)
    event = _event_by_id(events, pending.get("event_id"))
    if event is None:
        return PendingDecision(
            True,
            answer="推薦対象のイベント情報を再取得できませんでした。イベント名からもう一度探してみて。",
            clear=True,
        )

    awaiting = str(pending.get("awaiting") or "")
    if awaiting == "date":
        parsed = event_recommendation.parse_recommendation_date_answer(query, reference_date)
        if not parsed.is_date_like:
            return PendingDecision(False)
        if parsed.invalid or parsed.value is None:
            return PendingDecision(
                True,
                answer="日付を解釈できませんでした。例：11月3日、11/3、3日",
                next_state=dict(pending),
            )
        recommendation_date = parsed.value
        schedule = normalize_schedule(event)
        if not schedule.active_on(recommendation_date):
            return PendingDecision(
                True,
                answer=(
                    f"{_date_label(recommendation_date)}は選択中のイベントの開催期間外です。"
                    "その日の次の候補は計算しません。"
                ),
                clear=True,
            )
        if str(event.get("参加形式")) == event_recommendation.DROP_IN_ENTRY:
            return PendingDecision(
                True,
                answer="何時ごろ見終わる予定？",
                next_state=make_pending_state(
                    event,
                    awaiting="end_time",
                    recommendation_date=recommendation_date,
                ),
                event=event,
            )
        return PendingDecision(
            True,
            event=event,
            recommendation_date=recommendation_date,
            clear=True,
        )

    if awaiting == "end_time":
        parsed = event_recommendation.parse_recommendation_time_answer(query)
        if not parsed.is_time_like:
            return PendingDecision(False)
        if parsed.invalid or parsed.value is None:
            return PendingDecision(
                True,
                answer="イベントの開催時間内で、何時ごろ見終わる予定か教えてみて。",
                next_state=dict(pending),
            )
        try:
            recommendation_date = date.fromisoformat(str(pending.get("date")))
        except ValueError:
            return PendingDecision(
                True,
                answer="日付が確認できません。何日に行く予定か教えてみて。",
                clear=True,
            )
        schedule = normalize_schedule(event)
        end_datetime = datetime.combine(recommendation_date, parsed.value)
        if not schedule.active_on(recommendation_date) or not (
            schedule.starts_at(recommendation_date)
            <= end_datetime
            <= schedule.ends_at(recommendation_date)
        ):
            return PendingDecision(
                True,
                answer="イベントの開催時間内で、何時ごろ見終わる予定か教えてみて。",
                next_state=dict(pending),
            )
        return PendingDecision(
            True,
            event=event,
            recommendation_date=recommendation_date,
            selected_end_override=end_datetime,
            clear=True,
        )

    return PendingDecision(
        True,
        answer="推薦条件を確認できませんでした。イベント名からもう一度探してみて。",
        clear=True,
    )
