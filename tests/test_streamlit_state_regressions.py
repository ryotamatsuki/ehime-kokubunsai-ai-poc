from __future__ import annotations

from datetime import date

import event_search
from command_models import CommandPlan, CommandSlots
from command_orchestrator import CommandOrchestrator


def _event(event_id: str):
    return next(event for event in event_search.load_events() if str(event.get("id")) == event_id)


def test_selected_single_day_event_uses_its_event_date_not_poc_current_date():
    events = event_search.load_events()
    selected = _event("002")
    state = {
        "reference_date": "2028-11-03",
        "selected_event_id": "002",
        "last_result_ids": ["001", "002", "003"],
    }
    plan = CommandPlan(
        flow="recommend_next",
        slots=CommandSlots(reference_kind="selected"),
        confidence="high",
    )

    result = CommandOrchestrator(
        reference_date=date(2028, 11, 3),
        events=events,
    ).handle_query(
        "そのイベントのあと何か行ける？",
        state,
        command_plan=plan,
    )

    assert selected["日時"].startswith("2028-10-22")
    assert "2028年11月3日は選択中のイベントの開催期間外" not in result.message
    assert result.status == "ok"
    assert any(str(event.get("id")) == "003" for event in result.events)


def test_period_event_date_remains_in_trusted_command_when_asking_for_end_time():
    events = event_search.load_events()
    period_event = next(
        event
        for event in events
        if event.get("参加形式") == "随時入場"
        and "〜" in str(event.get("日時", ""))
    )
    event_id = str(period_event["id"])
    plan = CommandPlan(
        flow="recommend_next",
        slots=CommandSlots(
            reference_kind="selected",
            dates=("2028-11-03",),
        ),
        confidence="high",
    )
    state = {
        "reference_date": "2028-11-03",
        "selected_event_id": event_id,
        "last_result_ids": [event_id],
    }

    result = CommandOrchestrator(
        reference_date=date(2028, 11, 3),
        events=events,
    ).handle_query(
        "11月3日",
        state,
        command_plan=plan,
    )

    assert result.status == "clarification"
    assert result.pending is not None
    assert "time_after" in result.pending["missing_slots"]
    assert result.command.slots.dates == ("2028-11-03",)
