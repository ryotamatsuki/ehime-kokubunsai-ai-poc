from __future__ import annotations

from agent_orchestrator import build_writer_payload
from app_config import MAX_RESULT_SET_SIZE, MAX_WRITER_CANDIDATES, POC_REFERENCE_DATE
from command_models import CommandPlan, CommandSlots
from command_orchestrator import CommandOrchestrator
from event_search import load_events, search_events
from result_pagination import next_visible_count, normalize_visible_count, visible_items


def test_search_keeps_all_ordered_ids_for_the_28_match_query() -> None:
    result = search_events("子どもと楽しめるイベント")

    assert result.total_matches == 28
    assert len(result.events) == 28
    assert len(result.all_event_ids) == 28
    assert result.all_event_ids == [event["id"] for event in result.events]
    assert len(set(result.all_event_ids)) == 28


def test_pagination_starts_at_eight_and_has_no_extra_page_at_eight() -> None:
    items = list(range(8))

    assert normalize_visible_count(8, None) == 8
    assert visible_items(items, 8) == items
    assert next_visible_count(8, 8) == 8


def test_pagination_handles_five_and_nine_without_reordering() -> None:
    five = list(range(5))
    nine = list(range(9))

    assert visible_items(five, None) == five
    assert next_visible_count(5, 5) == 5
    assert visible_items(nine, None) == list(range(8))
    assert next_visible_count(9, 8) == 9
    assert visible_items(nine, 9) == nine


def test_28_items_progress_8_16_24_28_without_duplicates() -> None:
    items = list(range(28))
    counts = [8]
    while counts[-1] < len(items):
        counts.append(next_visible_count(len(items), counts[-1]))

    assert counts == [8, 16, 24, 28]
    displayed = visible_items(items, counts[-1])
    assert displayed == items
    assert len(displayed) == len(set(displayed)) == 28


def test_command_result_and_reference_state_cover_the_complete_set() -> None:
    orchestrator = CommandOrchestrator(events=load_events(), reference_date=POC_REFERENCE_DATE)
    result = orchestrator.handle_query(
        "子ども向けイベント",
        command_plan=CommandPlan("find_events", CommandSlots(audience="family")),
    )

    assert result.total_matches == 28
    assert len(result.events) == 28
    assert result.all_event_ids == [event["id"] for event in result.events]
    assert len(result.all_event_ids) <= MAX_RESULT_SET_SIZE

    state = {"last_result_ids": result.all_event_ids}
    for ordinal in (9, 20, 28):
        detail = orchestrator.handle_query(
            f"{ordinal}番目の料金",
            state,
            command_plan=CommandPlan(
                "event_detail",
                CommandSlots(
                    reference_kind="ordinal",
                    reference_index=ordinal,
                    detail_fields=("fee",),
                ),
            ),
        )
        assert detail.status == "ok"
        assert detail.events[0]["id"] == result.all_event_ids[ordinal - 1]


def test_refine_previous_uses_ids_beyond_the_first_page() -> None:
    orchestrator = CommandOrchestrator(events=load_events(), reference_date=POC_REFERENCE_DATE)
    first = orchestrator.handle_query(
        "子ども向けイベント",
        command_plan=CommandPlan("find_events", CommandSlots(audience="family")),
    )
    refined = orchestrator.handle_query(
        "その中から無料だけ",
        {"last_result_ids": first.all_event_ids},
        command_plan=CommandPlan(
            "find_events",
            CommandSlots(entry_free=True, refine_previous=True),
        ),
    )

    assert refined.status == "ok"
    assert "011" in {event["id"] for event in refined.events}
    assert "011" not in first.all_event_ids[:8]


def test_writer_candidate_budget_is_separate_from_result_set() -> None:
    result = search_events("子どもと楽しめるイベント")
    payload = build_writer_payload(
        "子どもと楽しめるイベント",
        {
            "answer_type": "list",
            "exact_event_ids": result.all_event_ids,
            "relaxed_event_ids": [],
        },
        result.events,
    )

    assert len(result.events) == 28
    assert len(payload["candidate_ids"]) == MAX_WRITER_CANDIDATES
