from __future__ import annotations

from app_config import POC_REFERENCE_DATE
from conversation_router import route_conversation
import event_search
from command_models import CommandPlan, CommandSlots
from command_orchestrator import CommandOrchestrator
from result_context import (
    classify_result_context_source,
    should_replace_result_set,
    transition_result_context,
)


def _records(ids: list[str]) -> list[dict[str, str]]:
    return [{"id": event_id} for event_id in ids]


def _transition(
    previous_ids: list[str],
    *,
    flow: str,
    source: str,
    new_ids: list[str],
    visible_count: int = 8,
):
    previous = _records(previous_ids)
    return transition_result_context(
        previous_results=previous,
        previous_result_ids=previous_ids,
        previous_near_results=[],
        previous_near_result_ids=[],
        previous_visible_count=visible_count,
        previous_near_visible_count=0,
        flow=flow,
        source=source,
        new_results=_records(new_ids),
        new_result_ids=new_ids,
        new_near_results=[],
        new_near_result_ids=[],
    )


def test_event_detail_preserves_the_full_search_context_across_turns() -> None:
    orchestrator = CommandOrchestrator()
    first = orchestrator.handle_query(
        "子ども向けイベント",
        command_plan=CommandPlan("find_events", CommandSlots(audience="family")),
    )
    state = {"last_result_ids": list(first.all_event_ids)}

    detail = orchestrator.handle_query(
        "20番目の料金",
        state,
        command_plan=CommandPlan(
            "event_detail",
            CommandSlots(
                reference_kind="ordinal",
                reference_index=20,
                detail_fields=("fee",),
            ),
        ),
    )
    after_detail = transition_result_context(
        previous_results=first.events,
        previous_result_ids=first.all_event_ids,
        previous_near_results=[],
        previous_near_result_ids=[],
        previous_visible_count=8,
        previous_near_visible_count=0,
        flow=detail.flow,
        source="command",
        new_results=detail.events,
        new_result_ids=detail.all_event_ids,
        new_near_results=[],
        new_near_result_ids=[],
    )

    assert detail.events[0]["id"] == first.all_event_ids[19]
    assert after_detail.result_ids == first.all_event_ids
    assert len(after_detail.result_ids) == 28

    next_detail = orchestrator.handle_query(
        "21番目の場所",
        {"last_result_ids": list(after_detail.result_ids)},
        command_plan=CommandPlan(
            "event_detail",
            CommandSlots(
                reference_kind="ordinal",
                reference_index=21,
                detail_fields=("place",),
            ),
        ),
    )
    assert next_detail.status == "ok"
    assert next_detail.events[0]["id"] == first.all_event_ids[20]


def test_ordinal_place_question_uses_the_router_reference_path() -> None:
    search = event_search.search_events(
        "子どもと楽しめるイベント",
        reference_date=POC_REFERENCE_DATE,
    )
    route = route_conversation(
        "21番目はどこ？",
        search.events,
        None,
        None,
        POC_REFERENCE_DATE,
    )

    assert route.action_type == "reference_followup"
    assert route.reference_index == 20
    assert route.selected_event is not None
    assert route.selected_event["id"] == search.all_event_ids[20]
    assert event_search.attribute_answer(route.selected_event, "place").endswith(
        "愛媛県県民文化会館（松山市）です。"
    )


def test_selected_event_is_separate_from_the_preserved_result_set() -> None:
    previous_ids = ["007", "008", "010", "016", "024"]
    update = _transition(
        previous_ids,
        flow="event_detail",
        source="command",
        new_ids=["024"],
    )

    selected_event_id = "024"
    assert selected_event_id == update.result_ids[4]
    assert update.result_ids == previous_ids


def test_ordinal_reference_uses_full_set_not_selected_event() -> None:
    ids = [f"{index:03d}" for index in range(1, 29)]
    update = _transition(
        ids,
        flow="event_detail",
        source="command",
        new_ids=[ids[19]],
    )
    assert update.result_ids[20] == "021"
    assert len(update.result_ids) == 28


def test_last_reference_survives_detail_turn() -> None:
    ids = [f"{index:03d}" for index in range(1, 29)]
    update = _transition(
        ids,
        flow="event_detail",
        source="command",
        new_ids=[ids[-1]],
    )
    assert update.result_ids[-1] == "028"


def test_detail_preserves_visible_page_boundary() -> None:
    ids = [f"{index:03d}" for index in range(1, 29)]
    update = _transition(
        ids,
        flow="event_detail",
        source="command",
        new_ids=["020"],
        visible_count=16,
    )
    assert update.result_ids == ids
    assert update.visible_count == 16


def test_refinement_replaces_context_after_detail() -> None:
    ids = [f"{index:03d}" for index in range(1, 29)]
    detail = _transition(
        ids,
        flow="event_detail",
        source="command",
        new_ids=["020"],
    )
    refined_ids = [event_id for event_id in ids if int(event_id) % 2 == 0]
    refined = _transition(
        list(detail.result_ids),
        flow="find_events",
        source="command",
        new_ids=refined_ids,
    )
    assert refined.result_ids == refined_ids
    assert refined.result_ids != ids
    assert refined.visible_count == 8


def test_real_command_refinement_still_uses_the_full_context_after_detail() -> None:
    orchestrator = CommandOrchestrator()
    first = orchestrator.handle_query(
        "子ども向けイベント",
        command_plan=CommandPlan("find_events", CommandSlots(audience="family")),
    )
    state = {"last_result_ids": list(first.all_event_ids)}
    detail = orchestrator.handle_query(
        "20番目の料金",
        state,
        command_plan=CommandPlan(
            "event_detail",
            CommandSlots(reference_kind="ordinal", reference_index=20, detail_fields=("fee",)),
        ),
    )
    after_detail = transition_result_context(
        previous_results=first.events,
        previous_result_ids=state["last_result_ids"],
        previous_near_results=[],
        previous_near_result_ids=[],
        previous_visible_count=16,
        previous_near_visible_count=0,
        flow=detail.flow,
        source="command",
        new_results=detail.events,
        new_result_ids=detail.all_event_ids,
        new_near_results=[],
        new_near_result_ids=[],
    )
    refined = orchestrator.handle_query(
        "その中から無料だけ",
        {"last_result_ids": list(after_detail.result_ids)},
        command_plan=CommandPlan(
            "find_events",
            CommandSlots(entry_free=True, refine_previous=True),
        ),
    )

    assert len(after_detail.result_ids) == 28
    assert refined.total_matches == 15
    assert len(refined.all_event_ids) == 15
    assert "029" in refined.all_event_ids


def test_new_search_replaces_the_old_context() -> None:
    old_ids = [f"{index:03d}" for index in range(1, 29)]
    new_ids = ["002", "007", "028"]
    update = _transition(
        old_ids,
        flow="find_events",
        source="command",
        new_ids=new_ids,
        visible_count=16,
    )
    assert update.result_ids == new_ids
    assert update.visible_count == 3


def test_general_faq_does_not_replace_search_context() -> None:
    ids = [f"{index:03d}" for index in range(1, 29)]
    update = _transition(
        ids,
        flow="general_faq",
        source="command",
        new_ids=[],
        visible_count=16,
    )
    assert update.result_ids == ids
    assert update.visible_count == 16


def test_direct_detail_without_prior_search_creates_a_one_event_context() -> None:
    update = _transition(
        [],
        flow="event_detail",
        source="command",
        new_ids=["007"],
    )
    assert update.result_ids == ["007"]
    assert update.visible_count == 1


def test_named_legacy_detail_is_a_new_search_but_command_detail_preserves() -> None:
    assert should_replace_result_set(flow="event_detail", source="legacy_search")
    assert not should_replace_result_set(flow="event_detail", source="command")


def test_preserving_recommendation_confirmation_keeps_the_current_page() -> None:
    ids = [f"{index:03d}" for index in range(1, 29)]
    update = _transition(
        ids,
        flow="recommend_next",
        source="preserving",
        new_ids=ids,
        visible_count=16,
    )
    assert not update.replace_result_set
    assert update.result_ids == ids
    assert update.visible_count == 16


def test_command_confirmation_with_unchanged_results_is_classified_as_preserving() -> None:
    assert (
        classify_result_context_source(
            command_flow="recommend_next",
            search_result_present=False,
            agentic_response_present=False,
            command_pending_response=True,
        )
        == "preserving"
    )
    assert (
        classify_result_context_source(
            command_flow="find_events",
            search_result_present=False,
            agentic_response_present=False,
        )
        == "command"
    )


def test_real_recommendation_is_not_preserving_even_if_ids_are_unchanged() -> None:
    assert (
        classify_result_context_source(
            command_flow="recommend_next",
            search_result_present=False,
            agentic_response_present=False,
            command_pending_response=False,
        )
        == "command"
    )


def test_pending_recommendation_error_is_preserving_without_id_heuristics() -> None:
    assert (
        classify_result_context_source(
            command_flow=None,
            search_result_present=False,
            agentic_response_present=False,
            pending_response_preserving=True,
        )
        == "preserving"
    )


def test_state_boundary_keeps_16_cards_for_confirmation_then_resets_new_search() -> None:
    old_ids = [f"{index:03d}" for index in range(1, 29)]
    confirmation_source = classify_result_context_source(
        command_flow="recommend_next",
        search_result_present=False,
        agentic_response_present=False,
        command_pending_response=True,
    )
    after_confirmation = transition_result_context(
        previous_results=_records(old_ids),
        previous_result_ids=old_ids,
        previous_near_results=[],
        previous_near_result_ids=[],
        previous_visible_count=16,
        previous_near_visible_count=0,
        flow="recommend_next",
        source=confirmation_source,
        new_results=_records(old_ids),
        new_result_ids=old_ids,
        new_near_results=[],
        new_near_result_ids=[],
    )
    new_ids = ["002", "007", "028"]
    new_search_source = classify_result_context_source(
        command_flow="find_events",
        search_result_present=False,
        agentic_response_present=False,
        command_pending_response=False,
    )
    after_new_search = transition_result_context(
        previous_results=after_confirmation.results,
        previous_result_ids=after_confirmation.result_ids,
        previous_near_results=[],
        previous_near_result_ids=[],
        previous_visible_count=after_confirmation.visible_count,
        previous_near_visible_count=0,
        flow="find_events",
        source=new_search_source,
        new_results=_records(new_ids),
        new_result_ids=new_ids,
        new_near_results=[],
        new_near_result_ids=[],
    )

    assert after_confirmation.visible_count == 16
    assert after_new_search.result_ids == new_ids
    assert after_new_search.visible_count == 3
