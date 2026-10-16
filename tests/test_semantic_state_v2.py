from __future__ import annotations

from command_models import CommandPlan, CommandSlots
from semantic_frame_v2 import SemanticFrame, SemanticReference
from semantic_state_v2 import reduce_semantic_frame


def _state_with_previous(**slots):
    plan = CommandPlan(
        flow="find_events",
        slots=CommandSlots.from_dict(slots),
        confidence="high",
    )
    return {
        "last_result_ids": ["001", "002", "003"],
        "last_command": plan.to_dict(),
    }


def test_fee_release_wins_over_lexical_free_match():
    state = _state_with_previous(entry_free=True, regions=["中予"])
    frame = SemanticFrame(intent="search", refine_previous=True, release=("fee",))
    reduced = reduce_semantic_frame(frame, "無料じゃなくてもいい。南予だけにして", state)
    assert reduced.ready
    assert reduced.plan.slots.regions == ("南予",)
    assert reduced.plan.slots.entry_free is None
    assert reduced.plan.slots.paid_only is None
    assert reduced.plan.slots.max_entry_fee is None


def test_weather_and_venue_release_do_not_flip_to_opposite_constraints():
    state = _state_with_previous(rain_preferred=True, venue="indoor")
    frame = SemanticFrame(
        intent="search",
        refine_previous=True,
        release=("rain", "venue"),
    )
    reduced = reduce_semantic_frame(frame, "雨は気にしないし、屋外でもいい", state)
    assert reduced.ready
    assert reduced.plan.slots.rain_preferred is None
    assert reduced.plan.slots.venue is None


def test_compound_explicit_constraints_are_grounded_by_python():
    frame = SemanticFrame(intent="search", experience_required=("seated",))
    reduced = reduce_semantic_frame(
        frame,
        "11/3に松山市で、無料で座って見られるイベント",
        {},
    )
    assert reduced.ready
    assert reduced.plan.slots.dates == ("2028-11-03",)
    assert reduced.plan.slots.municipalities == ("松山市",)
    assert reduced.plan.slots.entry_free is True
    assert "seated" in reduced.plan.slots.experience_required


def test_refinement_preserves_unmentioned_previous_constraints():
    state = _state_with_previous(entry_free=True, regions=["中予"])
    frame = SemanticFrame(intent="search", refine_previous=True)
    reduced = reduce_semantic_frame(frame, "その中から南予だけにして", state)
    assert reduced.ready
    assert reduced.plan.slots.regions == ("南予",)
    assert reduced.plan.slots.entry_free is True
    assert reduced.plan.slots.refine_previous is True


def test_reference_is_structured_not_reparsed_from_state():
    frame = SemanticFrame(
        intent="detail",
        reference=SemanticReference(kind="ordinal", index=2),
    )
    reduced = reduce_semantic_frame(frame, "2番目っていくら？", {})
    assert reduced.ready
    assert reduced.plan.slots.reference_kind == "ordinal"
    assert reduced.plan.slots.reference_index == 2


def test_data_gap_never_becomes_a_search_fact():
    frame = SemanticFrame(intent="search", data_gap="medical_safety")
    reduced = reduce_semantic_frame(frame, "認知症の父でも安心して行けるイベント", {})
    assert reduced.status == "data_limit"
    assert reduced.plan is None
    assert "安全性" in reduced.message


def test_refinement_without_previous_context_clarifies():
    frame = SemanticFrame(intent="search", refine_previous=True)
    reduced = reduce_semantic_frame(frame, "その中から南予だけ", {})
    assert reduced.status == "clarification"
    assert reduced.plan is None
