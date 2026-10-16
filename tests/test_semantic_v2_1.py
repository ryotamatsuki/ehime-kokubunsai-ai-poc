from __future__ import annotations

import json

import pytest

from semantic_frame_v2_1 import SparseFrameError, SparseSemanticFrame
from semantic_orchestrator_v2_1 import SemanticOperationsOrchestratorV21
from semantic_state_v2_1 import reduce_sparse_frame


def _state():
    return {
        "last_result_ids": ["001", "002", "003", "004", "005"],
        "last_result_count": 5,
        "has_last_search_context": True,
    }


def test_sparse_frame_requires_only_intent():
    frame = SparseSemanticFrame.from_json('{"intent":"search"}')
    assert frame.to_dict(sparse=True) == {"intent": "search"}


def test_sparse_frame_rejects_unknown_field():
    with pytest.raises(SparseFrameError):
        SparseSemanticFrame.from_json('{"intent":"search","unused":null}')


def test_specific_experience_unset_preserves_other_concept():
    state = {
        "last_result_ids": ["001"],
        "last_command": {
            "flow": "find_events",
            "slots": {
                "experience_required": ["seated", "watch_listen"],
                "refine_previous": False,
            },
            "confidence": "high",
        },
    }
    frame = SparseSemanticFrame.from_dict({
        "intent": "search",
        "scope": "previous",
        "unset": ["experience:seated"],
    })
    reduction = reduce_sparse_frame(frame, "条件を一つ外す", state)
    assert reduction.ready
    assert reduction.plan is not None
    assert reduction.plan.slots.experience_required == ("watch_listen",)


def test_fee_unset_wins_after_explicit_lexical_parse():
    state = {
        "last_result_ids": ["001"],
        "last_command": {
            "flow": "find_events",
            "slots": {"entry_free": True, "refine_previous": False},
            "confidence": "high",
        },
    }
    frame = SparseSemanticFrame.from_dict({"intent": "search", "scope": "previous", "unset": ["fee"]})
    reduction = reduce_sparse_frame(frame, "無料じゃなくてもいい", state)
    assert reduction.ready
    assert reduction.plan is not None
    assert reduction.plan.slots.entry_free is None
    assert reduction.plan.slots.paid_only is None
    assert reduction.plan.slots.max_entry_fee is None


def test_family_mode_clears_incidental_age_constraint():
    frame = SparseSemanticFrame.from_dict({"intent": "search", "audience_mode": "family"})
    reduction = reduce_sparse_frame(frame, "祖母と小学3年の孫で行きたい")
    assert reduction.ready
    assert reduction.plan is not None
    assert reduction.plan.slots.audience == "family"
    assert reduction.plan.slots.age_group is None
    assert reduction.plan.slots.age is None


def test_ordinal_detail_is_resolved_before_model():
    calls = 0

    def frame_call(_):
        nonlocal calls
        calls += 1
        return {"answer": '{"intent":"search"}'}

    result = SemanticOperationsOrchestratorV21(frame_call=frame_call).handle_query("2番目はいくら？", _state())
    assert calls == 0
    assert result.flow == "event_detail"
    assert result.slots["reference_kind"] == "ordinal"
    assert result.slots["reference_index"] == 2


def test_search_explanation_is_resolved_before_model():
    calls = 0

    def frame_call(_):
        nonlocal calls
        calls += 1
        return {"answer": '{"intent":"search"}'}

    result = SemanticOperationsOrchestratorV21(frame_call=frame_call).handle_query("どういう基準でこの結果になったの？", _state())
    assert calls == 0
    assert result.flow == "explain_search"


def test_refinement_scope_is_forced_from_trusted_router():
    seen = []

    def frame_call(payload):
        seen.append(payload)
        return {"answer": json.dumps({"intent": "search"}, ensure_ascii=False)}

    result = SemanticOperationsOrchestratorV21(frame_call=frame_call).handle_query("その中から南予だけにして", _state())
    assert len(seen) == 1
    assert result.flow == "find_events"
    assert result.slots["refine_previous"] is True
    assert "南予" in result.slots["regions"]


def test_generic_non_event_scope_is_rejected_before_model():
    calls = 0

    def frame_call(_):
        nonlocal calls
        calls += 1
        return {"answer": '{"intent":"search"}'}

    result = SemanticOperationsOrchestratorV21(frame_call=frame_call).handle_query("松山の居酒屋おすすめ教えて")
    assert calls == 0
    assert result.flow == "unsupported"
    assert result.status == "unsupported"
