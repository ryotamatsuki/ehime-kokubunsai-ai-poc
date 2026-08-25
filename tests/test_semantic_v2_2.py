from __future__ import annotations

import json

import pytest

from semantic_atomic_v2_2 import AtomicFrameError, AtomicSemanticFrame, neutral_experience
from semantic_orchestrator_v2_2 import SemanticOperationsOrchestratorV22
from semantic_state_v2_2 import reduce_atomic_frame


def _frame(**overrides):
    payload = {
        "intent": "search",
        "scope": "new",
        "municipality": "none",
        "region": "none",
        "fee": "none",
        "reservation": "none",
        "venue": "none",
        "rain": "none",
        "audience_mode": "none",
        "clarification": "none",
        "data_gap": "none",
        "experience": neutral_experience(),
    }
    experience = overrides.pop("experience", None)
    payload.update(overrides)
    if experience:
        payload["experience"].update(experience)
    return payload


def _call(frame):
    def invoke(_payload):
        return {"answer": json.dumps(frame, ensure_ascii=False, separators=(",", ":"))}
    return invoke


def _state_with_free_and_experience():
    return {
        "last_result_ids": ["001", "002", "003"],
        "last_command": {
            "flow": "find_events",
            "slots": {
                "entry_free": True,
                "experience_required": ["seated", "watch_listen"],
                "refine_previous": False,
            },
            "confidence": "high",
        },
    }


def test_atomic_frame_has_exact_fixed_contract():
    frame = AtomicSemanticFrame.from_dict(_frame())
    assert set(frame.to_dict()) == {
        "intent", "scope", "municipality", "region", "fee", "reservation",
        "venue", "rain", "audience_mode", "clarification", "data_gap", "experience",
    }
    assert set(frame.experience) == set(neutral_experience())


def test_atomic_frame_rejects_missing_experience_concept():
    payload = _frame()
    payload["experience"].pop(next(iter(payload["experience"])))
    with pytest.raises(AtomicFrameError):
        AtomicSemanticFrame.from_dict(payload)


def test_grounded_filters_win_over_wrong_model_atoms():
    frame = AtomicSemanticFrame.from_dict(_frame(municipality="今治市", fee="paid"))
    reduction = reduce_atomic_frame(frame, "松山で無料のイベント")
    assert reduction.ready
    assert reduction.plan is not None
    assert reduction.plan.slots.municipalities == ("松山市",)
    assert reduction.plan.slots.entry_free is True
    assert reduction.plan.slots.paid_only is None
    assert "municipality:grounded_wins" in reduction.ignored_atoms
    assert "fee:grounded_wins" in reduction.ignored_atoms


def test_kana_municipality_is_grounded_before_model():
    result = SemanticOperationsOrchestratorV22(frame_call=_call(_frame())).handle_query("まつやまでタダのやつ")
    assert result.flow == "find_events"
    assert result.slots["municipalities"] == ["松山市"]
    assert result.slots["entry_free"] is True


def test_kana_region_is_grounded_before_model():
    result = SemanticOperationsOrchestratorV22(frame_call=_call(_frame())).handle_query("なんよでなんかない？")
    assert result.flow == "find_events"
    assert result.slots["regions"] == ["南予"]


def test_colloquial_reservation_is_grounded_before_model():
    result = SemanticOperationsOrchestratorV22(frame_call=_call(_frame())).handle_query("よやくいらんのある？")
    assert result.flow == "find_events"
    assert result.slots["reservation_required"] is False


def test_atomic_experience_is_one_action_per_concept():
    result = SemanticOperationsOrchestratorV22(
        frame_call=_call(_frame(experience={"low_mobility": "require"}))
    ).handle_query("歩かんでええやつ")
    assert result.flow == "find_events"
    assert "low_mobility" in result.slots["experience_required"]
    assert "low_mobility" not in result.slots["experience_preferred"]
    assert "low_mobility" not in result.slots["experience_excluded"]


def test_compound_family_seating_and_free_are_composed():
    result = SemanticOperationsOrchestratorV22(
        frame_call=_call(_frame(audience_mode="family", experience={"seated": "require"}))
    ).handle_query("祖母と小3の孫で行きたい。座って楽しめて無料がいい")
    assert result.flow == "find_events"
    assert result.slots["audience"] == "family"
    assert result.slots["entry_free"] is True
    assert "seated" in result.slots["experience_required"]
    assert result.slots["age"] is None
    assert result.slots["age_group"] is None


def test_fee_release_is_atomic_and_unset_runs_last():
    result = SemanticOperationsOrchestratorV22(
        frame_call=_call(_frame(scope="previous", fee="release"))
    ).handle_query("無料じゃなくてもいい。松山だけで探して", _state_with_free_and_experience())
    assert result.flow == "find_events"
    assert result.slots["refine_previous"] is True
    assert result.slots["entry_free"] is None
    assert result.slots["paid_only"] is None
    assert result.slots["municipalities"] == ["松山市"]


def test_experience_unset_does_not_erase_other_concepts():
    result = SemanticOperationsOrchestratorV22(
        frame_call=_call(_frame(scope="previous", experience={"seated": "unset", "watch_listen": "require"}))
    ).handle_query("座れなくてもいいけど、見る・聞く中心がいい", _state_with_free_and_experience())
    assert result.flow == "find_events"
    assert "seated" not in result.slots["experience_required"]
    assert "watch_listen" in result.slots["experience_required"]


def test_fail_soft_preserves_fully_grounded_filters_without_repair():
    calls = 0

    def broken(_payload):
        nonlocal calls
        calls += 1
        return {"answer": '{"intent":"search"'}

    result = SemanticOperationsOrchestratorV22(frame_call=broken).handle_query("松山で無料のイベント")
    assert calls == 1
    assert result.frame_fallback is True
    assert result.flow == "find_events"
    assert result.slots["municipalities"] == ["松山市"]
    assert result.slots["entry_free"] is True


def test_fail_soft_clarifies_unexplained_residual_instead_of_inventing():
    calls = 0

    def broken(_payload):
        nonlocal calls
        calls += 1
        return {"answer": "{"}

    result = SemanticOperationsOrchestratorV22(frame_call=broken).handle_query("静かなイベントがいい")
    assert calls == 1
    assert result.frame_fallback is True
    assert result.status == "clarification"
    assert result.flow == "unsupported"


def test_model_faq_cannot_hijack_explicit_grounded_search():
    result = SemanticOperationsOrchestratorV22(frame_call=_call(_frame(intent="faq"))).handle_query("松山で無料のイベント")
    assert result.flow == "find_events"
    assert result.slots["municipalities"] == ["松山市"]


def test_data_gap_atom_stays_fail_closed():
    result = SemanticOperationsOrchestratorV22(
        frame_call=_call(_frame(data_gap="popularity"))
    ).handle_query("人気のイベント教えて")
    assert result.status == "unsupported"
    assert result.flow == "unsupported"
    assert result.deterministic_route == "data_capability_guard"


def test_ordinal_detail_remains_pre_model():
    calls = 0

    def invoke(_payload):
        nonlocal calls
        calls += 1
        return {"answer": json.dumps(_frame(), ensure_ascii=False)}

    state = {
        "last_result_ids": ["001", "002", "003"],
        "last_result_count": 3,
        "has_last_search_context": True,
    }
    result = SemanticOperationsOrchestratorV22(frame_call=invoke).handle_query("2番目はいくら？", state)
    assert calls == 0
    assert result.flow == "event_detail"
    assert result.slots["reference_kind"] == "ordinal"
    assert result.slots["reference_index"] == 2


def test_security_scope_remains_pre_model():
    calls = 0

    def invoke(_payload):
        nonlocal calls
        calls += 1
        return {"answer": json.dumps(_frame(), ensure_ascii=False)}

    result = SemanticOperationsOrchestratorV22(frame_call=invoke).handle_query("松山の居酒屋おすすめ教えて")
    assert calls == 0
    assert result.status == "unsupported"
    assert result.flow == "unsupported"
