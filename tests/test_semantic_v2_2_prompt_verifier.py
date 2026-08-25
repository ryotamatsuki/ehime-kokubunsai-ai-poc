from __future__ import annotations

from semantic_atomic_v2_2 import AtomicSemanticFrame, neutral_experience
from semantic_orchestrator_v2_2 import SemanticOperationsOrchestratorV22
from semantic_prompt_v2_2 import (
    ATOMIC_FEW_SHOT_EXAMPLES,
    build_atomic_frame_messages,
    build_minimal_atomic_payload,
)
from semantic_verifier_v2_2 import verify_atomic_frame


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
    return AtomicSemanticFrame.from_dict(payload)


def _state(**slot_overrides):
    slots = {
        "entry_free": True,
        "experience_required": ["seated"],
        "municipalities": ["松山市"],
    }
    slots.update(slot_overrides)
    return {
        "last_result_ids": ["001", "002"],
        "selected_event_id": "001",
        "last_command": {"flow": "find_events", "slots": slots, "confidence": "high"},
    }


def test_minimal_payload_never_exposes_result_or_selected_ids():
    payload = build_minimal_atomic_payload("前の候補から無料だけ", _state(), {"entry_free": True})
    assert payload["state"] == {"has_previous_results": True, "has_previous_command": True}
    rendered = repr(payload)
    assert "last_result_ids" not in rendered
    assert "selected_event_id" not in rendered
    assert "001" not in rendered


def test_few_shot_examples_are_small_spec_authored_contract_examples():
    assert 4 <= len(ATOMIC_FEW_SHOT_EXAMPLES) <= 8
    messages = build_atomic_frame_messages("南予で何か", None, {"regions": ["南予"]})
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert len([message for message in messages if message["role"] == "assistant"]) == len(ATOMIC_FEW_SHOT_EXAMPLES)
    assert "last_result_ids" not in "\n".join(message["content"] for message in messages)


def test_verifier_rejects_previous_scope_without_trusted_context():
    checked = verify_atomic_frame(_frame(scope="previous"), state=None, grounded={})
    assert checked.accepted is False
    assert checked.reason == "previous_scope_without_context"


def test_verifier_normalizes_release_to_previous_scope():
    checked = verify_atomic_frame(_frame(scope="new", fee="release"), state=_state(), grounded={})
    assert checked.accepted is True
    assert checked.frame is not None
    assert checked.frame.scope == "previous"
    assert checked.frame.fee == "release"
    assert "scope:release_implies_previous" in checked.normalized


def test_verifier_neutralizes_release_when_no_prior_constraint_of_that_group():
    state = _state(entry_free=None, paid_only=None, max_entry_fee=None)
    checked = verify_atomic_frame(_frame(scope="previous", fee="release"), state=state, grounded={})
    assert checked.accepted is True
    assert checked.frame is not None
    assert checked.frame.fee == "none"
    assert "fee:release_without_prior_constraint" in checked.ignored


def test_verifier_rejects_conflicting_municipality_region_atoms():
    checked = verify_atomic_frame(
        _frame(municipality="松山市", region="南予"),
        state=None,
        grounded={},
    )
    assert checked.accepted is False
    assert checked.reason == "municipality_region_conflict"


def test_verifier_keeps_specific_municipality_and_drops_matching_region():
    checked = verify_atomic_frame(
        _frame(municipality="松山市", region="中予"),
        state=None,
        grounded={},
    )
    assert checked.accepted is True
    assert checked.frame is not None
    assert checked.frame.municipality == "松山市"
    assert checked.frame.region == "none"
    assert "region:municipality_is_more_specific" in checked.normalized


def test_verifier_neutralizes_positive_atom_when_grounding_already_owns_field():
    checked = verify_atomic_frame(_frame(fee="paid"), state=None, grounded={"entry_free": True})
    assert checked.accepted is True
    assert checked.frame is not None
    assert checked.frame.fee == "none"
    assert "fee:grounded_wins" in checked.ignored


def test_verifier_normalizes_clarification_control_action():
    checked = verify_atomic_frame(_frame(intent="search", clarification="ambiguous_request"), state=None, grounded={})
    assert checked.accepted is True
    assert checked.frame is not None
    assert checked.frame.intent == "clarify"
    assert "intent:clarification_implies_clarify" in checked.normalized


def test_verifier_preserves_negative_experience_against_positive_lexical_grounding():
    checked = verify_atomic_frame(
        _frame(experience={"hands_on": "exclude"}),
        state=None,
        grounded={"experience_required": ["hands_on"]},
    )
    assert checked.accepted is True
    assert checked.frame is not None
    assert checked.frame.experience["hands_on"] == "exclude"


def test_orchestrator_fail_softs_semantic_verifier_rejection_without_second_model_call():
    calls = 0

    def invoke(_payload):
        nonlocal calls
        calls += 1
        return {"answer": __import__("json").dumps(_frame(scope="previous").to_dict(), ensure_ascii=False)}

    result = SemanticOperationsOrchestratorV22(frame_call=invoke).handle_query("工芸のイベント")
    assert calls == 1
    assert result.frame_fallback is True
    assert result.verification is not None
    assert result.verification.accepted is False
    assert result.frame_error is not None or result.reduction is not None
