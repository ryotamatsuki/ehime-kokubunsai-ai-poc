from __future__ import annotations

import json

import pytest

from semantic_atomic_v2_2 import neutral_experience
from semantic_atomic_v2_3 import ATOMIC_FRAME_JSON_SCHEMA_V23, AtomicSemanticFrameV23
from semantic_capability_registry_v2_3 import CAPABILITY_REGISTRY, lookup_capability
from semantic_evidence_v2_3 import AllowedSemanticAction, EvidenceRequest, SemanticResolution
from semantic_orchestrator_v2_3 import SemanticOperationsOrchestratorV23
from semantic_state_v2_3 import grounded_slots_from_query_v22, reduce_evidence_bounded_frame
from semantic_verifier_v2_3 import verify_evidence_bounded_frame


def _frame(*, evidence=EvidenceRequest.NONE, resolution=SemanticResolution.RESOLVED, **overrides):
    payload = {
        "intent": "search",
        "scope": "new",
        "evidence_request": evidence.value if isinstance(evidence, EvidenceRequest) else str(evidence),
        "semantic_resolution": resolution.value if isinstance(resolution, SemanticResolution) else str(resolution),
        "municipality": "none",
        "region": "none",
        "fee": "none",
        "reservation": "none",
        "venue": "none",
        "rain": "none",
        "audience_mode": "none",
        "experience": neutral_experience(),
    }
    experience = overrides.pop("experience", None)
    payload.update(overrides)
    if experience:
        payload["experience"].update(experience)
    return AtomicSemanticFrameV23.from_dict(payload)


def _call(frame):
    def invoke(_payload):
        return {"answer": json.dumps(frame.to_dict(), ensure_ascii=False, separators=(",", ":"))}
    return invoke


def _state():
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


def test_schema_is_closed_and_model_has_no_final_data_gap_or_clarification_fields():
    assert ATOMIC_FRAME_JSON_SCHEMA_V23["additionalProperties"] is False
    assert "evidence_request" in ATOMIC_FRAME_JSON_SCHEMA_V23["required"]
    assert "semantic_resolution" in ATOMIC_FRAME_JSON_SCHEMA_V23["required"]
    assert "data_gap" not in ATOMIC_FRAME_JSON_SCHEMA_V23["properties"]
    assert "clarification" not in ATOMIC_FRAME_JSON_SCHEMA_V23["properties"]


def test_registry_is_complete_for_every_closed_evidence_class():
    assert set(CAPABILITY_REGISTRY) == set(EvidenceRequest)
    assert lookup_capability(EvidenceRequest.RELATIONAL_SUITABILITY).allowed_semantic_action is AllowedSemanticAction.CLARIFY
    assert lookup_capability(EvidenceRequest.REALTIME_STATE).allowed_semantic_action is AllowedSemanticAction.DATA_GAP


def test_a_relational_suitability_only_derives_clarification():
    result = SemanticOperationsOrchestratorV23(
        frame_call=_call(_frame(evidence=EvidenceRequest.RELATIONAL_SUITABILITY))
    ).handle_query("初参加の同行者に向いている催しを知りたい")
    assert result.status == "clarification"
    assert result.flow == "unsupported"
    assert result.evidence_request == EvidenceRequest.RELATIONAL_SUITABILITY.value
    assert result.clarification_reason == EvidenceRequest.RELATIONAL_SUITABILITY.value


def test_b_relational_plus_explicit_seated_clarifies_and_keeps_seated():
    result = SemanticOperationsOrchestratorV23(
        frame_call=_call(_frame(evidence=EvidenceRequest.RELATIONAL_SUITABILITY))
    ).handle_query("同行者に合うもので、座って楽しめる催しを探したい")
    assert result.status == "clarification"
    assert "seated" in result.slots.get("experience_required", [])


def test_c_unsupported_suitability_proxy_experience_is_ignored():
    frame = _frame(
        evidence=EvidenceRequest.RELATIONAL_SUITABILITY,
        experience={"watch_listen": "prefer"},
    )
    checked = verify_evidence_bounded_frame(
        frame,
        query="初参加の同行者に向いている催しを知りたい",
        grounded={},
    )
    assert checked.accepted is True
    assert checked.frame is not None
    assert checked.frame.experience["watch_listen"] == "none"
    assert any("watch_listen" in item for item in checked.rejected_atoms)
    assert checked.silent_coercion_count == 0
    assert checked.silent_coercion_prevented_count == 1


def test_d_explicit_watch_listen_is_retained_by_trusted_grounding():
    result = SemanticOperationsOrchestratorV23(
        frame_call=_call(_frame(evidence=EvidenceRequest.SUPPORTED_ATTRIBUTE))
    ).handle_query("鑑賞中心の催し")
    assert result.flow == "find_events"
    assert "watch_listen" in result.slots["experience_required"]


def test_compositional_functional_expression_can_prove_supported_experience():
    frame = _frame(
        evidence=EvidenceRequest.SUPPORTED_ATTRIBUTE,
        experience={"low_mobility": "require"},
    )
    checked = verify_evidence_bounded_frame(
        frame,
        query="足腰への負担が少なく移動も少ない催し",
        grounded={},
    )
    assert checked.accepted and checked.frame is not None
    assert checked.frame.experience["low_mobility"] == "require"
    assert any("functional:mobility_load" in proof for proof in checked.grounding_proofs)


def test_e_subjective_judgment_derives_data_gap():
    result = SemanticOperationsOrchestratorV23(
        frame_call=_call(_frame(evidence=EvidenceRequest.SUBJECTIVE_JUDGMENT))
    ).handle_query("満足度が高いと評価できる催しを選びたい")
    assert result.status == "unsupported"
    assert result.data_gap_reason == EvidenceRequest.SUBJECTIVE_JUDGMENT.value


def test_absolute_guarantee_derives_data_gap():
    result = SemanticOperationsOrchestratorV23(
        frame_call=_call(_frame(evidence=EvidenceRequest.ABSOLUTE_GUARANTEE))
    ).handle_query("必ず期待どおりになる催しを選んで")
    assert result.status == "unsupported"
    assert result.data_gap_reason == EvidenceRequest.ABSOLUTE_GUARANTEE.value


def test_unresolved_signal_is_not_final_flow_but_python_derives_clarification():
    result = SemanticOperationsOrchestratorV23(
        frame_call=_call(_frame(
            evidence=EvidenceRequest.ABSOLUTE_GUARANTEE,
            resolution=SemanticResolution.AMBIGUOUS,
        ))
    ).handle_query("保証を求めつつ選択基準がまだ決まっていない")
    assert result.status == "clarification"
    assert result.clarification_reason == "semantic_resolution:ambiguous"
    assert result.data_gap_reason == EvidenceRequest.ABSOLUTE_GUARANTEE.value


def test_f_realtime_state_derives_data_gap():
    result = SemanticOperationsOrchestratorV23(
        frame_call=_call(_frame(evidence=EvidenceRequest.REALTIME_STATE))
    ).handle_query("今この瞬間の空き状況で選びたい")
    assert result.status == "unsupported"
    assert result.data_gap_reason == EvidenceRequest.REALTIME_STATE.value


def test_g_external_logistics_derives_data_gap():
    result = SemanticOperationsOrchestratorV23(
        frame_call=_call(_frame(evidence=EvidenceRequest.EXTERNAL_LOGISTICS))
    ).handle_query("会場の近くに駐車しやすい催しを知りたい")
    assert result.status == "unsupported"
    assert result.data_gap_reason == EvidenceRequest.EXTERNAL_LOGISTICS.value


def test_h_supported_fee_condition_executes_normal_search():
    result = SemanticOperationsOrchestratorV23(
        frame_call=_call(_frame(evidence=EvidenceRequest.SUPPORTED_ATTRIBUTE))
    ).handle_query("入場無料の催しを探したい")
    assert result.flow == "find_events"
    assert result.slots["entry_free"] is True


def test_i_previous_fee_release_removes_only_fee():
    frame = _frame(evidence=EvidenceRequest.SUPPORTED_ATTRIBUTE, scope="previous", fee="release")
    result = SemanticOperationsOrchestratorV23(frame_call=_call(frame)).handle_query(
        "前の候補は料金条件を外したい",
        _state(),
    )
    assert result.flow == "find_events"
    assert result.slots["entry_free"] is None
    assert "seated" in result.slots["experience_required"]


def test_j_same_turn_lexical_fee_false_positive_can_be_cancelled_by_release():
    query = "無料じゃなくてもいい"
    grounded = grounded_slots_from_query_v22(query)
    assert grounded.get("entry_free") is True
    frame = _frame(evidence=EvidenceRequest.SUPPORTED_ATTRIBUTE, fee="release")
    checked = verify_evidence_bounded_frame(frame, query=query, grounded=grounded)
    assert checked.accepted is True
    assert checked.frame is not None and checked.frame.fee == "release"
    reduced = reduce_evidence_bounded_frame(checked.frame, query)
    assert reduced.plan is None or reduced.plan.slots.entry_free is None


def test_k_release_without_prior_or_current_target_is_safe_noop():
    frame = _frame(evidence=EvidenceRequest.SUPPORTED_ATTRIBUTE, fee="release")
    checked = verify_evidence_bounded_frame(frame, query="催しを探したい", grounded={})
    assert checked.accepted is True
    assert checked.frame is not None and checked.frame.fee == "none"
    assert "fee:release_without_effect" in checked.ignored_atoms


def test_l_experience_release_is_concept_scoped():
    frame = _frame(
        evidence=EvidenceRequest.SUPPORTED_ATTRIBUTE,
        scope="previous",
        experience={"seated": "unset"},
    )
    result = SemanticOperationsOrchestratorV23(frame_call=_call(frame)).handle_query(
        "前の候補は座れる条件を外したい",
        _state(),
    )
    assert "seated" not in result.slots["experience_required"]
    assert "watch_listen" in result.slots["experience_required"]
    assert "experience:seated" in result.release_operations


def test_negative_experience_operation_outranks_same_turn_positive_parse():
    query = "体験は避けて、鑑賞中心がいい"
    grounded = grounded_slots_from_query_v22(query)
    frame = _frame(
        evidence=EvidenceRequest.SUPPORTED_ATTRIBUTE,
        experience={"hands_on": "exclude"},
    )
    checked = verify_evidence_bounded_frame(frame, query=query, grounded=grounded)
    assert checked.accepted and checked.frame is not None
    assert checked.frame.experience["hands_on"] == "exclude"


def test_m_unsupported_request_does_not_erase_explicit_supported_slot():
    result = SemanticOperationsOrchestratorV23(
        frame_call=_call(_frame(evidence=EvidenceRequest.REALTIME_STATE))
    ).handle_query("入場無料で、今空いている催しを知りたい")
    assert result.status == "unsupported"
    assert result.slots["entry_free"] is True


def test_n_invalid_model_output_fail_soft_preserves_trusted_grounding():
    calls = 0

    def broken(_payload):
        nonlocal calls
        calls += 1
        return {"answer": '{"intent":"search"'}

    result = SemanticOperationsOrchestratorV23(frame_call=broken).handle_query("松山で無料の催し")
    assert calls == 1
    assert result.frame_fallback is True
    assert result.status == "clarification"
    assert result.deterministic_grounding["municipalities"] == ["松山市"]
    assert result.deterministic_grounding["entry_free"] is True


def test_positive_model_atom_without_independent_grounding_is_never_adopted():
    frame = _frame(evidence=EvidenceRequest.UNSUPPORTED_FACT, fee="free")
    checked = verify_evidence_bounded_frame(frame, query="静かな催しがいい", grounded={})
    assert checked.accepted is True
    assert checked.frame is not None and checked.frame.fee == "none"
    assert checked.unsupported_inference_count == 0
    assert checked.silent_coercion_count == 0
    assert checked.unsupported_inference_prevented_count == 1
    assert checked.silent_coercion_prevented_count == 1


def test_previous_scope_without_context_remains_rejected():
    checked = verify_evidence_bounded_frame(
        _frame(scope="previous"),
        query="前の条件を変えたい",
        state=None,
        grounded={},
    )
    assert checked.accepted is False
    assert checked.reason == "previous_scope_without_context"


def test_v23_frame_rejects_extra_fields():
    payload = _frame().to_dict()
    payload["data_gap"] = "other"
    with pytest.raises(Exception):
        AtomicSemanticFrameV23.from_dict(payload)
