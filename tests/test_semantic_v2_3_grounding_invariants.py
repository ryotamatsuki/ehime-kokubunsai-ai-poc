from __future__ import annotations

from semantic_atomic_v2_2 import neutral_experience
from semantic_atomic_v2_3 import AtomicSemanticFrameV23
from semantic_evidence_v2_3 import EvidenceRequest, SemanticResolution
from semantic_grounding_v2_3 import prove_experience
from semantic_verifier_v2_3 import verify_evidence_bounded_frame


def _frame(**experience_actions: str) -> AtomicSemanticFrameV23:
    experience = neutral_experience()
    experience.update(experience_actions)
    return AtomicSemanticFrameV23(
        intent="search",
        scope="new",
        evidence_request=EvidenceRequest.SUPPORTED_ATTRIBUTE.value,
        semantic_resolution=SemanticResolution.RESOLVED.value,
        municipality="none",
        region="none",
        fee="none",
        reservation="none",
        venue="none",
        rain="none",
        audience_mode="none",
        experience=experience,
    )


def test_watch_listen_composition_is_grounded_without_fixture_phrase_matching():
    proof = prove_experience(
        "展示を落ち着いて見て、解説の音も聞いて楽しめる催し",
        "watch_listen",
        "require",
    )
    assert proof.grounded is True
    assert proof.source in {"controlled_vocabulary", "explicit_expression"}


def test_release_is_concept_scoped_and_does_not_cancel_independent_requirement():
    query = "着席にはこだわらない。ただし鑑賞して解説を聴くのが中心の催しがよい"
    seated = prove_experience(query, "seated", "require")
    watch = prove_experience(query, "watch_listen", "require")
    assert seated.grounded is False
    assert watch.grounded is True


def test_alias_based_exclusion_is_a_general_supported_operation():
    checked = verify_evidence_bounded_frame(
        _frame(hands_on="exclude"),
        query="前の候補からワークショップ系を除外したい",
        grounded={},
    )
    assert checked.accepted is True
    assert checked.frame is not None
    assert checked.frame.experience["hands_on"] == "exclude"
    assert any("excluded_composition" in proof for proof in checked.grounding_proofs)


def test_functional_posture_proof_requires_limitation_and_low_load_observation_together():
    positive = prove_experience(
        "足に痛みがあるので、負担なく鑑賞できる催し",
        "seated",
        "require",
    )
    limitation_only = prove_experience("足に痛みがある人向けの催し", "seated", "require")
    observation_only = prove_experience("負担なく鑑賞できる催し", "seated", "require")
    assert positive.grounded is True
    assert positive.rule == "functional:posture_from_lower_limb_load"
    assert limitation_only.grounded is False
    assert observation_only.grounded is False
