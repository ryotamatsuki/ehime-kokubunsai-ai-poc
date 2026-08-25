from __future__ import annotations

import pytest

from semantic_frame_v2 import SemanticFrame, SemanticFrameError, SemanticReference


def test_frame_accepts_compact_semantics_only():
    frame = SemanticFrame.from_dict(
        {
            "intent": "search",
            "refine_previous": True,
            "release": ["fee"],
            "experience_required": ["seated"],
            "experience_preferred": [],
            "experience_excluded": ["hands_on"],
            "reference": None,
            "clarification_reason": "none",
            "data_gap": "none",
            "confidence": "high",
        }
    )
    assert frame.intent == "search"
    assert frame.refine_previous is True
    assert frame.release == ("fee",)
    assert frame.experience_required == ("seated",)


def test_unknown_fields_are_rejected():
    with pytest.raises(SemanticFrameError):
        SemanticFrame.from_dict(
            {
                "intent": "search",
                "refine_previous": False,
                "release": [],
                "experience_required": [],
                "experience_preferred": [],
                "experience_excluded": [],
                "reference": None,
                "clarification_reason": "none",
                "data_gap": "none",
                "confidence": "medium",
                "entry_free": True,
            }
        )


def test_reference_contract_is_typed():
    assert SemanticReference.from_value({"kind": "ordinal", "index": 2}).index == 2
    with pytest.raises(SemanticFrameError):
        SemanticReference.from_value({"kind": "ordinal"})
    with pytest.raises(SemanticFrameError):
        SemanticReference.from_value({"kind": "selected", "index": 2})


def test_experience_overlap_is_rejected():
    with pytest.raises(SemanticFrameError):
        SemanticFrame(
            intent="search",
            experience_required=("seated",),
            experience_excluded=("seated",),
        )


def test_clarify_requires_a_reason():
    with pytest.raises(SemanticFrameError):
        SemanticFrame(intent="clarify")
