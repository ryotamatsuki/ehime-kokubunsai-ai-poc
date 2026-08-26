"""Closed evidence and constraint vocabularies for Semantic Operations v2.3.

The model may classify what kind of evidence the user is asking for.  It may
not decide whether the current product can substantiate that request, nor may
it translate an unsupported request into a different supported event
attribute.  Those decisions belong to the Python capability registry and
verifier.
"""

from __future__ import annotations

from enum import Enum


class EvidenceRequest(str, Enum):
    NONE = "none"
    SUPPORTED_ATTRIBUTE = "supported_attribute"
    RELATIONAL_SUITABILITY = "relational_suitability"
    SUBJECTIVE_JUDGMENT = "subjective_judgment"
    ABSOLUTE_GUARANTEE = "absolute_guarantee"
    REALTIME_STATE = "realtime_state"
    EXTERNAL_LOGISTICS = "external_logistics"
    UNSUPPORTED_FACT = "unsupported_fact"
    UNKNOWN_CAPABILITY = "unknown_capability"


class CapabilitySupportStatus(str, Enum):
    SUPPORTED = "supported"
    CLARIFY_TO_SUPPORTED_DIMENSIONS = "clarify_to_supported_dimensions"
    UNSUPPORTED_DATA_GAP = "unsupported_data_gap"
    EXTERNAL_DATA_REQUIRED = "external_data_required"


class AllowedSemanticAction(str, Enum):
    SEARCH = "search"
    CLARIFY = "clarify"
    DATA_GAP = "data_gap"


class ConstraintOperation(str, Enum):
    KEEP = "keep"
    SET = "set"
    EXCLUDE = "exclude"
    RELEASE = "release"


EVIDENCE_REQUEST_VALUES = frozenset(item.value for item in EvidenceRequest)
UNSUPPORTED_EVIDENCE_REQUESTS = frozenset({
    EvidenceRequest.SUBJECTIVE_JUDGMENT,
    EvidenceRequest.ABSOLUTE_GUARANTEE,
    EvidenceRequest.REALTIME_STATE,
    EvidenceRequest.EXTERNAL_LOGISTICS,
    EvidenceRequest.UNSUPPORTED_FACT,
    EvidenceRequest.UNKNOWN_CAPABILITY,
})
NON_COERCIBLE_EVIDENCE_REQUESTS = frozenset({
    EvidenceRequest.RELATIONAL_SUITABILITY,
    *UNSUPPORTED_EVIDENCE_REQUESTS,
})


def evidence_request(value: str | EvidenceRequest) -> EvidenceRequest:
    if isinstance(value, EvidenceRequest):
        return value
    try:
        return EvidenceRequest(str(value))
    except ValueError as exc:
        raise ValueError(f"unknown evidence request: {value!r}") from exc


__all__ = [
    "AllowedSemanticAction",
    "CapabilitySupportStatus",
    "ConstraintOperation",
    "EVIDENCE_REQUEST_VALUES",
    "EvidenceRequest",
    "NON_COERCIBLE_EVIDENCE_REQUESTS",
    "UNSUPPORTED_EVIDENCE_REQUESTS",
    "evidence_request",
]
