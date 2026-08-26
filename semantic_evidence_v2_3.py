"""Closed evidence, resolution and constraint vocabularies for v2.3.

The model may classify the requested evidence domain and whether the residual
utterance is semantically resolved.  It may not decide whether the product can
substantiate that evidence request, nor may it choose final flow/status.  Those
decisions belong to the Python capability registry, verifier and state machine.
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


class SemanticResolution(str, Enum):
    """Bounded interpretation signal; never a final routing decision."""

    RESOLVED = "resolved"
    UNDERSPECIFIED = "underspecified"
    CONDITIONAL = "conditional"
    AMBIGUOUS = "ambiguous"


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
SEMANTIC_RESOLUTION_VALUES = frozenset(item.value for item in SemanticResolution)
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


def semantic_resolution(value: str | SemanticResolution) -> SemanticResolution:
    if isinstance(value, SemanticResolution):
        return value
    try:
        return SemanticResolution(str(value))
    except ValueError as exc:
        raise ValueError(f"unknown semantic resolution: {value!r}") from exc


__all__ = [
    "AllowedSemanticAction",
    "CapabilitySupportStatus",
    "ConstraintOperation",
    "EVIDENCE_REQUEST_VALUES",
    "EvidenceRequest",
    "NON_COERCIBLE_EVIDENCE_REQUESTS",
    "SEMANTIC_RESOLUTION_VALUES",
    "SemanticResolution",
    "UNSUPPORTED_EVIDENCE_REQUESTS",
    "evidence_request",
    "semantic_resolution",
]
