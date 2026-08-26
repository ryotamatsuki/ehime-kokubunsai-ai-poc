"""Authoritative product-capability registry for Semantic Operations v2.3.

Supportability is a data/product fact, not an LLM judgment.  Every closed
EvidenceRequest therefore resolves here to one deterministic action and one
bounded response strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from semantic_evidence_v2_3 import (
    AllowedSemanticAction,
    CapabilitySupportStatus,
    EvidenceRequest,
    evidence_request,
)


@dataclass(frozen=True)
class CapabilitySpec:
    capability_key: EvidenceRequest
    support_status: CapabilitySupportStatus
    evidence_source: str
    allowed_semantic_action: AllowedSemanticAction
    clarification_strategy: str | None = None
    data_gap_strategy: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "capability_key": self.capability_key.value,
            "support_status": self.support_status.value,
            "evidence_source": self.evidence_source,
            "allowed_semantic_action": self.allowed_semantic_action.value,
            "clarification_strategy": self.clarification_strategy,
            "data_gap_strategy": self.data_gap_strategy,
        }


CAPABILITY_REGISTRY: Mapping[EvidenceRequest, CapabilitySpec] = {
    EvidenceRequest.NONE: CapabilitySpec(
        EvidenceRequest.NONE,
        CapabilitySupportStatus.SUPPORTED,
        "trusted deterministic state and event catalog",
        AllowedSemanticAction.SEARCH,
    ),
    EvidenceRequest.SUPPORTED_ATTRIBUTE: CapabilitySpec(
        EvidenceRequest.SUPPORTED_ATTRIBUTE,
        CapabilitySupportStatus.SUPPORTED,
        "event catalog / deterministic supported attributes",
        AllowedSemanticAction.SEARCH,
    ),
    EvidenceRequest.RELATIONAL_SUITABILITY: CapabilitySpec(
        EvidenceRequest.RELATIONAL_SUITABILITY,
        CapabilitySupportStatus.CLARIFY_TO_SUPPORTED_DIMENSIONS,
        "no direct person-or-group x event suitability field",
        AllowedSemanticAction.CLARIFY,
        clarification_strategy="supported_dimensions",
    ),
    EvidenceRequest.SUBJECTIVE_JUDGMENT: CapabilitySpec(
        EvidenceRequest.SUBJECTIVE_JUDGMENT,
        CapabilitySupportStatus.UNSUPPORTED_DATA_GAP,
        "catalog stores event facts, not guaranteed subjective utility",
        AllowedSemanticAction.DATA_GAP,
        data_gap_strategy="subjective_judgment",
    ),
    EvidenceRequest.ABSOLUTE_GUARANTEE: CapabilitySpec(
        EvidenceRequest.ABSOLUTE_GUARANTEE,
        CapabilitySupportStatus.UNSUPPORTED_DATA_GAP,
        "catalog cannot substantiate absolute real-world guarantees",
        AllowedSemanticAction.DATA_GAP,
        data_gap_strategy="absolute_guarantee",
    ),
    EvidenceRequest.REALTIME_STATE: CapabilitySpec(
        EvidenceRequest.REALTIME_STATE,
        CapabilitySupportStatus.EXTERNAL_DATA_REQUIRED,
        "static event catalog has no realtime feed",
        AllowedSemanticAction.DATA_GAP,
        data_gap_strategy="realtime_state",
    ),
    EvidenceRequest.EXTERNAL_LOGISTICS: CapabilitySpec(
        EvidenceRequest.EXTERNAL_LOGISTICS,
        CapabilitySupportStatus.EXTERNAL_DATA_REQUIRED,
        "external transport/parking/surrounding-service source is not connected",
        AllowedSemanticAction.DATA_GAP,
        data_gap_strategy="external_logistics",
    ),
    EvidenceRequest.UNSUPPORTED_FACT: CapabilitySpec(
        EvidenceRequest.UNSUPPORTED_FACT,
        CapabilitySupportStatus.UNSUPPORTED_DATA_GAP,
        "requested fact is outside the current event data model",
        AllowedSemanticAction.DATA_GAP,
        data_gap_strategy="unsupported_fact",
    ),
    EvidenceRequest.UNKNOWN_CAPABILITY: CapabilitySpec(
        EvidenceRequest.UNKNOWN_CAPABILITY,
        CapabilitySupportStatus.UNSUPPORTED_DATA_GAP,
        "request cannot be safely mapped to a registered evidence capability",
        AllowedSemanticAction.DATA_GAP,
        data_gap_strategy="unknown_capability",
    ),
}


_CLARIFICATION_MESSAGES = {
    "supported_dimensions": (
        "その条件を直接示す情報はありません。重視する点を、掲載データで確認できる条件から教えてください。"
        "例：座って楽しめる、あまり歩かない、見る・聞く中心。"
    ),
}

_DATA_GAP_MESSAGES = {
    "subjective_judgment": "その評価を客観的に保証できる情報は、現在のイベントデータにはありません。",
    "absolute_guarantee": "現実の体験を絶対に保証できる情報は、現在のイベントデータにはありません。",
    "realtime_state": "現在の混雑・空き状況などのリアルタイム情報は、この静的イベントデータでは確認できません。",
    "external_logistics": "交通・駐車・周辺利便性などの外部情報源は、現在のPoCには接続していません。",
    "unsupported_fact": "その事実を確認できる項目は、現在のイベントデータにはありません。",
    "unknown_capability": "その条件を根拠を保ったまま判定できるデータ項目を確認できませんでした。",
}


def lookup_capability(value: str | EvidenceRequest) -> CapabilitySpec:
    request = evidence_request(value)
    try:
        return CAPABILITY_REGISTRY[request]
    except KeyError as exc:  # defensive: registry completeness is unit-tested
        raise KeyError(f"evidence capability is not registered: {request.value}") from exc


def capability_message(spec: CapabilitySpec) -> str:
    if spec.allowed_semantic_action is AllowedSemanticAction.CLARIFY:
        return _CLARIFICATION_MESSAGES.get(
            str(spec.clarification_strategy),
            "確認できる条件をもう少し具体的に教えてください。",
        )
    if spec.allowed_semantic_action is AllowedSemanticAction.DATA_GAP:
        return _DATA_GAP_MESSAGES.get(
            str(spec.data_gap_strategy),
            "その条件を確認できるデータが現在のPoCにはありません。",
        )
    return ""


def registry_snapshot() -> dict[str, dict[str, str | None]]:
    return {
        key.value: CAPABILITY_REGISTRY[key].to_dict()
        for key in sorted(CAPABILITY_REGISTRY, key=lambda item: item.value)
    }


__all__ = [
    "CAPABILITY_REGISTRY",
    "CapabilitySpec",
    "capability_message",
    "lookup_capability",
    "registry_snapshot",
]
