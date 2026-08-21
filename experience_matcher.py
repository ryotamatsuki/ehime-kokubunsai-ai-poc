"""Deterministic matching of user experience concepts to Data Model v3 facts."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable, Mapping

import data_model_v3
import experience_preferences


def _event_id(event: Mapping[str, Any]) -> str:
    value = event.get("id")
    if value is not None and str(value).strip():
        return str(value).strip()
    url = str(event.get("公式URL", "")).rstrip("/")
    return url.rsplit("/", 1)[-1] if url else ""


@lru_cache(maxsize=1)
def _v3_events_by_id() -> dict[str, dict[str, Any]]:
    return {str(event["id"]): event for event in data_model_v3.load_events_v3()}


def _v3_event(event: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if event.get("data_model_version") == 3:
        return event
    return _v3_events_by_id().get(_event_id(event))


def experience_profile(event: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the v3 profile only when the field is hard-filter eligible."""

    enriched = _v3_event(event)
    if enriched is None:
        return None
    try:
        if not data_model_v3.hard_filter_eligible(enriched, "experience_profile"):
            return None
    except (data_model_v3.DataModelV3Error, TypeError, KeyError):
        return None
    profile = enriched.get("experience_profile")
    return profile if isinstance(profile, Mapping) else None


def _predicate_match(profile: Mapping[str, Any] | None, concept_id: str) -> bool | None:
    if profile is None:
        return None
    try:
        definition = experience_preferences.concept(concept_id)
    except experience_preferences.ExperienceVocabularyError:
        return None
    for field_name, allowed_values in definition.predicate.items():
        value = profile.get(field_name)
        if field_name == "engagement_modes":
            if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
                return None
            return bool(set(value) & set(allowed_values))
        if not isinstance(value, str) or value == "unknown":
            return None
        return value in allowed_values
    return None


def concept_match(event: Mapping[str, Any], concept_id: str) -> bool | None:
    """Return True/False/None for match/known-mismatch/unknown."""

    if concept_id not in experience_preferences.EXPERIENCE_CONCEPT_IDS:
        raise experience_preferences.ExperienceVocabularyError(f"unknown experience concept: {concept_id!r}")
    return _predicate_match(experience_profile(event), concept_id)


def matches_experience(
    event: Mapping[str, Any],
    *,
    required: Iterable[str] = (),
    excluded: Iterable[str] = (),
) -> bool:
    """Apply hard semantics; unknown required facts are not matches."""

    required_ids = experience_preferences.normalize_concept_ids(list(required), field_name="required")
    excluded_ids = experience_preferences.normalize_concept_ids(list(excluded), field_name="excluded")
    if set(required_ids) & set(excluded_ids):
        return False
    for concept_id in required_ids:
        if concept_match(event, concept_id) is not True:
            return False
    for concept_id in excluded_ids:
        # Unknown excluded facts do not remove an event.  Only a confirmed
        # positive fact can satisfy the exclusion predicate.
        if concept_match(event, concept_id) is True:
            return False
    return True


def preferred_match_count(event: Mapping[str, Any], preferred: Iterable[str] = ()) -> int:
    preferred_ids = experience_preferences.normalize_concept_ids(list(preferred), field_name="preferred")
    return sum(concept_match(event, concept_id) is True for concept_id in preferred_ids)


def describe_event_experience(event: Mapping[str, Any], query: str = "") -> str:
    """Answer selected-event experience questions from v3 facts only."""

    name = str(event.get("イベント名", "このイベント"))
    profile = experience_profile(event)
    if profile is None:
        return f"「{name}」の座席・歩行・体験特性は、このPoCデータでは確認できません。"
    normalized = experience_preferences.compact(query)
    posture = profile.get("posture")
    seating = profile.get("seating")
    mobility = profile.get("mobility_load")
    modes = profile.get("engagement_modes", [])

    if any(term in normalized for term in ("座", "着席", "立ちっぱなし")):
        if posture == "unknown":
            return f"「{name}」の座席・姿勢は、このPoCデータでは確認できません。"
        if posture == "mostly_seated":
            seating_text = {
                "guaranteed": "座席が確保されています",
                "available": "座席が利用できます",
                "limited": "座席は限られています",
                "none": "座席はありません",
            }.get(str(seating), "座席情報があります")
            return f"「{name}」は、座って楽しめる構成です。{seating_text}。"
        if posture == "mixed":
            return f"「{name}」は、座ったり立ったりする場面が混在する構成です。"
        if posture == "standing_or_walking":
            return f"「{name}」は、立ったり歩いたりする構成です。"
    if any(term in normalized for term in ("歩", "移動", "疲れ")):
        if mobility == "unknown":
            return f"「{name}」の移動量は、このPoCデータでは確認できません。"
        mobility_text = {
            "low": "移動は少なめです",
            "medium": "一定の移動があります",
            "high": "歩く・移動する量が多めです",
        }.get(str(mobility))
        if mobility_text:
            return f"「{name}」は、{mobility_text}。"
    if any(term in normalized for term in ("体験", "作", "ワークショップ", "手を動か")):
        return (
            f"「{name}」は、手を動かす体験型の要素があります。"
            if "hands_on" in modes
            else f"「{name}」は、このPoCデータでは手を動かす体験型とは確認できません。"
        )
    if any(term in normalized for term in ("参加型", "観客参加", "一緒に参加")):
        return (
            f"「{name}」は、観客参加型の要素があります。"
            if "audience_participation" in modes
            else f"「{name}」は、このPoCデータでは観客参加型とは確認できません。"
        )
    return f"「{name}」の体験特性は、座席={seating}、移動量={mobility}、参加方法={', '.join(map(str, modes))}です。"


__all__ = [
    "concept_match",
    "describe_event_experience",
    "experience_profile",
    "matches_experience",
    "preferred_match_count",
]
