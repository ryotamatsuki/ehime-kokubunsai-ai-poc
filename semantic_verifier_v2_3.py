"""Pure-Python Evidence-Bounded verifier for Semantic Operations v2.3.

A valid JSON frame is still untrusted.  Positive supported atoms survive only
when there is independent current-turn grounding or trusted prior state.  In
particular, relational/unsupported evidence requests can never create proxy
Experience or ordinary search filters.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from app_config import REGION_CITIES
import experience_preferences
from semantic_atomic_v2_3 import AtomicSemanticFrameV23, normalize_query_for_grounding
from semantic_capability_registry_v2_3 import lookup_capability
from semantic_evidence_v2_3 import EvidenceRequest, NON_COERCIBLE_EVIDENCE_REQUESTS, evidence_request


@dataclass(frozen=True)
class EvidenceBoundedVerification:
    accepted: bool
    frame: AtomicSemanticFrameV23 | None
    reason: str | None = None
    accepted_atoms: tuple[str, ...] = ()
    ignored_atoms: tuple[str, ...] = ()
    rejected_atoms: tuple[str, ...] = ()
    normalized: tuple[str, ...] = ()
    unsupported_inference_count: int = 0
    silent_coercion_prevented_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "accepted_atoms": list(self.accepted_atoms),
            "ignored_atoms": list(self.ignored_atoms),
            "rejected_atoms": list(self.rejected_atoms),
            "normalized": list(self.normalized),
            "unsupported_inference_count": self.unsupported_inference_count,
            "silent_coercion_prevented_count": self.silent_coercion_prevented_count,
            "frame": self.frame.to_dict() if self.frame is not None else None,
        }


def _nonempty(value: Any) -> bool:
    return value not in (None, "", [], (), {})


def _context_slots(state: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        return {}
    command = state.get("last_command")
    if not isinstance(command, Mapping):
        return {}
    slots = command.get("slots")
    return dict(slots) if isinstance(slots, Mapping) else {}


def _has_context(state: Mapping[str, Any] | None) -> bool:
    if not isinstance(state, Mapping):
        return False
    ids = state.get("last_result_ids")
    return bool(isinstance(ids, (list, tuple)) and ids) or isinstance(state.get("last_command"), Mapping)


def _has_any(source: Mapping[str, Any], *names: str) -> bool:
    return any(_nonempty(source.get(name)) for name in names)


def _experience_in(source: Mapping[str, Any], concept: str) -> bool:
    return any(
        concept in set(source.get(name) or ())
        for name in ("experience_required", "experience_preferred", "experience_excluded")
    )


def _municipality_region(municipality: str) -> str | None:
    for region, municipalities in REGION_CITIES.items():
        if municipality in municipalities:
            return region
    return None


def _explicit_location(query: str, value: str) -> bool:
    normalized = normalize_query_for_grounding(query)
    aliases = {value, value.removesuffix("市").removesuffix("町")}
    return any(alias and alias in normalized for alias in aliases)


def _explicit_scalar(query: str, atom: str, value: str) -> bool:
    text = normalize_query_for_grounding(query).replace(" ", "")
    if atom in {"municipality", "region"}:
        return _explicit_location(query, value)
    if atom == "fee":
        if value == "free":
            return bool(re.search(r"(?:入場|料金|参加費)?(?:無料|0円|タダ)", text)) and not bool(
                re.search(r"(?:無料|0円|タダ).{0,8}(?:じゃなくても|でなくても|でなくていい|こだわらない|問わない)", text)
            )
        if value == "paid":
            return bool(re.search(r"(?:有料|料金が?かかる|参加費が?かかる)", text)) and not bool(
                re.search(r"有料.{0,8}(?:でもいい|でなくても|こだわらない|問わない)", text)
            )
    if atom == "reservation":
        if value == "required":
            return bool(re.search(r"(?:要予約|予約(?:が)?(?:必要|いる|要る))", text))
        if value == "not_required":
            return bool(re.search(r"(?:予約不要|予約なし|予約(?:は)?(?:いらない|要らない))", text))
    if atom == "venue":
        return (value == "indoor" and bool(re.search(r"(?:屋内|室内)", text))) or (
            value == "outdoor" and bool(re.search(r"(?:屋外|野外)", text))
        )
    if atom == "rain":
        return value == "prefer" and bool(re.search(r"(?:雨天|雨でも|雨の日|雨でも安心)", text))
    return False


def _explicit_experience(query: str, concept: str, action: str) -> bool:
    resolved = experience_preferences.resolve_experience_query(query)
    if action == "require":
        return concept in resolved.required
    if action == "prefer":
        return concept in resolved.preferred
    if action == "exclude":
        return concept in resolved.excluded
    return False


def _prior_matches(previous: Mapping[str, Any], atom: str, value: str) -> bool:
    if atom == "municipality":
        return value in set(previous.get("municipalities") or ())
    if atom == "region":
        return value in set(previous.get("regions") or ())
    if atom == "fee":
        return (value == "free" and previous.get("entry_free") is True) or (
            value == "paid" and previous.get("paid_only") is True
        )
    if atom == "reservation":
        return (value == "required" and previous.get("reservation_required") is True) or (
            value == "not_required" and previous.get("reservation_required") is False
        )
    if atom == "venue":
        return previous.get("venue") == value
    if atom == "rain":
        return value == "prefer" and previous.get("rain_preferred") is True
    return False


def verify_evidence_bounded_frame(
    frame: AtomicSemanticFrameV23,
    *,
    query: str,
    state: Mapping[str, Any] | None = None,
    grounded: Mapping[str, Any] | None = None,
) -> EvidenceBoundedVerification:
    grounded_values = dict(grounded or {})
    previous = _context_slots(state)
    values = frame.to_dict()
    experience = dict(values["experience"])
    accepted_atoms: list[str] = []
    ignored_atoms: list[str] = []
    rejected_atoms: list[str] = []
    normalized: list[str] = []
    unsupported_inference_count = 0
    silent_coercion = 0

    if frame.scope == "previous" and not _has_context(state):
        return EvidenceBoundedVerification(False, None, reason="previous_scope_without_context")

    request = evidence_request(frame.evidence_request)
    try:
        lookup_capability(request)
    except (KeyError, ValueError):
        return EvidenceBoundedVerification(False, None, reason="capability_registry_mismatch")

    municipality = str(values["municipality"])
    region = str(values["region"])
    if municipality not in {"none", "release"} and region not in {"none", "release"}:
        actual_region = _municipality_region(municipality)
        if actual_region != region:
            return EvidenceBoundedVerification(False, None, reason="municipality_region_conflict")
        values["region"] = "none"
        normalized.append("region:municipality_is_more_specific")

    scalar_grounding = {
        "municipality": ("municipalities", "regions"),
        "region": ("municipalities", "regions"),
        "fee": ("entry_free", "paid_only", "max_entry_fee"),
        "reservation": ("reservation_required",),
        "venue": ("venue",),
        "rain": ("rain_preferred",),
        "audience_mode": ("audience", "age", "age_group", "age_intent"),
    }
    release_prior = dict(scalar_grounding)
    release_prior["municipality"] = ("municipalities", "regions")
    release_prior["region"] = ("municipalities", "regions")

    for atom, grounding_names in scalar_grounding.items():
        value = str(values[atom])
        if value == "none":
            continue
        if value == "release":
            if _has_any(previous, *release_prior[atom]) or _has_any(grounded_values, *grounding_names):
                accepted_atoms.append(f"{atom}:release")
            else:
                values[atom] = "none"
                ignored_atoms.append(f"{atom}:release_without_effect")
            continue

        # Deterministic current grounding owns the value.  The model atom is
        # redundant and is neutralized rather than allowed to overwrite it.
        if _has_any(grounded_values, *grounding_names):
            values[atom] = "none"
            ignored_atoms.append(f"{atom}:grounded_wins")
            continue
        if frame.scope == "previous" and _prior_matches(previous, atom, value):
            values[atom] = "none"
            ignored_atoms.append(f"{atom}:trusted_prior_wins")
            continue

        # Residual positive supplements require an independent explicit proof
        # in the utterance.  audience_mode deliberately has no lexical fallback
        # here: person/group properties are never proxies for catalog audience.
        if atom != "audience_mode" and _explicit_scalar(query, atom, value):
            accepted_atoms.append(f"{atom}:{value}")
            continue

        values[atom] = "none"
        rejected_atoms.append(f"{atom}:{value}:no_explicit_grounding")
        unsupported_inference_count += 1
        if request in NON_COERCIBLE_EVIDENCE_REQUESTS:
            silent_coercion += 1

    for concept, action in list(experience.items()):
        if action == "none":
            continue
        grounded_has = _experience_in(grounded_values, concept)
        prior_has = _experience_in(previous, concept)
        if action == "unset":
            if grounded_has or prior_has:
                accepted_atoms.append(f"experience.{concept}:release")
            else:
                experience[concept] = "none"
                ignored_atoms.append(f"experience.{concept}:release_without_effect")
            continue

        if grounded_has:
            experience[concept] = "none"
            ignored_atoms.append(f"experience.{concept}:grounded_wins")
            continue
        if frame.scope == "previous" and prior_has:
            experience[concept] = "none"
            ignored_atoms.append(f"experience.{concept}:trusted_prior_wins")
            continue
        if _explicit_experience(query, concept, action):
            accepted_atoms.append(f"experience.{concept}:{action}")
            continue

        experience[concept] = "none"
        rejected_atoms.append(f"experience.{concept}:{action}:no_explicit_grounding")
        unsupported_inference_count += 1
        if request in NON_COERCIBLE_EVIDENCE_REQUESTS:
            silent_coercion += 1

    values["experience"] = experience
    verified = AtomicSemanticFrameV23.from_dict(values)
    return EvidenceBoundedVerification(
        True,
        verified,
        accepted_atoms=tuple(accepted_atoms),
        ignored_atoms=tuple(ignored_atoms),
        rejected_atoms=tuple(rejected_atoms),
        normalized=tuple(normalized),
        unsupported_inference_count=unsupported_inference_count,
        silent_coercion_prevented_count=silent_coercion,
    )


__all__ = ["EvidenceBoundedVerification", "verify_evidence_bounded_frame"]
