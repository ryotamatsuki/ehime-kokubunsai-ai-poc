"""Pure-Python semantic verifier for Semantic Operations v2.2.

The atomic classifier is untrusted.  JSON-schema validity is necessary but not
sufficient: an atom must also be compatible with trusted deterministic
extraction and the bounded conversation state.  This module performs that
second check without another model call.

The verifier never invents event facts or natural-language meaning.  It only
normalizes finite control-state contradictions, ignores redundant model atoms,
or rejects unsafe transitions so the orchestrator can fail-soft.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from app_config import REGION_CITIES
from semantic_atomic_v2_2 import AtomicSemanticFrame


@dataclass(frozen=True)
class AtomicVerification:
    accepted: bool
    frame: AtomicSemanticFrame | None
    reason: str | None = None
    normalized: tuple[str, ...] = ()
    ignored: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "normalized": list(self.normalized),
            "ignored": list(self.ignored),
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


def _has_prior(slots: Mapping[str, Any], *names: str) -> bool:
    return any(_nonempty(slots.get(name)) for name in names)


def _grounded_has(grounded: Mapping[str, Any], *names: str) -> bool:
    return any(_nonempty(grounded.get(name)) for name in names)


def _experience_grounded(grounded: Mapping[str, Any], concept: str) -> bool:
    return any(
        concept in set(grounded.get(name) or ())
        for name in ("experience_required", "experience_preferred", "experience_excluded")
    )


def _experience_prior(slots: Mapping[str, Any], concept: str) -> bool:
    return any(
        concept in set(slots.get(name) or ())
        for name in ("experience_required", "experience_preferred", "experience_excluded")
    )


def _municipality_region(municipality: str) -> str | None:
    for region, municipalities in REGION_CITIES.items():
        if municipality in municipalities:
            return region
    return None


def verify_atomic_frame(
    frame: AtomicSemanticFrame,
    *,
    state: Mapping[str, Any] | None = None,
    grounded: Mapping[str, Any] | None = None,
) -> AtomicVerification:
    """Verify one parsed atomic frame against trusted state and grounding.

    This is a semantic/state verifier, not an utterance classifier.  It does
    not look for phrases in the user query.  When a safe deterministic value
    already exists, a positive model atom is neutralized.  Release operations
    are allowed only when a corresponding previous constraint exists.
    """

    grounded_values = dict(grounded or {})
    previous = _context_slots(state)
    has_context = _has_context(state)
    values = frame.to_dict()
    experience = dict(values["experience"])
    normalized: list[str] = []
    ignored: list[str] = []

    if frame.scope == "previous" and not has_context:
        return AtomicVerification(False, None, reason="previous_scope_without_context")

    # A release/unset operation is inherently a previous-state operation.  If
    # the classifier forgot scope=previous but trusted state exists, normalize
    # the scope rather than discarding the otherwise valid semantic action.
    has_release = any(
        value == "release"
        for value in (
            frame.municipality,
            frame.region,
            frame.fee,
            frame.reservation,
            frame.venue,
            frame.rain,
            frame.audience_mode,
        )
    ) or any(value == "unset" for value in experience.values())
    if has_release and not has_context:
        return AtomicVerification(False, None, reason="release_without_context")
    if has_release and values["scope"] != "previous":
        values["scope"] = "previous"
        normalized.append("scope:release_implies_previous")

    # Trusted current-turn grounding wins over positive model supplements.
    if _grounded_has(grounded_values, "municipalities", "regions"):
        if values["municipality"] not in {"none", "release"}:
            values["municipality"] = "none"
            ignored.append("municipality:grounded_wins")
        if values["region"] not in {"none", "release"}:
            values["region"] = "none"
            ignored.append("region:grounded_wins")
    if _grounded_has(grounded_values, "entry_free", "paid_only", "max_entry_fee") and values["fee"] not in {"none", "release"}:
        values["fee"] = "none"
        ignored.append("fee:grounded_wins")
    if grounded_values.get("reservation_required") is not None and values["reservation"] not in {"none", "release"}:
        values["reservation"] = "none"
        ignored.append("reservation:grounded_wins")
    if grounded_values.get("venue") is not None and values["venue"] not in {"none", "release"}:
        values["venue"] = "none"
        ignored.append("venue:grounded_wins")
    if grounded_values.get("rain_preferred") is not None and values["rain"] not in {"none", "release"}:
        values["rain"] = "none"
        ignored.append("rain:grounded_wins")

    # A municipality is more specific than its containing region.  If the two
    # model atoms disagree, reject instead of silently picking one.
    municipality = str(values["municipality"])
    region = str(values["region"])
    if municipality not in {"none", "release"} and region not in {"none", "release"}:
        actual_region = _municipality_region(municipality)
        if actual_region != region:
            return AtomicVerification(False, None, reason="municipality_region_conflict")
        values["region"] = "none"
        normalized.append("region:municipality_is_more_specific")

    # No-op releases are neutralized.  They should not turn a valid search into
    # a state mutation when there is nothing of that type to remove.
    release_groups = {
        "fee": ("entry_free", "paid_only", "max_entry_fee"),
        "reservation": ("reservation_required",),
        "venue": ("venue",),
        "rain": ("rain_preferred",),
        "audience_mode": ("audience", "age", "age_group", "age_intent"),
    }
    for atom, slot_names in release_groups.items():
        if values[atom] == "release" and not _has_prior(previous, *slot_names):
            values[atom] = "none"
            ignored.append(f"{atom}:release_without_prior_constraint")

    if values["municipality"] == "release" or values["region"] == "release":
        if not _has_prior(previous, "municipalities", "regions"):
            if values["municipality"] == "release":
                values["municipality"] = "none"
            if values["region"] == "release":
                values["region"] = "none"
            ignored.append("location:release_without_prior_constraint")

    for concept, action in list(experience.items()):
        if action in {"require", "prefer"} and _experience_grounded(grounded_values, concept):
            experience[concept] = "none"
            ignored.append(f"experience.{concept}:grounded_wins")
        elif action == "unset" and not _experience_prior(previous, concept):
            experience[concept] = "none"
            ignored.append(f"experience.{concept}:unset_without_prior_constraint")
    values["experience"] = experience

    # Clarification is a control action.  A model may identify the reason but
    # forget to align the intent; normalize the finite control fields.
    if values["clarification"] != "none" and values["intent"] != "clarify":
        values["intent"] = "clarify"
        normalized.append("intent:clarification_implies_clarify")

    verified = AtomicSemanticFrame.from_dict(values)
    return AtomicVerification(
        True,
        verified,
        normalized=tuple(normalized),
        ignored=tuple(ignored),
    )


__all__ = ["AtomicVerification", "verify_atomic_frame"]
