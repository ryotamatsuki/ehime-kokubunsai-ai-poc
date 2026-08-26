"""Pure-Python Evidence-Bounded verifier for Semantic Operations v2.3.

A syntactically valid model frame is still untrusted.  Positive supported atoms
survive only with independent current-turn grounding, a compositional explicit
proof, or trusted prior state.  Unsupported/relational requests therefore
cannot create proxy search filters.  Negative/release semantics are evaluated
before same-turn lexical positives so they can cancel deterministic false
positives without becoming opposite requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from app_config import REGION_CITIES
from semantic_atomic_v2_3 import AtomicSemanticFrameV23, normalize_query_for_grounding
from semantic_capability_registry_v2_3 import lookup_capability
from semantic_evidence_v2_3 import NON_COERCIBLE_EVIDENCE_REQUESTS, evidence_request
from semantic_grounding_v2_3 import GroundingProof, prove_audience, prove_experience


@dataclass(frozen=True)
class EvidenceBoundedVerification:
    accepted: bool
    frame: AtomicSemanticFrameV23 | None
    reason: str | None = None
    accepted_atoms: tuple[str, ...] = ()
    ignored_atoms: tuple[str, ...] = ()
    rejected_atoms: tuple[str, ...] = ()
    grounding_proofs: tuple[str, ...] = ()
    normalized: tuple[str, ...] = ()
    # These are post-verifier violations.  A correct verifier keeps them zero.
    unsupported_inference_count: int = 0
    silent_coercion_count: int = 0
    # Attempt telemetry records what the verifier prevented.
    unsupported_inference_prevented_count: int = 0
    silent_coercion_prevented_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "accepted_atoms": list(self.accepted_atoms),
            "ignored_atoms": list(self.ignored_atoms),
            "rejected_atoms": list(self.rejected_atoms),
            "grounding_proofs": list(self.grounding_proofs),
            "normalized": list(self.normalized),
            "unsupported_inference_count": self.unsupported_inference_count,
            "silent_coercion_count": self.silent_coercion_count,
            "unsupported_inference_prevented_count": self.unsupported_inference_prevented_count,
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


def _explicit_scalar(query: str, atom: str, value: str) -> GroundingProof:
    text = normalize_query_for_grounding(query).replace(" ", "")
    if atom in {"municipality", "region"} and _explicit_location(query, value):
        return GroundingProof(True, "explicit_expression", f"{atom}:canonical_location")
    if atom == "fee":
        if value == "free" and bool(re.search(r"(?:入場|料金|参加費)?(?:無料|0円|タダ)", text)) and not bool(
            re.search(r"(?:無料|0円|タダ).{0,8}(?:じゃなくても|でなくても|でなくていい|こだわらない|問わない)", text)
        ):
            return GroundingProof(True, "explicit_expression", "fee:free")
        if value == "paid" and bool(re.search(r"(?:有料|料金が?かかる|参加費が?かかる)", text)) and not bool(
            re.search(r"有料.{0,8}(?:でもいい|でなくても|こだわらない|問わない)", text)
        ):
            return GroundingProof(True, "explicit_expression", "fee:paid")
    if atom == "reservation":
        if value == "required" and re.search(r"(?:要予約|予約(?:が)?(?:必要|いる|要る))", text):
            return GroundingProof(True, "explicit_expression", "reservation:required")
        if value == "not_required" and re.search(r"(?:予約不要|予約なし|予約(?:は)?(?:いらない|要らない)|申し込み不要)", text):
            return GroundingProof(True, "explicit_expression", "reservation:not_required")
    if atom == "venue":
        if value == "indoor" and re.search(r"(?:屋内|室内)", text):
            return GroundingProof(True, "explicit_expression", "venue:indoor")
        if value == "outdoor" and re.search(r"(?:屋外|野外)", text):
            return GroundingProof(True, "explicit_expression", "venue:outdoor")
    if atom == "rain" and value == "prefer" and re.search(r"(?:雨天|雨でも|雨の日|雨でも安心)", text):
        return GroundingProof(True, "explicit_expression", "rain:prefer")
    if atom == "audience_mode":
        return prove_audience(query, value)
    return GroundingProof(False)


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
    if atom == "audience_mode":
        return previous.get("audience") == value
    return False


def _record_proof(proofs: list[str], atom: str, proof: GroundingProof) -> None:
    if proof.grounded:
        proofs.append(f"{atom}:{proof.source or 'explicit'}:{proof.rule or 'bounded'}")


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
    grounding_proofs: list[str] = []
    normalized: list[str] = []
    prevented_unsupported = 0
    prevented_coercion = 0

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

    for atom, grounding_names in scalar_grounding.items():
        value = str(values[atom])
        if value == "none":
            continue
        if value == "release":
            if _has_any(previous, *grounding_names) or _has_any(grounded_values, *grounding_names):
                accepted_atoms.append(f"{atom}:release")
                grounding_proofs.append(f"{atom}:release:existing_target")
            else:
                values[atom] = "none"
                ignored_atoms.append(f"{atom}:release_without_effect")
            continue

        proof = _explicit_scalar(query, atom, value)
        # Explicit bounded audience semantics may intentionally normalize a
        # coarse deterministic age-group parse (e.g. adult -> audience=adult).
        # Demographic proxy mappings are not proven by prove_audience().
        if atom == "audience_mode" and proof.grounded:
            accepted_atoms.append(f"{atom}:{value}")
            _record_proof(grounding_proofs, atom, proof)
            continue

        # For ordinary scalars, deterministic current grounding owns the
        # family and the model cannot overwrite it.
        if _has_any(grounded_values, *grounding_names):
            values[atom] = "none"
            ignored_atoms.append(f"{atom}:grounded_wins")
            continue
        if frame.scope == "previous" and _prior_matches(previous, atom, value):
            values[atom] = "none"
            ignored_atoms.append(f"{atom}:trusted_prior_wins")
            continue
        if proof.grounded:
            accepted_atoms.append(f"{atom}:{value}")
            _record_proof(grounding_proofs, atom, proof)
            continue

        values[atom] = "none"
        rejected_atoms.append(f"{atom}:{value}:no_explicit_grounding")
        prevented_unsupported += 1
        if request in NON_COERCIBLE_EVIDENCE_REQUESTS:
            prevented_coercion += 1

    for concept, action in list(experience.items()):
        if action == "none":
            continue
        grounded_has = _experience_in(grounded_values, concept)
        prior_has = _experience_in(previous, concept)
        if action == "unset":
            if grounded_has or prior_has:
                accepted_atoms.append(f"experience.{concept}:release")
                grounding_proofs.append(f"experience.{concept}:release:existing_target")
            else:
                experience[concept] = "none"
                ignored_atoms.append(f"experience.{concept}:release_without_effect")
            continue

        proof = prove_experience(query, concept, action)
        # Explicit exclusion is a negative operation.  It must outrank both a
        # same-turn lexical positive and a prior positive for the same concept.
        if action == "exclude" and proof.grounded:
            accepted_atoms.append(f"experience.{concept}:exclude")
            _record_proof(grounding_proofs, f"experience.{concept}", proof)
            continue

        if grounded_has:
            experience[concept] = "none"
            ignored_atoms.append(f"experience.{concept}:grounded_wins")
            continue
        if frame.scope == "previous" and prior_has:
            experience[concept] = "none"
            ignored_atoms.append(f"experience.{concept}:trusted_prior_wins")
            continue
        if proof.grounded:
            accepted_atoms.append(f"experience.{concept}:{action}")
            _record_proof(grounding_proofs, f"experience.{concept}", proof)
            continue

        experience[concept] = "none"
        rejected_atoms.append(f"experience.{concept}:{action}:no_explicit_grounding")
        prevented_unsupported += 1
        if request in NON_COERCIBLE_EVIDENCE_REQUESTS:
            prevented_coercion += 1

    values["experience"] = experience
    verified = AtomicSemanticFrameV23.from_dict(values)
    return EvidenceBoundedVerification(
        True,
        verified,
        accepted_atoms=tuple(accepted_atoms),
        ignored_atoms=tuple(ignored_atoms),
        rejected_atoms=tuple(rejected_atoms),
        grounding_proofs=tuple(grounding_proofs),
        normalized=tuple(normalized),
        unsupported_inference_count=0,
        silent_coercion_count=0,
        unsupported_inference_prevented_count=prevented_unsupported,
        silent_coercion_prevented_count=prevented_coercion,
    )


__all__ = ["EvidenceBoundedVerification", "verify_evidence_bounded_frame"]
