"""Atomic-to-trusted-state reducer for Semantic Operations v2.2.

The model never emits an arbitrary slot patch. Deterministic grounding wins
for every ordinary explicit filter. Atomic model outputs may only supplement
a missing bounded field or release an existing field. Experience concepts are
single-label actions, preventing cross-list conflicts by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

import event_search
from app_config import POC_REFERENCE_DATE
from command_models import CommandPlan
from semantic_atomic_v2_2 import AtomicSemanticFrame, neutral_experience, normalize_query_for_grounding
from semantic_frame_v2_1 import SparseSemanticFrame
from semantic_state_v2_1 import SparseReduction, grounded_slots_from_query, reduce_sparse_frame


@dataclass(frozen=True)
class AtomicReduction:
    status: str
    frame: AtomicSemanticFrame | None
    plan: CommandPlan | None = None
    message: str = ""
    grounded_slots: Mapping[str, Any] | None = None
    applied_atoms: Mapping[str, Any] | None = None
    ignored_atoms: tuple[str, ...] = ()
    applied_unset: tuple[str, ...] = ()
    normalized_query: str = ""
    fail_soft: bool = False
    fail_soft_reason: str | None = None

    @property
    def ready(self) -> bool:
        return self.status == "ready" and self.plan is not None


def grounded_slots_from_query_v22(query: str, reference_date: date = POC_REFERENCE_DATE) -> dict[str, Any]:
    values = grounded_slots_from_query(normalize_query_for_grounding(query), reference_date)
    # A canonical genre explicitly present in the query is also a safe thematic
    # soft term. Keeping it in topics preserves the existing command contract
    # without asking the model to emit arbitrary free text.
    genres = list(values.get("genres") or [])
    topics = list(values.get("topics") or [])
    for genre in genres:
        if genre not in topics:
            topics.append(genre)
    if topics:
        values["topics"] = topics
    return values


def _has_any(source: Mapping[str, Any], *names: str) -> bool:
    return any(source.get(name) not in (None, "", [], (), {}) for name in names)


def _experience_grounded(source: Mapping[str, Any], concept: str) -> bool:
    return any(
        concept in set(source.get(name) or ())
        for name in ("experience_required", "experience_preferred", "experience_excluded")
    )


def _sanitize_intent(frame: AtomicSemanticFrame, grounded: Mapping[str, Any], previous_scope: bool) -> str:
    if (grounded or previous_scope) and frame.intent in {"faq", "unsupported"} and frame.data_gap == "none" and frame.clarification == "none":
        return "search"
    return frame.intent


def atomic_to_sparse_frame(
    frame: AtomicSemanticFrame,
    query: str,
    state: Mapping[str, Any] | None = None,
    *,
    reference_date: date = POC_REFERENCE_DATE,
    previous_scope: bool = False,
) -> tuple[SparseSemanticFrame, dict[str, Any], dict[str, Any], tuple[str, ...]]:
    grounded = grounded_slots_from_query_v22(query, reference_date)
    set_slots: dict[str, Any] = {}
    # v2.1's reducer deliberately reparses the query. Its parser can recover a
    # canonical genre but intentionally keeps genre and topic separate. When
    # v2.2 has deterministically identified a canonical genre, carry the same
    # bounded value through the sparse supplement so both existing contract
    # fields survive without asking the model for arbitrary free text.
    if grounded.get("genres"):
        set_slots["genres"] = list(grounded.get("genres") or [])
        set_slots["topics"] = list(grounded.get("topics") or grounded.get("genres") or [])
    unset: list[str] = []
    require: list[str] = []
    prefer: list[str] = []
    exclude: list[str] = []
    ignored: list[str] = []

    if frame.municipality == "release" or frame.region == "release":
        unset.append("location")
    elif not _has_any(grounded, "municipalities", "regions"):
        if frame.municipality not in {"none", "release"}:
            set_slots["municipalities"] = [frame.municipality]
        elif frame.region not in {"none", "release"}:
            set_slots["regions"] = [frame.region]
    else:
        if frame.municipality not in {"none", "release"}:
            ignored.append("municipality:grounded_wins")
        if frame.region not in {"none", "release"}:
            ignored.append("region:grounded_wins")

    if frame.fee == "release":
        unset.append("fee")
    elif not _has_any(grounded, "entry_free", "paid_only", "max_entry_fee"):
        if frame.fee == "free":
            set_slots["entry_free"] = True
        elif frame.fee == "paid":
            set_slots["paid_only"] = True
    elif frame.fee != "none":
        ignored.append("fee:grounded_wins")

    if frame.reservation == "release":
        unset.append("reservation")
    elif grounded.get("reservation_required") is None:
        if frame.reservation == "required":
            set_slots["reservation_required"] = True
        elif frame.reservation == "not_required":
            set_slots["reservation_required"] = False
    elif frame.reservation != "none":
        ignored.append("reservation:grounded_wins")

    if frame.venue == "release":
        unset.append("venue")
    elif grounded.get("venue") is None:
        if frame.venue in {"indoor", "outdoor"}:
            set_slots["venue"] = frame.venue
    elif frame.venue != "none":
        ignored.append("venue:grounded_wins")

    if frame.rain == "release":
        unset.append("rain")
    elif grounded.get("rain_preferred") is None:
        if frame.rain == "prefer":
            set_slots["rain_preferred"] = True
    elif frame.rain != "none":
        ignored.append("rain:grounded_wins")

    audience_mode = None
    if frame.audience_mode == "release":
        unset.append("age")
    elif frame.audience_mode in {"family", "adult", "target"}:
        audience_mode = frame.audience_mode

    for concept, action in frame.experience.items():
        grounded_has = _experience_grounded(grounded, concept)
        # Negative/release semantics must outrank a lexical positive match from
        # the deterministic parser. This is exactly the residual semantic job
        # assigned to the atomic classifier.
        if action == "unset":
            unset.append(f"experience:{concept}")
        elif action == "exclude":
            exclude.append(concept)
        elif grounded_has:
            if action not in {"none", "exclude", "unset"}:
                ignored.append(f"experience.{concept}:grounded_wins")
        elif action == "require":
            require.append(concept)
        elif action == "prefer":
            prefer.append(concept)

    scope = "previous" if previous_scope or frame.scope == "previous" else "new"
    clarification = None if frame.clarification == "none" else frame.clarification
    data_gap = None if frame.data_gap == "none" else frame.data_gap
    intent = _sanitize_intent(frame, grounded, previous_scope)
    if clarification is not None:
        intent = "clarify"

    sparse = SparseSemanticFrame(
        intent=intent,
        scope=scope,
        set_slots=set_slots,
        unset=tuple(dict.fromkeys(unset)),
        require=tuple(require),
        prefer=tuple(prefer),
        exclude=tuple(exclude),
        clarification=clarification,
        data_gap=data_gap,
        audience_mode=audience_mode,
    )
    applied = {
        "set": set_slots,
        "unset": list(sparse.unset),
        "require": list(sparse.require),
        "prefer": list(sparse.prefer),
        "exclude": list(sparse.exclude),
        "audience_mode": audience_mode,
        "intent": intent,
        "scope": scope,
        "clarification": clarification,
        "data_gap": data_gap,
    }
    return sparse, grounded, applied, tuple(ignored)


def reduce_atomic_frame(
    frame: AtomicSemanticFrame,
    query: str,
    state: Mapping[str, Any] | None = None,
    *,
    reference_date: date = POC_REFERENCE_DATE,
    previous_scope: bool = False,
) -> AtomicReduction:
    normalized_query = normalize_query_for_grounding(query)
    sparse, grounded, applied, ignored = atomic_to_sparse_frame(
        frame,
        query,
        state,
        reference_date=reference_date,
        previous_scope=previous_scope,
    )
    reduced: SparseReduction = reduce_sparse_frame(
        sparse,
        normalized_query,
        state,
        reference_date=reference_date,
    )
    return AtomicReduction(
        status=reduced.status,
        frame=frame,
        plan=reduced.plan,
        message=reduced.message,
        grounded_slots=grounded,
        applied_atoms=applied,
        ignored_atoms=ignored,
        applied_unset=reduced.applied_unset,
        normalized_query=normalized_query,
    )


def _neutral_frame(*, intent: str = "search", scope: str = "new") -> AtomicSemanticFrame:
    return AtomicSemanticFrame(
        intent=intent,
        scope=scope,
        municipality="none",
        region="none",
        fee="none",
        reservation="none",
        venue="none",
        rain="none",
        audience_mode="none",
        clarification="none",
        data_gap="none",
        experience=neutral_experience(),
    )


_FAIL_SOFT_TYPED_FIELDS = (
    "dates", "municipalities", "regions", "genres",
    "experience_required", "experience_preferred", "experience_excluded",
    "audience", "age", "age_group", "age_intent", "venue",
    "entry_free", "paid_only", "max_entry_fee", "reservation_required",
    "rain_preferred", "time_slots", "time_after",
)


def fail_soft_reduce(
    query: str,
    state: Mapping[str, Any] | None = None,
    *,
    reference_date: date = POC_REFERENCE_DATE,
    previous_scope: bool = False,
    reason: str = "atomic_frame_invalid",
) -> AtomicReduction:
    """Preserve trusted typed grounding after a classifier failure.

    The fallback never treats arbitrary free-text topics as fully understood.
    If the deterministic parser has typed filters, they may execute only when
    no residual text remains. A trusted previous-scope operation may also be
    retained. Otherwise the safe fallback is clarification.
    """

    normalized = normalize_query_for_grounding(query)
    grounded = grounded_slots_from_query_v22(query, reference_date)
    residual = event_search.unknown_query_residuals(normalized)
    has_typed_grounding = _has_any(grounded, *_FAIL_SOFT_TYPED_FIELDS)
    has_context = bool(isinstance(state, Mapping) and (state.get("last_result_ids") or state.get("last_command")))
    scope = "previous" if previous_scope and has_context else "new"

    if residual or (not has_typed_grounding and scope != "previous"):
        return AtomicReduction(
            status="clarification",
            frame=None,
            message="一部の条件を確実に解釈できませんでした。条件を少し言い換えて教えてみて。",
            grounded_slots=grounded,
            applied_atoms={},
            normalized_query=normalized,
            fail_soft=True,
            fail_soft_reason=reason,
        )

    neutral = _neutral_frame(scope=scope)
    reduced = reduce_atomic_frame(
        neutral,
        query,
        state,
        reference_date=reference_date,
        previous_scope=previous_scope,
    )
    return AtomicReduction(
        status=reduced.status,
        frame=None,
        plan=reduced.plan,
        message=reduced.message,
        grounded_slots=reduced.grounded_slots,
        applied_atoms=reduced.applied_atoms,
        ignored_atoms=reduced.ignored_atoms,
        applied_unset=reduced.applied_unset,
        normalized_query=reduced.normalized_query,
        fail_soft=True,
        fail_soft_reason=reason,
    )


__all__ = [
    "AtomicReduction", "atomic_to_sparse_frame", "fail_soft_reduce",
    "grounded_slots_from_query_v22", "reduce_atomic_frame",
]
