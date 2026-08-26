"""Trusted state reduction and release tracing for Semantic Operations v2.3.

The mature v2.2 reducer remains the executable slot reducer.  v2.3 supplies a
sanitized Evidence-Bounded frame and removes model control over clarification
and data-gap.  Constraint operations are exposed explicitly for observability;
Experience release is always concept-scoped.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

from app_config import POC_REFERENCE_DATE
from semantic_atomic_v2_2 import AtomicSemanticFrame
from semantic_atomic_v2_3 import AtomicSemanticFrameV23
from semantic_evidence_v2_3 import ConstraintOperation
from semantic_state_v2_2 import (
    AtomicReduction,
    fail_soft_reduce,
    grounded_slots_from_query_v22,
    reduce_atomic_frame,
)


@dataclass(frozen=True)
class EvidenceBoundedReduction:
    atomic: AtomicReduction
    constraint_operations: Mapping[str, str]

    @property
    def status(self) -> str:
        return self.atomic.status

    @property
    def plan(self):
        return self.atomic.plan

    @property
    def message(self) -> str:
        return self.atomic.message

    @property
    def grounded_slots(self):
        return self.atomic.grounded_slots

    @property
    def applied_atoms(self):
        return self.atomic.applied_atoms

    @property
    def ignored_atoms(self):
        return self.atomic.ignored_atoms

    @property
    def applied_unset(self):
        return self.atomic.applied_unset

    @property
    def fail_soft(self) -> bool:
        return self.atomic.fail_soft

    @property
    def fail_soft_reason(self) -> str | None:
        return self.atomic.fail_soft_reason

    @property
    def ready(self) -> bool:
        return self.atomic.ready


def adapt_v23_to_v22(frame: AtomicSemanticFrameV23) -> AtomicSemanticFrame:
    """Drop evidence control metadata before entering the trusted v2.2 reducer."""

    return AtomicSemanticFrame(
        intent=frame.intent,
        scope=frame.scope,
        municipality=frame.municipality,
        region=frame.region,
        fee=frame.fee,
        reservation=frame.reservation,
        venue=frame.venue,
        rain=frame.rain,
        audience_mode=frame.audience_mode,
        clarification="none",
        data_gap="none",
        experience=dict(frame.experience),
    )


def derive_constraint_operations(frame: AtomicSemanticFrameV23) -> dict[str, str]:
    operations: dict[str, str] = {}
    for atom in ("municipality", "region", "fee", "reservation", "venue", "rain", "audience_mode"):
        value = str(getattr(frame, atom))
        if value == "release":
            operations[atom] = ConstraintOperation.RELEASE.value
        elif value == "none":
            operations[atom] = ConstraintOperation.KEEP.value
        else:
            operations[atom] = ConstraintOperation.SET.value
    for concept, action in frame.experience.items():
        key = f"experience:{concept}"
        if action == "unset":
            operations[key] = ConstraintOperation.RELEASE.value
        elif action == "exclude":
            operations[key] = ConstraintOperation.EXCLUDE.value
        elif action in {"require", "prefer"}:
            operations[key] = ConstraintOperation.SET.value
        else:
            operations[key] = ConstraintOperation.KEEP.value
    return operations


def reduce_evidence_bounded_frame(
    frame: AtomicSemanticFrameV23,
    query: str,
    state: Mapping[str, Any] | None = None,
    *,
    reference_date: date = POC_REFERENCE_DATE,
    previous_scope: bool = False,
) -> EvidenceBoundedReduction:
    reduced = reduce_atomic_frame(
        adapt_v23_to_v22(frame),
        query,
        state,
        reference_date=reference_date,
        previous_scope=previous_scope,
    )
    return EvidenceBoundedReduction(reduced, derive_constraint_operations(frame))


def fail_soft_reduce_v23(
    query: str,
    state: Mapping[str, Any] | None = None,
    *,
    reference_date: date = POC_REFERENCE_DATE,
    previous_scope: bool = False,
    reason: str = "atomic_v23_frame_invalid",
) -> EvidenceBoundedReduction:
    reduced = fail_soft_reduce(
        query,
        state,
        reference_date=reference_date,
        previous_scope=previous_scope,
        reason=reason,
    )
    return EvidenceBoundedReduction(reduced, {})


__all__ = [
    "EvidenceBoundedReduction",
    "adapt_v23_to_v22",
    "derive_constraint_operations",
    "fail_soft_reduce_v23",
    "grounded_slots_from_query_v22",
    "reduce_evidence_bounded_frame",
]
