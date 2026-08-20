"""Conservative deterministic age semantics for the PoC event dataset.

The event JSON remains the source of truth.  This module only normalizes the
age phrases that actually exist in the current 30-event dataset and never
turns a recommendation into an eligibility restriction.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping


@dataclass(frozen=True)
class AgeSemantics:
    eligible_min_age: int | None
    eligible_max_age: int | None
    recommended_min_age: int | None
    recommended_max_age: int | None
    child_friendly: bool
    unknown: bool


AGE_GROUP_RANGES: dict[str, tuple[int, int | None]] = {
    "preschool": (0, 5),
    "elementary": (6, 11),
    "junior_high": (12, 14),
    "high_school": (15, 17),
    "adult": (18, None),
}

KNOWN_AGE_LABELS = frozenset({"年齢制限なし", "小学生以上推奨", "高校生以上推奨"})
KNOWN_TARGET_LABELS = frozenset({"どなたでも（子ども・家族での参加を想定）", "一般向け"})


def parse_event_age_semantics(event: Mapping[str, Any]) -> AgeSemantics:
    """Normalize only age wording present in the current dataset.

    ``小学生以上推奨`` and ``高校生以上推奨`` are recommendations, not
    participation requirements.  Unknown wording is left unknown rather than
    inferred from surrounding prose.
    """

    guide = event.get("参加案内")
    guide = guide if isinstance(guide, Mapping) else {}
    age_label = str(guide.get("対象年齢", "")).strip()
    child_friendly = event.get("子ども向け") is True

    if age_label == "年齢制限なし":
        return AgeSemantics(None, None, None, None, child_friendly, False)
    if age_label == "小学生以上推奨":
        return AgeSemantics(None, None, 6, None, child_friendly, False)
    if age_label == "高校生以上推奨":
        return AgeSemantics(None, None, 15, None, child_friendly, False)
    return AgeSemantics(None, None, None, None, child_friendly, True)


def extract_query_age(query: str) -> int | None:
    match = re.search(r"(?<!\d)(\d{1,2})(?:歳|才)", query)
    if not match:
        return None
    value = int(match.group(1))
    return value if 0 <= value <= 120 else None


def extract_query_age_group(query: str) -> str | None:
    groups: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("preschool", ("未就学児", "幼稚園児", "保育園児", "幼児", "小さい子")),
        ("elementary", ("小学生", "小学児童")),
        ("junior_high", ("中学生", "中学")),
        ("high_school", ("高校生", "高校")),
        ("adult", ("大人", "成人")),
    )
    for canonical, terms in groups:
        if any(term in query for term in terms):
            return canonical
    return None


def infer_query_age_intent(query: str) -> str | None:
    """Infer only the two canonical age intents used by deterministic tools."""

    if extract_query_age(query) is None and extract_query_age_group(query) is None:
        return None
    if any(
        term in query
        for term in (
            "参加でき",
            "でも参加",
            "参加可能",
            "入れる",
            "入場でき",
            "子と行ける",
            "子どもと行ける",
            "一緒に行ける",
        )
    ):
        return "eligible"
    return "recommended"


def _requested_range(age: int | None, age_group: str | None) -> tuple[int, int | None] | None:
    if age is not None:
        if isinstance(age, bool) or not isinstance(age, int) or not 0 <= age <= 120:
            return None
        return age, age
    if age_group is None:
        return None
    return AGE_GROUP_RANGES.get(age_group)


def _range_is_outside(
    requested: tuple[int, int | None],
    minimum: int | None,
    maximum: int | None,
) -> bool:
    req_min, req_max = requested
    if minimum is not None and req_max is not None and req_max < minimum:
        return True
    if maximum is not None and req_min > maximum:
        return True
    return False


def age_match_tier(
    event: Mapping[str, Any],
    *,
    age: int | None = None,
    age_group: str | None = None,
    age_intent: str | None = None,
) -> int:
    """Return 2=strong, 1=reference candidate, 0=exclude.

    Unknown data never means ineligible.  Recommendation mismatch can lower a
    candidate to reference status, but only explicit eligibility bounds could
    hard-exclude an otherwise eligible attendee.
    """

    requested = _requested_range(age, age_group)
    if requested is None:
        return 2 if age is None and age_group is None else 0
    if age_intent not in {None, "recommended", "eligible"}:
        return 0

    semantics = parse_event_age_semantics(event)
    if _range_is_outside(requested, semantics.eligible_min_age, semantics.eligible_max_age):
        return 0

    if age_intent == "eligible":
        if semantics.unknown:
            return 1
        if _range_is_outside(requested, semantics.recommended_min_age, semantics.recommended_max_age):
            return 1
        if requested[0] < 18 and not semantics.child_friendly:
            return 1
        return 2

    # Recommendation queries require child compatibility for child ages, but a
    # recommendation threshold is not interpreted as a participation ban.
    if requested[0] < 18 and not semantics.child_friendly:
        return 0
    if semantics.unknown:
        return 1
    if _range_is_outside(requested, semantics.recommended_min_age, semantics.recommended_max_age):
        return 1
    return 2
