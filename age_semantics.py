"""Deterministic age semantics for query and event-side data.

Age words are a search constraint, not a free-text keyword.  This module keeps
the small canonical vocabulary at the boundary and deliberately distinguishes
recommended age from an explicit participation-eligibility limit.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Mapping


AGE_GROUPS = frozenset(
    {"preschool", "elementary", "junior_high", "high_school", "adult"}
)
AGE_INTENTS = frozenset({"recommended", "eligible"})

_PRESCHOOL_TERMS = ("未就学児", "幼稚園児", "保育園児", "幼児")
_ELEMENTARY_TERMS = ("小学生", "小学校", "小学")
_JUNIOR_HIGH_TERMS = ("中学生", "中学校", "中学")
_HIGH_SCHOOL_TERMS = ("高校生", "高校")
_ADULT_TERMS = ("大人", "成人")
_AGE_RE = re.compile(r"(?P<age>\d{1,2})\s*(?:歳|才)")


@dataclass(frozen=True)
class QueryAgeSemantics:
    """Canonical age meaning extracted from a user query."""

    age: int | None = None
    age_group: str | None = None
    age_intent: str | None = None
    matched_terms: tuple[str, ...] = ()

    @property
    def recognized(self) -> bool:
        return self.age is not None or self.age_group is not None


@dataclass(frozen=True)
class EventAgeSemantics:
    """Age facts available in one events.json record.

    The PoC data expresses lower bounds as recommendations.  They therefore
    populate ``recommended_min_age`` only.  ``eligible_min_age`` is reserved
    for a future explicit participation-eligibility statement and remains
    ``None`` for the current dataset.
    """

    recommended_min_age: int | None = None
    recommended_max_age: int | None = None
    eligible_min_age: int | None = None
    eligible_max_age: int | None = None
    child_friendly: bool = False
    unknown: bool = False


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def age_group_for_age(age: int) -> str:
    if age <= 5:
        return "preschool"
    if age <= 12:
        return "elementary"
    if age <= 15:
        return "junior_high"
    if age <= 18:
        return "high_school"
    return "adult"


def query_age_semantics(query: str) -> QueryAgeSemantics:
    """Map Japanese age expressions to the canonical query representation."""

    normalized = _normalize(str(query))
    age_match = _AGE_RE.search(normalized)
    age = int(age_match.group("age")) if age_match else None
    matched: list[str] = []
    if age_match:
        matched.append(age_match.group(0))

    group: str | None = age_group_for_age(age) if age is not None else None
    term_groups = (
        ("preschool", _PRESCHOOL_TERMS),
        ("elementary", _ELEMENTARY_TERMS),
        ("junior_high", _JUNIOR_HIGH_TERMS),
        ("high_school", _HIGH_SCHOOL_TERMS),
        ("adult", _ADULT_TERMS),
    )
    for candidate, terms in term_groups:
        for term in terms:
            if term in normalized:
                group = candidate
                matched.append(term)
                break
        if group == candidate and matched:
            break

    if not matched:
        return QueryAgeSemantics()

    intent = "eligible" if any(
        phrase in normalized
        for phrase in ("でも参加できる", "でも参加でき", "参加できる", "参加可能")
    ) else "recommended"
    return QueryAgeSemantics(
        age=age,
        age_group=group,
        age_intent=intent,
        matched_terms=tuple(dict.fromkeys(matched)),
    )


def is_age_query_term(value: Any) -> bool:
    """Return whether a residual parser token is an age expression."""

    text = _normalize(str(value))
    return bool(_AGE_RE.fullmatch(text) or text in {
        *(_PRESCHOOL_TERMS + _ELEMENTARY_TERMS + _JUNIOR_HIGH_TERMS),
        *(_HIGH_SCHOOL_TERMS + _ADULT_TERMS),
    })


def event_age_semantics(event: Mapping[str, Any]) -> EventAgeSemantics:
    """Interpret only age wording present in the structured event record."""

    guide = event.get("参加案内")
    guide = guide if isinstance(guide, Mapping) else {}
    age_text = _normalize(str(guide.get("対象年齢", "")))
    target = _normalize(str(guide.get("対象", "")))
    child_friendly = event.get("子ども向け") is True

    if age_text == "年齢制限なし":
        return EventAgeSemantics(child_friendly=child_friendly)
    if age_text == "小学生以上推奨":
        return EventAgeSemantics(recommended_min_age=6, child_friendly=child_friendly)
    if age_text == "高校生以上推奨":
        return EventAgeSemantics(recommended_min_age=16, child_friendly=child_friendly)

    # The current data has only the values above.  Do not infer eligibility
    # from a free-form target such as "一般向け".
    return EventAgeSemantics(
        child_friendly=child_friendly,
        unknown=bool(age_text or target),
    )


def _group_age_bounds(group: str | None) -> tuple[int, int] | None:
    return {
        "preschool": (0, 5),
        "elementary": (6, 12),
        "junior_high": (13, 15),
        "high_school": (16, 18),
        "adult": (19, 120),
    }.get(group)


def event_age_match(
    event: Mapping[str, Any],
    *,
    age: int | None = None,
    age_group: str | None = None,
    age_intent: str | None = None,
) -> str:
    """Classify a query/event age match as strong, reference, or excluded."""

    age_group = {
        "小学生": "elementary",
        "child": "preschool",
        "children": "preschool",
        "子ども": "preschool",
        "こども": "preschool",
    }.get(age_group, age_group)
    if age is None and age_group is None:
        return "none"
    semantics = event_age_semantics(event)
    if not semantics.child_friendly:
        return "excluded"
    if age is not None and semantics.eligible_min_age is not None and age < semantics.eligible_min_age:
        return "excluded"
    if age is not None and semantics.eligible_max_age is not None and age > semantics.eligible_max_age:
        return "excluded"
    bounds = _group_age_bounds(age_group)
    if bounds and semantics.eligible_min_age is not None and bounds[1] < semantics.eligible_min_age:
        return "excluded"

    requested_age = age if age is not None else (bounds[1] if bounds else None)
    if semantics.unknown:
        return "unknown"
    if semantics.recommended_min_age is not None and requested_age is not None:
        if requested_age < semantics.recommended_min_age:
            # "推奨" is not an eligibility restriction.  Keep it as a
            # reference candidate rather than claiming the event is invalid.
            return "reference"
    return "strong"


def matches_event_age(
    event: Mapping[str, Any],
    *,
    age: int | None = None,
    age_group: str | None = None,
    age_intent: str | None = None,
) -> bool:
    """Apply safe hard matching while preserving recommendation semantics."""

    return event_age_match(
        event,
        age=age,
        age_group=age_group,
        age_intent=age_intent,
    ) != "excluded"
