"""Deterministic clarification for ambiguous user-suitability language.

Demographic labels such as "高齢者向け" are not event facts in the current
PoC.  They must not be silently converted to "adult" or relaxed into unrelated
near matches.  The safe behavior is to ask which grounded experience property
the user actually cares about, then let the existing Experience Preferences
pipeline perform the deterministic match.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

import conversation_recovery
import event_search
import experience_preferences


MAX_SUITABILITY_QUERY_LENGTH = 1200
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")

# Longest phrases first so stripping does not leave a dangling "向け".
_SENIOR_SUITABILITY_PHRASES = (
    "高齢者向け",
    "高齢の方向け",
    "高齢の人向け",
    "シニア向け",
    "お年寄り向け",
    "年配の方向け",
    "年配者向け",
    "老人向け",
    "高齢者",
    "高齢の方",
    "高齢の人",
    "シニア",
    "お年寄り",
    "年配の方",
    "年配者",
    "老人",
)


@dataclass(frozen=True)
class SuitabilityDecision:
    has_suitability_marker: bool
    needs_clarification: bool
    sanitized_query: str
    experience_required: tuple[str, ...] = ()
    experience_preferred: tuple[str, ...] = ()
    experience_excluded: tuple[str, ...] = ()

    @property
    def should_strip_suitability_marker(self) -> bool:
        return self.has_suitability_marker and not self.needs_clarification


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", _normalize(value))


def _has_senior_marker(query: str) -> bool:
    compact = _compact(query)
    return any(_compact(phrase) in compact for phrase in _SENIOR_SUITABILITY_PHRASES)


def strip_suitability_markers(query: str) -> str:
    """Remove only the ambiguous demographic label, preserving real filters."""

    value = _normalize(query)
    for phrase in _SENIOR_SUITABILITY_PHRASES:
        value = value.replace(phrase, " ")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"^[、,・/\s]+|[、,・/\s]+$", "", value)
    return value.strip()


def analyze_suitability_request(query: str) -> SuitabilityDecision:
    """Classify a query without inferring needs from age or demographics."""

    if not isinstance(query, str):
        return SuitabilityDecision(False, False, "")
    normalized = _normalize(query)
    if (
        not normalized
        or len(normalized) > MAX_SUITABILITY_QUERY_LENGTH
        or _CONTROL_RE.search(normalized)
    ):
        return SuitabilityDecision(False, False, normalized)

    has_marker = _has_senior_marker(normalized)
    if not has_marker:
        return SuitabilityDecision(False, False, normalized)

    # Security/product-boundary guards keep precedence over a friendly
    # clarification.  An injected or clearly out-of-domain request must still
    # reach the existing security path.
    if (
        event_search.classify_intent(normalized) in {"injection", "out_of_scope"}
        or conversation_recovery.is_domain_out_of_scope(normalized)
    ):
        return SuitabilityDecision(True, False, normalized)

    experience = experience_preferences.resolve_experience_query(normalized)
    sanitized = strip_suitability_markers(normalized)
    return SuitabilityDecision(
        has_suitability_marker=True,
        needs_clarification=not experience.recognized,
        sanitized_query=sanitized,
        experience_required=tuple(experience.required),
        experience_preferred=tuple(experience.preferred),
        experience_excluded=tuple(experience.excluded),
    )


def clarification_message(query: str) -> str:
    """Return a bounded prompt using only properties present in Data Model v3."""

    decision = analyze_suitability_request(query)
    base = decision.sanitized_query
    extra = ""
    if base and base not in {"イベント", "イベントを探して", "イベント探して"}:
        extra = " 日付や地域など、ほかに指定した条件も一緒に入れてね。"
    return (
        "高齢の方向けですね。「高齢者向け」だけでは、何を重視するか一意に決められんけん、"
        "もう1つ条件を教えてみて。たとえば「座って楽しめる」「あまり歩かず楽しめる」"
        "「見る・聞く中心」で絞れるよ。複数でも大丈夫です。"
        + extra
    )
