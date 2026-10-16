"""Relational demographic clarification invariant for Semantic v2.1.

The current event data has no senior-suitability fact.  A demographic subject
such as "高齢の母" therefore cannot be converted to adult/age/experience
filters unless the utterance also states a grounded Experience need.  This is
a grammatical invariant over a bounded demographic class, not a list of full
utterances.
"""

from __future__ import annotations

import re
import unicodedata

import experience_preferences


_SENIOR_RELATION = re.compile(
    r"(?:高齢|年配|シニア|お年寄り|老人)"
    r"(?:の)?(?:母|父|祖母|祖父|親|両親|家族|人|方|同行者)?"
)
_SUITABILITY_PREDICATE = re.compile(
    r"(?:楽しめ|おすすめ|向け|行きやす|連れて(?:行|い)け|参加しやす|合う|適した)"
)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value)))


def needs_relational_demographic_clarification(query: str) -> bool:
    text = _normalize(query)
    if not text or not _SENIOR_RELATION.search(text) or not _SUITABILITY_PREDICATE.search(text):
        return False
    # An explicit grounded need such as "高齢の父と、あまり歩かないもの"
    # is actionable; the demographic marker itself is ignored rather than
    # triggering another question.
    return not experience_preferences.resolve_experience_query(query).recognized


__all__ = ["needs_relational_demographic_clarification"]
