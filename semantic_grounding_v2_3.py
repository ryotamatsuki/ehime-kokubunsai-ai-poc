"""Compositional grounding proofs for Semantic Operations v2.3.

The deterministic parser remains the first source of truth. This module covers
bounded residual expressions that explicitly describe an existing supported
dimension without enumerating complete user utterances. It proves only the
semantic relation between an expression and an existing catalog dimension; it
never maps demographics, personality, expertise or diagnoses directly to a
supported filter.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

import experience_preferences


@dataclass(frozen=True)
class GroundingProof:
    grounded: bool
    source: str | None = None
    rule: str | None = None


# Preference markers describe strength, not event facts. Generic ``がいい`` is
# intentionally absent: ``鑑賞中心がいい`` is still a direct requested mode,
# while explicit weak markers such as ``できれば`` remain preferences.
_PREFERENCE = re.compile(r"(?:できれば|できたら|なるべく|できるだけ|ほうがいい|方がいい|方がよい|希望|疲れにく|疲れない)")

# Functional-load grammar. These are dimension-level predicates, not complete
# fixture sentences. Demographic labels are deliberately absent.
_SEATED_DIRECT = re.compile(r"(?:座(?:る|れ|っ|り|席)|着席|椅子|いす|イス|すわ(?:る|れ|っ))")
_STANDING_LIMIT = re.compile(r"(?:立(?:つ|て|ち|た)|立ちっぱなし).{0,8}(?:無理|難|できない|出来ない|つら|辛|嫌|いや|避け|ない)")
_LOWER_LIMB_LIMIT = re.compile(r"(?:膝|ひざ|足腰|脚|足).{0,10}(?:悪|弱|痛|不自由|負担|つら|辛)")
_EASY_OBSERVATION = re.compile(r"(?:無理なく|楽に|負担なく|負担少なく).{0,8}(?:見|観覧|鑑賞)")
_LOW_MOBILITY_DIRECT = re.compile(
    r"(?:歩(?:く|き|か)|移動|足腰|脚|膝|ひざ).{0,12}"
    r"(?:少な|短|弱|悪|苦手|つら|辛|無理|疲|しんど|ない|なく|ず|んで|不要|要ら|いら)"
    r"|(?:あまり|なるべく|できるだけ).{0,6}(?:歩|移動)"
)
_LOW_EXERTION = re.compile(r"(?:疲れ|疲労|しんど).{0,8}(?:少な|にく|ない|ず|軽|避け)")
_WATCH_LISTEN = re.compile(
    r"(?:鑑賞|観覧|見る|見たり|見て|見られ|聞く|聞いたり|聞いて|聴く|聴いたり|聴いて).{0,14}"
    r"(?:中心|メイン|だけ|楽し|落ち着|主体)"
    r"|(?:見る|見たり|見て|見られ).{0,10}(?:聞く|聞いたり|聞いて|聴く|聴いたり|聴いて).{0,10}(?:でき|楽し|中心|メイン|主体)"
    r"|(?:聞く|聞いたり|聞いて|聴く|聴いたり|聴いて).{0,10}(?:見る|見たり|見て|見られ).{0,10}(?:でき|楽し|中心|メイン|主体)"
    r"|(?:見る|見たり|聞く|聞いたり|聴く|聴いたり)(?:か|・|や|と)(?:見る|聞く|聴く).{0,8}(?:中心|メイン|主体)"
)
_HANDS_ON = re.compile(r"(?:手作り|制作|工作|ワークショップ|作(?:る|り|れ|って)|体験(?:する|でき|した|型))")
_WALK_EXPLORE = re.compile(r"(?:まち歩き|街歩き|散策|歩いて(?:巡|回)|徒歩で(?:巡|回))")
_PARTICIPATION = re.compile(r"(?:参加型|観客参加|一緒に参加|みんなで参加)")

_ADULT = re.compile(r"(?:大人|成人)(?:向け|が|で|だけ|中心|対象|楽し)")
_FAMILY = re.compile(r"(?:家族|親子|子ども連れ|こども連れ|子連れ|孫と|祖父母と孫)")

# Release and exclusion must be concept-scoped. A release of ``seated`` in the
# first clause must never neutralize an independent watch/listen requirement in
# the second clause.
_RELEASE_PATTERNS = {
    "seated": re.compile(r"(?:座(?:れ|ら|って|る|り)|着席).{0,8}(?:なくても(?:いい|大丈夫)|なくて(?:いい|大丈夫)|不要|こだわらない|問わない)"),
    "low_mobility": re.compile(r"(?:歩(?:く|いて)|移動).{0,8}(?:でも(?:いい|大丈夫|構わない)|こだわらない|問わない)"),
    "hands_on": re.compile(r"(?:体験型|体験|手作り|制作).{0,8}(?:じゃなくても(?:いい|大丈夫)|でなくても(?:いい|大丈夫)|こだわらない|問わない)"),
    "walk_explore": re.compile(r"(?:歩くイベント|まち歩き|街歩き|散策).{0,8}(?:でも(?:いい|大丈夫|構わない)|こだわらない|問わない)"),
    "audience_participation": re.compile(r"(?:参加型|観客参加).{0,8}(?:じゃなくても(?:いい|大丈夫)|でなくても(?:いい|大丈夫)|こだわらない|問わない)"),
}
_EXCLUSION_MARKER = r"(?:外(?:す|して|したい|せ)|除(?:く|いて|外)|以外|避け(?:る|て|たい)|嫌|いや|苦手|じゃないもの|でないもの)"


def _text(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value))).lower()


def _released_for_concept(text: str, concept: str) -> bool:
    pattern = _RELEASE_PATTERNS.get(concept)
    return bool(pattern and pattern.search(text))


def _explicit_exclusion_for_concept(text: str, concept: str) -> bool:
    """Recognize exclusion compositionally around canonical concept aliases."""

    try:
        aliases = experience_preferences.concept(concept).aliases
    except Exception:
        return False
    for alias in sorted(aliases, key=len, reverse=True):
        escaped = re.escape(alias)
        if re.search(rf"{escaped}.{{0,8}}{_EXCLUSION_MARKER}", text):
            return True
        if re.search(rf"{_EXCLUSION_MARKER}.{{0,8}}{escaped}", text):
            return True
    # Residual broad forms that intentionally refer to the existing hands-on
    # dimension even when the controlled alias is morphologically shortened.
    if concept == "hands_on" and re.search(rf"(?:体験|手作り|制作|工作).{{0,8}}{_EXCLUSION_MARKER}", text):
        return True
    return False


def prove_audience(query: str, value: str) -> GroundingProof:
    text = _text(query)
    if value == "adult" and _ADULT.search(text):
        return GroundingProof(True, "explicit_expression", "audience:adult")
    if value == "family" and _FAMILY.search(text):
        return GroundingProof(True, "explicit_expression", "audience:family")
    # target is intentionally not inferred from arbitrary demographic language.
    return GroundingProof(False)


def prove_experience(query: str, concept: str, action: str) -> GroundingProof:
    text = _text(query)
    resolved = experience_preferences.resolve_experience_query(query)
    if action == "require" and concept in resolved.required:
        return GroundingProof(True, "controlled_vocabulary", f"experience:{concept}:required")
    if action == "prefer" and concept in resolved.preferred:
        return GroundingProof(True, "controlled_vocabulary", f"experience:{concept}:preferred")
    if action == "exclude" and concept in resolved.excluded:
        return GroundingProof(True, "controlled_vocabulary", f"experience:{concept}:excluded")

    # Do not upgrade/downgrade an already recognized preference-strength signal.
    if action == "require" and concept in resolved.preferred:
        return GroundingProof(False)
    if action == "prefer" and concept in resolved.required:
        return GroundingProof(False)

    if action == "exclude":
        if _explicit_exclusion_for_concept(text, concept):
            return GroundingProof(True, "explicit_expression", f"experience:{concept}:excluded_composition")
        return GroundingProof(False)

    if action not in {"require", "prefer"} or _released_for_concept(text, concept):
        return GroundingProof(False)

    preferred = bool(_PREFERENCE.search(text))
    if action == "prefer" and not preferred:
        return GroundingProof(False)

    if concept == "seated":
        if _SEATED_DIRECT.search(text) or _STANDING_LIMIT.search(text):
            return GroundingProof(True, "explicit_expression", "functional:posture")
        # A lower-limb limitation combined with an explicit request to observe
        # without load is a functional posture requirement, not a demographic
        # proxy. The condition must contain both parts; neither alone proves it.
        if _LOWER_LIMB_LIMIT.search(text) and _EASY_OBSERVATION.search(text):
            return GroundingProof(True, "explicit_expression", "functional:posture_from_lower_limb_load")
    if concept == "low_mobility":
        if _LOW_MOBILITY_DIRECT.search(text):
            return GroundingProof(True, "explicit_expression", "functional:mobility_load")
        if action == "prefer" and _LOW_EXERTION.search(text):
            return GroundingProof(True, "explicit_expression", "functional:low_exertion_preference")
    if concept == "watch_listen" and _WATCH_LISTEN.search(text):
        return GroundingProof(True, "explicit_expression", "functional:watch_listen")
    if concept == "hands_on" and _HANDS_ON.search(text) and not _explicit_exclusion_for_concept(text, concept):
        return GroundingProof(True, "explicit_expression", "functional:hands_on")
    if concept == "walk_explore" and _WALK_EXPLORE.search(text):
        return GroundingProof(True, "explicit_expression", "functional:walk_explore")
    if concept == "audience_participation" and _PARTICIPATION.search(text):
        return GroundingProof(True, "explicit_expression", "functional:audience_participation")
    return GroundingProof(False)


__all__ = ["GroundingProof", "prove_audience", "prove_experience"]
