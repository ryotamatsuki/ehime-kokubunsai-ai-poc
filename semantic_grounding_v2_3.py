"""Compositional grounding proofs for Semantic Operations v2.3.

The deterministic parser remains the first source of truth.  This module covers
bounded residual expressions that explicitly describe an existing supported
dimension without enumerating complete user utterances.  It proves only the
semantic relation between an expression and an existing catalog dimension; it
never infers a need from demographics, personality, expertise or diagnoses.
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


_FALSE_POSITIVE_TOLERANCE = re.compile(
    r"(?:じゃなくても(?:いい|大丈夫)|でなくても(?:いい|大丈夫)|でも(?:いい|大丈夫|構わない)|"
    r"は(?:どうでもいい|気にしない)|こだわらない|問わない)"
)
_PREFERENCE = re.compile(r"(?:できれば|なるべく|ほうがいい|方がいい|がいい|希望|疲れにく|疲れない)")

# Functional-load grammar.  These are dimension-level predicates, not complete
# test utterances.  Demographic words are deliberately absent.
_SEATED_DIRECT = re.compile(r"(?:座(?:る|れ|っ|り|席)|着席|椅子|いす|イス|すわ(?:る|れ|っ))")
_STANDING_LIMIT = re.compile(r"(?:立(?:つ|て|ち|た)|立ちっぱなし).{0,8}(?:無理|難|できない|出来ない|つら|辛|嫌|いや|避け|ない)")
_LOW_MOBILITY_DIRECT = re.compile(
    r"(?:歩(?:く|き|か)|移動|足腰|脚|膝|ひざ).{0,12}"
    r"(?:少な|短|弱|悪|苦手|つら|辛|無理|疲|しんど|ない|なく|ず|んで|不要|要ら|いら)"
    r"|(?:あまり|なるべく|できるだけ).{0,6}(?:歩|移動)"
)
_LOW_EXERTION = re.compile(r"(?:疲れ|疲労|しんど).{0,8}(?:少な|にく|ない|ず|軽|避け)")
_WATCH_LISTEN = re.compile(
    r"(?:鑑賞|観覧|見る|見たり|見て|聞く|聞いたり|聴く|聴いたり).{0,14}"
    r"(?:中心|メイン|だけ|楽し|落ち着|主体)"
    r"|(?:見る|見たり|聞く|聞いたり|聴く|聴いたり)(?:か|・|や|と)(?:見る|聞く|聴く).{0,8}(?:中心|メイン|主体)"
)
_HANDS_ON = re.compile(r"(?:手作り|制作|工作|ワークショップ|作(?:る|り|れ|って)|体験(?:する|でき|した|型))")
_HANDS_ON_NEGATIVE = re.compile(r"(?:体験|手作り|制作|工作|作る).{0,8}(?:苦手|嫌|いや|避け|じゃない|でない|以外|除外|除いて)")
_WALK_EXPLORE = re.compile(r"(?:まち歩き|街歩き|散策|歩いて(?:巡|回)|徒歩で(?:巡|回))")
_PARTICIPATION = re.compile(r"(?:参加型|観客参加|一緒に参加|みんなで参加)")

_ADULT = re.compile(r"(?:大人|成人)(?:向け|が|で|だけ|中心|対象|楽し)")
_FAMILY = re.compile(r"(?:家族|親子|子ども連れ|こども連れ|子連れ|孫と|祖父母と孫)")


def _text(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value))).lower()


def _released(text: str) -> bool:
    return bool(_FALSE_POSITIVE_TOLERANCE.search(text))


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

    if action == "exclude":
        if concept == "hands_on" and _HANDS_ON_NEGATIVE.search(text):
            return GroundingProof(True, "explicit_expression", "hands_on:negative")
        return GroundingProof(False)

    if action not in {"require", "prefer"} or _released(text):
        return GroundingProof(False)

    preferred = bool(_PREFERENCE.search(text))
    if action == "prefer" and not preferred:
        return GroundingProof(False)
    if action == "require" and preferred:
        # A preference must not be silently upgraded to a hard requirement.
        return GroundingProof(False)

    if concept == "seated" and (_SEATED_DIRECT.search(text) or _STANDING_LIMIT.search(text)):
        return GroundingProof(True, "explicit_expression", "functional:posture")
    if concept == "low_mobility":
        if _LOW_MOBILITY_DIRECT.search(text):
            return GroundingProof(True, "explicit_expression", "functional:mobility_load")
        if action == "prefer" and _LOW_EXERTION.search(text):
            return GroundingProof(True, "explicit_expression", "functional:low_exertion_preference")
    if concept == "watch_listen" and _WATCH_LISTEN.search(text):
        return GroundingProof(True, "explicit_expression", "functional:watch_listen")
    if concept == "hands_on" and _HANDS_ON.search(text) and not _HANDS_ON_NEGATIVE.search(text):
        return GroundingProof(True, "explicit_expression", "functional:hands_on")
    if concept == "walk_explore" and _WALK_EXPLORE.search(text):
        return GroundingProof(True, "explicit_expression", "functional:walk_explore")
    if concept == "audience_participation" and _PARTICIPATION.search(text):
        return GroundingProof(True, "explicit_expression", "functional:audience_participation")
    return GroundingProof(False)


__all__ = ["GroundingProof", "prove_audience", "prove_experience"]
