"""Controlled natural-language semantics for event experience preferences.

The user-side concepts in this module are deliberately separate from the
event-side ``ExperienceProfile``.  This module only answers: "what kind of
experience did the user request?"  It never decides whether an event really
has that property.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parent
VOCABULARY_PATH = ROOT / "data" / "experience_vocabulary.json"
SCHEMA_VERSION = 1
MAX_EXPERIENCE_ITEMS = 8
_CONCEPT_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,31}$")
_PREDICATE_FIELDS = frozenset({"posture", "seating", "mobility_load", "engagement_modes"})
_PREDICATE_VALUES = {
    "posture": frozenset({"mostly_seated", "mixed", "standing_or_walking", "unknown"}),
    "seating": frozenset({"guaranteed", "available", "limited", "none", "unknown"}),
    "mobility_load": frozenset({"low", "medium", "high", "unknown"}),
    "engagement_modes": frozenset({"watch", "listen", "hands_on", "audience_participation", "walk_explore"}),
}

# These modifiers describe the user's strength of preference, not event facts.
_PREFERRED_MARKERS = (
    "できれば",
    "できたら",
    "なるべく",
    "できるだけ",
    "ほうがいい",
    "方がいい",
    "方がよい",
)
_EXPERIENCE_MODIFIER_PHRASES = _PREFERRED_MARKERS

# Release phrases are checked before aliases so ``座ってなくてもいい`` does
# not accidentally match the shorter ``座って`` alias.
_RELEASE_PHRASES = (
    "座ってなくてもいい",
    "座ってなくても大丈夫",
    "座っていなくてもいい",
    "座れなくてもいい",
    "座れなくても大丈夫",
    "座れなくても構わない",
    "座らなくてもいい",
    "歩くイベントでも大丈夫",
    "歩くイベントでも構わない",
    "歩いてもいい",
    "歩いても大丈夫",
    "体験型じゃなくていい",
    "体験型でなくてもいい",
    "参加型じゃなくていい",
    "参加型でなくてもいい",
)

_EXCLUSION_PHRASES = (
    ("歩くイベントは嫌", "walk_explore"),
    ("歩くイベントはいや", "walk_explore"),
    ("歩くイベントを除いて", "walk_explore"),
    ("歩くイベント以外", "walk_explore"),
    ("まち歩きは嫌", "walk_explore"),
    ("まち歩きはいや", "walk_explore"),
    ("まち歩きは除いて", "walk_explore"),
    ("まち歩き以外", "walk_explore"),
    ("まち歩きじゃないもの", "walk_explore"),
    ("座るイベントは嫌", "seated"),
    ("座るイベント以外", "seated"),
    ("体験型は嫌", "hands_on"),
    ("体験型以外", "hands_on"),
    ("体験型じゃないもの", "hands_on"),
    ("参加型は嫌", "audience_participation"),
    ("参加型以外", "audience_participation"),
    ("参加型じゃないもの", "audience_participation"),
)


class ExperienceVocabularyError(ValueError):
    """Raised when the controlled vocabulary or semantic slots are invalid."""


@dataclass(frozen=True)
class ExperienceConcept:
    id: str
    label: str
    aliases: tuple[str, ...]
    predicate: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class ExperienceQuery:
    """User-side normalized experience intent."""

    required: tuple[str, ...] = ()
    preferred: tuple[str, ...] = ()
    excluded: tuple[str, ...] = ()
    matched_phrases: tuple[str, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        concepts = valid_concept_ids()
        for field_name in ("required", "preferred", "excluded"):
            value = getattr(self, field_name)
            if not isinstance(value, (list, tuple)):
                raise ExperienceVocabularyError(f"{field_name} must be an array")
            if len(value) > MAX_EXPERIENCE_ITEMS:
                raise ExperienceVocabularyError(f"{field_name} has too many concepts")
            normalized: list[str] = []
            for item in value:
                if not isinstance(item, str) or item not in concepts:
                    raise ExperienceVocabularyError(f"unknown experience concept: {item!r}")
                if item in normalized:
                    raise ExperienceVocabularyError(f"duplicate experience concept: {item!r}")
                normalized.append(item)
            object.__setattr__(self, field_name, tuple(normalized))
        overlap = (set(self.required) & set(self.preferred)) | (set(self.required) & set(self.excluded)) | (set(self.preferred) & set(self.excluded))
        if overlap:
            raise ExperienceVocabularyError(f"conflicting experience concepts: {sorted(overlap)}")
        phrases = tuple(
            item.strip()
            for item in self.matched_phrases
            if isinstance(item, str) and item.strip()
        )
        object.__setattr__(self, "matched_phrases", tuple(dict.fromkeys(phrases)))

    @property
    def recognized(self) -> bool:
        return bool(self.required or self.preferred or self.excluded)

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "required": list(self.required),
            "preferred": list(self.preferred),
            "excluded": list(self.excluded),
        }


def _normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def compact(value: Any) -> str:
    return re.sub(r"\s+", "", _normalize(value))


def _validate_vocabulary(raw: Any) -> dict[str, ExperienceConcept]:
    if not isinstance(raw, Mapping) or set(raw) != {"schema_version", "concepts"}:
        raise ExperienceVocabularyError("experience vocabulary root is invalid")
    if raw["schema_version"] != SCHEMA_VERSION or not isinstance(raw["concepts"], list):
        raise ExperienceVocabularyError("experience vocabulary schema is invalid")
    concepts: dict[str, ExperienceConcept] = {}
    seen_aliases: set[str] = set()
    for index, item in enumerate(raw["concepts"]):
        if not isinstance(item, Mapping) or set(item) != {"id", "label", "aliases", "predicate"}:
            raise ExperienceVocabularyError(f"concept {index} is invalid")
        concept_id = item["id"]
        label = item["label"]
        aliases = item["aliases"]
        predicate = item["predicate"]
        if not isinstance(concept_id, str) or not _CONCEPT_ID_RE.fullmatch(concept_id) or concept_id in concepts:
            raise ExperienceVocabularyError(f"concept {index} has an invalid or duplicate id")
        if not isinstance(label, str) or not label.strip():
            raise ExperienceVocabularyError(f"concept {concept_id} label is invalid")
        if (
            not isinstance(aliases, list)
            or not aliases
            or not all(isinstance(alias, str) for alias in aliases)
            or len(aliases) != len(set(aliases))
        ):
            raise ExperienceVocabularyError(f"concept {concept_id} aliases are invalid")
        normalized_aliases: list[str] = []
        for alias in aliases:
            if not isinstance(alias, str):
                raise ExperienceVocabularyError(f"concept {concept_id} has an invalid alias")
            normalized_alias = compact(alias)
            if not normalized_alias or len(normalized_alias) > 80:
                raise ExperienceVocabularyError(f"concept {concept_id} has an invalid alias")
            if normalized_alias in seen_aliases:
                raise ExperienceVocabularyError(f"duplicate alias: {normalized_alias}")
            seen_aliases.add(normalized_alias)
            normalized_aliases.append(normalized_alias)
        if not isinstance(predicate, Mapping) or not predicate or set(predicate) - _PREDICATE_FIELDS:
            raise ExperienceVocabularyError(f"concept {concept_id} predicate is invalid")
        normalized_predicate: dict[str, tuple[str, ...]] = {}
        for field_name, values in predicate.items():
            if (
                not isinstance(values, list)
                or not values
                or not all(isinstance(value, str) for value in values)
                or len(values) != len(set(values))
            ):
                raise ExperienceVocabularyError(f"concept {concept_id} predicate values are invalid")
            if not all(isinstance(value, str) and value in _PREDICATE_VALUES[field_name] for value in values):
                raise ExperienceVocabularyError(f"concept {concept_id} predicate value is invalid")
            normalized_predicate[field_name] = tuple(values)
        concepts[concept_id] = ExperienceConcept(
            id=concept_id,
            label=label.strip(),
            aliases=tuple(normalized_aliases),
            predicate=normalized_predicate,
        )
    if not concepts:
        raise ExperienceVocabularyError("experience vocabulary must not be empty")
    return concepts


@lru_cache(maxsize=1)
def load_vocabulary(path: str | Path = VOCABULARY_PATH) -> dict[str, ExperienceConcept]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperienceVocabularyError("experience vocabulary could not be loaded") from exc
    return _validate_vocabulary(raw)


def valid_concept_ids() -> frozenset[str]:
    return frozenset(load_vocabulary())


EXPERIENCE_CONCEPT_IDS = valid_concept_ids()


def concept(value: str) -> ExperienceConcept:
    try:
        return load_vocabulary()[value]
    except KeyError as exc:
        raise ExperienceVocabularyError(f"unknown experience concept: {value!r}") from exc


def normalize_concept_ids(value: Iterable[str] | None, *, field_name: str = "experience") -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ExperienceVocabularyError(f"{field_name} must be an array of concept IDs")
    if len(value) > MAX_EXPERIENCE_ITEMS:
        raise ExperienceVocabularyError(f"{field_name} has too many concepts")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or item not in valid_concept_ids():
            raise ExperienceVocabularyError(f"{field_name} contains unknown concept: {item!r}")
        if item in result:
            raise ExperienceVocabularyError(f"{field_name} contains duplicate concept: {item!r}")
        result.append(item)
    return tuple(result)


def _preferred_context(text: str, alias: str) -> bool:
    index = text.find(alias)
    if index < 0:
        return False
    prefix = text[max(0, index - 18):index]
    suffix = text[index + len(alias):index + len(alias) + 12]
    return any(marker in prefix for marker in _PREFERRED_MARKERS) or any(
        marker in suffix for marker in ("ほうがいい", "方がいい", "方がよい")
    ) or any(marker in alias for marker in ("疲れにくそう", "そうなもの"))


def _remove_phrases(text: str, phrases: Iterable[str]) -> str:
    result = text
    for phrase in sorted({compact(item) for item in phrases}, key=len, reverse=True):
        if phrase:
            result = result.replace(phrase, " ")
    return result


def resolve_experience_query(query: str) -> ExperienceQuery:
    """Resolve high-confidence Japanese experience expressions deterministically."""

    text = compact(query)
    if not text:
        return ExperienceQuery()

    excluded: list[str] = []
    excluded_phrases: list[str] = []
    for phrase, concept_id in _EXCLUSION_PHRASES:
        phrase_compact = compact(phrase)
        if phrase_compact in text:
            excluded.append(concept_id)
            excluded_phrases.append(phrase_compact)

    working = _remove_phrases(text, (*_RELEASE_PHRASES, *excluded_phrases))
    required: list[str] = []
    preferred: list[str] = []
    matched: list[str] = []
    for concept_id, item in load_vocabulary().items():
        for alias in sorted(item.aliases, key=len, reverse=True):
            if alias not in working:
                continue
            if concept_id in excluded:
                break
            matched.append(alias)
            if _preferred_context(text, alias):
                if concept_id not in preferred:
                    preferred.append(concept_id)
            elif concept_id not in required:
                required.append(concept_id)
            break

    # A direct positive request takes precedence over a weaker duplicate
    # alias.  Conflicting expressions are treated conservatively as no hard
    # preference rather than inventing a feasible event set.
    overlap = set(required) & set(preferred)
    if overlap:
        required = [value for value in required if value not in overlap]
        preferred = [value for value in preferred if value not in overlap]
    excluded_set = set(excluded)
    required = [value for value in required if value not in excluded_set]
    preferred = [value for value in preferred if value not in excluded_set]
    return ExperienceQuery(
        required=tuple(required),
        preferred=tuple(preferred),
        excluded=tuple(dict.fromkeys(excluded)),
        matched_phrases=tuple(dict.fromkeys(matched)),
    )


def is_experience_phrase(value: Any) -> bool:
    text = compact(value)
    if not text:
        return False
    if any(compact(phrase) in text for phrase in (*_RELEASE_PHRASES, *(phrase for phrase, _ in _EXCLUSION_PHRASES))):
        return True
    return any(alias in text for item in load_vocabulary().values() for alias in item.aliases)


def has_release_phrase(value: Any) -> bool:
    """Return whether the text explicitly releases an experience constraint."""

    text = compact(value)
    return bool(text and any(compact(phrase) in text for phrase in _RELEASE_PHRASES))


def remove_experience_phrases(value: Any) -> str:
    """Remove recognized experience language before legacy soft-term parsing."""

    text = _normalize(value)
    phrases = [
        *[alias for item in load_vocabulary().values() for alias in item.aliases],
        *_RELEASE_PHRASES,
        *[phrase for phrase, _ in _EXCLUSION_PHRASES],
        *_EXPERIENCE_MODIFIER_PHRASES,
    ]
    return _remove_phrases(text, phrases)


def labels_for(query: ExperienceQuery | Mapping[str, Any]) -> tuple[str, ...]:
    if not isinstance(query, ExperienceQuery):
        query = ExperienceQuery(
            required=tuple(query.get("required", query.get("experience_required", ()))),
            preferred=tuple(query.get("preferred", query.get("experience_preferred", ()))),
            excluded=tuple(query.get("excluded", query.get("experience_excluded", ()))),
        )
    return tuple(
        concept(value).label
        for value in (*query.required, *query.preferred, *query.excluded)
    )


def render_result_message(
    total_matches: int,
    *,
    required: Iterable[str] = (),
    preferred: Iterable[str] = (),
    excluded: Iterable[str] = (),
) -> str:
    query = ExperienceQuery(tuple(required), tuple(preferred), tuple(excluded))
    if query.required:
        labels = "・".join(labels_for(query))
        return f"{labels}条件を満たすことを確認できるイベントを探したよ。{total_matches}件見つかったけん、カードを見てみて。"
    if query.preferred:
        labels = "・".join(labels_for(query))
        return f"できるだけ{labels}条件に合うものを上位にしたよ。{total_matches}件見つかったけん、カードを見てみて。"
    if query.excluded:
        labels = "・".join(labels_for(query))
        return f"{labels}条件に当たるイベントを除いて探したよ。{total_matches}件見つかったけん、カードを見てみて。"
    return f"条件に合うイベントが{total_matches}件見つかりました。カードを見てみて。"


__all__ = [
    "EXPERIENCE_CONCEPT_IDS",
    "ExperienceConcept",
    "ExperienceQuery",
    "ExperienceVocabularyError",
    "MAX_EXPERIENCE_ITEMS",
    "VOCABULARY_PATH",
    "concept",
    "compact",
    "is_experience_phrase",
    "has_release_phrase",
    "labels_for",
    "load_vocabulary",
    "normalize_concept_ids",
    "remove_experience_phrases",
    "render_result_message",
    "resolve_experience_query",
    "valid_concept_ids",
]
