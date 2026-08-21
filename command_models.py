"""Semantic command contracts for the event-guide PoC.

The command generator is an untrusted boundary.  This module contains the
small, dependency-free-ish value objects used between that boundary and the
deterministic flow executor.  It intentionally does not know how to search
events or execute tools.

The public ``from_dict``/``parse_command_plan`` entry points are strict:
unknown fields and values that are outside the command contract are rejected
instead of being silently ignored.  The dataclasses also validate direct
construction so callers cannot bypass the same boundary accidentally.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import re
from typing import Any, Mapping

import age_semantics
import experience_preferences
from app_config import GENRE_ALIASES, MAX_RESULT_SET_SIZE, REGION_CITIES


# The flow names are the only semantic actions the command generator may
# request.  Tool names deliberately do not appear here; the mapping lives in
# flow_registry.py.
FLOW_NAMES = frozenset(
    {
        "find_events",
        "count_events",
        "event_detail",
        "recommend_next",
        "recommend_similar",
        "plan_event_pair",
        "explain_search",
        "explain_result",
        "general_faq",
        "unsupported",
    }
)

CONFIDENCE_VALUES = frozenset({"high", "medium", "low"})
AUDIENCE_VALUES = frozenset(
    {"family", "preschool", "elementary", "junior_high", "high_school", "adult"}
)
AGE_GROUP_VALUES = frozenset(
    {"preschool", "elementary", "junior_high", "high_school", "adult"}
)
AGE_INTENT_VALUES = frozenset({"recommended", "eligible"})
VENUE_VALUES = frozenset({"indoor", "outdoor"})
TIME_SLOT_VALUES = frozenset({"午前", "午後", "夕方"})
REFERENCE_KIND_VALUES = frozenset({"ordinal", "event_name", "selected", "last_result"})
GENRE_VALUES = frozenset(GENRE_ALIASES)

CANONICAL_REGIONS = frozenset(REGION_CITIES)
CANONICAL_MUNICIPALITIES = frozenset(
    municipality for municipalities in REGION_CITIES.values() for municipality in municipalities
)

# Reference indexes address the complete bounded result set, not the number of
# cards visible on the first UI page.  The index remains 1-based
# ("2番目" -> 2).
MAX_REFERENCE_INDEX = MAX_RESULT_SET_SIZE
MAX_VISIT_COUNT = 2
MAX_TEXT_LENGTH = 240
MAX_SHORT_TEXT_LENGTH = 64
MAX_TOPIC_LENGTH = 64
MAX_COLLECTION_ITEMS = 32
MAX_TOPIC_ITEMS = 8

# Conversation scaffolding is not a content topic.  This is a bounded schema
# safety rule, not a growing natural-language synonym table.
NON_TOPIC_MARKERS = (
    "含めて",
    "一緒に",
    "楽しみたい",
    "楽しめる",
    "行きたい",
    "行ける",
    "探して",
    "おすすめ",
    "イベント",
    "建物の中",
    "建物内",
    "屋内",
    "屋外",
)

# event_details.py is the source of truth for participation fields.  The
# high-level fields are included as they are also valid facts for an
# event_detail flow (日時、場所、料金、概要など).
try:
    from event_details import DETAIL_FIELDS as PARTICIPATION_DETAIL_FIELDS
except ImportError:  # pragma: no cover - useful when this file is copied alone
    PARTICIPATION_DETAIL_FIELDS = (
        "application_required",
        "application_deadline",
        "capacity",
        "target",
        "parking",
        "public_transport",
        "rain_policy",
        "wheelchair",
        "accessible_toilet",
        "sign_language",
        "contact",
        "fee_detail",
    )

EVENT_DETAIL_FIELDS = (
    "datetime",
    "place",
    "fee",
    "genre",
    "child_friendly",
    "venue",
    "overview",
    *PARTICIPATION_DETAIL_FIELDS,
    # Stable Japanese aliases used by the existing event-card/detail layer.
    "日時",
    "場所",
    "料金",
    "ジャンル",
    "概要",
    "対象",
    "申込",
    "アクセス",
    "雨天",
    "experience_profile",
)
ALLOWED_DETAIL_FIELDS = frozenset(EVENT_DETAIL_FIELDS)

# Public aliases make the schema discoverable to prompt/adapter code without
# requiring callers to know the implementation's internal names.
ALLOWED_AUDIENCES = AUDIENCE_VALUES
ALLOWED_AGE_GROUPS = AGE_GROUP_VALUES
ALLOWED_AGE_INTENTS = AGE_INTENT_VALUES
ALLOWED_VENUES = VENUE_VALUES
ALLOWED_TIME_SLOTS = TIME_SLOT_VALUES
ALLOWED_REFERENCE_KINDS = REFERENCE_KIND_VALUES
ALLOWED_GENRES = GENRE_VALUES
VALID_MUNICIPALITIES = CANONICAL_MUNICIPALITIES
VALID_REGIONS = CANONICAL_REGIONS
ALLOWED_EXPERIENCE_CONCEPTS = experience_preferences.EXPERIENCE_CONCEPT_IDS
EXPERIENCE_SLOT_FIELDS = frozenset(
    {"experience_required", "experience_preferred", "experience_excluded"}
)


class CommandValidationError(ValueError):
    """Raised when an untrusted command is outside the semantic contract."""

    def __init__(self, message: str, *, path: str | None = None) -> None:
        self.path = path
        rendered = f"{path}: {message}" if path else message
        super().__init__(rendered)


# A shorter name is convenient for callers and preserves the normal
# ``ValueError`` catch behavior.
ValidationError = CommandValidationError


_SLOT_COLLECTION_FIELDS = (
    "dates",
    "municipalities",
    "regions",
    "genres",
    "topics",
    "experience_required",
    "experience_preferred",
    "experience_excluded",
    "time_slots",
    "detail_fields",
)
_BOOLEAN_SLOT_FIELDS = (
    "entry_free",
    "paid_only",
    "reservation_required",
    "rain_preferred",
    "refine_previous",
)


def _fail(message: str, path: str | None = None) -> None:
    raise CommandValidationError(message, path=path)


def _check_unknown_fields(raw: Mapping[str, Any], allowed: frozenset[str], *, path: str) -> None:
    unknown = [key for key in raw if key not in allowed]
    if unknown:
        _fail(f"unknown field(s): {sorted(map(str, unknown))}", path)


def _text(
    value: Any,
    *,
    path: str,
    max_length: int = MAX_TEXT_LENGTH,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        _fail("must be a string", path)
    if "\x00" in value or any(ord(character) < 32 for character in value):
        _fail("must not contain control characters", path)
    normalized = value.strip()
    if not allow_empty and not normalized:
        _fail("must not be empty", path)
    if len(normalized) > max_length:
        _fail(f"must be at most {max_length} characters", path)
    return normalized


def _optional_text(
    value: Any,
    *,
    path: str,
    max_length: int = MAX_TEXT_LENGTH,
) -> str | None:
    if value is None:
        return None
    return _text(value, path=path, max_length=max_length)


def _strict_int(value: Any, *, path: str, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("must be an integer", path)
    if minimum is not None and value < minimum:
        _fail(f"must be at least {minimum}", path)
    if maximum is not None and value > maximum:
        _fail(f"must be at most {maximum}", path)
    return value


def _optional_int(
    value: Any,
    *,
    path: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    if value is None:
        return None
    return _strict_int(value, path=path, minimum=minimum, maximum=maximum)


def _optional_bool(value: Any, *, path: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        _fail("must be a boolean", path)
    return value


def _strict_date(value: Any, *, path: str) -> str:
    if not isinstance(value, str):
        _fail("must be an ISO date string (YYYY-MM-DD)", path)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        _fail("must be an ISO date string (YYYY-MM-DD)", path)
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        _fail("is not a valid calendar date", path)
    if parsed.isoformat() != value:
        _fail("must use the canonical YYYY-MM-DD form", path)
    return value


def _collection(
    value: Any,
    *,
    path: str,
    max_items: int = MAX_COLLECTION_ITEMS,
    item_max_length: int = MAX_TEXT_LENGTH,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        _fail("must be an array of strings", path)
    if len(value) > max_items:
        _fail(f"must contain at most {max_items} items", path)
    result: list[str] = []
    for index, item in enumerate(value):
        result.append(_text(item, path=f"{path}[{index}]", max_length=item_max_length))
    return tuple(result)


def _normalize_direct_collections(instance: Any) -> None:
    """Normalize list inputs from direct construction to the tuple contract."""

    for field_name in _SLOT_COLLECTION_FIELDS:
        value = getattr(instance, field_name)
        if not isinstance(value, (list, tuple)):
            _fail("must be an array/tuple", f"slots.{field_name}")
        if len(value) > MAX_COLLECTION_ITEMS:
            _fail(
                f"must contain at most {MAX_COLLECTION_ITEMS} items",
                f"slots.{field_name}",
            )
        object.__setattr__(instance, field_name, tuple(value))


@dataclass(frozen=True)
class CommandSlots:
    """Validated semantic parameters for one registered flow."""

    dates: tuple[str, ...] = ()
    municipalities: tuple[str, ...] = ()
    regions: tuple[str, ...] = ()
    genres: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    experience_required: tuple[str, ...] = ()
    experience_preferred: tuple[str, ...] = ()
    experience_excluded: tuple[str, ...] = ()

    audience: str | None = None
    age: int | None = None
    age_group: str | None = None
    age_intent: str | None = None

    venue: str | None = None
    entry_free: bool | None = None
    paid_only: bool | None = None
    max_entry_fee: int | None = None
    reservation_required: bool | None = None
    rain_preferred: bool | None = None

    time_slots: tuple[str, ...] = ()
    time_after: int | None = None

    visit_count: int | None = None

    reference_kind: str | None = None
    reference_index: int | None = None
    event_name: str | None = None

    detail_fields: tuple[str, ...] = ()
    refine_previous: bool = False

    def __post_init__(self) -> None:
        _normalize_direct_collections(self)

        dates = tuple(
            _strict_date(value, path=f"slots.dates[{index}]")
            for index, value in enumerate(self.dates)
        )
        municipalities = tuple(
            _text(value, path=f"slots.municipalities[{index}]", max_length=32)
            for index, value in enumerate(self.municipalities)
        )
        if any(value not in CANONICAL_MUNICIPALITIES for value in municipalities):
            invalid = next(value for value in municipalities if value not in CANONICAL_MUNICIPALITIES)
            _fail(f"unknown municipality {invalid!r}; use a canonical Ehime municipality", "slots.municipalities")

        regions = tuple(
            _text(value, path=f"slots.regions[{index}]", max_length=16)
            for index, value in enumerate(self.regions)
        )
        if any(value not in CANONICAL_REGIONS for value in regions):
            invalid = next(value for value in regions if value not in CANONICAL_REGIONS)
            _fail(f"unknown region {invalid!r}", "slots.regions")

        genres = tuple(
            _text(value, path=f"slots.genres[{index}]", max_length=MAX_SHORT_TEXT_LENGTH)
            for index, value in enumerate(self.genres)
        )
        if any(value not in GENRE_VALUES for value in genres):
            invalid = next(value for value in genres if value not in GENRE_VALUES)
            _fail(f"unknown genre {invalid!r}", "slots.genres")
        topics = tuple(
            _text(value, path=f"slots.topics[{index}]", max_length=MAX_TOPIC_LENGTH)
            for index, value in enumerate(self.topics)
        )
        if len(topics) > MAX_TOPIC_ITEMS:
            _fail(f"must contain at most {MAX_TOPIC_ITEMS} items", "slots.topics")
        if any(
            marker in topic
            for topic in topics
            for marker in NON_TOPIC_MARKERS
        ):
            _fail("conversation scaffolding is not a topic", "slots.topics")
        if any(experience_preferences.is_experience_phrase(topic) for topic in topics):
            _fail(
                "experience language must use an experience slot",
                "slots.topics",
            )

        def _experience_slot(field_name: str) -> tuple[str, ...]:
            try:
                return experience_preferences.normalize_concept_ids(
                    getattr(self, field_name),
                    field_name=field_name,
                )
            except experience_preferences.ExperienceVocabularyError as exc:
                _fail(str(exc), f"slots.{field_name}")
            raise AssertionError("unreachable")

        experience_required = _experience_slot("experience_required")
        experience_preferred = _experience_slot("experience_preferred")
        experience_excluded = _experience_slot("experience_excluded")
        try:
            experience_preferences.ExperienceQuery(
                required=experience_required,
                preferred=experience_preferred,
                excluded=experience_excluded,
            )
        except experience_preferences.ExperienceVocabularyError as exc:
            _fail(str(exc), "slots.experience")

        time_slots = tuple(
            _text(value, path=f"slots.time_slots[{index}]", max_length=16)
            for index, value in enumerate(self.time_slots)
        )
        if any(value not in TIME_SLOT_VALUES for value in time_slots):
            invalid = next(value for value in time_slots if value not in TIME_SLOT_VALUES)
            _fail(f"unknown time slot {invalid!r}", "slots.time_slots")

        detail_fields = tuple(
            _text(value, path=f"slots.detail_fields[{index}]", max_length=MAX_SHORT_TEXT_LENGTH)
            for index, value in enumerate(self.detail_fields)
        )
        if any(value not in ALLOWED_DETAIL_FIELDS for value in detail_fields):
            invalid = next(value for value in detail_fields if value not in ALLOWED_DETAIL_FIELDS)
            _fail(f"unknown detail field {invalid!r}", "slots.detail_fields")

        # The validators above strip surrounding whitespace.  Store their
        # canonical values so to_dict() never emits a subtly different shape.
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "municipalities", municipalities)
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "genres", genres)
        object.__setattr__(self, "topics", topics)
        object.__setattr__(self, "experience_required", experience_required)
        object.__setattr__(self, "experience_preferred", experience_preferred)
        object.__setattr__(self, "experience_excluded", experience_excluded)
        object.__setattr__(self, "time_slots", time_slots)
        object.__setattr__(self, "detail_fields", detail_fields)

        if self.audience is not None:
            audience = _text(self.audience, path="slots.audience", max_length=32)
            if audience not in AUDIENCE_VALUES:
                _fail(f"must be one of {sorted(AUDIENCE_VALUES)}", "slots.audience")
            object.__setattr__(self, "audience", audience)

        age = _optional_int(self.age, path="slots.age", minimum=0, maximum=120)
        object.__setattr__(self, "age", age)

        if self.age_group is not None:
            age_group = _text(self.age_group, path="slots.age_group", max_length=32)
            if age_group not in AGE_GROUP_VALUES:
                _fail(f"must be one of {sorted(AGE_GROUP_VALUES)}", "slots.age_group")
            object.__setattr__(self, "age_group", age_group)

        numeric_group = age_semantics.age_group_for_age(age) if age is not None else None
        requested_groups = {
            group
            for group in (
                None if self.audience in (None, "family") else self.audience,
                self.age_group,
                numeric_group,
            )
            if group is not None
        }
        if len(requested_groups) > 1:
            _fail(
                "audience, age, and age_group contain conflicting constraints",
                "slots",
            )
        if self.audience == "family" and "adult" in requested_groups:
            _fail(
                "family cannot be combined with an adult age constraint",
                "slots",
            )

        if self.age_intent is not None:
            age_intent = _text(self.age_intent, path="slots.age_intent", max_length=32)
            if age_intent not in AGE_INTENT_VALUES:
                _fail(f"must be one of {sorted(AGE_INTENT_VALUES)}", "slots.age_intent")
            object.__setattr__(self, "age_intent", age_intent)

        if self.venue is not None:
            venue = _text(self.venue, path="slots.venue", max_length=16)
            if venue not in VENUE_VALUES:
                _fail(f"must be one of {sorted(VENUE_VALUES)}", "slots.venue")
            object.__setattr__(self, "venue", venue)

        for field_name in _BOOLEAN_SLOT_FIELDS:
            value = getattr(self, field_name)
            if field_name == "refine_previous":
                if not isinstance(value, bool):
                    _fail("must be a boolean", f"slots.{field_name}")
            else:
                _optional_bool(value, path=f"slots.{field_name}")

        if self.entry_free is True and self.paid_only is True:
            _fail("entry_free and paid_only cannot both be true", "slots")

        object.__setattr__(
            self,
            "max_entry_fee",
            _optional_int(self.max_entry_fee, path="slots.max_entry_fee", minimum=0),
        )
        object.__setattr__(
            self,
            "time_after",
            _optional_int(self.time_after, path="slots.time_after", minimum=0, maximum=24 * 60),
        )
        object.__setattr__(
            self,
            "visit_count",
            _optional_int(self.visit_count, path="slots.visit_count", minimum=1, maximum=MAX_VISIT_COUNT),
        )
        object.__setattr__(
            self,
            "reference_index",
            _optional_int(
                self.reference_index,
                path="slots.reference_index",
                minimum=1,
                maximum=MAX_REFERENCE_INDEX,
            ),
        )
        object.__setattr__(
            self,
            "reference_kind",
            _optional_text(self.reference_kind, path="slots.reference_kind", max_length=MAX_SHORT_TEXT_LENGTH),
        )
        if self.reference_kind is not None and self.reference_kind not in REFERENCE_KIND_VALUES:
            _fail(
                f"must be one of {sorted(REFERENCE_KIND_VALUES)}",
                "slots.reference_kind",
            )
        object.__setattr__(
            self,
            "event_name",
            _optional_text(self.event_name, path="slots.event_name", max_length=MAX_TEXT_LENGTH),
        )

    @classmethod
    def from_dict(cls, raw: Any) -> "CommandSlots":
        """Build and strictly validate slots from a JSON-like object."""

        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, Mapping):
            _fail("must be an object", "slots")
        allowed = frozenset(field.name for field in cls.__dataclass_fields__.values())
        _check_unknown_fields(raw, allowed, path="slots")
        return cls(**dict(raw))

    @classmethod
    def from_json(cls, raw: str) -> "CommandSlots":
        """Parse a JSON object and validate it as ``CommandSlots``."""

        parsed = _parse_json_object(raw, path="slots")
        return cls.from_dict(parsed)

    def validate(self) -> "CommandSlots":
        """Return this already-validated immutable value object."""

        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of the slots."""

        return {
            "dates": list(self.dates),
            "municipalities": list(self.municipalities),
            "regions": list(self.regions),
            "genres": list(self.genres),
            "topics": list(self.topics),
            "experience_required": list(self.experience_required),
            "experience_preferred": list(self.experience_preferred),
            "experience_excluded": list(self.experience_excluded),
            "audience": self.audience,
            "age": self.age,
            "age_group": self.age_group,
            "age_intent": self.age_intent,
            "venue": self.venue,
            "entry_free": self.entry_free,
            "paid_only": self.paid_only,
            "max_entry_fee": self.max_entry_fee,
            "reservation_required": self.reservation_required,
            "rain_preferred": self.rain_preferred,
            "time_slots": list(self.time_slots),
            "time_after": self.time_after,
            "visit_count": self.visit_count,
            "reference_kind": self.reference_kind,
            "reference_index": self.reference_index,
            "event_name": self.event_name,
            "detail_fields": list(self.detail_fields),
            "refine_previous": self.refine_previous,
        }

    def to_json(self, **kwargs: Any) -> str:
        """Serialize slots as JSON without exposing non-JSON Python types."""

        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)


@dataclass(frozen=True)
class CommandPlan:
    """A semantic flow request; no raw tool name is accepted here."""

    flow: str
    slots: CommandSlots
    confidence: str = "medium"

    def __post_init__(self) -> None:
        if not isinstance(self.flow, str) or not self.flow:
            _fail("must be a non-empty string", "flow")
        if self.flow not in FLOW_NAMES:
            _fail(f"unknown flow {self.flow!r}", "flow")
        if not isinstance(self.slots, CommandSlots):
            _fail("must be a CommandSlots object", "slots")
        if not isinstance(self.confidence, str) or self.confidence not in CONFIDENCE_VALUES:
            _fail(f"must be one of {sorted(CONFIDENCE_VALUES)}", "confidence")
        if self.flow == "plan_event_pair":
            if len(self.slots.dates) > 1:
                _fail(
                    "plan_event_pair accepts exactly one date",
                    "slots.dates",
                )
            if self.slots.visit_count not in (None, 2):
                _fail(
                    "plan_event_pair requires visit_count=2",
                    "slots.visit_count",
                )

    @classmethod
    def from_dict(cls, raw: Any) -> "CommandPlan":
        """Build and strictly validate a command plan from a JSON object."""

        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, Mapping):
            _fail("must be an object", "command")
        allowed = frozenset({"flow", "slots", "confidence"})
        _check_unknown_fields(raw, allowed, path="command")
        if "flow" not in raw:
            _fail("is required", "command.flow")
        if "slots" not in raw:
            _fail("is required", "command.slots")
        confidence = raw.get("confidence", "medium")
        return cls(
            flow=raw["flow"],
            slots=CommandSlots.from_dict(raw["slots"]),
            confidence=confidence,
        )

    @classmethod
    def from_json(cls, raw: str) -> "CommandPlan":
        """Parse a JSON object and validate it as ``CommandPlan``."""

        return cls.from_dict(_parse_json_object(raw, path="command"))

    def validate(self) -> "CommandPlan":
        """Validate the registered-flow boundary and return this plan."""

        return validate_command_plan(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "flow": self.flow,
            "slots": self.slots.to_dict(),
            "confidence": self.confidence,
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate field {key!r}", "json")
        result[key] = value
    return result


def _parse_json_object(raw: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(raw, str):
        _fail("must be a JSON string", path)
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    if not text:
        _fail("must not be empty", path)
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda value: (_fail(f"invalid JSON constant {value!r}", "json")),
        )
    except json.JSONDecodeError as exc:
        _fail(f"invalid JSON: {exc.msg}", path)
    if not isinstance(parsed, Mapping):
        _fail("must contain a JSON object", path)
    return parsed


def parse_command_plan(raw: Any) -> CommandPlan:
    """Parse a mapping or JSON response into a validated command plan."""

    if isinstance(raw, CommandPlan):
        return validate_command_plan(raw)
    if isinstance(raw, str):
        return CommandPlan.from_json(raw).validate()
    return CommandPlan.from_dict(raw).validate()


# Short aliases for adapters that use the generic command terminology.
parse_command = parse_command_plan


def validate_command_slots(raw: Any) -> CommandSlots:
    """Validate a slots mapping or return an already validated instance."""

    return CommandSlots.from_dict(raw)


def validate_command_plan(raw: Any) -> CommandPlan:
    """Validate a command and ensure its flow exists in the registry.

    Importing the registry lazily avoids an import cycle while retaining a
    single runtime check against the actual registry rather than merely the
    duplicated type-level set.
    """

    plan = raw if isinstance(raw, CommandPlan) else CommandPlan.from_dict(raw)
    try:
        from flow_registry import FLOW_REGISTRY
    except ImportError:  # pragma: no cover - only relevant outside the repo
        registry_names = FLOW_NAMES
    else:
        registry_names = frozenset(FLOW_REGISTRY)
    if plan.flow not in registry_names:
        _fail(f"flow {plan.flow!r} is not registered", "flow")
    return plan


def is_valid_command_plan(raw: Any) -> bool:
    """Return whether ``raw`` satisfies the complete command contract."""

    try:
        validate_command_plan(raw)
    except (CommandValidationError, TypeError, ValueError):
        return False
    return True


def try_parse_command_plan(raw: Any) -> CommandPlan | None:
    """Parse a command, returning ``None`` instead of raising on bad output."""

    try:
        return parse_command_plan(raw)
    except (CommandValidationError, TypeError, ValueError):
        return None


COMMAND_SLOT_FIELDS = frozenset(CommandSlots.__dataclass_fields__)
COMMAND_PLAN_FIELDS = frozenset(CommandPlan.__dataclass_fields__)
BOOLEAN_SLOT_FIELDS = frozenset(_BOOLEAN_SLOT_FIELDS)


__all__ = [
    "AGE_GROUP_VALUES",
    "AGE_INTENT_VALUES",
    "ALLOWED_AGE_GROUPS",
    "ALLOWED_AGE_INTENTS",
    "ALLOWED_AUDIENCES",
    "ALLOWED_DETAIL_FIELDS",
    "ALLOWED_EXPERIENCE_CONCEPTS",
    "ALLOWED_GENRES",
    "ALLOWED_REFERENCE_KINDS",
    "ALLOWED_TIME_SLOTS",
    "ALLOWED_VENUES",
    "AUDIENCE_VALUES",
    "BOOLEAN_SLOT_FIELDS",
    "CANONICAL_MUNICIPALITIES",
    "CANONICAL_REGIONS",
    "COMMAND_PLAN_FIELDS",
    "COMMAND_SLOT_FIELDS",
    "CommandPlan",
    "CommandSlots",
    "CommandValidationError",
    "CONFIDENCE_VALUES",
    "EVENT_DETAIL_FIELDS",
    "EXPERIENCE_SLOT_FIELDS",
    "GENRE_VALUES",
    "FLOW_NAMES",
    "MAX_REFERENCE_INDEX",
    "MAX_VISIT_COUNT",
    "PARTICIPATION_DETAIL_FIELDS",
    "REFERENCE_KIND_VALUES",
    "TIME_SLOT_VALUES",
    "VALID_MUNICIPALITIES",
    "VALID_REGIONS",
    "ValidationError",
    "is_valid_command_plan",
    "parse_command",
    "parse_command_plan",
    "try_parse_command_plan",
    "validate_command_plan",
    "validate_command_slots",
]
