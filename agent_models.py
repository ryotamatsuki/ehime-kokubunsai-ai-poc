"""Typed contracts for the bounded Agentic Search layer.

The planner and writer exchange JSON, but the application never executes
planner-generated code.  These small dataclasses make the boundary explicit
and keep event facts in deterministic Python structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping


ALLOWED_TOOL_NAMES = frozenset(
    {
        "search_events",
        "count_events",
        "get_event_detail",
        "recommend_next_events",
        "recommend_similar_events",
        "search_faq",
    }
)

ALLOWED_FILTER_KEYS = frozenset(
    {
        "dates",
        "municipalities",
        "regions",
        "genres",
        "genre_groups",
        "age",
        "age_group",
        "age_intent",
        "child_friendly",
        "venue",
        "entry_free",
        "paid_only",
        "max_entry_fee",
        "reservation_required",
        "rain_preferred",
        "time_slots",
        "time_after",
        "soft_terms",
        "event_id",
        "event_ids",
        "reference_date",
        "selected_event_id",
        "selected_end_override",
        "query",
    }
)


def _string_list(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    return tuple(item.strip() for item in value if item.strip())


@dataclass(frozen=True)
class SearchSpec:
    search_id: str
    tool: str
    purpose: str
    filters: dict[str, Any] = field(default_factory=dict)
    relaxed: bool = False
    relaxed_fields: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: Any) -> "SearchSpec":
        if not isinstance(raw, Mapping):
            raise ValueError("search specification must be an object")
        search_id = raw.get("search_id")
        tool = raw.get("tool")
        purpose = raw.get("purpose", "exact")
        filters = raw.get("filters", {})
        if not isinstance(search_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", search_id):
            raise ValueError("invalid search_id")
        if not isinstance(tool, str) or tool not in ALLOWED_TOOL_NAMES:
            raise ValueError("unknown tool")
        if not isinstance(purpose, str) or not purpose.strip():
            raise ValueError("invalid search purpose")
        if not isinstance(filters, Mapping):
            raise ValueError("filters must be an object")
        unknown = set(filters) - ALLOWED_FILTER_KEYS
        if unknown:
            raise ValueError(f"unknown filter keys: {sorted(unknown)}")
        relaxed = raw.get("relaxed", False)
        if not isinstance(relaxed, bool):
            raise ValueError("relaxed must be boolean")
        relaxed_fields = _string_list(raw.get("relaxed_fields", []), "relaxed_fields")
        return cls(
            search_id=search_id,
            tool=tool,
            purpose=purpose,
            filters=dict(filters),
            relaxed=relaxed,
            relaxed_fields=relaxed_fields,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "search_id": self.search_id,
            "tool": self.tool,
            "purpose": self.purpose,
            "filters": dict(self.filters),
            "relaxed": self.relaxed,
            "relaxed_fields": list(self.relaxed_fields),
        }


@dataclass(frozen=True)
class SearchPlan:
    intent: str
    answer_type: str
    searches: tuple[SearchSpec, ...]
    confidence: str = "medium"
    allow_replan: bool = False

    @classmethod
    def from_dict(cls, raw: Any) -> "SearchPlan":
        if not isinstance(raw, Mapping):
            raise ValueError("search plan must be an object")
        intent = raw.get("intent", "discover")
        answer_type = raw.get("answer_type", "list")
        searches = raw.get("searches")
        confidence = raw.get("confidence", "medium")
        allow_replan = raw.get("allow_replan", False)
        if not isinstance(intent, str) or not intent.strip():
            raise ValueError("invalid plan intent")
        if answer_type not in {"list", "count", "detail"}:
            raise ValueError("invalid answer_type")
        if not isinstance(searches, list) or not searches:
            raise ValueError("a plan needs at least one search")
        if not isinstance(confidence, str):
            raise ValueError("invalid confidence")
        if not isinstance(allow_replan, bool):
            raise ValueError("allow_replan must be boolean")
        parsed = tuple(SearchSpec.from_dict(item) for item in searches)
        return cls(
            intent=intent,
            answer_type=answer_type,
            searches=parsed,
            confidence=confidence,
            allow_replan=allow_replan,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "answer_type": self.answer_type,
            "searches": [item.to_dict() for item in self.searches],
            "confidence": self.confidence,
            "allow_replan": self.allow_replan,
        }


@dataclass(frozen=True)
class ToolResult:
    search_id: str
    purpose: str
    total_matches: int
    events: list[dict[str, Any]] = field(default_factory=list)
    all_event_ids: list[str] = field(default_factory=list)
    relaxed: bool = False
    relaxed_fields: tuple[str, ...] = ()
    message: str = ""


@dataclass(frozen=True)
class MergedResults:
    exact_events: list[dict[str, Any]] = field(default_factory=list)
    relaxed_events: list[dict[str, Any]] = field(default_factory=list)
    display_candidates: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class WriterOutput:
    lead: str
    recommended_event_ids: tuple[str, ...] = ()
    reasons: tuple[dict[str, str], ...] = ()
    follow_up: str | None = None


@dataclass(frozen=True)
class AgenticResponse:
    answer_type: str
    total_matches: int
    exact_events: list[dict[str, Any]] = field(default_factory=list)
    relaxed_events: list[dict[str, Any]] = field(default_factory=list)
    lead: str = ""
    follow_up: str | None = None
    relaxed_fields: tuple[str, ...] = ()
    planner_used: bool = True
    planner_rounds: int = 0
    search_count: int = 0
    recommended_event_ids: tuple[str, ...] = ()
