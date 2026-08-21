"""State transitions for search-result context and visible pagination.

The application keeps three independent concepts:

* the ordered search-result context used by ordinal/refinement references;
* the selected event used by pronoun/detail questions; and
* the number of cards currently visible in the UI.

This module owns only the first and third concepts.  A detail answer may
change the selected event while leaving this context untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from app_config import RESULT_PAGE_SIZE
from result_pagination import normalize_visible_count


RESULT_SET_REPLACING_FLOWS = frozenset(
    {
        "find_events",
        "count_events",
        "recommend_next",
        "recommend_similar",
        "plan_event_pair",
        "event_pair",
        "pair_recommendation",
        "agentic_search",
    }
)

RESULT_SET_PRESERVING_FLOWS = frozenset(
    {
        "event_detail",
        "reference_followup",
        "detail_followup",
        "general_faq",
        "generic_scope",
        "nearby",
        "recommend_next_without_selection",
        "recommend_similar_without_selection",
    }
)


@dataclass(frozen=True)
class ResultContextTransition:
    """The persisted search context after one conversation turn."""

    results: list[dict[str, Any]]
    result_ids: list[str]
    near_results: list[dict[str, Any]]
    near_result_ids: list[str]
    visible_count: int
    near_visible_count: int
    replace_result_set: bool


def _copy_records(events: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    if not events:
        return []
    return [dict(event) for event in events]


def _ordered_ids(
    values: Sequence[Any] | None,
    events: Sequence[Mapping[str, Any]] | None = None,
) -> list[str]:
    raw_values = list(values or [])
    if not raw_values and events:
        raw_values = [event.get("id") for event in events]
    ordered: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        normalized = str(value).strip()
        if normalized and normalized not in seen:
            ordered.append(normalized)
            seen.add(normalized)
    return ordered


def should_replace_result_set(*, flow: str | None, source: str | None) -> bool:
    """Classify whether the current turn creates a new result context.

    ``source`` distinguishes a Command ``event_detail`` from a legacy named
    event search that happens to answer a detail field.  The latter is a new
    search and is therefore allowed to replace the context.
    """

    normalized_flow = str(flow or "").strip()
    normalized_source = str(source or "").strip()
    if normalized_source == "preserving":
        return False
    if normalized_source in {"legacy_search", "agentic"}:
        return True
    if normalized_flow in RESULT_SET_PRESERVING_FLOWS:
        return False
    return normalized_flow in RESULT_SET_REPLACING_FLOWS


def classify_result_context_source(
    *,
    command_flow: str | None,
    search_result_present: bool,
    agentic_response_present: bool,
    command_handled: bool,
    pending_handled: bool,
    flow: str | None,
    previous_result_ids: Sequence[Any] | None,
    result_ids: Sequence[Any] | None,
) -> str:
    """Select the state-source label used by the Streamlit boundary.

    A Command recommendation may return a clarification that carries the
    recommendation flow name but leaves the previous cards untouched.  Only
    that unchanged response is preserving; a real recommendation result still
    replaces the context.
    """

    previous = _ordered_ids(previous_result_ids)
    current = _ordered_ids(result_ids)
    if command_flow is not None:
        if (
            current == previous
            and command_flow
            in RESULT_SET_PRESERVING_FLOWS | {"recommend_next", "recommend_similar"}
        ):
            return "preserving"
        return "command"
    if agentic_response_present:
        return "agentic"
    if search_result_present:
        return "legacy_search"
    if (command_handled or pending_handled) and current == previous:
        return "preserving"
    return "router"


def transition_result_context(
    *,
    previous_results: Sequence[Mapping[str, Any]] | None,
    previous_result_ids: Sequence[Any] | None,
    previous_near_results: Sequence[Mapping[str, Any]] | None,
    previous_near_result_ids: Sequence[Any] | None,
    previous_visible_count: int | None,
    previous_near_visible_count: int | None,
    flow: str | None,
    source: str | None,
    new_results: Sequence[Mapping[str, Any]] | None,
    new_result_ids: Sequence[Any] | None,
    new_near_results: Sequence[Mapping[str, Any]] | None,
    new_near_result_ids: Sequence[Any] | None,
    page_size: int = RESULT_PAGE_SIZE,
) -> ResultContextTransition:
    """Apply one explicit result-context transition.

    Search/refinement/recommendation flows replace the context.  Detail and
    FAQ flows preserve it while the caller updates ``selected_event``
    separately.  Visible counts follow the same rule: a replacement starts
    at the first page, while a preserving turn keeps the current page.
    """

    previous_ids = _ordered_ids(previous_result_ids, previous_results)
    previous_near_ids = _ordered_ids(previous_near_result_ids, previous_near_results)
    incoming_ids = _ordered_ids(new_result_ids, new_results)
    incoming_near_ids = _ordered_ids(new_near_result_ids, new_near_results)
    replace = should_replace_result_set(flow=flow, source=source)
    # A direct detail query with no prior search context still needs a
    # one-event context so its card and selected-event state remain usable.
    if not replace and not previous_ids and incoming_ids:
        replace = True

    if replace:
        results = _copy_records(new_results)
        result_ids = incoming_ids
        near_results = _copy_records(new_near_results)
        near_result_ids = incoming_near_ids
        visible_count = normalize_visible_count(
            len(result_ids), None, page_size=page_size
        )
        near_visible_count = normalize_visible_count(
            len(near_result_ids), None, page_size=page_size
        )
    else:
        results = _copy_records(previous_results)
        result_ids = previous_ids
        near_results = _copy_records(previous_near_results)
        near_result_ids = previous_near_ids
        visible_count = normalize_visible_count(
            len(result_ids), previous_visible_count, page_size=page_size
        )
        near_visible_count = normalize_visible_count(
            len(near_result_ids), previous_near_visible_count, page_size=page_size
        )

    return ResultContextTransition(
        results=results,
        result_ids=result_ids,
        near_results=near_results,
        near_result_ids=near_result_ids,
        visible_count=visible_count,
        near_visible_count=near_visible_count,
        replace_result_set=replace,
    )


__all__ = [
    "RESULT_SET_PRESERVING_FLOWS",
    "RESULT_SET_REPLACING_FLOWS",
    "ResultContextTransition",
    "classify_result_context_source",
    "should_replace_result_set",
    "transition_result_context",
]
