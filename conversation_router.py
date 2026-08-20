"""Pure conversation-routing decisions for the Streamlit PoC.

The router only decides which deterministic branch should handle a turn.  It
does not call Modal, mutate Streamlit state, or generate event facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Sequence

import event_details
import event_recommendation
import event_search
import faq_search


@dataclass(frozen=True)
class ConversationRoute:
    action_type: str
    detail_field: str | None = None
    reference_index: int | None = None
    selected_event: Mapping[str, Any] | None = None
    faq_match: faq_search.FAQMatch | None = None
    search_required: bool = False
    recommendation_mode: str | None = None


def contains_named_event_context(query: str) -> bool:
    normalized = event_search.normalize_query(query)
    return any(
        token in normalized
        for event in event_search.load_events()
        for token in (
            str(event["イベント名"]).replace("【PoC架空】", ""),
            *[
                str(alias)
                for alias in event.get("aliases", [])
                if len(str(alias)) >= 2
            ],
        )
        if len(token) >= 2
    )


def named_event_match(query: str) -> Mapping[str, Any] | None:
    """Return a unique event explicitly named in the current query."""

    normalized = event_search.normalize_query(query)
    matches: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for event in event_search.load_events():
        event_id = str(event.get("id") or event["公式URL"])
        tokens = (
            str(event["イベント名"]).replace("【PoC架空】", ""),
            *[
                str(alias)
                for alias in event.get("aliases", [])
                if len(str(alias)) >= 2
            ],
        )
        if any(len(token) >= 2 and token in normalized for token in tokens):
            if event_id not in seen:
                matches.append(event)
                seen.add(event_id)
    return matches[0] if len(matches) == 1 else None


def is_pronoun_reference(query: str) -> bool:
    normalized = event_search.normalize_query(query)
    return "それ" in normalized or "そのイベント" in normalized


def is_detail_followup_without_new_search(
    query: str,
    detail_field: str | None,
) -> bool:
    if not detail_field or contains_named_event_context(query):
        return False
    normalized = event_search.normalize_query(query)
    explicit_search_terms = (
        "イベント",
        "探し",
        "今日",
        "本日",
        "明日",
        "今週",
        "来週",
        "月",
        "市",
        "町",
        "東予",
        "中予",
        "南予",
        "似た",
        "同じ系統",
    )
    return not any(term in normalized for term in explicit_search_terms)


def is_general_faq_context(
    query: str,
    faq_match: faq_search.FAQMatch | None,
    reference_date: date,
) -> bool:
    if faq_match is None:
        return False
    normalized = event_search.normalize_query(query)
    named_event = contains_named_event_context(query)
    filters = event_search.parse_query(normalized, reference_date)
    has_event_context = bool(
        named_event or filters.dates or filters.city_groups or filters.region_groups
    )

    if faq_match.faq_id in {"faq-001", "faq-003", "faq-008"} and not has_event_context:
        return True
    if faq_match.faq_id == "faq-002" and not has_event_context and "イベント" not in normalized:
        return True
    if event_details.detect_detail_field(query) and not named_event and not any(
        term in normalized
        for term in (
            "イベント",
            "探し",
            "おすすめ",
            "楽しめる",
            "楽しみたい",
            "見たい",
            "行きたい",
        )
    ):
        return True
    if has_event_context or any(
        term in normalized
        for term in (
            "イベント",
            "探し",
            "今日の",
            "明日の",
            "月",
            "市",
            "町",
            "東予",
            "中予",
            "南予",
        )
    ):
        return False
    return not filters.soft_terms and not filters.dates and not filters.city_groups and not filters.region_groups


def _resolve_recommendation_seed(
    query: str,
    last_results: Sequence[Mapping[str, Any]],
    selected_event: Mapping[str, Any] | None,
    named_event: Mapping[str, Any] | None,
    reference_index: int | None,
) -> Mapping[str, Any] | None:
    """Resolve a recommendation seed after the recommendation intent wins."""

    if named_event is not None:
        return named_event
    if reference_index is not None and 0 <= reference_index < len(last_results):
        return last_results[reference_index]
    if is_pronoun_reference(query):
        return selected_event or (last_results[0] if len(last_results) == 1 else None)
    return selected_event or (last_results[0] if len(last_results) == 1 else None)


def route_conversation(
    query: str,
    last_results: Sequence[Mapping[str, Any]],
    selected_event: Mapping[str, Any] | None,
    last_filters: Mapping[str, Any] | None,
    reference_date: date,
) -> ConversationRoute:
    """Return the deterministic branch that should handle one user turn."""

    detail_field = event_details.detect_detail_field(query)
    faq_match = faq_search.find_faq(query)
    reference_index = event_search.resolve_reference_index(query, len(last_results))
    named_event = named_event_match(query)

    if event_recommendation.is_next_query(query):
        target = _resolve_recommendation_seed(
            query, last_results, selected_event, named_event, reference_index
        )
        if target is not None:
            return ConversationRoute(
                "recommend_next",
                reference_index=reference_index,
                selected_event=target,
                faq_match=faq_match,
                recommendation_mode="next",
            )
        if is_general_faq_context(query, faq_match, reference_date):
            return ConversationRoute("general_faq", faq_match=faq_match)
        return ConversationRoute(
            "recommend_next_without_selection",
            reference_index=reference_index,
            faq_match=faq_match,
            recommendation_mode="next",
        )

    if event_recommendation.is_similar_query(query):
        target = _resolve_recommendation_seed(
            query, last_results, selected_event, named_event, reference_index
        )
        if target is not None:
            return ConversationRoute(
                "recommend_similar",
                reference_index=reference_index,
                selected_event=target,
                faq_match=faq_match,
                recommendation_mode="similar",
            )
        if is_general_faq_context(query, faq_match, reference_date):
            return ConversationRoute("general_faq", faq_match=faq_match)
        return ConversationRoute(
            "recommend_similar_without_selection",
            reference_index=reference_index,
            faq_match=faq_match,
            recommendation_mode="similar",
        )

    if last_results and (reference_index is not None or is_pronoun_reference(query)):
        target = None
        if reference_index is not None and 0 <= reference_index < len(last_results):
            target = last_results[reference_index]
        elif selected_event is not None:
            target = selected_event
        elif len(last_results) == 1:
            target = last_results[0]
        return ConversationRoute(
            "reference_followup",
            detail_field=detail_field,
            reference_index=reference_index,
            selected_event=target,
            faq_match=faq_match,
        )

    if detail_field and is_detail_followup_without_new_search(query, detail_field) and (
        selected_event is not None or len(last_results) == 1
    ):
        return ConversationRoute(
            "detail_followup",
            detail_field=detail_field,
            selected_event=selected_event or last_results[0],
            faq_match=faq_match,
        )

    if event_search.asks_for_nearby(query):
        if is_general_faq_context(query, faq_match, reference_date):
            return ConversationRoute("general_faq", faq_match=faq_match)
        return ConversationRoute("nearby", faq_match=faq_match)

    if event_search.classify_intent(query) in {"injection", "out_of_scope"}:
        return ConversationRoute("scope_search", faq_match=faq_match, search_required=True)

    if is_general_faq_context(query, faq_match, reference_date):
        return ConversationRoute("general_faq", faq_match=faq_match)

    if not event_search.looks_like_event_query(query):
        return ConversationRoute("generic_scope", faq_match=faq_match)

    return ConversationRoute("search", faq_match=faq_match, search_required=True)
