"""Bounded Planner/Writer adapters for the Agentic Search layer.

The adapters treat Modal output as untrusted JSON.  A failed or malformed
model response becomes a deterministic local plan or writer fallback; it
never becomes executable Python or an event fact.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping

import event_search
from agent_models import SearchPlan, SearchSpec, WriterOutput
from app_config import POC_REFERENCE_DATE


MAX_PLAN_SEARCHES = 3
MAX_FILTER_ITEMS = 20
MAX_STRING_LENGTH = 240
_COUNT_HINTS = ("何件", "いくつ", "件数", "何個", "どれくらい", "どのくらい", "どの程度")


@dataclass(frozen=True)
class ModalConfig:
    url: str
    key: str
    secret: str


class PlannerError(RuntimeError):
    """Raised internally when a planner response cannot be used."""


def _extract_json(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    return dict(parsed) if isinstance(parsed, dict) else None


def _call_modal_json(config: ModalConfig | None, payload: Mapping[str, Any]) -> dict[str, Any] | None:
    if config is None or not all(isinstance(value, str) and value.strip() for value in (config.url, config.key, config.secret)):
        return None
    try:
        import requests
    except ImportError:
        return None
    try:
        response = requests.post(
            config.url,
            headers={
                "Modal-Key": config.key,
                "Modal-Secret": config.secret,
                "Content-Type": "application/json",
            },
            json=dict(payload),
            timeout=300,
            allow_redirects=False,
        )
        if response.status_code < 200 or response.status_code >= 300:
            return None
        body = response.json()
        if not isinstance(body, dict) or body.get("service_id") != "ehime-kokubunsai-ai-poc":
            return None
        return _extract_json(body.get("answer"))
    except (requests.RequestException, ValueError, TypeError):
        return None


def _validate_filter_value(value: Any, depth: int = 0) -> bool:
    if depth > 2:
        return False
    if value is None or isinstance(value, (bool, int, float)):
        return True
    if isinstance(value, str):
        return len(value) <= MAX_STRING_LENGTH and "\x00" not in value
    if isinstance(value, list):
        return len(value) <= MAX_FILTER_ITEMS and all(_validate_filter_value(item, depth + 1) for item in value)
    return False


def validate_search_plan(raw: Any) -> SearchPlan | None:
    """Validate planner JSON and enforce the bounded search contract."""

    try:
        plan = SearchPlan.from_dict(raw)
    except (TypeError, ValueError):
        return None
    if len(plan.searches) > MAX_PLAN_SEARCHES:
        return None
    for spec in plan.searches:
        if not all(_validate_filter_value(value) for value in spec.filters.values()):
            return None
        if spec.relaxed and not spec.relaxed_fields:
            return None
    return plan


def _clean_soft_terms(values: list[str], query: str) -> list[str]:
    generic = {
        "イベント", "ある", "ありますか", "探して", "探す", "楽しめる", "楽しみたい",
        "行ける", "行きたい", "おすすめ", "どれくらい", "どのくらい", "どの程度",
        "何件", "いくつ", "件数", "何個", "好き", "興味", "がいい", "が好き",
    }
    terms: list[str] = []
    for value in values:
        cleaned = re.sub(r"(?:が)?好き|(?:に)?興味(?:がある)?", "", value).strip()
        cleaned = re.sub(r"イベント|楽しめる|楽しみたい|行ける|行きたい", "", cleaned).strip()
        if cleaned and cleaned not in generic and len(cleaned) >= 2 and cleaned not in terms:
            terms.append(cleaned)
    # Age-only discovery has no soft term; do not pass the full user sentence
    # to the deterministic matcher as a pseudo-keyword.
    return terms[:MAX_FILTER_ITEMS]


def fallback_search_plan(query: str, reference_date=POC_REFERENCE_DATE) -> SearchPlan:
    """Create a safe local plan when Modal is unavailable or returns bad JSON."""

    parsed = event_search.parse_query(query, reference_date)
    filters: dict[str, Any] = {}
    if parsed.dates:
        filters["dates"] = list(parsed.dates)
    municipalities = [city for group in parsed.city_groups for city in group]
    regions = [region for group in parsed.region_groups for region in group]
    if municipalities:
        filters["municipalities"] = municipalities
    if regions:
        filters["regions"] = regions
    if parsed.genres:
        filters["genres"] = list(parsed.genres)
        filters["genre_groups"] = [list(group) for group in parsed.genre_groups]
    if parsed.venue:
        filters["venue"] = parsed.venue
    if parsed.child_friendly:
        filters["child_friendly"] = True
    if parsed.rain_preferred:
        filters["rain_preferred"] = True
    if parsed.entry_free:
        filters["entry_free"] = True
    if parsed.paid_only:
        filters["paid_only"] = True
    if parsed.max_entry_fee is not None:
        filters["max_entry_fee"] = parsed.max_entry_fee
    if parsed.time_slots:
        filters["time_slots"] = list(parsed.time_slots)
    if parsed.time_after is not None:
        filters["time_after"] = parsed.time_after

    age_match = re.search(r"(\d{1,2})歳", event_search.normalize_query(query))
    if age_match:
        filters["age"] = int(age_match.group(1))
        filters["age_intent"] = "recommended"
        filters["child_friendly"] = True

    soft_terms = _clean_soft_terms(list(parsed.soft_terms), query)
    if soft_terms:
        filters["soft_terms"] = soft_terms

    is_count = any(hint in query for hint in _COUNT_HINTS)
    tool = "count_events" if is_count else "search_events"
    return SearchPlan(
        intent="count" if is_count else "discover",
        answer_type="count" if is_count else "list",
        searches=(
            SearchSpec(
                search_id="s1",
                tool=tool,
                purpose="exact",
                filters=filters,
            ),
        ),
        confidence="medium",
        allow_replan=bool(filters.get("soft_terms")),
    )


def _fallback_replan(previous_plan: SearchPlan) -> SearchPlan | None:
    if not previous_plan.allow_replan:
        return None
    searches: list[SearchSpec] = []
    for previous in previous_plan.searches:
        filters = dict(previous.filters)
        removed = []
        for key in ("soft_terms",):
            if key in filters:
                filters.pop(key, None)
                removed.append(key)
        if not removed:
            continue
        searches.append(
            SearchSpec(
                search_id=f"{previous.search_id}-relaxed",
                tool=previous.tool,
                purpose="relaxed",
                filters=filters,
                relaxed=True,
                relaxed_fields=tuple(removed),
            )
        )
    if not searches:
        return None
    return SearchPlan(
        intent=previous_plan.intent,
        answer_type=previous_plan.answer_type,
        searches=tuple(searches),
        confidence="medium",
        allow_replan=False,
    )


def request_search_plan(
    context: Mapping[str, Any],
    modal_config: ModalConfig | None = None,
) -> SearchPlan:
    payload = {
        "mode": "planner",
        "query": str(context.get("query", ""))[:1200],
        "state": {
            "reference_date": str(context.get("reference_date", POC_REFERENCE_DATE.isoformat())),
            "selected_event_id": context.get("selected_event_id"),
            "last_result_ids": list(context.get("last_result_ids", []))[:20],
            "last_filters": dict(context.get("last_filters") or {}),
        },
    }
    for _ in range(2):
        plan = validate_search_plan(_call_modal_json(modal_config, payload))
        if plan is not None:
            return plan
    return fallback_search_plan(str(context.get("query", "")), POC_REFERENCE_DATE)


def request_replan(
    query: str,
    previous_plan: SearchPlan,
    result_summary: Mapping[str, Any],
    modal_config: ModalConfig | None = None,
) -> SearchPlan | None:
    payload = {
        "mode": "planner",
        "query": query[:1200],
        "state": {"previous_plan": previous_plan.to_dict(), "result_summary": dict(result_summary)},
        "replan": True,
    }
    plan = validate_search_plan(_call_modal_json(modal_config, payload))
    return plan if plan is not None else _fallback_replan(previous_plan)


def validate_writer_output(raw: Any, allowed_event_ids: set[str]) -> WriterOutput | None:
    if not isinstance(raw, Mapping):
        return None
    lead = raw.get("lead")
    follow_up = raw.get("follow_up")
    ids = raw.get("recommended_event_ids", [])
    reasons = raw.get("reasons", [])
    if not isinstance(lead, str) or not lead.strip() or len(lead) > 600:
        return None
    if follow_up is not None and (not isinstance(follow_up, str) or len(follow_up) > 300):
        return None
    if not isinstance(ids, list) or not all(isinstance(item, str) and item in allowed_event_ids for item in ids):
        return None
    if not isinstance(reasons, list) or len(reasons) > 8:
        return None
    parsed_reasons: list[dict[str, str]] = []
    for item in reasons:
        if not isinstance(item, Mapping) or item.get("event_id") not in allowed_event_ids or not isinstance(item.get("reason"), str):
            return None
        if len(str(item["reason"])) > 300:
            return None
        parsed_reasons.append({"event_id": str(item["event_id"]), "reason": str(item["reason"])})
    # Writer output is language only.  Dates, fees and URLs are rendered from
    # JSON cards by Streamlit, so reject fact-like leakage at this boundary.
    combined = " ".join([lead, follow_up or "", *(item["reason"] for item in parsed_reasons)])
    if re.search(r"https?://|www\.|20\d{2}[-年/]\d{1,2}|\d{1,4}円|\d{1,2}:\d{2}|\d{1,4}件", combined):
        return None
    return WriterOutput(
        lead=lead.strip(),
        recommended_event_ids=tuple(ids),
        reasons=tuple(parsed_reasons),
        follow_up=follow_up.strip() if isinstance(follow_up, str) and follow_up.strip() else None,
    )


def request_writer(
    writer_input: Mapping[str, Any],
    modal_config: ModalConfig | None = None,
) -> WriterOutput | None:
    allowed_ids = {str(item) for item in writer_input.get("candidate_ids", [])}
    payload = {"mode": "writer", "writer_input": dict(writer_input)}
    return validate_writer_output(_call_modal_json(modal_config, payload), allowed_ids)
