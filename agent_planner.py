"""Bounded Planner/Writer adapters for the Agentic Search layer.

The adapters treat Modal output as untrusted JSON.  A failed or malformed
model response becomes a deterministic local plan or writer fallback; it
never becomes executable Python or an event fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import re
from typing import Any, Mapping

import age_semantics
import experience_preferences
from command_generator import (
    CommandGenerationResult,
    CommandPlan,
    DEFAULT_COMMAND_FORMAT,
    generate_command,
)
import event_search
from agent_models import SearchPlan, SearchSpec, WriterOutput
from app_config import (
    CITY_ALIASES,
    GENRE_ALIASES,
    MAX_RESULT_SET_SIZE,
    POC_REFERENCE_DATE,
    REGION_CITIES,
)


MAX_PLAN_SEARCHES = 3
MAX_FILTER_ITEMS = 20
MAX_STRING_LENGTH = 240
_COUNT_HINTS = ("何件", "いくつ", "件数", "何個", "どれくらい", "どのくらい", "どの程度")
_VALID_VENUES = frozenset({"屋内", "室内", "屋外", "indoor", "outdoor"})
_VALID_TIME_SLOTS = frozenset({"午前", "午後", "夕方"})
# The matcher has a single child-friendly boolean, not a family audience
# taxonomy.  Do not accept labels that would otherwise be silently ignored.
_VALID_AGE_GROUPS = frozenset(
    {
        *age_semantics.AGE_GROUPS,
        # Keep legacy aliases accepted at the validation boundary for old
        # planner fixtures; fallback plans always emit canonical values.
        "child", "children", "小学生", "子ども", "こども",
    }
)
_VALID_AGE_INTENTS = frozenset({*age_semantics.AGE_INTENTS, "対象", "おすすめ", "推奨"})
_RELAXABLE_FILTERS = frozenset(
    {"soft_terms", "genres", "genre_groups", "child_friendly", "venue", "rain_preferred", "entry_free", "max_entry_fee"}
)
_WRITER_FACT_PATTERNS = re.compile(
    r"https?://|www\.|20\d{2}[-年/]\d{1,2}|[0-9０-９]{1,2}[月/]\s*[0-9０-９]{1,2}日?|"
    r"[0-9０-９]{1,2}月[0-9０-９]{1,2}日|(?:午前|午後)?\s*[0-9０-９]{1,2}時(?:半|[0-9０-９]{1,2}分)?|"
    r"\d{1,4}円|\d{1,2}:\d{2}|\d{1,4}件|"
    r"無料|有料|予約|申込|申し込み|屋内|屋外|室内|会場|場所|開催|日時|時間|料金|"
    r"駐車場|雨天|雨でも|車いす|公式|電話|住所|市|町|対象|入場"
)


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


def _call_modal_raw(config: ModalConfig | None, payload: Mapping[str, Any]) -> Any:
    """Return the untrusted Modal answer for command parse/repair handling."""

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
        return body.get("answer")
    except (requests.RequestException, ValueError, TypeError):
        return None


def _call_modal_json(config: ModalConfig | None, payload: Mapping[str, Any]) -> dict[str, Any] | None:
    return _extract_json(_call_modal_raw(config, payload))


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


def _validate_filter_semantics(spec: SearchSpec) -> bool:
    """Reject planner values that the deterministic matcher would ignore."""

    filters = spec.filters
    if spec.tool in {"search_events", "count_events"} and not filters and not spec.relaxed:
        return False
    if spec.tool == "get_event_detail":
        ids = filters.get("event_ids")
        if filters.get("event_id") is None and not (isinstance(ids, list) and ids):
            return False
        return True
    if spec.tool in {"recommend_next_events", "recommend_similar_events"}:
        return bool(filters.get("selected_event_id") or filters.get("event_id"))
    if spec.tool == "search_faq":
        return isinstance(filters.get("query"), str) and bool(filters["query"].strip())
    if spec.tool not in {"search_events", "count_events"}:
        return False

    allowed_search_filters = {
        "dates", "municipalities", "regions", "genres", "genre_groups", "age", "age_group",
        "age_intent", "child_friendly", "venue", "entry_free", "paid_only", "max_entry_fee",
        "reservation_required", "rain_preferred", "time_slots", "time_after", "soft_terms",
        "experience_required", "experience_preferred", "experience_excluded",
    }
    if set(filters) - allowed_search_filters:
        return False

    if "dates" in filters:
        dates = filters["dates"]
        if not isinstance(dates, list) or not dates:
            return False
        try:
            if not all(isinstance(value, str) for value in dates):
                return False
            [date.fromisoformat(value) for value in dates]
        except ValueError:
            return False
    if "municipalities" in filters:
        municipalities = filters["municipalities"]
        valid_cities = set(CITY_ALIASES) | set(CITY_ALIASES.values())
        if not isinstance(municipalities, list) or not municipalities or not all(value in valid_cities for value in municipalities):
            return False
    if "regions" in filters:
        regions = filters["regions"]
        if not isinstance(regions, list) or not regions or not all(value in REGION_CITIES for value in regions):
            return False
    if "genres" in filters:
        genres = filters["genres"]
        if not isinstance(genres, list) or not genres or not all(value in GENRE_ALIASES for value in genres):
            return False
    if "genre_groups" in filters:
        groups = filters["genre_groups"]
        if not isinstance(groups, list) or not groups or not all(
            isinstance(group, list) and group and all(value in GENRE_ALIASES for value in group)
            for group in groups
        ):
            return False
    if "age" in filters:
        age = filters["age"]
        if isinstance(age, bool) or not isinstance(age, int) or not 0 <= age <= 120:
            return False
    if "age_group" in filters and filters["age_group"] not in _VALID_AGE_GROUPS:
        return False
    if "age_intent" in filters and filters["age_intent"] not in _VALID_AGE_INTENTS:
        return False
    if "age_intent" in filters and not any(
        key in filters for key in ("age", "age_group", "child_friendly")
    ):
        # An intent label alone has no effect in the deterministic matcher.
        return False
    if "venue" in filters and filters["venue"] not in _VALID_VENUES:
        return False
    for key in ("child_friendly", "entry_free", "paid_only", "reservation_required", "rain_preferred"):
        if key in filters and not isinstance(filters[key], bool):
            return False
    # These predicates are only implemented for their positive form.  A
    # false-only planner value would be accepted but ignored by the matcher.
    for key in ("child_friendly", "entry_free", "paid_only", "rain_preferred"):
        if key in filters and filters[key] is not True:
            return False
    if "max_entry_fee" in filters:
        fee = filters["max_entry_fee"]
        if isinstance(fee, bool) or not isinstance(fee, int) or fee < 0:
            return False
    if "time_slots" in filters:
        slots = filters["time_slots"]
        if not isinstance(slots, list) or not slots or not all(slot in _VALID_TIME_SLOTS for slot in slots):
            return False
    if "time_after" in filters:
        after = filters["time_after"]
        if isinstance(after, bool) or not isinstance(after, int) or not 0 <= after <= 24 * 60:
            return False
    if "soft_terms" in filters:
        terms = filters["soft_terms"]
        if not isinstance(terms, list) or not terms or not all(isinstance(term, str) and term.strip() for term in terms):
            return False
    for field_name in (
        "experience_required",
        "experience_preferred",
        "experience_excluded",
    ):
        if field_name in filters:
            try:
                experience_preferences.normalize_concept_ids(
                    filters[field_name],
                    field_name=field_name,
                )
            except experience_preferences.ExperienceVocabularyError:
                return False
    try:
        experience_preferences.ExperienceQuery(
            required=filters.get("experience_required", ()),
            preferred=filters.get("experience_preferred", ()),
            excluded=filters.get("experience_excluded", ()),
        )
    except experience_preferences.ExperienceVocabularyError:
        return False
    return True


def validate_search_plan(raw: Any) -> SearchPlan | None:
    """Validate planner JSON and enforce the bounded search contract."""

    try:
        plan = SearchPlan.from_dict(raw)
    except (TypeError, ValueError):
        return None
    if plan.intent not in {"discover", "count"} or plan.confidence not in {"high", "medium", "low"}:
        return None
    if len(plan.searches) > MAX_PLAN_SEARCHES:
        return None
    for spec in plan.searches:
        if not all(_validate_filter_value(value) for value in spec.filters.values()):
            return None
        if not _validate_filter_semantics(spec):
            return None
        if spec.relaxed and (
            not spec.relaxed_fields
            or spec.purpose != "relaxed"
            or not set(spec.relaxed_fields).issubset(_RELAXABLE_FILTERS)
        ):
            return None
        if not spec.relaxed and spec.relaxed_fields:
            return None
    return plan


def validate_replan_plan(raw: Any, previous_plan: SearchPlan) -> SearchPlan | None:
    """Accept only a bounded, explicitly weaker plan than the prior plan."""

    plan = validate_search_plan(raw.to_dict()) if isinstance(raw, SearchPlan) else validate_search_plan(raw)
    if plan is None or not plan.searches:
        return None
    previous_specs = list(previous_plan.searches)
    for spec in plan.searches:
        if not spec.relaxed or not spec.relaxed_fields:
            return None
        candidates = [previous for previous in previous_specs if previous.tool == spec.tool]
        if not candidates:
            return None
        previous = candidates[0]
        changed = {
            key
            for key in set(previous.filters) | set(spec.filters)
            if previous.filters.get(key) != spec.filters.get(key)
        }
        if not changed or changed != set(spec.relaxed_fields):
            return None
        if any(key not in previous.filters for key in spec.filters):
            return None
        if not all(_is_relaxation(key, previous.filters.get(key), spec.filters.get(key)) for key in changed):
            return None
    return plan


def _is_relaxation(key: str, previous: Any, current: Any) -> bool:
    """Return whether a changed filter is strictly weaker for the matcher."""

    if key == "soft_terms":
        if current is None:
            return isinstance(previous, list) and bool(previous)
        if not isinstance(previous, list) or not isinstance(current, list):
            return False
        return len(current) < len(previous) and set(current).issubset(set(previous))

    if key in {"child_friendly", "venue", "rain_preferred", "entry_free"}:
        # The only supported relaxation is removing a positive predicate.
        return current is None and previous in {True, "屋内", "室内", "indoor", "屋外", "outdoor"}

    if key == "max_entry_fee":
        if current is None:
            return isinstance(previous, int) and not isinstance(previous, bool)
        if isinstance(previous, bool) or isinstance(current, bool):
            return False
        return isinstance(previous, int) and isinstance(current, int) and current > previous

    if key == "genres":
        if current is None:
            return isinstance(previous, list) and bool(previous)
        if not isinstance(previous, list) or not isinstance(current, list):
            return False
        return len(current) < len(previous) and set(current).issubset(set(previous))

    if key == "genre_groups":
        if current is None:
            return isinstance(previous, list) and bool(previous)
        if not isinstance(previous, list) or not isinstance(current, list) or len(current) > len(previous):
            return False
        used: set[int] = set()
        strictly_weaker = len(current) < len(previous)
        for new_group in current:
            if not isinstance(new_group, list):
                return False
            match_index = next(
                (
                    index
                    for index, old_group in enumerate(previous)
                    if index not in used
                    and isinstance(old_group, list)
                    and set(old_group).issubset(set(new_group))
                ),
                None,
            )
            if match_index is None:
                return False
            used.add(match_index)
            if set(new_group) != set(previous[match_index]):
                strictly_weaker = True
        return strictly_weaker

    return False


def _clean_soft_terms(values: list[str], query: str) -> list[str]:
    generic = {
        "イベント", "ある", "ありますか", "探して", "探す", "楽しめる", "楽しみたい",
        "行ける", "行きたい", "おすすめ", "どれくらい", "どのくらい", "どの程度",
        "何件", "いくつ", "いくつくらい", "何件くらい", "何個くらい", "件数", "何個", "好き", "興味", "がいい", "が好き",
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
    if parsed.invalid_date:
        # The matcher treats malformed date filters as a safe zero-match
        # condition.  Never drop an invalid date and accidentally broaden to
        # the whole 30-event dataset.
        filters["dates"] = ["invalid-date"]
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
    if parsed.reservation_required is not None:
        filters["reservation_required"] = parsed.reservation_required
    if parsed.time_slots:
        filters["time_slots"] = list(parsed.time_slots)
    if parsed.time_after is not None:
        filters["time_after"] = parsed.time_after
    if parsed.experience_required:
        filters["experience_required"] = list(parsed.experience_required)
    if parsed.experience_preferred:
        filters["experience_preferred"] = list(parsed.experience_preferred)
    if parsed.experience_excluded:
        filters["experience_excluded"] = list(parsed.experience_excluded)

    age_query = age_semantics.query_age_semantics(query)
    if age_query.age is not None:
        filters["age"] = age_query.age
    if age_query.age_group is not None:
        filters["age_group"] = age_query.age_group
    if age_query.age_intent is not None:
        filters["age_intent"] = age_query.age_intent
    if age_query.recognized and age_semantics.child_friendly_for_request(
        age=age_query.age,
        age_group=age_query.age_group,
    ) is True:
        filters["child_friendly"] = True

    # These expressions are intentionally semantic constraints.  Keep the
    # exact vocabulary in the fallback so a failed Modal call cannot turn
    # them into soft keywords or silently discard them.
    normalized_query = event_search.normalize_query(query)
    if parsed.venue is None and any(term in normalized_query for term in ("建物の中", "建物内", "中でやる")):
        filters["venue"] = "屋内"

    soft_terms = _clean_soft_terms(list(parsed.soft_terms), query)
    if soft_terms:
        filters["soft_terms"] = soft_terms

    # A release-only utterance is not a request to list the whole catalogue.
    # Keep the fallback executable but make it a deterministic zero-match,
    # out-of-catalog date sentinel; the Agentic orchestrator routes this case
    # to the normal ``needs_condition`` response before planning.
    if not filters and experience_preferences.has_release_phrase(query):
        filters["dates"] = ["1900-01-01"]

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
        allow_replan=bool(filters.get("soft_terms")) and not any(
            filters.get(key)
            for key in (
                "experience_required",
                "experience_excluded",
            )
        ),
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
    query = str(context.get("query", ""))[:1200]
    deterministic_experience = experience_preferences.resolve_experience_query(query)

    def reconcile(plan: SearchPlan) -> SearchPlan:
        """Apply explicit deterministic experience intent to every search."""

        release = experience_preferences.has_release_phrase(query)
        if not deterministic_experience.recognized and not release:
            return plan
        searches: list[SearchSpec] = []
        for spec in plan.searches:
            if spec.tool not in {"search_events", "count_events"}:
                return fallback_search_plan(query, POC_REFERENCE_DATE)
            filters = dict(spec.filters)
            if release:
                for key in (
                    "experience_required",
                    "experience_preferred",
                    "experience_excluded",
                ):
                    filters.pop(key, None)
            else:
                filters.update(
                    {
                        "experience_required": list(deterministic_experience.required),
                        "experience_preferred": list(deterministic_experience.preferred),
                        "experience_excluded": list(deterministic_experience.excluded),
                    }
                )
            terms = filters.get("soft_terms")
            if isinstance(terms, list):
                cleaned = [
                    value
                    for value in (
                        experience_preferences.remove_experience_phrases(term)
                        for term in terms
                    )
                    if value.strip()
                ]
                if cleaned:
                    filters["soft_terms"] = cleaned
                else:
                    filters.pop("soft_terms", None)
            searches.append(
                SearchSpec(
                    search_id=spec.search_id,
                    tool=spec.tool,
                    purpose=spec.purpose,
                    filters=filters,
                    relaxed=spec.relaxed,
                    relaxed_fields=spec.relaxed_fields,
                )
            )
        return SearchPlan(
            intent=plan.intent,
            answer_type=plan.answer_type,
            searches=tuple(searches),
            confidence=plan.confidence,
            allow_replan=plan.allow_replan and not any(
                search.filters.get(key)
                for search in searches
                for key in ("experience_required", "experience_excluded")
            ),
        )

    payload = {
        "mode": "planner",
        "query": query,
        "state": {
            "reference_date": str(context.get("reference_date", POC_REFERENCE_DATE.isoformat())),
            "selected_event_id": context.get("selected_event_id"),
            "last_result_ids": list(context.get("last_result_ids", []))[:MAX_RESULT_SET_SIZE],
            "last_filters": dict(context.get("last_filters") or {}),
        },
    }
    for _ in range(2):
        plan = validate_search_plan(_call_modal_json(modal_config, payload))
        if plan is not None:
            try:
                reconciled = reconcile(plan)
                # Re-validate after deterministic reconciliation.  Removing
                # model-supplied experience text from soft_terms can make a
                # previously valid-looking plan empty or otherwise invalid;
                # never execute that untrusted intermediate shape.
                validated = validate_search_plan(reconciled.to_dict())
                if validated is not None:
                    return validated
            except (TypeError, ValueError, experience_preferences.ExperienceVocabularyError):
                break
    return fallback_search_plan(query, POC_REFERENCE_DATE)


def request_command_result(
    context: Mapping[str, Any],
    modal_config: ModalConfig | None = None,
    *,
    output_format: str = DEFAULT_COMMAND_FORMAT,
) -> CommandGenerationResult:
    """Generate a validated semantic command without touching SearchPlan.

    The existing planner path remains the compatibility path for the current
    Agentic Search orchestrator.  This API is the isolated command-mode
    boundary that a later flow executor can adopt.
    """

    query = str(context.get("query", ""))[:1200]
    state = {
        "reference_date": context.get("reference_date", POC_REFERENCE_DATE.isoformat()),
        "selected_event_id": context.get("selected_event_id"),
        "last_result_ids": list(context.get("last_result_ids", []))[:MAX_RESULT_SET_SIZE],
        "last_command": context.get("last_command"),
        "active_flow": context.get("active_flow"),
        "pending_slots": context.get("pending_slots"),
    }
    return generate_command(
        query,
        state,
        call=lambda payload: _call_modal_raw(modal_config, payload),
        output_format=output_format,
    )


def request_command_plan(
    context: Mapping[str, Any],
    modal_config: ModalConfig | None = None,
    *,
    output_format: str = DEFAULT_COMMAND_FORMAT,
) -> CommandPlan:
    """Return only the typed Flow+Slots command for a caller."""

    return request_command_result(context, modal_config, output_format=output_format).plan


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
    plan = validate_replan_plan(_call_modal_json(modal_config, payload), previous_plan)
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
    if _WRITER_FACT_PATTERNS.search(combined):
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
