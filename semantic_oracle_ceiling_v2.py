"""Oracle ceiling for the existing Semantic Operations v2 lower layers.

This is deliberately evaluated only against the already-exposed frozen v1
100-case regression set.  The sealed v2.1 holdout is not opened here.

The oracle supplies the semantic meaning that the model was expected to infer,
while deterministic parsing, state reduction and trusted execution remain real.
This separates model/frame errors from lower-layer architecture errors before
v2.1 is redesigned.
"""

from __future__ import annotations

from collections import Counter
import json
from typing import Any, Mapping

from semantic_frame_v2 import SemanticFrame, SemanticReference
from unexpected_utterances_eval import load_dataset
from unexpected_utterances_v2_eval import evaluate_case_v2


FLOW_TO_INTENT = {
    "find_events": "search",
    "count_events": "count",
    "event_detail": "detail",
    "recommend_next": "next",
    "recommend_similar": "similar",
    "plan_event_pair": "pair",
    "explain_search": "explain_search",
    "explain_result": "explain_result",
    "general_faq": "faq",
    "unsupported": "unsupported",
}

FORBIDDEN_TO_RELEASE = {
    "entry_free": "fee",
    "paid_only": "fee",
    "max_entry_fee": "fee",
    "venue": "venue",
    "rain_preferred": "rain",
    "reservation_required": "reservation",
    "experience_required": "experience",
    "experience_preferred": "experience",
    "experience_excluded": "experience",
    "audience": "age",
    "age": "age",
    "age_group": "age",
    "age_intent": "age",
    "municipalities": "location",
    "regions": "location",
    "dates": "date",
    "time_slots": "time",
    "time_after": "time",
    "genres": "topic",
    "topics": "topic",
}

DATA_GAP_BY_RISK = {
    "crowding": "crowding",
    "noise": "noise",
    "wheelchair": "wheelchair_access",
    "wheelchair_access": "wheelchair_access",
    "medical": "medical_safety",
    "medical_safety": "medical_safety",
    "parking": "parking_distance",
    "parking_distance": "parking_distance",
    "toilet": "toilet_proximity",
    "toilet_proximity": "toilet_proximity",
    "weather": "weather_guarantee",
    "weather_guarantee": "weather_guarantee",
    "social": "social_fit",
    "social_fit": "social_fit",
}


def _intent_for(case: Mapping[str, Any]) -> str:
    expected = case.get("expected", {})
    flows = list(expected.get("allowed_flows", [])) if isinstance(expected, Mapping) else []
    statuses = set(expected.get("allowed_statuses", [])) if isinstance(expected, Mapping) else set()
    behavior = str(case.get("expected_behavior", ""))
    if behavior == "clarify_need" or ("clarification" in statuses and flows == ["unsupported"]):
        return "clarify"
    for flow in flows:
        if flow in FLOW_TO_INTENT and flow != "unsupported":
            return FLOW_TO_INTENT[flow]
    return "unsupported"


def _reference_from_required(required: Mapping[str, Any]) -> SemanticReference | None:
    kind = required.get("reference_kind")
    index = required.get("reference_index")
    event_name = required.get("event_name")
    if kind is None and index is None and event_name is None:
        return None
    if kind is None and index is not None:
        kind = "ordinal"
    if kind is None and event_name is not None:
        kind = "event_name"
    return SemanticReference(kind=str(kind), index=index, event_name=event_name)


def _release_groups(case: Mapping[str, Any]) -> tuple[str, ...]:
    expected = case.get("expected", {})
    forbidden = list(expected.get("forbidden_slots", [])) if isinstance(expected, Mapping) else []
    groups: list[str] = []
    for field_name in forbidden:
        group = FORBIDDEN_TO_RELEASE.get(str(field_name))
        if group and group not in groups:
            groups.append(group)
    return tuple(groups)


def _data_gap(case: Mapping[str, Any]) -> str:
    if str(case.get("category")) != "data_gap_boundary":
        return "none"
    risk = str(case.get("risk", "")).lower()
    for marker, gap in DATA_GAP_BY_RISK.items():
        if marker in risk:
            return gap
    return "other"


def oracle_frame(case: Mapping[str, Any]) -> SemanticFrame:
    expected = case.get("expected", {})
    required = dict(expected.get("required_slots", {})) if isinstance(expected, Mapping) else {}
    intent = _intent_for(case)
    clarification_reason = "none"
    if intent == "clarify":
        clarification_reason = (
            "ambiguous_suitability"
            if str(case.get("category")) == "ambiguous_suitability"
            else "ambiguous_request"
        )
    return SemanticFrame(
        intent=intent,
        refine_previous=bool(required.get("refine_previous", False)),
        release=_release_groups(case),
        experience_required=tuple(required.get("experience_required", [])),
        experience_preferred=tuple(required.get("experience_preferred", [])),
        experience_excluded=tuple(required.get("experience_excluded", [])),
        reference=_reference_from_required(required),
        clarification_reason=clarification_reason,
        data_gap=_data_gap(case),
        confidence="high",
    )


def evaluate_oracle(dataset: Mapping[str, Any] | None = None) -> dict[str, Any]:
    dataset = dict(dataset or load_dataset())
    rows: list[dict[str, Any]] = []
    for case in dataset.get("cases", []):
        frame = oracle_frame(case)

        def invoke(_: Mapping[str, Any], *, _frame: SemanticFrame = frame) -> Mapping[str, str]:
            return {"answer": json.dumps(_frame.to_dict(), ensure_ascii=False, separators=(",", ":"))}

        rows.append(evaluate_case_v2(case, invoke, format_enforcer="baseline"))

    category = Counter()
    category_pass = Counter()
    failures = Counter()
    for row in rows:
        category[str(row.get("category"))] += 1
        category_pass[str(row.get("category"))] += int(bool(row.get("machine_pass")))
        for failure in row.get("failures", []):
            failures[str(failure)] += 1
    return {
        "architecture": "semantic-operations-v2-oracle-ceiling",
        "dataset": dataset.get("version"),
        "cases": len(rows),
        "machine_pass": sum(bool(row.get("machine_pass")) for row in rows),
        "machine_pass_rate": round(sum(bool(row.get("machine_pass")) for row in rows) / len(rows), 4) if rows else 0.0,
        "category": {
            name: {"cases": count, "pass": category_pass[name]}
            for name, count in sorted(category.items())
        },
        "failure_checks": dict(failures),
        "failed_ids": [str(row.get("id")) for row in rows if not row.get("machine_pass")],
    }


if __name__ == "__main__":
    print(json.dumps(evaluate_oracle(), ensure_ascii=False, indent=2, sort_keys=True))
