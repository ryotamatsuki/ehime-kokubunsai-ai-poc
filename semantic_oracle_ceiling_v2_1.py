"""Oracle ceiling for Semantic Operations v2.1 using only frozen v1 regression.

The sealed 200-case holdout is intentionally not imported or opened. The
oracle supplies perfect high-level semantics and canonical explicit-filter
patches so failures measure the reducer/router/executor ceiling rather than
model or lexical-normalization quality.
"""

from __future__ import annotations

from collections import Counter
import json
from typing import Any, Mapping

from semantic_frame_v2_1 import SET_SLOT_FIELDS, SparseReference, SparseSemanticFrame
from semantic_orchestrator_v2_1 import SemanticOperationsOrchestratorV21
from unexpected_utterances_eval import _forbidden_present, _seed_state, _slot_subset, load_dataset


FLOW_TO_INTENT = {
    "find_events": "search", "count_events": "count", "event_detail": "detail",
    "recommend_next": "next", "recommend_similar": "similar", "plan_event_pair": "pair",
    "explain_search": "explain_search", "explain_result": "explain_result",
    "general_faq": "faq", "unsupported": "unsupported",
}

FIELD_TO_UNSET = {
    "entry_free": "fee", "paid_only": "fee", "max_entry_fee": "fee",
    "venue": "venue", "rain_preferred": "rain", "reservation_required": "reservation",
    "audience": "age", "age": "age", "age_group": "age", "age_intent": "age",
    "municipalities": "location", "regions": "location", "dates": "date",
    "time_slots": "time", "time_after": "time", "genres": "topic", "topics": "topic",
    "experience_required": "experience_all", "experience_preferred": "experience_all",
    "experience_excluded": "experience_all",
}
GROUP_MEMBERS = {
    "fee": {"entry_free", "paid_only", "max_entry_fee"},
    "venue": {"venue"}, "rain": {"rain_preferred"},
    "reservation": {"reservation_required"},
    "age": {"audience", "age", "age_group", "age_intent"},
    "location": {"municipalities", "regions"}, "date": {"dates"},
    "time": {"time_slots", "time_after"}, "topic": {"genres", "topics"},
    "experience_all": {"experience_required", "experience_preferred", "experience_excluded"},
}

DATA_GAP_MARKERS = {
    "crowd": "crowding", "noise": "noise", "wheelchair": "wheelchair_access",
    "medical": "medical_safety", "dementia": "medical_safety", "pregnan": "medical_safety",
    "parking": "parking_distance", "toilet": "toilet_proximity", "weather": "weather_guarantee",
    "social": "social_fit", "popularity": "popularity", "fame": "fame",
    "duration": "duration_fit", "localness": "localness",
}


def _intent(case: Mapping[str, Any]) -> str:
    expected = case.get("expected", {})
    flows = list(expected.get("allowed_flows", [])) if isinstance(expected, Mapping) else []
    statuses = set(expected.get("allowed_statuses", [])) if isinstance(expected, Mapping) else set()
    if str(case.get("expected_behavior")) == "clarify_need" or (flows == ["unsupported"] and "clarification" in statuses):
        return "clarify"
    for flow in flows:
        if flow in FLOW_TO_INTENT and flow != "unsupported":
            return FLOW_TO_INTENT[flow]
    return "unsupported"


def _gap(case: Mapping[str, Any]) -> str | None:
    if str(case.get("category")) != "data_gap_boundary":
        return None
    risk = str(case.get("risk", "")).lower()
    for marker, value in DATA_GAP_MARKERS.items():
        if marker in risk:
            return value
    return "other"


def _oracle_unset(required: Mapping[str, Any], forbidden: list[Any]) -> tuple[str, ...]:
    groups: list[str] = []
    required_fields = {str(key) for key, value in required.items() if value not in (None, "", [], (), {})}
    for field_name in forbidden:
        group = FIELD_TO_UNSET.get(str(field_name))
        if not group or group in groups:
            continue
        # A positive replacement in the same semantic family already removes
        # the conflicting prior/parser value. Do not erase that positive value.
        if required_fields & GROUP_MEMBERS[group]:
            continue
        groups.append(group)
    return tuple(groups)


def oracle_sparse_frame(case: Mapping[str, Any]) -> SparseSemanticFrame:
    expected = case.get("expected", {})
    required = dict(expected.get("required_slots", {})) if isinstance(expected, Mapping) else {}
    forbidden = list(expected.get("forbidden_slots", [])) if isinstance(expected, Mapping) else []

    reference = None
    kind = required.get("reference_kind")
    index = required.get("reference_index")
    event_name = required.get("event_name")
    if kind is not None or index is not None or event_name is not None:
        if kind is None:
            kind = "ordinal" if index is not None else "event_name"
        reference = SparseReference(kind=str(kind), index=index, event_name=event_name)

    intent = _intent(case)
    clarification = None
    if intent == "clarify":
        clarification = "ambiguous_suitability" if str(case.get("category")) == "ambiguous_suitability" else "ambiguous_request"

    audience = required.get("audience")
    audience_mode = str(audience) if audience in {"family", "adult"} else None
    if audience_mode is None and any(required.get(name) is not None for name in ("age", "age_group", "age_intent")):
        audience_mode = "target"

    set_slots = {
        str(key): value
        for key, value in required.items()
        if str(key) in SET_SLOT_FIELDS and value not in (None, "", [], (), {})
    }

    return SparseSemanticFrame(
        intent=intent,
        scope="previous" if required.get("refine_previous") else "new",
        set_slots=set_slots,
        unset=_oracle_unset(required, forbidden),
        require=tuple(required.get("experience_required", [])),
        prefer=tuple(required.get("experience_preferred", [])),
        exclude=tuple(required.get("experience_excluded", [])),
        reference=reference,
        clarification=clarification,
        data_gap=_gap(case),
        audience_mode=audience_mode,
    )


def evaluate_oracle_v21(dataset: Mapping[str, Any] | None = None) -> dict[str, Any]:
    dataset = dict(dataset or load_dataset())
    rows: list[dict[str, Any]] = []
    for case in dataset.get("cases", []):
        frame = oracle_sparse_frame(case)
        calls = 0

        def invoke(_: Mapping[str, Any], *, _frame: SparseSemanticFrame = frame) -> Mapping[str, str]:
            nonlocal calls
            calls += 1
            return {"answer": json.dumps(_frame.to_dict(sparse=True), ensure_ascii=False, separators=(",", ":"))}

        result = SemanticOperationsOrchestratorV21(frame_call=invoke).handle_query(
            str(case.get("query", "")),
            _seed_state(str(case.get("context", "none"))),
        )
        expected = dict(case.get("expected", {}))
        failures: list[str] = []
        flows = list(expected.get("allowed_flows", []))
        if flows and result.flow not in flows:
            failures.append("flow")
        statuses = list(expected.get("allowed_statuses", []))
        if statuses and result.status not in statuses:
            failures.append("status")
        required = dict(expected.get("required_slots", {}))
        if required:
            _, slot_failures = _slot_subset(result.slots, required)
            failures.extend(slot_failures)
        for name in expected.get("forbidden_slots", []):
            if _forbidden_present(str(name), result.slots.get(str(name))):
                failures.append(f"forbidden:{name}")
        max_calls = expected.get("max_modal_calls")
        if max_calls is not None and calls > int(max_calls):
            failures.append("max_modal_calls")
        if expected.get("must_not_auto_relax") and result.near_events:
            failures.append("auto_relax")
        rows.append({
            "id": case.get("id"), "category": case.get("category"), "pass": not failures,
            "failures": failures, "flow": result.flow, "status": result.status,
            "slots": result.slots, "model_calls": calls, "route": result.deterministic_route,
        })

    categories = Counter(str(row["category"]) for row in rows)
    category_pass = Counter(str(row["category"]) for row in rows if row["pass"])
    failure_checks = Counter(item for row in rows for item in row["failures"])
    return {
        "architecture": "semantic-operations-v2.1-oracle-ceiling",
        "dataset": dataset.get("version"),
        "cases": len(rows),
        "machine_pass": sum(bool(row["pass"]) for row in rows),
        "machine_pass_rate": round(sum(bool(row["pass"]) for row in rows) / len(rows), 4) if rows else 0.0,
        "category": {name: {"cases": count, "pass": category_pass[name]} for name, count in sorted(categories.items())},
        "failure_checks": dict(failure_checks),
        "failed": [
            {
                "id": row["id"], "failures": row["failures"], "route": row["route"],
                "flow": row["flow"], "status": row["status"], "slots": row["slots"],
            }
            for row in rows if not row["pass"]
        ],
        "zero_model_call_cases": sum(row["model_calls"] == 0 for row in rows),
    }


if __name__ == "__main__":
    print(json.dumps(evaluate_oracle_v21(), ensure_ascii=False, indent=2, sort_keys=True))
