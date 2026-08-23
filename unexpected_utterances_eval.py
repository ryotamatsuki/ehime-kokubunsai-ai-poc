"""Evaluation helpers for Unexpected User Utterances v1.

This module is deliberately separate from production routing. It validates the
frozen exploratory fixture and scores a live run without changing application
behavior. The first baseline should be executed before any production tuning
against the fixture.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import conversation_recovery
import event_search
from app_config import MAX_RESULT_SET_SIZE, POC_REFERENCE_DATE
from command_models import CommandSlots, FLOW_NAMES
from command_orchestrator import CommandOrchestrator
from conversation_router import route_conversation
from live_semantic_command_eval import RemoteCommandCall, canonical_flow


EXPECTED_CASE_COUNT = 100
ALLOWED_CONTEXTS = {"none", "search"}
ALLOWED_STATUSES = {
    "ok", "clarification", "unsupported", "unavailable", "invalid_command",
    "execution_error", "reset",
}
DEFAULT_FIXTURE = Path(__file__).resolve().parent / "tests" / "data" / "unexpected_utterances_v1" / "manifest.json"


def load_dataset(path: str | Path = DEFAULT_FIXTURE) -> dict[str, Any]:
    manifest_path = Path(path)
    dataset = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = list(dataset.get("files", []))
    if files:
        cases: list[dict[str, Any]] = []
        for filename in files:
            part = json.loads((manifest_path.parent / str(filename)).read_text(encoding="utf-8"))
            cases.extend(list(part.get("cases", [])))
        dataset["cases"] = cases
    return dataset


def validate_dataset(dataset: Mapping[str, Any]) -> dict[str, Any]:
    if dataset.get("schema_version") != 1:
        raise ValueError("unexpected utterance dataset schema_version must be 1")
    cases = list(dataset.get("cases", []))
    if len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError(f"expected {EXPECTED_CASE_COUNT} cases, got {len(cases)}")

    ids = [str(case.get("id", "")) for case in cases]
    expected_ids = [f"UU-{index:03d}" for index in range(1, EXPECTED_CASE_COUNT + 1)]
    if ids != expected_ids:
        raise ValueError("case IDs must be sequential UU-001..UU-100")
    queries = [str(case.get("query", "")).strip() for case in cases]
    if any(not query for query in queries) or len(set(queries)) != len(queries):
        raise ValueError("queries must be non-empty and unique")

    target_counts = Counter(dataset.get("category_targets", {}))
    actual_counts = Counter(str(case.get("category", "")) for case in cases)
    # Counter(mapping) uses mapping values as counts, which is exactly the fixture shape.
    if actual_counts != target_counts:
        raise ValueError(f"category distribution drift: {dict(actual_counts)} != {dict(target_counts)}")

    slot_names = set(CommandSlots.__dataclass_fields__)
    manual_count = 0
    for case in cases:
        if case.get("context", "none") not in ALLOWED_CONTEXTS:
            raise ValueError(f"{case['id']}: unsupported context seed")
        expected = case.get("expected")
        if not isinstance(expected, Mapping):
            raise ValueError(f"{case['id']}: expected must be an object")
        flows = list(expected.get("allowed_flows", []))
        if any(flow not in FLOW_NAMES for flow in flows):
            raise ValueError(f"{case['id']}: unknown allowed flow")
        statuses = list(expected.get("allowed_statuses", []))
        if any(status not in ALLOWED_STATUSES for status in statuses):
            raise ValueError(f"{case['id']}: unknown allowed status")
        required = dict(expected.get("required_slots", {}))
        unknown_required = set(required) - slot_names
        if unknown_required:
            raise ValueError(f"{case['id']}: unknown required slots {sorted(unknown_required)}")
        # Reuse the production contract solely as a fixture-value validator.
        CommandSlots.from_dict(required)
        forbidden = list(expected.get("forbidden_slots", []))
        if any(name not in slot_names for name in forbidden):
            raise ValueError(f"{case['id']}: unknown forbidden slot")
        max_calls = expected.get("max_modal_calls")
        if max_calls is not None and (isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 0):
            raise ValueError(f"{case['id']}: max_modal_calls must be a non-negative integer")
        manual = bool(case.get("manual_review"))
        manual_count += int(manual)
        if manual and not case.get("review_focus"):
            raise ValueError(f"{case['id']}: manual-review case needs review_focus")
        if not manual and not flows and not statuses and not required and not forbidden:
            raise ValueError(f"{case['id']}: automatic case has no machine-checkable expectation")

    return {
        "version": dataset.get("version"),
        "cases": len(cases),
        "manual_review_cases": manual_count,
        "automatic_only_cases": len(cases) - manual_count,
        "category_counts": dict(actual_counts),
        "frozen_against_main_sha": dataset.get("frozen_against_main_sha"),
    }


def _seed_state(context: str) -> dict[str, Any]:
    if context != "search":
        return {}
    events = event_search.load_events()[:5]
    ids = [str(event.get("id", "")) for event in events if event.get("id")]
    search_context = conversation_recovery.build_search_context(
        "unexpected utterance evaluation seed",
        {},
        events,
        result_ids=ids,
        total_matches=len(ids),
    )
    return {
        "last_result_ids": ids,
        "last_result_count": len(ids),
        "last_action": "find_events",
        "has_last_search_context": True,
        "last_search_context": search_context.to_dict(),
    }


def _router_fast_path(query: str, state: Mapping[str, Any]) -> tuple[str | None, Any]:
    by_id = {
        str(event.get("id")): event
        for event in event_search.load_events()
        if event.get("id")
    }
    previous = [
        by_id.get(str(event_id))
        for event_id in list(state.get("last_result_ids", []))[:MAX_RESULT_SET_SIZE]
    ]
    previous = [event for event in previous if isinstance(event, Mapping)]
    route = route_conversation(
        query,
        previous,
        None,
        state.get("last_filters"),
        POC_REFERENCE_DATE,
    )
    action = getattr(route, "action_type", None)
    detail_field = getattr(route, "detail_field", None)
    if action in {"detail_followup", "explain_search", "explain_result", "clarify_reference"} or (
        action == "reference_followup" and detail_field is not None
    ):
        if action in {"detail_followup", "reference_followup"}:
            action = "event_detail"
        return str(action), route
    return None, route


def _normalize(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in value.items()}
    return value


def _slot_subset(actual: Mapping[str, Any], required: Mapping[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for key, expected_value in required.items():
        actual_value = _normalize(actual.get(key))
        expected_value = _normalize(expected_value)
        if isinstance(expected_value, list):
            if not isinstance(actual_value, list) or not set(expected_value).issubset(set(actual_value)):
                failures.append(f"slot:{key}")
        elif actual_value != expected_value:
            failures.append(f"slot:{key}")
    return not failures, failures


def _forbidden_present(key: str, value: Any) -> bool:
    if value in (None, "", [], (), {}):
        return False
    if isinstance(value, bool):
        if key == "reservation_required":
            return True  # both True and False are meaningful reservation filters
        return value  # other false booleans are effectively absent/default
    return True


def evaluate_case(
    case: Mapping[str, Any],
    invoke: Callable[[Mapping[str, Any]], Any],
    *,
    include_raw: bool = False,
    format_enforcer: str = "baseline",
) -> dict[str, Any]:
    query = str(case.get("query", ""))
    state = _seed_state(str(case.get("context", "none")))
    fast_flow, route = _router_fast_path(query, state)
    remote = RemoteCommandCall(invoke, format_enforcer=format_enforcer)
    result = None
    if fast_flow is not None:
        actual_flow = canonical_flow(fast_flow)
        actual_status = "clarification" if fast_flow == "clarify_reference" else "ok"
        actual_slots: dict[str, Any] = {}
        near_count = 0
        message = ""
    else:
        result = CommandOrchestrator(modal_call=remote, reference_date=POC_REFERENCE_DATE).handle_query(query, state)
        actual_flow = canonical_flow(result.flow)
        actual_status = result.status
        actual_slots = result.command.slots.to_dict()
        near_count = len(result.near_events)
        message = result.message

    expected = dict(case.get("expected", {}))
    checks: dict[str, Any] = {}
    failures: list[str] = []

    allowed_flows = list(expected.get("allowed_flows", []))
    if allowed_flows:
        checks["flow"] = actual_flow in allowed_flows
        if not checks["flow"]:
            failures.append("flow")

    allowed_statuses = list(expected.get("allowed_statuses", []))
    if allowed_statuses and result is not None:
        checks["status"] = actual_status in allowed_statuses
        if not checks["status"]:
            failures.append("status")
    elif allowed_statuses:
        checks["status"] = "skipped_fast_path"

    required_slots = dict(expected.get("required_slots", {}))
    if required_slots and result is not None:
        ok, slot_failures = _slot_subset(actual_slots, required_slots)
        checks["required_slots"] = ok
        failures.extend(slot_failures)
    elif required_slots:
        checks["required_slots"] = "skipped_fast_path"

    forbidden_slots = list(expected.get("forbidden_slots", []))
    if forbidden_slots and result is not None:
        present = [name for name in forbidden_slots if _forbidden_present(name, actual_slots.get(name))]
        checks["forbidden_slots"] = not present
        failures.extend(f"forbidden:{name}" for name in present)
    elif forbidden_slots:
        checks["forbidden_slots"] = "skipped_fast_path"

    max_calls = expected.get("max_modal_calls")
    if max_calls is not None:
        checks["max_modal_calls"] = len(remote.calls) <= int(max_calls)
        if not checks["max_modal_calls"]:
            failures.append("max_modal_calls")

    if expected.get("must_not_auto_relax"):
        checks["no_auto_relax"] = near_count == 0
        if not checks["no_auto_relax"]:
            failures.append("auto_relax")

    machine_pass = not failures
    manual = bool(case.get("manual_review"))
    verdict = "manual_pending" if manual and machine_pass else "fail" if failures else "pass"
    stats = remote.stats()
    if not include_raw:
        stats.pop("raw_output", None)

    return {
        "id": case.get("id"),
        "category": case.get("category"),
        "risk": case.get("risk"),
        "query": query,
        "expected_behavior": case.get("expected_behavior"),
        "actual_flow": actual_flow,
        "actual_status": actual_status,
        "fast_path_hit": fast_flow is not None,
        "route_action": getattr(route, "action_type", None),
        "actual_slots": actual_slots,
        "near_event_count": near_count,
        "modal_calls": len(remote.calls),
        "checks": checks,
        "machine_pass": machine_pass,
        "manual_review": manual,
        "review_focus": list(case.get("review_focus", [])),
        "verdict": verdict,
        "failures": failures,
        "message": str(message)[:1000],
        "stats": stats,
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    category = defaultdict(lambda: {"cases": 0, "machine_pass": 0, "manual_pending": 0})
    risks = Counter()
    for row in rows:
        bucket = category[str(row.get("category"))]
        bucket["cases"] += 1
        bucket["machine_pass"] += int(bool(row.get("machine_pass")))
        bucket["manual_pending"] += int(row.get("verdict") == "manual_pending")
        for failure in row.get("failures", []):
            risks[str(failure)] += 1
    return {
        "cases": len(rows),
        "machine_pass_cases": sum(bool(row.get("machine_pass")) for row in rows),
        "machine_pass_rate": round(sum(bool(row.get("machine_pass")) for row in rows) / len(rows), 4) if rows else 0.0,
        "manual_review_queue": sum(bool(row.get("manual_review")) for row in rows),
        "manual_pending": sum(row.get("verdict") == "manual_pending" for row in rows),
        "category_summary": dict(category),
        "machine_failure_checks": dict(risks),
    }


def evaluate_dataset(
    dataset: Mapping[str, Any],
    invoke: Callable[[Mapping[str, Any]], Any],
    *,
    limit: int = EXPECTED_CASE_COUNT,
    include_raw: bool = False,
    format_enforcer: str = "baseline",
) -> dict[str, Any]:
    contract = validate_dataset(dataset)
    selected = list(dataset.get("cases", []))[: max(0, int(limit))]
    rows = [
        evaluate_case(case, invoke, include_raw,format_enforcer=format_enforcer)
        for case in selected
    ]
    return {
        "dataset": dataset.get("version"),
        "frozen_against_main_sha": dataset.get("frozen_against_main_sha"),
        "reference_date": dataset.get("reference_date"),
        "fixture_contract": contract,
        "summary": summarize(rows),
        "rows": rows,
    }


__all__ = ["evaluate_case", "evaluate_dataset", "load_dataset", "summarize", "validate_dataset"]

if __name__ == "__main__":
    print(json.dumps(validate_dataset(load_dataset()), ensure_ascii=False, indent=2, sort_keys=True))
