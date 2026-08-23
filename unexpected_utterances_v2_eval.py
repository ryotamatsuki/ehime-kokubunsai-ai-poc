"""Frozen Unexpected User Utterances v1 evaluator for Semantic Operations v2.

This evaluator is deliberately parallel to the baseline evaluator. It consumes
exactly the same frozen 100 cases and scoring expectations, but routes each
utterance through ``SemanticOperationsOrchestratorV2``.

No production tuning should be performed while a v2 run is in progress.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable, Mapping, Sequence

from app_config import POC_REFERENCE_DATE
from semantic_orchestrator_v2 import SemanticOperationsOrchestratorV2
from unexpected_utterances_eval import (
    _forbidden_present,
    _seed_state,
    _slot_subset,
    load_dataset,
    validate_dataset,
)


class RemoteFrameCall:
    def __init__(
        self,
        invoke: Callable[[Mapping[str, Any]], Any],
        *,
        format_enforcer: str = "lmfe",
        include_raw: bool = False,
    ) -> None:
        self.invoke = invoke
        self.format_enforcer = format_enforcer
        self.include_raw = include_raw
        self.calls: list[dict[str, Any]] = []

    def __call__(self, payload: Mapping[str, Any]) -> Any:
        request = dict(payload)
        request["format_enforcer"] = self.format_enforcer
        response = self.invoke(request)
        raw_answer = (
            response.get("answer")
            if isinstance(response, Mapping) and isinstance(response.get("answer"), str)
            else response if isinstance(response, str) else None
        )
        self.calls.append(
            {
                "repair": bool(request.get("repair")),
                "raw_output": raw_answer if self.include_raw else None,
            }
        )
        return response

    def stats(self) -> dict[str, Any]:
        first = self.calls[0] if self.calls else None
        second = self.calls[1] if len(self.calls) > 1 else None
        return {
            "calls": len(self.calls),
            "first_pass_returned_text": bool(first and first.get("raw_output") is not None)
            if self.include_raw
            else bool(first),
            "repair_called": bool(second),
        }


def _normalize(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in value.items()}
    return value


def evaluate_case_v2(
    case: Mapping[str, Any],
    invoke: Callable[[Mapping[str, Any]], Any],
    *,
    include_raw: bool = False,
    format_enforcer: str = "lmfe",
) -> dict[str, Any]:
    query = str(case.get("query", ""))
    state = _seed_state(str(case.get("context", "none")))
    remote = RemoteFrameCall(
        invoke,
        format_enforcer=format_enforcer,
        include_raw=include_raw,
    )
    result = SemanticOperationsOrchestratorV2(
        frame_call=remote,
        reference_date=POC_REFERENCE_DATE,
    ).handle_query(query, state)

    actual_flow = result.flow
    actual_status = result.status
    actual_slots = _normalize(result.slots)
    near_count = len(result.near_events)

    expected = dict(case.get("expected", {}))
    checks: dict[str, Any] = {}
    failures: list[str] = []

    allowed_flows = list(expected.get("allowed_flows", []))
    if allowed_flows:
        checks["flow"] = actual_flow in allowed_flows
        if not checks["flow"]:
            failures.append("flow")

    allowed_statuses = list(expected.get("allowed_statuses", []))
    if allowed_statuses:
        checks["status"] = actual_status in allowed_statuses
        if not checks["status"]:
            failures.append("status")

    required_slots = dict(expected.get("required_slots", {}))
    if required_slots:
        ok, slot_failures = _slot_subset(actual_slots, required_slots)
        checks["required_slots"] = ok
        failures.extend(slot_failures)

    forbidden_slots = list(expected.get("forbidden_slots", []))
    if forbidden_slots:
        present = [
            name
            for name in forbidden_slots
            if _forbidden_present(name, actual_slots.get(name))
        ]
        checks["forbidden_slots"] = not present
        failures.extend(f"forbidden:{name}" for name in present)

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

    frame = result.frame
    first_pass_frame_valid = frame is not None and result.frame_attempts == 1
    repair_success = frame is not None and result.frame_attempts == 2

    return {
        "id": case.get("id"),
        "category": case.get("category"),
        "risk": case.get("risk"),
        "query": query,
        "expected_behavior": case.get("expected_behavior"),
        "actual_flow": actual_flow,
        "actual_status": actual_status,
        "actual_slots": actual_slots,
        "near_event_count": near_count,
        "frame_calls": len(remote.calls),
        "frame_attempts": result.frame_attempts,
        "frame_repaired": result.frame_repaired,
        "first_pass_frame_valid": first_pass_frame_valid,
        "repair_success": repair_success,
        "deterministic_route": result.deterministic_route,
        "frame": frame.to_dict() if frame is not None else None,
        "reduction_status": result.reduction.status if result.reduction is not None else None,
        "applied_release_groups": (
            list(result.reduction.applied_release_groups)
            if result.reduction is not None
            else []
        ),
        "checks": checks,
        "machine_pass": machine_pass,
        "manual_review": manual,
        "review_focus": list(case.get("review_focus", [])),
        "verdict": verdict,
        "failures": failures,
        "message": str(result.message)[:1000],
        "stats": remote.stats(),
    }


def summarize_v2(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    category = defaultdict(
        lambda: {
            "cases": 0,
            "machine_pass": 0,
            "manual_pending": 0,
            "frame_calls": 0,
        }
    )
    failure_checks = Counter()
    for row in rows:
        bucket = category[str(row.get("category"))]
        bucket["cases"] += 1
        bucket["machine_pass"] += int(bool(row.get("machine_pass")))
        bucket["manual_pending"] += int(row.get("verdict") == "manual_pending")
        bucket["frame_calls"] += int(row.get("frame_calls", 0))
        for failure in row.get("failures", []):
            failure_checks[str(failure)] += 1

    model_rows = [row for row in rows if int(row.get("frame_attempts", 0)) > 0]
    return {
        "cases": len(rows),
        "machine_pass_cases": sum(bool(row.get("machine_pass")) for row in rows),
        "machine_pass_rate": round(
            sum(bool(row.get("machine_pass")) for row in rows) / len(rows),
            4,
        )
        if rows
        else 0.0,
        "manual_review_queue": sum(bool(row.get("manual_review")) for row in rows),
        "manual_pending": sum(row.get("verdict") == "manual_pending" for row in rows),
        "frame_model_cases": len(model_rows),
        "first_pass_frame_valid": sum(
            bool(row.get("first_pass_frame_valid")) for row in model_rows
        ),
        "repair_success": sum(bool(row.get("repair_success")) for row in model_rows),
        "total_frame_calls": sum(int(row.get("frame_calls", 0)) for row in rows),
        "category_summary": dict(category),
        "machine_failure_checks": dict(failure_checks),
    }


def evaluate_dataset_v2(
    dataset: Mapping[str, Any],
    invoke: Callable[[Mapping[str, Any]], Any],
    *,
    limit: int = 100,
    include_raw: bool = False,
    format_enforcer: str = "lmfe",
) -> dict[str, Any]:
    contract = validate_dataset(dataset)
    selected = list(dataset.get("cases", []))[: max(0, int(limit))]
    rows = [
        evaluate_case_v2(
            case,
            invoke,
            include_raw=include_raw,
            format_enforcer=format_enforcer,
        )
        for case in selected
    ]
    return {
        "architecture": "semantic-operations-v2",
        "dataset": dataset.get("version"),
        "frozen_against_main_sha": dataset.get("frozen_against_main_sha"),
        "reference_date": dataset.get("reference_date"),
        "fixture_contract": contract,
        "format_enforcer": format_enforcer,
        "summary": summarize_v2(rows),
        "rows": rows,
    }


__all__ = [
    "RemoteFrameCall",
    "evaluate_case_v2",
    "evaluate_dataset_v2",
    "load_dataset",
    "summarize_v2",
]
