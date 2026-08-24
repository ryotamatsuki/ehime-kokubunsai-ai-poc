"""Observable frozen-v1 evaluator for Semantic Operations v2.1.

This module intentionally evaluates only the already-exposed Unexpected User
Utterances v1 regression set.  It never discovers or opens the sealed v2.1
holdout.  The live harness injects Sarashina frame generation; this evaluator
records raw frame output (when enabled), client/server latency and token counts
without changing the scoring contract.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import statistics
import time
from typing import Any, Callable, Mapping, Sequence

from app_config import POC_REFERENCE_DATE
from semantic_orchestrator_v2_1 import SemanticOperationsOrchestratorV21
from unexpected_utterances_eval import (
    _forbidden_present,
    _seed_state,
    _slot_subset,
    load_dataset,
    validate_dataset,
)


FROZEN_V1_MANIFEST = "tests/data/unexpected_utterances_v1/manifest.json"
FROZEN_V1_VERSION = "unexpected-user-utterances-v1"


class ObservableRemoteFrameCall:
    """Wrap one remote frame backend and preserve evaluation telemetry."""

    def __init__(
        self,
        invoke: Callable[[Mapping[str, Any]], Any],
        *,
        format_enforcer: str = "lmfe",
        include_raw: bool = True,
    ) -> None:
        self.invoke = invoke
        self.format_enforcer = format_enforcer
        self.include_raw = include_raw
        self.calls: list[dict[str, Any]] = []

    def __call__(self, payload: Mapping[str, Any]) -> Any:
        request = dict(payload)
        request["format_enforcer"] = self.format_enforcer
        started = time.perf_counter()
        try:
            response = self.invoke(request)
            invoke_error = None
        except Exception as exc:
            response = None
            invoke_error = f"{type(exc).__name__}: {exc}"[:500]
        client_elapsed_ms = (time.perf_counter() - started) * 1000

        answer = (
            response.get("answer")
            if isinstance(response, Mapping) and isinstance(response.get("answer"), str)
            else response if isinstance(response, str) else None
        )
        service_observability = (
            dict(response.get("observability", {}))
            if isinstance(response, Mapping) and isinstance(response.get("observability"), Mapping)
            else {}
        )
        response_error = (
            str(response.get("error"))[:500]
            if isinstance(response, Mapping) and response.get("error")
            else invoke_error
        )
        self.calls.append(
            {
                "attempt": len(self.calls) + 1,
                "repair": bool(request.get("repair")),
                "raw_output": answer if self.include_raw else None,
                "client_elapsed_ms": round(client_elapsed_ms, 3),
                "service": service_observability,
                "error": response_error,
            }
        )
        if invoke_error is not None:
            raise RuntimeError(invoke_error)
        return response

    def stats(self) -> dict[str, Any]:
        server_total = [
            float(call["service"].get("server_total_ms", 0.0))
            for call in self.calls
            if call.get("service", {}).get("server_total_ms") is not None
        ]
        generation = [
            float(call["service"].get("generation_ms", 0.0))
            for call in self.calls
            if call.get("service", {}).get("generation_ms") is not None
        ]
        prompt_tokens = sum(
            int(call.get("service", {}).get("prompt_tokens", 0) or 0)
            for call in self.calls
        )
        generated_tokens = sum(
            int(call.get("service", {}).get("generated_tokens", 0) or 0)
            for call in self.calls
        )
        return {
            "calls": len(self.calls),
            "repair_called": any(bool(call.get("repair")) for call in self.calls),
            "client_elapsed_ms": round(sum(float(call.get("client_elapsed_ms", 0.0)) for call in self.calls), 3),
            "server_total_ms": round(sum(server_total), 3),
            "generation_ms": round(sum(generation), 3),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "raw_capture": self.include_raw,
            "calls_detail": list(self.calls),
        }


def load_frozen_v1_dataset() -> dict[str, Any]:
    dataset = dict(load_dataset(FROZEN_V1_MANIFEST))
    if dataset.get("version") != FROZEN_V1_VERSION:
        raise ValueError("unexpected frozen-v1 dataset version")
    contract = validate_dataset(dataset)
    if int(contract.get("total_cases", 0)) != 100:
        raise ValueError("frozen-v1 dataset must contain exactly 100 cases")
    return dataset


def _normalize(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in value.items()}
    return value


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return round(ordered[0], 3)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 3)


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    data = [float(value) for value in values]
    if not data:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "mean_ms": round(statistics.fmean(data), 3),
        "median_ms": round(statistics.median(data), 3),
        "p95_ms": _percentile(data, 0.95),
        "max_ms": round(max(data), 3),
    }


def evaluate_case_v21(
    case: Mapping[str, Any],
    invoke: Callable[[Mapping[str, Any]], Any],
    *,
    include_raw: bool = True,
    format_enforcer: str = "lmfe",
) -> dict[str, Any]:
    query = str(case.get("query", ""))
    state = _seed_state(str(case.get("context", "none")))
    remote = ObservableRemoteFrameCall(
        invoke,
        format_enforcer=format_enforcer,
        include_raw=include_raw,
    )
    result = SemanticOperationsOrchestratorV21(
        frame_call=remote,
        reference_date=POC_REFERENCE_DATE,
    ).handle_query(query, state)

    actual_slots = _normalize(result.slots)
    expected = dict(case.get("expected", {}))
    failures: list[str] = []
    checks: dict[str, Any] = {}

    flows = list(expected.get("allowed_flows", []))
    if flows:
        checks["flow"] = result.flow in flows
        if not checks["flow"]:
            failures.append("flow")

    statuses = list(expected.get("allowed_statuses", []))
    if statuses:
        checks["status"] = result.status in statuses
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
            name for name in forbidden_slots
            if _forbidden_present(str(name), actual_slots.get(str(name)))
        ]
        checks["forbidden_slots"] = not present
        failures.extend(f"forbidden:{name}" for name in present)

    max_calls = expected.get("max_modal_calls")
    if max_calls is not None:
        checks["max_modal_calls"] = len(remote.calls) <= int(max_calls)
        if not checks["max_modal_calls"]:
            failures.append("max_modal_calls")

    if expected.get("must_not_auto_relax"):
        checks["no_auto_relax"] = len(result.near_events) == 0
        if not checks["no_auto_relax"]:
            failures.append("auto_relax")

    machine_pass = not failures
    manual = bool(case.get("manual_review"))
    verdict = "manual_pending" if manual and machine_pass else "fail" if failures else "pass"
    frame = result.frame
    telemetry = remote.stats()

    return {
        "id": case.get("id"),
        "category": case.get("category"),
        "risk": case.get("risk"),
        "query": query,
        "expected_behavior": case.get("expected_behavior"),
        "actual_flow": result.flow,
        "actual_status": result.status,
        "actual_slots": actual_slots,
        "near_event_count": len(result.near_events),
        "deterministic_route": result.deterministic_route,
        "frame_attempts": result.frame_attempts,
        "frame_repaired": result.frame_repaired,
        "frame_error": result.frame_error,
        "first_pass_frame_valid": frame is not None and result.frame_attempts == 1,
        "repair_success": frame is not None and result.frame_attempts == 2,
        "frame": frame.to_dict(sparse=True) if frame is not None else None,
        "reduction_status": result.reduction.status if result.reduction is not None else None,
        "grounded_slots": _normalize(result.reduction.grounded_slots) if result.reduction is not None else None,
        "applied_unset": list(result.reduction.applied_unset) if result.reduction is not None else [],
        "orchestrator_latency_ms": round(float(result.latency_ms), 3),
        "observability": telemetry,
        "checks": checks,
        "machine_pass": machine_pass,
        "manual_review": manual,
        "review_focus": list(case.get("review_focus", [])),
        "verdict": verdict,
        "failures": failures,
        "message": str(result.message)[:1000],
    }


def summarize_v21(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    category = defaultdict(lambda: {"cases": 0, "machine_pass": 0, "manual_pending": 0, "frame_calls": 0})
    failures = Counter()
    for row in rows:
        bucket = category[str(row.get("category"))]
        bucket["cases"] += 1
        bucket["machine_pass"] += int(bool(row.get("machine_pass")))
        bucket["manual_pending"] += int(row.get("verdict") == "manual_pending")
        bucket["frame_calls"] += int(row.get("observability", {}).get("calls", 0))
        for failure in row.get("failures", []):
            failures[str(failure)] += 1

    model_rows = [row for row in rows if int(row.get("frame_attempts", 0)) > 0]
    model_obs = [row.get("observability", {}) for row in model_rows]
    total_calls = sum(int(obs.get("calls", 0)) for obs in model_obs)
    repair_calls = sum(int(obs.get("calls", 0)) - 1 for obs in model_obs if int(obs.get("calls", 0)) > 1)
    passed = sum(bool(row.get("machine_pass")) for row in rows)

    return {
        "cases": len(rows),
        "machine_pass_cases": passed,
        "machine_pass_rate": round(passed / len(rows), 4) if rows else 0.0,
        "manual_review_queue": sum(bool(row.get("manual_review")) for row in rows),
        "manual_pending": sum(row.get("verdict") == "manual_pending" for row in rows),
        "frame_model_cases": len(model_rows),
        "zero_model_call_cases": len(rows) - len(model_rows),
        "first_pass_frame_valid": sum(bool(row.get("first_pass_frame_valid")) for row in model_rows),
        "repair_success": sum(bool(row.get("repair_success")) for row in model_rows),
        "total_frame_calls": total_calls,
        "repair_calls": repair_calls,
        "repair_rate_per_model_case": round(repair_calls / len(model_rows), 4) if model_rows else 0.0,
        "prompt_tokens": sum(int(obs.get("prompt_tokens", 0)) for obs in model_obs),
        "generated_tokens": sum(int(obs.get("generated_tokens", 0)) for obs in model_obs),
        "latency": {
            "orchestrator": _latency_summary([float(row.get("orchestrator_latency_ms", 0.0)) for row in rows]),
            "remote_client_model_cases": _latency_summary([float(obs.get("client_elapsed_ms", 0.0)) for obs in model_obs]),
            "server_total_model_cases": _latency_summary([float(obs.get("server_total_ms", 0.0)) for obs in model_obs]),
            "generation_model_cases": _latency_summary([float(obs.get("generation_ms", 0.0)) for obs in model_obs]),
        },
        "category_summary": dict(category),
        "machine_failure_checks": dict(failures),
    }


def evaluate_frozen_v1_v21(
    invoke: Callable[[Mapping[str, Any]], Any],
    *,
    limit: int = 100,
    include_raw: bool = True,
    format_enforcer: str = "lmfe",
) -> dict[str, Any]:
    if format_enforcer not in {"lmfe", "baseline"}:
        raise ValueError("format_enforcer must be lmfe or baseline")
    dataset = load_frozen_v1_dataset()
    contract = validate_dataset(dataset)
    selected = list(dataset.get("cases", []))[: max(0, min(100, int(limit)))]
    rows = [
        evaluate_case_v21(case, invoke, include_raw=include_raw, format_enforcer=format_enforcer)
        for case in selected
    ]
    return {
        "architecture": "semantic-operations-v2.1",
        "dataset": dataset.get("version"),
        "frozen_against_main_sha": dataset.get("frozen_against_main_sha"),
        "reference_date": dataset.get("reference_date"),
        "fixture_contract": contract,
        "format_enforcer": format_enforcer,
        "raw_frame_capture": bool(include_raw),
        "summary": summarize_v21(rows),
        "rows": rows,
    }


__all__ = [
    "FROZEN_V1_MANIFEST",
    "FROZEN_V1_VERSION",
    "ObservableRemoteFrameCall",
    "evaluate_case_v21",
    "evaluate_frozen_v1_v21",
    "load_frozen_v1_dataset",
    "summarize_v21",
]
