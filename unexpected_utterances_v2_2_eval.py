"""Observable frozen-v1 evaluator for Semantic Operations v2.2.

This evaluator is deliberately restricted to the already-exposed 100-case v1
regression suite. It does not discover, import, decompress or execute the
sealed 200-case holdout. The live backend may attach uncertainty telemetry,
but no uncertainty threshold affects behavior in this phase.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import statistics
from typing import Any, Callable, Mapping, Sequence

from app_config import POC_REFERENCE_DATE
from semantic_orchestrator_v2_2 import SemanticOperationsOrchestratorV22
from unexpected_utterances_eval import _forbidden_present, _seed_state, _slot_subset, validate_dataset
from unexpected_utterances_v2_1_eval import (
    ObservableRemoteFrameCall,
    _latency_summary,
    _normalize,
    load_frozen_v1_dataset,
)


def _numeric_summary(values: Sequence[float]) -> dict[str, float | None]:
    data = sorted(float(value) for value in values)
    if not data:
        return {"mean": None, "median": None, "p95": None, "min": None, "max": None}
    if len(data) == 1:
        p95 = data[0]
    else:
        position = (len(data) - 1) * 0.95
        lower = int(position)
        upper = min(lower + 1, len(data) - 1)
        weight = position - lower
        p95 = data[lower] * (1.0 - weight) + data[upper] * weight
    return {
        "mean": round(statistics.fmean(data), 6),
        "median": round(statistics.median(data), 6),
        "p95": round(p95, 6),
        "min": round(data[0], 6),
        "max": round(data[-1], 6),
    }


def evaluate_case_v22(
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
    result = SemanticOperationsOrchestratorV22(
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
    telemetry = remote.stats()
    verification = result.verification.to_dict() if result.verification is not None else None

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
        "frame_fallback": result.frame_fallback,
        "frame_error": result.frame_error,
        "atomic_frame_valid": result.frame is not None,
        "frame": result.frame.to_dict() if result.frame is not None else None,
        "semantic_verifier": verification,
        "reduction_status": result.reduction.status if result.reduction is not None else None,
        "grounded_slots": _normalize(result.reduction.grounded_slots) if result.reduction is not None else None,
        "applied_atoms": _normalize(result.reduction.applied_atoms) if result.reduction is not None else None,
        "ignored_atoms": list(result.reduction.ignored_atoms) if result.reduction is not None else [],
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


def summarize_v22(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
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
    passed = sum(bool(row.get("machine_pass")) for row in rows)
    verifier_rows = [row for row in model_rows if isinstance(row.get("semantic_verifier"), Mapping)]

    uncertainty_fields = 0
    uncertainty_margins: list[float] = []
    for obs in model_obs:
        for call in obs.get("calls_detail", []):
            service = call.get("service", {}) if isinstance(call, Mapping) else {}
            uncertainty = service.get("uncertainty", {}) if isinstance(service, Mapping) else {}
            fields = uncertainty.get("fields", {}) if isinstance(uncertainty, Mapping) else {}
            if isinstance(fields, Mapping):
                uncertainty_fields += len(fields)
                for item in fields.values():
                    if isinstance(item, Mapping) and item.get("min_margin") is not None:
                        uncertainty_margins.append(float(item["min_margin"]))

    return {
        "cases": len(rows),
        "machine_pass_cases": passed,
        "machine_pass_rate": round(passed / len(rows), 4) if rows else 0.0,
        "manual_review_queue": sum(bool(row.get("manual_review")) for row in rows),
        "manual_pending": sum(row.get("verdict") == "manual_pending" for row in rows),
        "frame_model_cases": len(model_rows),
        "zero_model_call_cases": len(rows) - len(model_rows),
        "valid_atomic_frames": sum(bool(row.get("atomic_frame_valid")) for row in model_rows),
        "semantic_verifier_cases": len(verifier_rows),
        "semantic_verifier_rejected": sum(
            row.get("semantic_verifier", {}).get("accepted") is False for row in verifier_rows
        ),
        "semantic_verifier_normalized": sum(
            bool(row.get("semantic_verifier", {}).get("normalized")) for row in verifier_rows
        ),
        "fail_soft_cases": sum(bool(row.get("frame_fallback")) for row in model_rows),
        "total_frame_calls": sum(int(obs.get("calls", 0)) for obs in model_obs),
        "repair_calls": 0,
        "prompt_tokens": sum(int(obs.get("prompt_tokens", 0)) for obs in model_obs),
        "generated_tokens": sum(int(obs.get("generated_tokens", 0)) for obs in model_obs),
        "uncertainty": {
            "fields_observed": uncertainty_fields,
            "field_margin_samples": len(uncertainty_margins),
            "field_margin": _numeric_summary(uncertainty_margins),
            "behavioral_threshold_enabled": False,
        },
        "latency": {
            "orchestrator": _latency_summary([float(row.get("orchestrator_latency_ms", 0.0)) for row in rows]),
            "remote_client_model_cases": _latency_summary([float(obs.get("client_elapsed_ms", 0.0)) for obs in model_obs]),
            "server_total_model_cases": _latency_summary([float(obs.get("server_total_ms", 0.0)) for obs in model_obs]),
            "generation_model_cases": _latency_summary([float(obs.get("generation_ms", 0.0)) for obs in model_obs]),
        },
        "category_summary": dict(category),
        "machine_failure_checks": dict(failures),
    }


def evaluate_frozen_v1_v22(
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
        evaluate_case_v22(case, invoke, include_raw=include_raw, format_enforcer=format_enforcer)
        for case in selected
    ]
    return {
        "architecture": "semantic-operations-v2.2-atomic-fail-soft",
        "dataset": dataset.get("version"),
        "frozen_against_main_sha": dataset.get("frozen_against_main_sha"),
        "reference_date": dataset.get("reference_date"),
        "fixture_contract": contract,
        "format_enforcer": format_enforcer,
        "raw_frame_capture": bool(include_raw),
        "uncertainty_is_observability_only": True,
        "summary": summarize_v22(rows),
        "rows": rows,
    }


__all__ = ["evaluate_case_v22", "evaluate_frozen_v1_v22", "summarize_v22"]
