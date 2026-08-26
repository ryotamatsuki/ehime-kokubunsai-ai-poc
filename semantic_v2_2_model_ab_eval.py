"""Frozen-v1 live A/B evaluator for Semantic Operations v2.2.

This is an evaluation-only runner.  It does not alter the Semantic Operations
architecture, prompt, schema, verifier, reducer, executor, generation policy,
or the exposed Frozen v1 fixture.  It calls the two already-deployed
authenticated Modal web functions through the existing UI adapter and records
the complete per-case observation needed for a paired model comparison.

The sealed v2.1 payload is deliberately not imported or referenced by the
runner.  Its manifest/hash boundary is checked by the workflow, outside this
module.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from app_config import POC_REFERENCE_DATE
from semantic_atomic_v2_2 import AtomicSemanticFrame, ATOMIC_FRAME_JSON_SCHEMA
from semantic_model_registry import (
    MODEL_BY_KEY,
    MODEL_SPECS,
    SemanticModelSpec,
    resolve_model_url,
)
from semantic_orchestrator_v2_2 import SemanticOperationsOrchestratorV22
from semantic_state_v2_2 import grounded_slots_from_query_v22
from semantic_v2_2_ui import SemanticEndpointConfig, post_atomic_frame
from unexpected_utterances_eval import (
    _forbidden_present,
    _seed_state,
    _slot_subset,
    validate_dataset,
)
from unexpected_utterances_v2_1_eval import (
    ObservableRemoteFrameCall,
    _normalize,
    _latency_summary,
    load_frozen_v1_dataset,
)


EXPECTED_CASES = 100
FORMAT_ENFORCER = "lmfe"
MAX_NEW_TOKENS = 220
REPETITION_PENALTY = 1.02
MODEL_KEYS = ("sarashina-2.2-3b", "llm-jp-4-8b")
ARCHITECTURE = "semantic-operations-v2.2-atomic-fail-soft"
HOLDOUT_PAYLOAD_SHA256 = "c844dda17248c0e7f16cd2985652e62bb0f8b601bf21196d6801479580899c92"
ERROR_CLUSTERS = (
    "intent_error",
    "scope_error",
    "municipality_error",
    "region_error",
    "fee_error",
    "reservation_error",
    "venue_error",
    "rain_error",
    "audience_error",
    "experience_require_error",
    "experience_prefer_error",
    "experience_exclude_error",
    "experience_unset_release_error",
    "clarification_error",
    "data_gap_error",
    "reference_error",
    "context_state_error",
    "flow_status_mismatch",
    "unsupported_inference",
    "verifier_rejection",
    "fail_soft_overuse",
    "fail_soft_underuse",
    "executor_mismatch",
    "json_invalid",
    "schema_invalid",
    "empty_output",
    "truncated_output",
    "timeout",
    "transport_failure",
    "modal_cold_start_failure",
    "application_semantic_failure",
    "other",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return {str(k): _jsonable(v) for k, v in vars(value).items()}
    return str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return _json_default(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _corpus_sha256() -> str:
    """Hash the exact Frozen v1 manifest and ordered case files."""

    root = Path("tests/data/unexpected_utterances_v1")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    for relative in ["manifest.json", *list(manifest.get("files", []))]:
        path = root / str(relative)
        raw = path.read_bytes()
        encoded_name = str(relative).encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _file_hashes() -> dict[str, str]:
    return {
        "prompt_sha256": _sha256_file(Path("semantic_prompt_v2_2.py")),
        "schema_sha256": _sha256_bytes(
            json.dumps(ATOMIC_FRAME_JSON_SCHEMA, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
        "verifier_sha256": _sha256_file(Path("semantic_verifier_v2_2.py")),
        "reducer_sha256": _sha256_file(Path("semantic_state_v2_2.py")),
        "orchestrator_sha256": _sha256_file(Path("semantic_orchestrator_v2_2.py")),
        "atomic_schema_module_sha256": _sha256_file(Path("semantic_atomic_v2_2.py")),
    }


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    data = sorted(float(value) for value in values)
    if not data:
        return None
    if len(data) == 1:
        return round(data[0], 3)
    position = (len(data) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(data) - 1)
    weight = position - lower
    return round(data[lower] * (1.0 - weight) + data[upper] * weight, 3)


def _numeric_summary(values: Sequence[float]) -> dict[str, float | None]:
    data = sorted(float(value) for value in values)
    if not data:
        return {"mean": None, "median": None, "p90": None, "p95": None, "min": None, "max": None}
    return {
        "mean": round(statistics.fmean(data), 3),
        "median": round(statistics.median(data), 3),
        "p90": _percentile(data, 0.90),
        "p95": _percentile(data, 0.95),
        "min": round(data[0], 3),
        "max": round(data[-1], 3),
    }


def _state_summary(state: Mapping[str, Any], context: str) -> dict[str, Any]:
    command = state.get("last_command") if isinstance(state, Mapping) else None
    return {
        "fixture_context": context,
        "has_previous_results": bool(state.get("last_result_ids")) if isinstance(state, Mapping) else False,
        "previous_result_count": len(state.get("last_result_ids", [])) if isinstance(state, Mapping) and isinstance(state.get("last_result_ids"), list) else 0,
        "has_previous_command": isinstance(command, Mapping),
        "previous_active_flow": state.get("active_flow") if isinstance(state, Mapping) else None,
    }


def _raw_frame_details(raw: Any, generated_tokens: int | None) -> dict[str, Any]:
    text = str(raw) if isinstance(raw, str) else ""
    detail: dict[str, Any] = {
        "non_empty": bool(text.strip()),
        "json_valid": False,
        "schema_valid": False,
        "json_object": None,
        "structural_error": None,
        "truncated": False,
    }
    if not text.strip():
        detail["structural_error"] = "empty_output"
        return detail
    try:
        parsed = json.loads(text)
        detail["json_valid"] = True
        detail["json_object"] = parsed
    except (TypeError, ValueError, json.JSONDecodeError):
        if generated_tokens is not None and int(generated_tokens) >= MAX_NEW_TOKENS:
            detail["truncated"] = True
            detail["structural_error"] = "truncated_output"
        else:
            detail["structural_error"] = "json_invalid"
        return detail
    try:
        AtomicSemanticFrame.from_json(text)
        detail["schema_valid"] = True
    except (TypeError, ValueError) as exc:
        if generated_tokens is not None and int(generated_tokens) >= MAX_NEW_TOKENS and not text.rstrip().endswith("}"):
            detail["truncated"] = True
            detail["structural_error"] = "truncated_output"
        else:
            detail["structural_error"] = "schema_invalid"
        detail["schema_error"] = str(exc)[:500]
    return detail


def _execution_summary(result: Any) -> dict[str, Any]:
    command_result = result.command_result
    if command_result is None:
        return {
            "final_operation": None,
            "final_status": result.status,
            "final_flow": result.flow,
            "final_slots": _normalize(result.slots),
            "result_ids": [],
            "result_count": None,
            "near_result_ids": [],
            "clarification": result.message if result.status == "clarification" else None,
            "data_gap": _normalize(result.slots).get("data_gap") if isinstance(result.slots, Mapping) else None,
            "assistant_answer": result.message,
        }
    ids = list(command_result.all_event_ids or [])
    if not ids:
        ids = [str(event.get("id")) for event in command_result.events if isinstance(event, Mapping) and event.get("id")]
    near_ids = list(command_result.all_near_event_ids or [])
    if not near_ids:
        near_ids = [str(event.get("id")) for event in command_result.near_events if isinstance(event, Mapping) and event.get("id")]
    return {
        "final_operation": command_result.flow,
        "final_status": command_result.status,
        "final_flow": command_result.flow,
        "final_slots": _normalize(command_result.slots),
        "result_ids": ids,
        "result_count": command_result.total_matches if command_result.total_matches is not None else len(command_result.events),
        "near_result_ids": near_ids,
        "clarification": command_result.question or (command_result.message if command_result.status == "clarification" else None),
        "data_gap": _normalize(command_result.slots).get("data_gap") if isinstance(command_result.slots, Mapping) else None,
        "assistant_answer": command_result.message or result.message,
    }


def _verification_summary(result: Any) -> dict[str, Any] | None:
    verification = result.verification
    if verification is None:
        return None
    if not verification.accepted:
        status = "rejection"
    elif verification.normalized or verification.ignored:
        status = "correction"
    else:
        status = "pass"
    return {
        "status": status,
        "accepted": bool(verification.accepted),
        "reason": verification.reason,
        "normalized": list(verification.normalized),
        "ignored": list(verification.ignored),
        "accepted_atoms": _normalize(verification.frame.to_dict()) if verification.frame is not None else None,
    }


def _reduction_summary(result: Any) -> dict[str, Any] | None:
    reduction = result.reduction
    if reduction is None:
        return None
    frame = reduction.frame
    experience = frame.to_dict().get("experience", {}) if frame is not None else {}
    return {
        "status": reduction.status,
        "ready": bool(reduction.ready),
        "fail_soft": bool(reduction.fail_soft),
        "fail_soft_reason": reduction.fail_soft_reason,
        "message": reduction.message,
        "grounded_slots": _normalize(reduction.grounded_slots),
        "applied_atoms": _normalize(reduction.applied_atoms),
        "ignored_atoms": list(reduction.ignored_atoms),
        "released_slots": list(reduction.applied_unset),
        "experience_operations": _normalize(experience),
        "final_slots": _normalize(result.slots),
    }


def _map_slot_failure(key: str) -> str:
    if key in {"intent"}:
        return "intent_error"
    if key in {"scope", "refine_previous"}:
        return "scope_error"
    if key in {"municipalities", "municipality"}:
        return "municipality_error"
    if key in {"regions", "region"}:
        return "region_error"
    if key in {"entry_free", "paid_only", "max_entry_fee", "fee"}:
        return "fee_error"
    if key in {"reservation_required", "reservation"}:
        return "reservation_error"
    if key in {"venue"}:
        return "venue_error"
    if key in {"rain_preferred", "rain"}:
        return "rain_error"
    if key in {"audience", "age", "age_group", "age_intent", "audience_mode"}:
        return "audience_error"
    if key == "clarification":
        return "clarification_error"
    if key in {"data_gap"}:
        return "data_gap_error"
    if key.startswith("experience_required"):
        return "experience_require_error"
    if key.startswith("experience_preferred"):
        return "experience_prefer_error"
    if key.startswith("experience_excluded"):
        return "experience_exclude_error"
    if key.startswith("experience") or key in {"release", "unset"}:
        return "experience_unset_release_error"
    if key.startswith("reference"):
        return "reference_error"
    if key in {"last_result_ids", "last_command", "active_flow"}:
        return "context_state_error"
    return "application_semantic_failure"


def _failure_clusters(row: Mapping[str, Any]) -> list[str]:
    clusters: list[str] = []
    for call in row.get("observability", {}).get("calls_detail", []):
        error = str(call.get("error") or "").lower()
        if not error:
            continue
        if "timeout" in error:
            clusters.append("timeout")
        elif "cold" in error or "setup" in error or "load" in error:
            clusters.append("modal_cold_start_failure")
        elif "http" in error or "request" in error or "transport" in error or "redirect" in error:
            clusters.append("transport_failure")
        else:
            clusters.append("application_semantic_failure")

    structural = row.get("structural", {})
    if structural.get("structural_error"):
        clusters.append(str(structural["structural_error"]))
    verification = row.get("semantic_verifier") or {}
    if verification.get("status") == "rejection":
        clusters.append("verifier_rejection")

    for failure in row.get("failures", []):
        text = str(failure)
        if text in {"flow", "status"}:
            clusters.append("flow_status_mismatch")
        elif text.startswith("slot:"):
            clusters.append(_map_slot_failure(text[5:]))
        elif text.startswith("forbidden:"):
            clusters.append(_map_slot_failure(text[10:]))
        elif text == "auto_relax":
            clusters.append("fail_soft_underuse")
        else:
            clusters.append("application_semantic_failure")

    if row.get("frame_fallback"):
        expected_statuses = set(row.get("expected", {}).get("allowed_statuses", []))
        actual_status = row.get("execution", {}).get("final_status")
        if actual_status in {"clarification", "unsupported", "invalid_command"} and actual_status not in expected_statuses:
            clusters.append("fail_soft_overuse")
        elif not row.get("machine_pass"):
            clusters.append("fail_soft_underuse")

    if not row.get("machine_pass") and not clusters:
        clusters.append("other")
    ordered: list[str] = []
    for item in clusters:
        if item not in ordered:
            ordered.append(item)
    return ordered or ([] if row.get("machine_pass") else ["other"])


def _evaluate_case(
    case: Mapping[str, Any],
    spec: SemanticModelSpec,
    config: SemanticEndpointConfig,
    *,
    execution_index: int,
    model_order_index: int,
) -> dict[str, Any]:
    query = str(case.get("query", ""))
    context = str(case.get("context", "none"))
    state = _seed_state(context)
    grounded = grounded_slots_from_query_v22(query, POC_REFERENCE_DATE)
    remote = ObservableRemoteFrameCall(
        lambda payload: post_atomic_frame(config, payload),
        format_enforcer=FORMAT_ENFORCER,
        include_raw=True,
    )
    started_at = _now()
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
        present = [name for name in forbidden_slots if _forbidden_present(str(name), actual_slots.get(str(name)))]
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
    service_calls = telemetry.get("calls_detail", [])
    first_service = service_calls[0].get("service", {}) if service_calls else {}
    raw = service_calls[0].get("raw_output") if service_calls else None
    generated_tokens = first_service.get("generated_tokens") if isinstance(first_service, Mapping) else None
    structural = _raw_frame_details(raw, generated_tokens)
    verification = _verification_summary(result)
    reduction = _reduction_summary(result)
    execution = _execution_summary(result)

    row: dict[str, Any] = {
        "execution_index": execution_index,
        "case_order_index": int(str(case.get("id", "UU-000"))[-3:]),
        "case_id": case.get("id"),
        "category": case.get("category"),
        "risk": case.get("risk"),
        "query": query,
        "context": context,
        "model_key": spec.key,
        "model_id": spec.model_id,
        "endpoint_url": config.url,
        "model_order_index": model_order_index,
        "previous_state_summary": _state_summary(state, context),
        "deterministic_route": result.deterministic_route,
        "deterministic_grounded_slots": _normalize(grounded),
        "model_called": bool(remote.calls),
        "model_call_decision": "model_called" if remote.calls else "zero_model_deterministic",
        "prompt_tokens": int(telemetry.get("prompt_tokens", 0) or 0),
        "generated_tokens": int(telemetry.get("generated_tokens", 0) or 0),
        "raw_model_response": raw,
        "raw_atomic_frame": raw,
        "raw_atomic_frame_object": structural.get("json_object"),
        "structural": structural,
        "atomic_frame_valid": bool(result.frame is not None),
        "verified_atomic_frame": result.frame.to_dict() if result.frame is not None else None,
        "frame_attempts": result.frame_attempts,
        "frame_fallback": result.frame_fallback,
        "frame_error": result.frame_error,
        "semantic_verifier": verification,
        "accepted_atoms": verification.get("accepted_atoms") if verification else None,
        "ignored_atoms": verification.get("ignored", []) if verification else list(result.reduction.ignored_atoms) if result.reduction else [],
        "reducer": reduction,
        "execution": execution,
        "final_operation": execution.get("final_operation"),
        "final_status": execution.get("final_status"),
        "result_ids": execution.get("result_ids", []),
        "result_count": execution.get("result_count"),
        "clarification": execution.get("clarification"),
        "data_gap": execution.get("data_gap"),
        "final_assistant_answer": execution.get("assistant_answer"),
        "orchestrator_latency_ms": round(float(result.latency_ms), 3),
        "observability": telemetry,
        "expected": expected,
        "expected_behavior": case.get("expected_behavior"),
        "review_focus": list(case.get("review_focus", [])),
        "machine_pass": machine_pass,
        "manual_review": manual,
        "manual_verdict": "PENDING" if manual else "NOT_REQUIRED",
        "verdict": verdict,
        "checks": checks,
        "failures": failures,
        "started_at": started_at,
        "finished_at": _now(),
    }
    row["failure_clusters"] = _failure_clusters(row)
    return row


def _model_latency(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    called = [row for row in rows if row.get("model_called")]
    generation = [float(row.get("observability", {}).get("generation_ms", 0.0) or 0.0) for row in called]
    client = [float(row.get("observability", {}).get("client_elapsed_ms", 0.0) or 0.0) for row in called]
    server = [float(row.get("observability", {}).get("server_total_ms", 0.0) or 0.0) for row in called]
    first_generation = generation[:1]
    warm_generation = generation[1:]
    setup = [
        float(row.get("observability", {}).get("calls_detail", [{}])[0].get("service", {}).get("container_setup_ms", 0.0) or 0.0)
        for row in called
        if row.get("observability", {}).get("calls_detail")
    ]
    return {
        "model_inference_generation_ms": _numeric_summary(generation),
        "client_request_ms": _numeric_summary(client),
        "server_total_ms": _numeric_summary(server),
        "first_model_call_generation_ms": _numeric_summary(first_generation),
        "subsequent_model_call_generation_ms": _numeric_summary(warm_generation),
        "reported_container_setup_ms": _numeric_summary(setup),
    }


def _model_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    called = [row for row in rows if row.get("model_called")]
    clusters = Counter(cluster for row in rows for cluster in row.get("failure_clusters", []))
    category: dict[str, dict[str, Any]] = defaultdict(lambda: {"cases": 0, "machine_pass": 0, "manual_pending": 0, "model_called": 0})
    for row in rows:
        bucket = category[str(row.get("category"))]
        bucket["cases"] += 1
        bucket["machine_pass"] += int(bool(row.get("machine_pass")))
        bucket["manual_pending"] += int(row.get("manual_verdict") == "PENDING")
        bucket["model_called"] += int(bool(row.get("model_called")))
    verifier = Counter((row.get("semantic_verifier") or {}).get("status") for row in rows if row.get("semantic_verifier"))
    structural_valid = sum(int(bool(row.get("structural", {}).get("schema_valid"))) for row in called)
    json_valid = sum(int(bool(row.get("structural", {}).get("json_valid"))) for row in called)
    empty = sum(int(row.get("structural", {}).get("structural_error") == "empty_output") for row in called)
    invalid = sum(int(row.get("structural", {}).get("structural_error") in {"json_invalid", "schema_invalid"}) for row in called)
    truncated = sum(int(row.get("structural", {}).get("structural_error") == "truncated_output") for row in called)
    prompt_tokens = sum(int(row.get("prompt_tokens", 0) or 0) for row in rows)
    generated_tokens = sum(int(row.get("generated_tokens", 0) or 0) for row in rows)
    return {
        "cases": len(rows),
        "machine_pass": sum(int(bool(row.get("machine_pass"))) for row in rows),
        "machine_pass_rate": round(sum(int(bool(row.get("machine_pass"))) for row in rows) / len(rows), 4) if rows else 0.0,
        "manual_review_cases": sum(int(bool(row.get("manual_review"))) for row in rows),
        "manual": {
            "pass": None,
            "borderline": None,
            "fail": None,
            "pending": sum(int(bool(row.get("manual_review"))) for row in rows),
        },
        "model_called_cases": len(called),
        "zero_model_deterministic_cases": len(rows) - len(called),
        "total_calls": sum(int(row.get("observability", {}).get("calls", 0) or 0) for row in rows),
        "calls_per_case": round(sum(int(row.get("observability", {}).get("calls", 0) or 0) for row in rows) / len(rows), 4) if rows else 0.0,
        "json_parse_success": json_valid,
        "atomic_schema_valid": structural_valid,
        "structural_valid_denominator": len(called),
        "invalid_frame": invalid,
        "empty_output": empty,
        "truncated_output": truncated,
        "verifier": dict(verifier),
        "fail_soft_cases": sum(int(bool(row.get("frame_fallback"))) for row in rows),
        "total_prompt_tokens": prompt_tokens,
        "mean_prompt_tokens_per_model_call": round(prompt_tokens / len(called), 3) if called else 0.0,
        "total_generated_tokens": generated_tokens,
        "mean_generated_tokens_per_model_call": round(generated_tokens / len(called), 3) if called else 0.0,
        "latency": _model_latency(rows),
        "category_summary": dict(category),
        "failure_clusters": dict(clusters),
    }


def _exact_mcnemar_p(b: int, c: int) -> float | None:
    n = int(b) + int(c)
    if n == 0:
        return None
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n)
    return round(min(1.0, 2.0 * tail), 8)


def _frame_diff(left: Mapping[str, Any] | None, right: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return {"available": False, "reason": "one_or_both_raw_frames_not_json_objects"}
    fields = ["intent", "scope", "municipality", "region", "fee", "reservation", "venue", "rain", "audience_mode", "clarification", "data_gap", "experience"]
    diff: dict[str, Any] = {"available": True, "changed": {}}
    for field in fields:
        if left.get(field) != right.get(field):
            diff["changed"][field] = {"sarashina": left.get(field), "llm_jp": right.get(field)}
    return diff


def _paired(rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    left = {str(row["case_id"]): row for row in rows_by_model["sarashina-2.2-3b"]}
    right = {str(row["case_id"]): row for row in rows_by_model["llm-jp-4-8b"]}
    pair_rows: list[dict[str, Any]] = []
    for case_id in sorted(left, key=lambda value: int(value.split("-")[-1])):
        a, b = left[case_id], right[case_id]
        ap, bp = bool(a.get("machine_pass")), bool(b.get("machine_pass"))
        if ap and bp:
            bucket = "both_pass"
        elif ap and not bp:
            bucket = "sarashina_only_pass"
        elif not ap and bp:
            bucket = "llm_jp_only_pass"
        else:
            bucket = "both_fail"
        pair_rows.append({
            "case_id": case_id,
            "category": a.get("category"),
            "query": a.get("query"),
            "sarashina_pass": ap,
            "llm_jp_pass": bp,
            "bucket": bucket,
            "sarashina_failure_clusters": a.get("failure_clusters", []),
            "llm_jp_failure_clusters": b.get("failure_clusters", []),
            "sarashina_raw_frame": a.get("raw_atomic_frame"),
            "llm_jp_raw_frame": b.get("raw_atomic_frame"),
            "atom_diff": _frame_diff(a.get("raw_atomic_frame_object"), b.get("raw_atomic_frame_object")),
        })
    counts = Counter(row["bucket"] for row in pair_rows)
    b = counts.get("sarashina_only_pass", 0)
    c = counts.get("llm_jp_only_pass", 0)
    return {
        "counts": {
            "both_pass": counts.get("both_pass", 0),
            "sarashina_only_pass": b,
            "llm_jp_only_pass": c,
            "both_fail": counts.get("both_fail", 0),
        },
        "mcnemar_exact": {
            "sarashina_only_pass_b": b,
            "llm_jp_only_pass_c": c,
            "discordant_pairs": b + c,
            "two_sided_p_value": _exact_mcnemar_p(b, c),
        },
        "improved_cases": [row for row in pair_rows if row["bucket"] == "llm_jp_only_pass"],
        "regressed_cases": [row for row in pair_rows if row["bucket"] == "sarashina_only_pass"],
        "rows": pair_rows,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")


def _write_paired_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = [
        "case_id", "category", "query", "sarashina_pass", "llm_jp_pass", "bucket",
        "sarashina_primary_failure", "llm_jp_primary_failure",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "case_id": row.get("case_id"),
                "category": row.get("category"),
                "query": row.get("query"),
                "sarashina_pass": row.get("sarashina_pass"),
                "llm_jp_pass": row.get("llm_jp_pass"),
                "bucket": row.get("bucket"),
                "sarashina_primary_failure": (row.get("sarashina_failure_clusters") or [""])[0],
                "llm_jp_primary_failure": (row.get("llm_jp_failure_clusters") or [""])[0],
            })


def _report(
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    paired: Mapping[str, Any],
    *,
    valid: bool = True,
    abort_reason: str | None = None,
) -> str:
    a = summary.get("models", {}).get("sarashina-2.2-3b", {})
    b = summary.get("models", {}).get("llm-jp-4-8b", {})
    counts = paired.get("counts", {})
    diagnosis = "PENDING_MANUAL_REVIEW"
    if not valid:
        diagnosis = "D_INCONCLUSIVE_ABORTED"
    lines = [
        "# Semantic Operations v2.2 Frozen v1 Live A/B Evaluation",
        "",
        "## 1. Conclusion",
        "",
        f"Evaluation validity: **{'VALID' if valid else 'INVALID / ABORTED'}**",
        f"Sarashina 2.2 3B: **{a.get('machine_pass', '—')}/100 machine PASS**",
        f"LLM-jp 4 8B: **{b.get('machine_pass', '—')}/100 machine PASS**",
        f"Decision: **{diagnosis}**",
        "",
        "Manual review is recorded as PENDING in this first artifact. The final report is updated only after the 33 manual-review cases per model are reviewed under the same rubric.",
        *( [f"Abort reason: {abort_reason}"] if abort_reason else [] ),
        "",
        "## 2. Evaluation integrity",
        "",
        f"- Architecture frozen SHA: `{manifest.get('architecture_frozen_sha')}`",
        f"- Evaluation branch: `{manifest.get('evaluation_branch')}`",
        f"- Frozen v1 corpus: `{manifest.get('frozen_v1_version')}`, {manifest.get('cases')} cases",
        f"- Frozen v1 corpus SHA-256: `{manifest.get('frozen_v1_corpus_sha256')}`",
        "- Same prompt, schema, LMFE, verifier, reducer, executor, generation settings, order policy, and scorer",
        "- No prompt/rule/few-shot/architecture/expected-result tuning during the run",
        "- Sealed v2.1 200-case payload: NOT OPENED / NOT RUN",
        "",
        "## 3. Total result",
        "",
        "| Model | Machine PASS | Model-called | Zero-call deterministic | Structural valid | Median generation ms | p95 generation ms | Generated tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Sarashina 2.2 3B | {a.get('machine_pass', '—')}/100 | {a.get('model_called_cases', '—')} | {a.get('zero_model_deterministic_cases', '—')} | {a.get('atomic_schema_valid', '—')}/{a.get('structural_valid_denominator', '—')} | {((a.get('latency') or {}).get('model_inference_generation_ms') or {}).get('median', '—')} | {((a.get('latency') or {}).get('model_inference_generation_ms') or {}).get('p95', '—')} | {a.get('total_generated_tokens', '—')} |",
        f"| LLM-jp 4 8B | {b.get('machine_pass', '—')}/100 | {b.get('model_called_cases', '—')} | {b.get('zero_model_deterministic_cases', '—')} | {b.get('atomic_schema_valid', '—')}/{b.get('structural_valid_denominator', '—')} | {((b.get('latency') or {}).get('model_inference_generation_ms') or {}).get('median', '—')} | {((b.get('latency') or {}).get('model_inference_generation_ms') or {}).get('p95', '—')} | {b.get('total_generated_tokens', '—')} |",
        "",
        "## 4. Category table",
        "",
        "| Category | Cases | Sarashina | LLM-jp | Difference |",
        "|---|---:|---:|---:|---:|",
    ]
    cats = sorted(set((a.get("category_summary") or {}) | (b.get("category_summary") or {})))
    for cat in cats:
        ac = (a.get("category_summary") or {}).get(cat, {})
        bc = (b.get("category_summary") or {}).get(cat, {})
        lines.append(f"| {cat} | {ac.get('cases', bc.get('cases', '—'))} | {ac.get('machine_pass', '—')} | {bc.get('machine_pass', '—')} | {bc.get('machine_pass', 0) - ac.get('machine_pass', 0)} |")
    lines.extend([
        "",
        "## 5. Pairwise result",
        "",
        f"- Both PASS: {counts.get('both_pass', '—')}",
        f"- Sarashina only PASS: {counts.get('sarashina_only_pass', '—')}",
        f"- LLM-jp only PASS: {counts.get('llm_jp_only_pass', '—')}",
        f"- Both FAIL: {counts.get('both_fail', '—')}",
        f"- McNemar exact two-sided p-value: {(paired.get('mcnemar_exact') or {}).get('two_sided_p_value', '—')}",
        "",
        "## 6. Structural validity",
        "",
        f"- Sarashina: JSON `{a.get('json_parse_success', '—')}`, Atomic schema `{a.get('atomic_schema_valid', '—')}/{a.get('structural_valid_denominator', '—')}`, invalid `{a.get('invalid_frame', '—')}`, empty `{a.get('empty_output', '—')}`, truncated `{a.get('truncated_output', '—')}`",
        f"- LLM-jp: JSON `{b.get('json_parse_success', '—')}`, Atomic schema `{b.get('atomic_schema_valid', '—')}/{b.get('structural_valid_denominator', '—')}`, invalid `{b.get('invalid_frame', '—')}`, empty `{b.get('empty_output', '—')}`, truncated `{b.get('truncated_output', '—')}`",
        "",
        "## 7. Failure clusters",
        "",
        f"- Sarashina: `{json.dumps(a.get('failure_clusters', {}), ensure_ascii=False, sort_keys=True)}`",
        f"- LLM-jp: `{json.dumps(b.get('failure_clusters', {}), ensure_ascii=False, sort_keys=True)}`",
        "",
        "## 8. Latency / tokens",
        "",
        f"- Sarashina latency: `{json.dumps(a.get('latency', {}), ensure_ascii=False, sort_keys=True)}`",
        f"- LLM-jp latency: `{json.dumps(b.get('latency', {}), ensure_ascii=False, sort_keys=True)}`",
        f"- Sarashina tokens: prompt total `{a.get('total_prompt_tokens', '—')}`, generated total `{a.get('total_generated_tokens', '—')}`",
        f"- LLM-jp tokens: prompt total `{b.get('total_prompt_tokens', '—')}`, generated total `{b.get('total_generated_tokens', '—')}`",
        "",
        "## 9. Important improved cases",
        "",
        f"LLM-jp-only PASS cases: `{', '.join(row.get('case_id', '') for row in paired.get('improved_cases', [])) or 'none'}`",
        "The final report must explain the changed raw Atomic atoms for each materially relevant pair; it must not infer that the improvement is caused by parameter count alone.",
        "",
        "## 10. Important regressions",
        "",
        f"Sarashina-only PASS cases: `{', '.join(row.get('case_id', '') for row in paired.get('regressed_cases', [])) or 'none'}`",
        "",
        "## 11. Architecture vs model bottleneck diagnosis",
        "",
        "Pending completion of manual review and failure-cluster inspection. Shared failures with valid raw frames will be treated as architecture/data-semantics candidates; model-specific atom differences will be treated as model-sensitive evidence.",
        "",
        "## 12. Recommended next action",
        "",
        "Do not merge PR #45 or run the sealed holdout until the final A/B report has been reviewed and one of A/B/C/D is selected.",
        "",
        "## Final required one-screen numbers",
        "",
        f"- Sarashina: Machine PASS {a.get('machine_pass', '—')}/100; Manual PASS/BORDERLINE/FAIL pending; Structural {a.get('atomic_schema_valid', '—')}/{a.get('structural_valid_denominator', '—')}; Median {((a.get('latency') or {}).get('model_inference_generation_ms') or {}).get('median', '—')} ms; p95 {((a.get('latency') or {}).get('model_inference_generation_ms') or {}).get('p95', '—')} ms; Generated tokens {a.get('total_generated_tokens', '—')}",
        f"- LLM-jp: Machine PASS {b.get('machine_pass', '—')}/100; Manual PASS/BORDERLINE/FAIL pending; Structural {b.get('atomic_schema_valid', '—')}/{b.get('structural_valid_denominator', '—')}; Median {((b.get('latency') or {}).get('model_inference_generation_ms') or {}).get('median', '—')} ms; p95 {((b.get('latency') or {}).get('model_inference_generation_ms') or {}).get('p95', '—')} ms; Generated tokens {b.get('total_generated_tokens', '—')}",
        f"- Pairwise: Both {counts.get('both_pass', '—')}; Sarashina-only {counts.get('sarashina_only_pass', '—')}; LLM-jp-only {counts.get('llm_jp_only_pass', '—')}; Both fail {counts.get('both_fail', '—')}",
        f"- Final diagnosis: {diagnosis}",
    ])
    return "\n".join(lines) + "\n"


def _manifest(out_dir: Path, *, architecture_frozen_sha: str, evaluation_branch: str, smoke: Mapping[str, Any], started: str, finished: str) -> dict[str, Any]:
    hashes = _file_hashes()
    return {
        "repository": os.environ.get("GITHUB_REPOSITORY", "ryotamatsuki/ehime-kokubunsai-ai-poc"),
        "evaluation_branch": evaluation_branch,
        "architecture_frozen_sha": architecture_frozen_sha,
        "frozen_v1_version": "unexpected-user-utterances-v1",
        "frozen_v1_cases": EXPECTED_CASES,
        "frozen_v1_corpus_sha256": _corpus_sha256(),
        **hashes,
        "model_ids": {spec.key: spec.model_id for spec in MODEL_SPECS},
        "model_keys": list(MODEL_KEYS),
        "generation_settings": {
            "format_enforcer": FORMAT_ENFORCER,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "repetition_penalty": REPETITION_PENALTY,
            "repair_generation": False,
            "model_calls_per_residual_turn_max": 1,
        },
        "retry_policy": {
            "semantic_quality_retry": False,
            "transport_retry": False,
            "modal_redirect_hops_max": 4,
            "one_request_per_model_called_case": True,
        },
        "prompt_contract": {
            "same_prompt": True,
            "same_few_shot": True,
            "same_schema": True,
            "same_lmfe": True,
            "same_verifier_reducer_executor": True,
        },
        "order_policy": "odd_case_sarashina_then_llmjp; even_case_llmjp_then_sarashina",
        "modal_endpoint_identifiers": {
            spec.key: {
                "url": spec.endpoint_url,
                "model_key": spec.key,
                "model_id": spec.model_id,
            }
            for spec in MODEL_SPECS
        },
        "smoke": smoke,
        "start_timestamp": started,
        "end_timestamp": finished,
        "holdout_opened": False,
        "holdout_run": False,
        "holdout_payload_sha256": HOLDOUT_PAYLOAD_SHA256,
        "production_main_modified": False,
        "production_deployed": False,
    }


def _smoke(spec: SemanticModelSpec, config: SemanticEndpointConfig) -> dict[str, Any]:
    started = time.perf_counter()
    body = post_atomic_frame(
        config,
        {
            "query": "松山市で無料の屋内イベントを探して",
            "state": {},
            "grounded": {},
        },
    )
    answer = body.get("answer") if isinstance(body, Mapping) else None
    parsed = AtomicSemanticFrame.from_json(str(answer)) if isinstance(answer, str) else None
    obs = body.get("observability", {}) if isinstance(body, Mapping) else {}
    assert body.get("service_id") == "ehime-kokubunsai-semantic-v2-2-api"
    assert body.get("model_key") == spec.key
    assert body.get("model_id") == spec.model_id
    assert isinstance(answer, str) and answer.strip()
    assert parsed is not None
    assert obs.get("format_enforcer") == FORMAT_ENFORCER
    return {
        "excluded_from_score": True,
        "model_key": spec.key,
        "model_id": spec.model_id,
        "endpoint_url": config.url,
        "http_success": True,
        "non_empty_answer": True,
        "json_parse_success": True,
        "atomic_schema_valid": True,
        "lmfe": obs.get("format_enforcer") == FORMAT_ENFORCER,
        "prompt_tokens": obs.get("prompt_tokens"),
        "generated_tokens": obs.get("generated_tokens"),
        "container_setup_ms": obs.get("container_setup_ms"),
        "model_load_ms": obs.get("model_load_ms"),
        "server_inference_ms": obs.get("generation_ms"),
        "client_elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "cache_verified_via_successful_model_load": bool(obs.get("model_load_ms") is not None),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="artifacts/semantic-v2-2-frozen-v1-ab")
    parser.add_argument("--architecture-frozen-sha", required=True)
    parser.add_argument("--evaluation-branch", default=os.environ.get("GITHUB_REF_NAME", "unknown"))
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = _now()
    secret_map = dict(os.environ)
    modal_key = str(os.environ.get("MODAL_KEY", "")).strip()
    modal_secret = str(os.environ.get("MODAL_SECRET", "")).strip()
    if not modal_key or not modal_secret:
        raise RuntimeError("MODAL_KEY and MODAL_SECRET are required")

    configs = {
        spec.key: SemanticEndpointConfig(
            model=spec,
            url=resolve_model_url(spec, secret_map),
            key=modal_key,
            secret=modal_secret,
        )
        for spec in MODEL_SPECS
    }
    if args.skip_smoke:
        smoke_path = out_dir / "smoke.json"
        if not smoke_path.exists():
            raise RuntimeError("--skip-smoke requires a completed smoke.json")
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    else:
        smoke = {}
        for spec in MODEL_SPECS:
            smoke[spec.key] = _smoke(spec, configs[spec.key])
        _write_json(out_dir / "smoke.json", smoke)
    if args.smoke_only:
        return 0

    dataset = load_frozen_v1_dataset()
    contract = validate_dataset(dataset)
    if int(contract.get("cases", 0)) != EXPECTED_CASES:
        raise RuntimeError(f"Frozen v1 contract is not 100 cases: {contract}")
    cases = list(dataset.get("cases", []))
    if len(cases) != EXPECTED_CASES:
        raise RuntimeError("Frozen v1 case count drift")

    rows_by_model: dict[str, list[dict[str, Any]]] = {key: [] for key in MODEL_KEYS}
    execution_index = 0
    raw_paths = {key: out_dir / ("sarashina_raw.jsonl" if key == "sarashina-2.2-3b" else "llmjp_raw.jsonl") for key in MODEL_KEYS}
    handles = {key: path.open("w", encoding="utf-8") for key, path in raw_paths.items()}
    try:
        for case in cases:
            numeric_id = int(str(case.get("id", "UU-000"))[-3:])
            ordered_specs = list(MODEL_SPECS) if numeric_id % 2 else list(reversed(MODEL_SPECS))
            for model_order_index, spec in enumerate(ordered_specs):
                execution_index += 1
                row = _evaluate_case(
                    case,
                    spec,
                    configs[spec.key],
                    execution_index=execution_index,
                    model_order_index=model_order_index,
                )
                rows_by_model[spec.key].append(row)
                handles[spec.key].write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")
                handles[spec.key].flush()
    finally:
        for handle in handles.values():
            handle.close()

    finished = _now()
    paired = _paired(rows_by_model)
    summary = {
        "valid": True,
        "architecture": ARCHITECTURE,
        "models": {key: _model_summary(rows_by_model[key]) for key in MODEL_KEYS},
        "paired": paired,
        "manual_review": {
            "rubric": "PASS/BORDERLINE/FAIL for all manual_review=true cases, response-level review for unsupported inference, fabricated facts, guarantees, and UX quality",
            "cases_per_model": sum(int(bool(row.get("manual_review"))) for row in rows_by_model[MODEL_KEYS[0]]),
            "status": "PENDING",
        },
    }
    manifest = _manifest(
        out_dir,
        architecture_frozen_sha=args.architecture_frozen_sha,
        evaluation_branch=args.evaluation_branch,
        smoke=smoke,
        started=started,
        finished=finished,
    )
    manifest["fixture_contract"] = contract
    _write_json(out_dir / "manifest.json", manifest)
    _write_json(out_dir / "environment.json", {
        "python": sys.version,
        "github_sha": os.environ.get("GITHUB_SHA"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "modal_endpoint_urls": {key: configs[key].url for key in MODEL_KEYS},
        "architecture_frozen_sha": args.architecture_frozen_sha,
        "holdout_opened": False,
        "holdout_payload_sha256": HOLDOUT_PAYLOAD_SHA256,
    })
    _write_json(out_dir / "summary.json", summary)
    _write_json(out_dir / "failure_clusters.json", {key: summary["models"][key]["failure_clusters"] for key in MODEL_KEYS})
    _write_json(out_dir / "latency_tokens.json", {key: {
        "latency": summary["models"][key]["latency"],
        "tokens": {
            "total_prompt_tokens": summary["models"][key]["total_prompt_tokens"],
            "mean_prompt_tokens_per_model_call": summary["models"][key]["mean_prompt_tokens_per_model_call"],
            "total_generated_tokens": summary["models"][key]["total_generated_tokens"],
            "mean_generated_tokens_per_model_call": summary["models"][key]["mean_generated_tokens_per_model_call"],
        },
    } for key in MODEL_KEYS})
    _write_jsonl(out_dir / "sarashina_raw.jsonl", rows_by_model["sarashina-2.2-3b"])
    _write_jsonl(out_dir / "llmjp_raw.jsonl", rows_by_model["llm-jp-4-8b"])
    _write_paired_csv(out_dir / "paired_results.csv", paired["rows"])
    (out_dir / "REPORT.md").write_text(_report(manifest, summary, paired), encoding="utf-8")
    _write_json(out_dir / "evaluation_status.json", {
        "valid": True,
        "formal_cases": EXPECTED_CASES,
        "model_conditions": EXPECTED_CASES * 2,
        "started": started,
        "finished": finished,
        "holdout_opened": False,
        "holdout_run": False,
    })
    print(json.dumps({
        "valid": True,
        "sarashina": summary["models"]["sarashina-2.2-3b"]["machine_pass"],
        "llm_jp": summary["models"]["llm-jp-4-8b"]["machine_pass"],
        "pairwise": paired["counts"],
        "holdout_opened": False,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
