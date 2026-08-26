"""Frozen-v1 evaluator extensions for Evidence-Bounded Semantics v2.3.

The evaluator only loads the already-exposed frozen-v1 development/regression
corpus.  It never imports, decompresses, searches or executes the sealed
holdout.  Evidence-boundary violations are first-class machine failures.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

from app_config import POC_REFERENCE_DATE
from semantic_capability_registry_v2_3 import lookup_capability
from semantic_evidence_v2_3 import AllowedSemanticAction, EvidenceRequest
from semantic_orchestrator_v2_3 import SemanticOperationsOrchestratorV23
from unexpected_utterances_eval import _forbidden_present, _seed_state, _slot_subset, validate_dataset
from unexpected_utterances_v2_1_eval import ObservableRemoteFrameCall, _latency_summary, _normalize, load_frozen_v1_dataset


class ManualRubricVerdict(str, Enum):
    PASS = "PASS"
    BORDERLINE = "BORDERLINE"
    FAIL = "FAIL"


def evidence_boundary_checks(result) -> tuple[dict[str, bool], list[str]]:
    checks: dict[str, bool] = {}
    failures: list[str] = []
    verification = result.verification

    # An ungrounded atom can be attempted by a model, but v2.3 must never
    # adopt it into the trusted reducer.  rejected_atoms are therefore not a
    # failure by themselves; they are evidence that the guard did its job.
    adopted_unsupported = 0
    silent_coercion = 0
    if verification is not None and verification.frame is not None:
        rejected = set(verification.rejected_atoms)
        for atom in verification.accepted_atoms:
            if atom in rejected:
                adopted_unsupported += 1
        # A non-coercible request may coexist with an independently grounded
        # explicit condition.  Only an adopted atom lacking proof is a silent
        # coercion; the verifier makes this count structurally zero.
        silent_coercion = adopted_unsupported

    checks["unsupported_inference"] = adopted_unsupported == 0
    checks["silent_coercion"] = silent_coercion == 0
    if adopted_unsupported:
        failures.append("unsupported_inference")
    if silent_coercion:
        failures.append("silent_coercion")

    action = None
    if result.frame is not None:
        action = lookup_capability(result.frame.evidence_request).allowed_semantic_action
    missed_data_gap = action is AllowedSemanticAction.DATA_GAP and not bool(result.data_gap_reason)
    false_data_gap = action is AllowedSemanticAction.SEARCH and bool(result.data_gap_reason)
    clarification_required = action is AllowedSemanticAction.CLARIFY
    clarification_actual = bool(result.clarification_reason)

    checks["missed_data_gap"] = not missed_data_gap
    checks["false_data_gap"] = not false_data_gap
    checks["clarification_precision"] = not (clarification_actual and not clarification_required and result.clarification_reason != "fail_soft")
    checks["clarification_recall"] = not (clarification_required and not clarification_actual)
    if missed_data_gap:
        failures.append("missed_data_gap")
    if false_data_gap:
        failures.append("false_data_gap")
    if not checks["clarification_precision"]:
        failures.append("false_clarification")
    if not checks["clarification_recall"]:
        failures.append("missed_clarification")
    return checks, failures


def manual_review_rubric(row: Mapping[str, Any]) -> ManualRubricVerdict:
    failures = set(str(item) for item in row.get("failures", []))
    hard = {
        "unsupported_inference", "silent_coercion", "missed_data_gap",
        "guarantee_fabrication", "unsupported_realtime_assertion",
    }
    if failures & hard:
        return ManualRubricVerdict.FAIL
    if failures:
        return ManualRubricVerdict.BORDERLINE
    return ManualRubricVerdict.PASS


def evaluate_case_v23(
    case: Mapping[str, Any],
    invoke: Callable[[Mapping[str, Any]], Any],
    *,
    include_raw: bool = True,
    format_enforcer: str = "lmfe",
) -> dict[str, Any]:
    query = str(case.get("query", ""))
    state = _seed_state(str(case.get("context", "none")))
    remote = ObservableRemoteFrameCall(invoke, format_enforcer=format_enforcer, include_raw=include_raw)
    result = SemanticOperationsOrchestratorV23(frame_call=remote, reference_date=POC_REFERENCE_DATE).handle_query(query, state)

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

    boundary_checks, boundary_failures = evidence_boundary_checks(result)
    checks.update(boundary_checks)
    failures.extend(boundary_failures)
    failures = list(dict.fromkeys(failures))
    machine_pass = not failures
    manual = bool(case.get("manual_review"))
    verdict = "manual_pending" if manual and machine_pass else "fail" if failures else "pass"
    verification = result.verification.to_dict() if result.verification is not None else None

    row = {
        "id": case.get("id"),
        "category": case.get("category"),
        "risk": case.get("risk"),
        "query": query,
        "expected_behavior": case.get("expected_behavior"),
        "actual_flow": result.flow,
        "actual_status": result.status,
        "actual_slots": actual_slots,
        "evidence_request": result.evidence_request,
        "capability": result.capability.to_dict() if result.capability is not None else None,
        "clarification_reason": result.clarification_reason,
        "data_gap_reason": result.data_gap_reason,
        "semantic_verifier": verification,
        "accepted_atoms": list(result.accepted_atoms),
        "ignored_atoms": list(result.ignored_atoms),
        "rejected_atoms": list(result.rejected_atoms),
        "silent_coercion_prevented_count": result.verification.silent_coercion_prevented_count if result.verification else 0,
        "unsupported_inference_prevented_count": result.verification.unsupported_inference_count if result.verification else 0,
        "release_operations": list(result.release_operations),
        "frame_attempts": result.frame_attempts,
        "frame_fallback": result.frame_fallback,
        "frame_error": result.frame_error,
        "atomic_frame_valid": result.frame is not None,
        "frame": result.frame.to_dict() if result.frame is not None else None,
        "deterministic_grounding": _normalize(result.deterministic_grounding),
        "orchestrator_latency_ms": round(float(result.latency_ms), 3),
        "observability": remote.stats(),
        "checks": checks,
        "machine_pass": machine_pass,
        "manual_review": manual,
        "review_focus": list(case.get("review_focus", [])),
        "verdict": verdict,
        "failures": failures,
        "message": str(result.message)[:1000],
    }
    row["manual_rubric"] = manual_review_rubric(row).value
    return row


def summarize_v23(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    category = defaultdict(lambda: {"cases": 0, "machine_pass": 0})
    failures = Counter()
    for row in rows:
        bucket = category[str(row.get("category"))]
        bucket["cases"] += 1
        bucket["machine_pass"] += int(bool(row.get("machine_pass")))
        failures.update(str(item) for item in row.get("failures", []))
    passed = sum(bool(row.get("machine_pass")) for row in rows)
    model_rows = [row for row in rows if int(row.get("frame_attempts", 0)) > 0]
    clarification_required = [row for row in model_rows if (row.get("capability") or {}).get("allowed_semantic_action") == "clarify"]
    clarification_actual = [row for row in model_rows if row.get("clarification_reason")]
    data_gap_required = [row for row in model_rows if (row.get("capability") or {}).get("allowed_semantic_action") == "data_gap"]
    return {
        "cases": len(rows),
        "machine_pass_cases": passed,
        "machine_pass_rate": round(passed / len(rows), 4) if rows else 0.0,
        "unsupported_inference_count": sum("unsupported_inference" in row.get("failures", []) for row in rows),
        "silent_coercion_count": sum("silent_coercion" in row.get("failures", []) for row in rows),
        "missed_data_gap_count": sum("missed_data_gap" in row.get("failures", []) for row in rows),
        "false_data_gap_count": sum("false_data_gap" in row.get("failures", []) for row in rows),
        "clarification_precision": round(sum(bool(row.get("clarification_reason")) and (row.get("capability") or {}).get("allowed_semantic_action") == "clarify" for row in model_rows) / len(clarification_actual), 4) if clarification_actual else 1.0,
        "clarification_recall": round(sum(bool(row.get("clarification_reason")) for row in clarification_required) / len(clarification_required), 4) if clarification_required else 1.0,
        "semantic_constraint_accuracy": round(sum(not any(str(f).startswith("required_slots") or str(f).startswith("forbidden:") for f in row.get("failures", [])) for row in rows) / len(rows), 4) if rows else 0.0,
        "silent_coercion_prevented_count": sum(int(row.get("silent_coercion_prevented_count", 0)) for row in rows),
        "unsupported_inference_prevented_count": sum(int(row.get("unsupported_inference_prevented_count", 0)) for row in rows),
        "total_frame_calls": sum(int((row.get("observability") or {}).get("calls", 0)) for row in model_rows),
        "repair_calls": 0,
        "prompt_tokens": sum(int((row.get("observability") or {}).get("prompt_tokens", 0)) for row in model_rows),
        "generated_tokens": sum(int((row.get("observability") or {}).get("generated_tokens", 0)) for row in model_rows),
        "latency": {"orchestrator": _latency_summary([float(row.get("orchestrator_latency_ms", 0.0)) for row in rows])},
        "data_gap_required_cases": len(data_gap_required),
        "category_summary": dict(category),
        "machine_failure_checks": dict(failures),
    }


def evaluate_frozen_v1_v23(
    invoke: Callable[[Mapping[str, Any]], Any],
    *,
    limit: int = 100,
    include_raw: bool = True,
    format_enforcer: str = "lmfe",
) -> dict[str, Any]:
    if format_enforcer != "lmfe":
        raise ValueError("v2.3 frozen evaluation requires lmfe")
    dataset = load_frozen_v1_dataset()
    contract = validate_dataset(dataset)
    selected = list(dataset.get("cases", []))[: max(0, min(100, int(limit)))]
    rows = [evaluate_case_v23(case, invoke, include_raw=include_raw, format_enforcer=format_enforcer) for case in selected]
    return {
        "architecture": "semantic-operations-v2.3-evidence-bounded",
        "dataset": dataset.get("version"),
        "fixture_contract": contract,
        "format_enforcer": format_enforcer,
        "raw_frame_capture": bool(include_raw),
        "summary": summarize_v23(rows),
        "rows": rows,
    }


__all__ = [
    "ManualRubricVerdict",
    "evaluate_case_v23",
    "evaluate_frozen_v1_v23",
    "evidence_boundary_checks",
    "manual_review_rubric",
    "summarize_v23",
]
