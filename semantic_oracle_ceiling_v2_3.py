"""Oracle ceiling for Semantic Operations v2.3 on exposed frozen-v1 only.

This module never imports or opens the sealed 200-case holdout.  It translates
the existing specification-authored v2.2 oracle meaning into the v2.3 closed
EvidenceRequest contract and sends it through the real verifier/state machine.
"""

from __future__ import annotations

from collections import Counter
import json
from typing import Any, Mapping

from semantic_atomic_v2_3 import ATOMIC_INTENT_VALUES_V23, AtomicSemanticFrameV23
from semantic_evidence_v2_3 import EvidenceRequest
from semantic_oracle_ceiling_v2_2 import oracle_atomic_frame
from semantic_orchestrator_v2_3 import SemanticOperationsOrchestratorV23
from unexpected_utterances_eval import _forbidden_present, _seed_state, _slot_subset, load_dataset


_DATA_GAP_EVIDENCE = {
    "crowding": EvidenceRequest.REALTIME_STATE,
    "parking_distance": EvidenceRequest.EXTERNAL_LOGISTICS,
    "toilet_proximity": EvidenceRequest.EXTERNAL_LOGISTICS,
    "weather_guarantee": EvidenceRequest.ABSOLUTE_GUARANTEE,
    "popularity": EvidenceRequest.SUBJECTIVE_JUDGMENT,
    "fame": EvidenceRequest.SUBJECTIVE_JUDGMENT,
    "noise": EvidenceRequest.UNSUPPORTED_FACT,
    "wheelchair_access": EvidenceRequest.UNSUPPORTED_FACT,
    "medical_safety": EvidenceRequest.UNSUPPORTED_FACT,
    "duration_fit": EvidenceRequest.UNSUPPORTED_FACT,
    "localness": EvidenceRequest.UNSUPPORTED_FACT,
    "social_fit": EvidenceRequest.RELATIONAL_SUITABILITY,
    "other": EvidenceRequest.UNKNOWN_CAPABILITY,
}


def oracle_atomic_frame_v23(case: Mapping[str, Any]) -> AtomicSemanticFrameV23:
    old = oracle_atomic_frame(case)
    if old.clarification == "ambiguous_suitability":
        evidence = EvidenceRequest.RELATIONAL_SUITABILITY
    elif old.data_gap != "none":
        evidence = _DATA_GAP_EVIDENCE.get(old.data_gap, EvidenceRequest.UNKNOWN_CAPABILITY)
    else:
        has_supported_atom = any(
            value != "none"
            for value in (
                old.municipality, old.region, old.fee, old.reservation,
                old.venue, old.rain, old.audience_mode,
            )
        ) or any(value != "none" for value in old.experience.values())
        evidence = EvidenceRequest.SUPPORTED_ATTRIBUTE if has_supported_atom else EvidenceRequest.NONE

    intent = old.intent if old.intent in ATOMIC_INTENT_VALUES_V23 else "search"
    return AtomicSemanticFrameV23(
        intent=intent,
        scope=old.scope,
        evidence_request=evidence.value,
        municipality=old.municipality,
        region=old.region,
        fee=old.fee,
        reservation=old.reservation,
        venue=old.venue,
        rain=old.rain,
        audience_mode=old.audience_mode,
        experience=dict(old.experience),
    )


def evaluate_oracle_v23(dataset: Mapping[str, Any] | None = None) -> dict[str, Any]:
    dataset = dict(dataset or load_dataset())
    rows: list[dict[str, Any]] = []
    for case in dataset.get("cases", []):
        frame = oracle_atomic_frame_v23(case)
        calls = 0

        def invoke(_: Mapping[str, Any], *, _frame: AtomicSemanticFrameV23 = frame) -> Mapping[str, str]:
            nonlocal calls
            calls += 1
            return {"answer": json.dumps(_frame.to_dict(), ensure_ascii=False, separators=(",", ":"))}

        result = SemanticOperationsOrchestratorV23(frame_call=invoke).handle_query(
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
            "id": case.get("id"),
            "category": case.get("category"),
            "pass": not failures,
            "failures": failures,
            "flow": result.flow,
            "status": result.status,
            "slots": result.slots,
            "model_calls": calls,
            "route": result.deterministic_route,
            "frame_fallback": result.frame_fallback,
            "evidence_request": result.evidence_request,
            "rejected_atoms": list(result.rejected_atoms),
        })

    categories = Counter(str(row["category"]) for row in rows)
    category_pass = Counter(str(row["category"]) for row in rows if row["pass"])
    failure_checks = Counter(item for row in rows for item in row["failures"])
    passed = sum(bool(row["pass"]) for row in rows)
    return {
        "architecture": "semantic-operations-v2.3-evidence-bounded-oracle-ceiling",
        "dataset": dataset.get("version"),
        "cases": len(rows),
        "machine_pass": passed,
        "machine_pass_rate": round(passed / len(rows), 4) if rows else 0.0,
        "category": {name: {"cases": count, "pass": category_pass[name]} for name, count in sorted(categories.items())},
        "failure_checks": dict(failure_checks),
        "failed": [
            {
                "id": row["id"], "failures": row["failures"], "route": row["route"],
                "flow": row["flow"], "status": row["status"], "slots": row["slots"],
                "evidence_request": row["evidence_request"], "rejected_atoms": row["rejected_atoms"],
            }
            for row in rows if not row["pass"]
        ],
        "zero_model_call_cases": sum(row["model_calls"] == 0 for row in rows),
        "fail_soft_cases": sum(bool(row["frame_fallback"]) for row in rows),
    }


if __name__ == "__main__":
    print(json.dumps(evaluate_oracle_v23(), ensure_ascii=False, indent=2, sort_keys=True))


__all__ = ["evaluate_oracle_v23", "oracle_atomic_frame_v23"]
