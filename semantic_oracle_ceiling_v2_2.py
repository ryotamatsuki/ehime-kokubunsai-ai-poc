"""Oracle ceiling for Semantic Operations v2.2 on the exposed frozen-v1 set.

The sealed 200-case holdout is never imported or opened. The oracle converts
the already-existing v2.1 oracle meaning into the smaller v2.2 atomic contract;
therefore any failure identifies an actual representational/routing ceiling in
v2.2 rather than Sarashina generation quality.
"""

from __future__ import annotations

from collections import Counter
import json
from typing import Any, Mapping

from semantic_atomic_v2_2 import ATOMIC_INTENT_VALUES, AtomicSemanticFrame, EXPERIENCE_CONCEPTS, neutral_experience
from semantic_oracle_ceiling_v2_1 import oracle_sparse_frame
from semantic_orchestrator_v2_2 import SemanticOperationsOrchestratorV22
from unexpected_utterances_eval import _forbidden_present, _seed_state, _slot_subset, load_dataset


def oracle_atomic_frame(case: Mapping[str, Any]) -> AtomicSemanticFrame:
    sparse = oracle_sparse_frame(case)
    set_slots = dict(sparse.set_slots or {})
    experience = neutral_experience()
    for concept in sparse.require:
        experience[concept] = "require"
    for concept in sparse.prefer:
        experience[concept] = "prefer"
    for concept in sparse.exclude:
        experience[concept] = "exclude"
    for token in sparse.unset:
        if token == "experience_all":
            for concept in EXPERIENCE_CONCEPTS:
                experience[concept] = "unset"
        elif token.startswith("experience:"):
            concept = token.split(":", 1)[1]
            if concept in experience:
                experience[concept] = "unset"

    municipality = "none"
    region = "none"
    fee = "none"
    reservation = "none"
    venue = "none"
    rain = "none"
    audience_mode = sparse.audience_mode or "none"

    if "location" in sparse.unset:
        municipality = "release"
    elif set_slots.get("municipalities"):
        municipality = str(list(set_slots["municipalities"])[0])
    elif set_slots.get("regions"):
        region = str(list(set_slots["regions"])[0])

    if "fee" in sparse.unset:
        fee = "release"
    elif set_slots.get("entry_free") is True:
        fee = "free"
    elif set_slots.get("paid_only") is True:
        fee = "paid"

    if "reservation" in sparse.unset:
        reservation = "release"
    elif set_slots.get("reservation_required") is True:
        reservation = "required"
    elif set_slots.get("reservation_required") is False:
        reservation = "not_required"

    if "venue" in sparse.unset:
        venue = "release"
    elif set_slots.get("venue") in {"indoor", "outdoor"}:
        venue = str(set_slots["venue"])

    if "rain" in sparse.unset:
        rain = "release"
    elif set_slots.get("rain_preferred") is True:
        rain = "prefer"

    if "age" in sparse.unset:
        audience_mode = "release"

    intent = sparse.intent if sparse.intent in ATOMIC_INTENT_VALUES else "search"
    clarification = sparse.clarification or "none"
    data_gap = sparse.data_gap or "none"

    return AtomicSemanticFrame(
        intent=intent,
        scope=sparse.scope,
        municipality=municipality,
        region=region,
        fee=fee,
        reservation=reservation,
        venue=venue,
        rain=rain,
        audience_mode=audience_mode,
        clarification=clarification,
        data_gap=data_gap,
        experience=experience,
    )


def evaluate_oracle_v22(dataset: Mapping[str, Any] | None = None) -> dict[str, Any]:
    dataset = dict(dataset or load_dataset())
    rows: list[dict[str, Any]] = []
    for case in dataset.get("cases", []):
        frame = oracle_atomic_frame(case)
        calls = 0

        def invoke(_: Mapping[str, Any], *, _frame: AtomicSemanticFrame = frame) -> Mapping[str, str]:
            nonlocal calls
            calls += 1
            return {"answer": json.dumps(_frame.to_dict(), ensure_ascii=False, separators=(",", ":"))}

        result = SemanticOperationsOrchestratorV22(frame_call=invoke).handle_query(
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
        })

    categories = Counter(str(row["category"]) for row in rows)
    category_pass = Counter(str(row["category"]) for row in rows if row["pass"])
    failure_checks = Counter(item for row in rows for item in row["failures"])
    passed = sum(bool(row["pass"]) for row in rows)
    return {
        "architecture": "semantic-operations-v2.2-atomic-oracle-ceiling",
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
            }
            for row in rows if not row["pass"]
        ],
        "zero_model_call_cases": sum(row["model_calls"] == 0 for row in rows),
        "fail_soft_cases": sum(bool(row["frame_fallback"]) for row in rows),
    }


if __name__ == "__main__":
    print(json.dumps(evaluate_oracle_v22(), ensure_ascii=False, indent=2, sort_keys=True))
